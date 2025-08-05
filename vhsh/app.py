import sys
import time
from typing import Generic, Callable
from collections import deque
from textwrap import dedent
from pathlib import Path
from dataclasses import dataclass

from imgui.integrations.glfw import GlfwRenderer

from .types import UniformLike, UniformT, Controller, App
from .window import Window
from .scene import ParameterParserError, Scene
from .renderer import ShaderCompileError, Renderer
from .gui import GUI
from .midi import MIDIController
from .microphone import Microphone
from .watch import FileWatcher


class Time:

    def __init__(self, running: bool = True):
        self._running = running
        self._start =  self.now()
        self._last_time = 0
        self._offset = 0

    def now(self):
        return time.monotonic()

    @property
    def running(self) -> bool:
        return self._running

    @running.setter
    def running(self, start: bool):
        if start == self._running:
            return
        if start:
            self._offset += self.now() - self._last_time
            self._running = True
        else:
            self._last_time = self.now()
            self._running = False

    def __call__(self) -> float:
        current_time = self.now() if self.running else self._last_time
        return current_time - self._start - self._offset


@dataclass
class SystemParameter(UniformLike, Generic[UniformT]):
    """Pass a function that returns a value to update the Parameter with.

    Pass None if value should be kept constant.
    """
    name: str
    type: str
    value: UniformT
    # TODO why are recursive types not working?
    update: Callable[[App], UniformT | None]

    def __post_init__(self):
        # wrap `.update()` so that it sets .value,
        # but can be passed as `update=`

        self._update = self.update

        def update(renderer: App):
            value = self._update(renderer)
            if value is not None:
                self.value = value
            return value

        self.update = update


class VHSh:

    def __init__(self,
                 scenes: list[Path],
                 width: int = 1280,
                 height: int = 720,
                 watch: bool = False,
                 midi: bool = False,
                 midi_mapping: dict = {},
                 microphone: bool = False):
        # need to be defined upfront for __del__() before glfw/imgui init can fail
        self.renderer: Renderer = None  # type: ignore
        self.gui: GUI = None  # type: ignore
        self._file_watcher: FileWatcher = None  # type: ignore
        self._midi_listener: MIDIManager = None  # type: ignore
        self._microphone: Microphone = None  # type: ignore
        self.error: ShaderCompileError | ParameterParserError | None = None

        self.time = Time()
        self.frame_times = deque([1.0], maxlen=100)

        self.window = Window(self.__class__.__name__, width, height)

        self.gui = GUI(app=self, renderer=GlfwRenderer, window=self.window.handler)
        self._show_gui = True

        self.scenes = [Scene(path) for path in scenes]
        self._scene_index = 0  # initializes @property .scene
        self.system_parameters: dict[str, SystemParameter] = dict(
            u_Resolution=SystemParameter(
                "u_Resolution", type="vec2", value=(0., 0.),
                update=lambda app: app.window.size
            ),
            u_Time=SystemParameter(
                "u_Time", type="float", value=0.,
                update=lambda app: app.time()
            ),
        )

        self.controllers: list[Controller] = []

        if midi:
            self.controllers.append(
                MIDIController(self, system_mapping=midi_mapping))
        if watch:
            self.controllers.append(FileWatcher(self))

        for controller in self.controllers:
            controller.start()

        num_levels = Microphone.NUM_LEVELS
        if microphone:
            self._microphone = Microphone()
            self._microphone.start()

            num_levels = len(self._microphone.levels)
        self.system_parameters["u_Microphone"] = SystemParameter(
            "u_Microphone",
            type=f"float[{num_levels}]",
            value=(0.) * num_levels,
            update=lambda app=self: (app._microphone.levels
                                     if app._microphone else None)
        )

        self.renderer = Renderer(list(self.system_parameters.values()))

        print("scenes:", [f"{scene.name} [{scene.path}]"
                          for scene in self.scenes])

        # TODO move set_scene() into reload(), fall back to test image shader here
        try:
            self.load_scene(self.scene, verbose=False)
        except (ParameterParserError, ShaderCompileError) as e:
            self._print_error(e)
            sys.exit(1)

    @property
    def scene(self) -> Scene:
        return self.scenes[self.scene_index]

    @property
    def scene_index(self) -> int:
        return self._scene_index

    @scene_index.setter
    def scene_index(self, value: int):
        self._scene_index = value
        self.scene.preset_index = 0
        # TODO maybe make reload explicit? with `changed` return from imgui
        self.reload(clear=True)

    def prev_scene(self, n=1):
        self.scene_index = (self.scene_index - n) % len(self.scenes)

    def next_scene(self, n=1):
        self.scene_index = (self.scene_index + n) % len(self.scenes)

    def reload(self, clear: bool = True):
        self.scene.reload()

        try:
            self.load_scene(self.scene, clear=clear)
        except ShaderCompileError as e:
            self.error = e
            self._print_error(e)
        else:
            self.error = None
            print(f"\x1b[2;32mOK: \x1b[2;37m{self.scene.path}\x1b[0;0m")

    def load_scene(self,
                   scene: Scene,
                   verbose: bool = True,
                   clear: bool = False):
        if verbose:
            print()
            print("scene:", scene.name)
            print("presets:", [p.name for p in self.scene.presets])
            print("current preset:", self.scene.presets[scene.preset_index].name)
            print("parameters:")
            for parameter in self.scene.parameters.values():
                print(" ", parameter)

        parameters = [*self.system_parameters.values(),
                      *self.scene.parameters.values()]
        self.renderer.set_shader(scene.source, parameters, clear=clear)

    # error(), custom ShaderCompileError attrs
    def _print_error(self, e: Exception | str):
        try:
            lines = str(e).strip().splitlines()
            if len(lines) == 2:
                flex, error = lines
            else:
                error = lines[0]
                flex = ""
            parts = error.split(':')
            title = parts[0].strip()
            col = parts[1].strip()
            line = parts[2].strip()
            offender = parts[3].strip()
            message = ':'.join(parts[4:])
            print(f"\x1b[1;31m{title}: \x1b[0;0m"
                  f"\x1b[2;37m{self.scene.path}:\x1b[0;0m"
                  f"\x1b[1;37m{col}:{line} \x1b[0;0m"
                  f"\x1b[2;37m({offender})\x1b[0;0m"
                  f"\x1b[0;37m:{message}\x1b[0;0m"
                  f"\x1b[2;37m ({flex})\x1b[0;0m")
                  # white on red: [0;37;41m
        except IndexError:
            print(e)

    def run(self):
        last_time = self.time.now()
        num_frames = 0
        try:
            if (not self.renderer
                    or not self.gui
                    or self.gui._renderer is None):
                raise RuntimeError("glfw imgui renderer not initialized!")

            while not self.window.should_close():
                # TODO -> renderer.frame_times, .update()? maybe not
                # TODO fix should use own high precion timer?
                # TODO correct unit?
                current_time = self.time.now()
                num_frames += 1
                if current_time - last_time >= 0.1:
                    self.frame_times.append(100/num_frames)
                    num_frames = 0
                    last_time += 0.1

                self.window.update()
                self.gui.process_inputs()

                for system_parameter in self.system_parameters.values():
                    system_parameter.update(self)

                for controller in self.controllers:
                    controller.update_pre()

                self.renderer.update((*self.system_parameters.values(),
                                      *self.scene.parameters.values()))
                self.renderer.render()

                for controller in self.controllers:
                    controller.update_post()

                self.gui.update()
                self.gui.render()

                self.window.swap_buffers()

        except KeyboardInterrupt:
            pass
        finally:
            self.shutdown()

    def shutdown(self):
        if self.renderer is not None:
            self.renderer.shutdown()
        if self.gui is not None:
            self.gui.shutdown()
        self.window.close()

        for controller in self.controllers:
            if controller.is_alive():
                controller.stop()
                controller.join()

        if self._microphone is not None:
            if self._microphone.is_alive():
                self._microphone.stop()
                self._microphone.join()

    def __del__(self):
        self.shutdown()
