__version__ = "0.1.0"

import sys
import time
from typing import Generic, Callable
from collections import deque
from threading import Thread, Event
from pprint import pprint
from textwrap import dedent
from pathlib import Path
from dataclasses import dataclass

from imgui.integrations.glfw import GlfwRenderer

from .types import UniformValue, UniformLike, UniformT
from .window import Window
from .scene import ParameterParserError, Scene
from .renderer import ShaderCompileError, Renderer
from .gui import GUI
from .midi import MIDIManager
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
    def running(self) -> int:
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
    update: Callable[["VHShRenderer"], UniformT | None]

    def __post_init__(self):
        # wrap `.update()` so that it sets .value,
        # but can be passed as `update=`

        self._update = self.update

        def update(renderer: "VHShRenderer"):
            value = self._update(renderer)
            if value is not None:
                self.value = value
            return value

        self.update = update


class VHShRenderer:

    # TODO use ths as class name
    NAME = "VideoHomeShader"

    # TODO scene.DEFAULT_SCENE: Scene
    FRAGMENT_SHADER = dedent("""\
        void main() {
            vec2 pos = gl_FragCoord.xy / u_Resolution;
            FragColor = vec4(pos.x, pos.y, 1.0 - (pos.x + pos.y) / 2.0, 1.0);
        }
        """
    )

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

        self.time = Time()
        self._frame_times = deque([1.0], maxlen=100)

        self.window = Window(self.NAME, width, height)

        self.gui = GUI(app=self, renderer=GlfwRenderer, window=self.window.handler)
        self._show_gui = True

        self._file_changed = Event()
        # TODO handle Scenes not shader_paths
        self.scenes = [Scene(path) for path in scenes]
        self._scene_index = 0  # initializes @property .scene
        self.system_parameters: dict[str, SystemParameter] = dict(
            u_Resolution=SystemParameter(
                "u_Resolution", type="vec2", value=(0., 0.),
                update=lambda app: app.window.size
            ),
            u_Time=SystemParameter(
                "u_Time", type="float", value=0.,
                update=lambda app: self.time()
            ),
        )

        self._file_watcher_stop = Event()
        self._file_watcher = FileWatcher(scenes, self._file_changed)
        self._file_watcher.current = self.scene.path
        if watch:
            self._file_watcher.start()

        self._midi_mapping: dict[int, str] = {}
        if midi:
            self._midi_listener = MIDIManager(app=self,
                                              system_mapping=midi_mapping,)
            self._midi_listener.start()

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
        self._error = None

        print("scenes:", [f"{scene.name} [{scene.path}]"
                          for scene in self.scenes])

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
        self._file_changed.set()

    def prev_scene(self, n=1):
        self.scene_index = (self.scene_index - n) % len(self.scenes)

    def next_scene(self, n=1):
        self.scene_index = (self.scene_index + n) % len(self.scenes)

    # TODO replace with self.parameters with magic?
    def set_parameter_value(self,
                            name: str,
                            value: UniformValue,
                            normalized: bool = False):
        self.renderer.update_uniform(name, value, normalized=normalized)

    def get_midi_mapping(self, cc: int) -> str:
        return self._midi_mapping[cc]

    def reload(self):
        # TODO somehow all of this property magic makes this very complicated.
        # have distinct reload_method and set_scene
        print("VHSh.reload")
        self.scene.reload()

        self._file_changed.clear()
        # TODO this needs to work even without file_watcher
        clear = self.scene.path != self._file_watcher.current
        if clear:
            self._file_watcher.current = self.scene.path

        # TODO now this crashes upon receiving file change
        try:
            self.load_scene(self.scene, clear=clear)
        except ShaderCompileError as e:
            self._error = e
            self._print_error(e)
        else:
            self._error = None
            print("\x1b[2;32mOK:"
                    f" \x1b[2;37m{self.scene.path}"
                    "\x1b[0;0m")

    def load_scene(self,
                   scene: Scene,
                   verbose: bool = True,
                   clear: bool = False):
        self.preset_index = 0
        # TODO @property?
        current_preset = self.scene.presets[scene.preset_index]

        if verbose:
            print()
            print("scene:", scene.name)
            print("presets:", [p.name for p in self.scene.presets])
            print("current preset:", current_preset.name)

        # TODO handle updating with current value correctly
        # for parameter in current_preset.parameters.values():
        #     if parameter.name in self.uniforms:
        #         parameter.value = self.uniforms[parameter.name].value
        #     # <current>
        #     if (self.preset_index == 0 and uniform.name in self.uniforms):
        #         uniform.value = self.uniforms[uniform.name].value
        #     # presets
        #     if self.preset_index == len(self.presets) - 1:
        #             uniform.value = self.uniforms[uniform.name].value

        self._midi_mapping = {}
        if verbose:
            print("parameters:")
        for parameter in self.scene.parameters.values():
            if verbose:
                print(" ", parameter)

            if parameter.midi is not None:
                self._midi_mapping[parameter.midi] = parameter.name
        if verbose and self._midi_listener:
            print("midi_mapping:")
            pprint(self._midi_mapping)

        parameters = [*self.system_parameters.values(),
                      *self.scene.parameters.values()]
        self.renderer.set_shader(scene.source, parameters, clear=clear)

    # error()
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
                self.window.update()
                self.gui.process_inputs()

                # TODO -> renderer.frame_times, .update()? maybe not
                # TODO fix should use own high precion timer?
                # TODO correct unit?
                current_time = self.time.now()
                num_frames += 1
                if current_time - last_time >= 0.1:
                    self._frame_times.append(100/num_frames)
                    num_frames = 0
                    last_time += 0.1

                if self._file_changed.is_set():
                    self.reload()

                for system_parameter in self.system_parameters.values():
                    system_parameter.update(self)
                self.renderer.update((*self.system_parameters.values(),
                                      *self.scene.parameters.values()))
                self.renderer.render()

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

        if self._file_watcher is not None:
            if self._file_watcher.is_alive():
                self._file_watcher_stop.set()
                self._file_watcher.join()

        if self._midi_listener is not None:
            if self._midi_listener.is_alive():
                self._midi_listener.stop()
                self._midi_listener.join()

        if self._microphone is not None:
            if self._microphone.is_alive():
                self._microphone.stop()
                self._microphone.join()

    def __del__(self):
        self.shutdown()
