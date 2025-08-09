import time
import logging
from collections import deque
from pathlib import Path

from imgui.integrations.glfw import GlfwRenderer

from .types import Controller, SystemParameter, Color
from .window import Window
from .scene import ParameterParserError, Scene
from .renderer import ShaderCompileError, Renderer
from .gui import GUI
from .midi import MIDIController
from .microphone import Microphone
from .watch import FileWatcher


logger = logging.getLogger(__name__)


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


class VHSh:

    def __init__(self,
                 scenes: list[Path],
                 width: int = 1280,
                 height: int = 720,
                 midi_mapping: dict = {},
                 microphone: bool = False):
        # need to be defined upfront for __del__() before glfw/imgui init can fail
        self.renderer: Renderer = None  # type: ignore
        self.gui: GUI = None  # type: ignore
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

        self.controllers: dict[str, Controller] = dict(
            FileWatcher=FileWatcher(self),
            MIDIController=MIDIController(self, system_mapping=midi_mapping),
            Microphone=Microphone(self, enabled=microphone)
        )


        for controller in self.controllers.values():
            controller.start()

        self.renderer = Renderer(list(self.system_parameters.values()))

        logger.info("scenes: %s",
                    [f"{scene.name} [{scene.path}]" for scene in self.scenes])
        self.load()

    @property
    def scene(self) -> Scene:
        return self.scenes[self.scene_index]

    @property
    def scene_index(self) -> int:
        return self._scene_index

    @scene_index.setter
    def scene_index(self, value: int):
        logger.debug("VHSh.scene_index.setter: %i", value)

        self._scene_index = value
        self.scene.preset_index = 0
        # TODO maybe make reload explicit? with `changed` return from imgui
        self.load(clear=True)

    def prev_scene(self, n=1):
        self.scene_index = (self.scene_index - n) % len(self.scenes)

    def next_scene(self, n=1):
        self.scene_index = (self.scene_index + n) % len(self.scenes)

    def load(self, clear: bool = True):
        logger.info("scene: %s", self.scene.name)
        logger.info("presets: %s", [p.name for p in self.scene.presets])
        logger.info("parameters: %s", self.scene.presets[self.scene.preset_index])

        self.scene.reload()
        parameters = [*self.system_parameters.values(),
                      *self.scene.parameters.values()]
        try:
            self.renderer.set_shader(self.scene.source, parameters, clear=clear)
        except ShaderCompileError as e:
            self.error = e
            self._print_error(e)
        else:
            self.error = None
            logger.info(f"{Color.GREEN + Color.Style.BOLD}OK{Color.RESET}:"
                        f" {self.scene.path}")

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
            logger.error(f"\x1b[2;37m{self.scene.path}:\x1b[0;0m"
                         f"\x1b[1;37m{col}:{line} \x1b[0;0m"
                  f"\x1b[2;37m({offender})\x1b[0;0m"
                  f"\x1b[0;37m:{message}\x1b[0;0m"
                  f"\x1b[2;37m ({flex})\x1b[0;0m")
                  # white on red: [0;37;41m
        except IndexError:
            logger.error(e, exc_info=True)

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

                for controller in self.controllers.values():
                    controller.update_pre()

                if not self.error:
                    self.renderer.update((*self.system_parameters.values(),
                                        *self.scene.parameters.values()))
                    self.renderer.render()

                for controller in self.controllers.values():
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

        for controller in self.controllers.values():
            if controller.is_alive():
                controller.stop()
                controller.join()

    def __del__(self):
        self.shutdown()
