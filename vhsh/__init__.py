__version__ = "0.1.0"

import sys
import re
from collections import deque
from threading import Thread, Event
from pprint import pprint
from textwrap import dedent
from pathlib import Path

import glfw
# `watchfiles` imported conditionally in VHShRenderer._watch_file()

from .types import UniformValue, get_shader_title
from .window import Window
from .scene import ParameterParserError, Scene, Preset, Parameter, SystemParameter
from .renderer import ShaderCompileError, Renderer
from .gui import GUI
from .midi import MIDIManager
from .microphone import Microphone


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
                 shader_paths: list[Path],
                 width: int = 1280,
                 height: int = 720,
                 watch: bool = False,
                 midi: bool = False,
                 midi_mapping: dict = {},
                 microphone: bool = False):
        # need to be defined upfront for __del__() before glfw/imgui init can fail
        self.renderer: Renderer = None  # type: ignore
        self.gui: GUI = None  # type: ignore
        self._file_watcher: Thread = None  # type: ignore
        self._midi_listener: MIDIManager = None  # type: ignore
        self._microphone: Microphone = None  # type: ignore
        self._glfw_imgui_renderer = None

        # class Time: .now(), .start(), .stop(), running()
        self._start_time = glfw.get_time()
        self._time_running = True
        self._frame_times = deque([1.0], maxlen=100)

        self.window = Window(self.NAME, width, height)

        self.gui = GUI(self)
        self._show_gui = True

        self._file_changed = Event()
        # TODO handle Scenes not shader_paths
        self._shader_paths = shader_paths
        self._shader_index = 0  # initializes @property ._shader_path
        self.__shader_path = self._shader_path
        self.presets: list[Preset] = []
        self._preset_index = 0
        self._new_preset_name = ""
        self.parameters: dict[str, Parameter] = {}
        self.system_parameters: dict[str, SystemParameter] = dict(
            u_Resolution=SystemParameter(
                "u_Resolution", type="vec2", value=(0., 0.),
                update=lambda app: app.window.size
            ),
            u_Time=SystemParameter(
                "u_Time", type="float", value=0.,
                update=lambda app: (glfw.get_time() if self._time_running
                                    else None)
            ),
        )

        self._file_watcher_stop = Event()
        if watch:
            self._file_watcher = Thread(target=self._watch_file,
                                        name="VHSh.file_watcher",
                                        args=(self._shader_path,))
            self._file_watcher.start()

        self._midi_mapping: dict[int, str] = {}
        if midi:
            self._midi_listener = MIDIManager(actions=self,
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

        print("scenes:", [get_shader_title(s) for s in self._shader_paths])

        try:
            scene = Scene(self._shader_path)
            self.set_scene(scene, verbose=False)
        except (ParameterParserError, ShaderCompileError) as e:
            self._print_error(e)
            sys.exit(1)

    def _watch_file(self, filename: str):
        from watchfiles import watch
        print(f"Watching for changes in '{filename}'...")
        for _ in watch(filename, stop_event=self._file_watcher_stop):
            # print(f"'{filename}' changed!")
            self._file_changed.set()

    @property
    def _shader_path(self) -> Path:
        return self._shader_paths[self._shader_index]

    @property
    def _shader_index(self) -> int:
        return self.__shader_index

    @_shader_index.setter
    def _shader_index(self, value: int):
        self.__shader_index = value
        self._preset_index = 0
        self._file_changed.set()

    def prev_shader(self, n=1):
        self._shader_index = (self._shader_index - n) % len(self._shader_paths)

    def next_shader(self, n=1):
        self._shader_index = (self._shader_index + n) % len(self._shader_paths)

    def set_time_running(self, value: bool):
        self._time_running = value

    def set_show_gui(self, value: bool):
        self._show_gui = value

    # TODO replace with self.parameters with magic?
    def set_parameter_value(self,
                            name: str,
                            value: UniformValue,
                            normalized: bool = False):
        self.renderer.update_uniform(name, value, normalized=normalized)

    def get_midi_mapping(self, cc: int) -> str:
        return self._midi_mapping[cc]

    def _reload_scene(self):
            scene = Scene(self._shader_path)

            self._file_changed.clear()
            if (clear := self._shader_path != self.__shader_path):
                self.__shader_path = self._shader_path

            try:
                self.set_scene(scene, clear=clear)
            except ShaderCompileError as e:
                self._error = e
                self._print_error(e)
            else:
                self._error = None
                print("\x1b[2;32mOK:"
                      f" \x1b[2;37m{self._shader_path}"
                      "\x1b[0;0m")

    @property
    def preset_index(self):
        return self._preset_index

    @preset_index.setter
    def preset_index(self, value):
        # TODO why was this here?
        # if self._preset_index == 0:
        #     # TODO abstract uniform from parameters and add here
        #     self.presets[0].parameters = self.parameters

        self._preset_index = value % len(self.presets)
        print()
        print("current preset:", self.presets[self.preset_index].name)

        self._midi_mapping = {}
        for parameter in self.presets[self.preset_index].parameters.values():
            if parameter.midi is not None:
                self._midi_mapping[parameter.midi] = parameter.name
            print(" ", parameter)

        self.parameters = self.presets[self._preset_index].parameters

    def set_scene(self, scene: Scene, verbose: bool = True, clear: bool = False):
        self.presets = scene.presets
        # initializes self.parameters
        self.preset_index = 0
        # TODO @property?
        current_preset = self.presets[self.preset_index]

        if verbose:
            print()
            print("scene:", scene.name)
            print("presets:", [p.name for p in self.presets])
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
        for parameter in current_preset.parameters.values():
            if verbose:
                print(" ", parameter)

            if parameter.midi is not None:
                self._midi_mapping[parameter.midi] = parameter.name

        if verbose and self._midi_listener:
            print("midi_mapping:")
            pprint(self._midi_mapping)

        parameters = [*self.system_parameters.values(),
                      *self.parameters.values()]
        self.renderer.set_shader(scene.source, parameters)

    def prev_preset(self, n: int = 1):
        self.preset_index = (self.preset_index - n) % len(self.presets)

    def next_preset(self, n: int = 1):
        self.preset_index = (self.preset_index + n) % len(self.presets)

    # TODO move to scenes.Scene
    def write_file(self,
                   presets: bool = True,
                   uniforms: bool = False,
                   new_preset: str | None = None):
        with open(self._shader_path) as f:
            shader_src = f.read()

        if presets:
            with self._uniform_lock:
                if new_preset is not None:
                    # TODO proper convertion
                    parameters = {name: Parameter(**uniform.__dict__)
                                  for name, uniform in self.uniforms.items()}
                    self.presets.append(Preset(name=new_preset, parameters=parameters))
                    self._preset_index = len(self.presets) - 1
                for uniform in self.uniforms.values():
                    self.presets[self._preset_index]['uniforms'] = \
                        self.uniforms.copy()

            presets_s = ""
            for preset in self.presets[1:]:
                presets_s += f"/// // {preset['name']}\n"
                presets_s += '\n'.join(
                    f"/// {u}" for u in preset['uniforms'].values()
                    if u.name not in self.FRAGMENT_SHADER_PREAMBLE
                ) + '\n'

            lines = [line for line in shader_src.splitlines()
                        if not line.startswith('///')]
            shader_src = '\n'.join(lines) + '\n'

            shader_src = presets_s + shader_src

            if self._preset_index == 0:
                uniforms = True

        if uniforms:
            with self._uniform_lock:
                for uniform in self.uniforms.values():
                    if uniform.name in self.FRAGMENT_SHADER_PREAMBLE:
                        continue
                    uniform.default = uniform.value
                    print(uniform)
                    shader_src = re.sub(f'^uniform \\w+ {uniform.name}.*$', str(uniform),
                                        shader_src,
                                        flags=re.MULTILINE)

        with open(self._shader_path, 'w') as f:
            f.write(shader_src)
        print(f"wrote {'uniform values' if uniforms else ''}{'presets' if presets else ''} to '{self._shader_path}'")

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
                  f"\x1b[2;37m{self._shader_path}:\x1b[0;0m"
                  f"\x1b[1;37m{col}:{line} \x1b[0;0m"
                  f"\x1b[2;37m({offender})\x1b[0;0m"
                  f"\x1b[0;37m:{message}\x1b[0;0m"
                  f"\x1b[2;37m ({flex})\x1b[0;0m")
                  # white on red: [0;37;41m
        except IndexError:
            print(e)

    def run(self):
        last_time = glfw.get_time()  # TODO maybe time.monotonic_ns()
        num_frames = 0
        try:
            if (not self.renderer
                    or not self.gui
                    or self.gui._glfw_imgui_renderer is None):
                raise RuntimeError("glfw imgui renderer not initialized!")

            while not self.window.should_close():
                self.window.update()
                self.gui.process_inputs()

                # TODO -> renderer.frame_times, .update()
                current_time = glfw.get_time()
                num_frames += 1
                if current_time - last_time >= 0.1:
                    self._frame_times.append(100/num_frames)
                    num_frames = 0
                    last_time += 0.1

                if self._file_changed.is_set():
                    self._reload_scene()

                for system_parameter in self.system_parameters.values():
                    system_parameter.update(self)
                self.renderer.update((*self.system_parameters.values(),
                                      *self.parameters.values()))
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
