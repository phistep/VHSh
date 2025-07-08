__version__ = "0.1.0"

import sys
import re
import time
from collections import deque
from typing import TYPE_CHECKING
from threading import Thread, Event, Lock
from pprint import pprint
from textwrap import dedent
from time import sleep

import glfw

# `watchfiles` imported conditionally in VHShRenderer._watch_file()
# `mido` imported conditionally in VHShRenderer._midi_listen()
if TYPE_CHECKING:
    import pyaudio

from .microphone import Microphone
from .gl_types import UniformValue, UniformT
from .shader import (
    ShaderCompileError, UniformIntializationError, ProgramLinkError,
    Uniform, Renderer
)
from .midi import MIDIManager
from .common import Actions, get_shader_title
from .gui import GUI


class VHShRenderer(Actions):

    NAME = "Video Home Shader"

    FRAGMENT_SHADER_PREAMBLE = dedent("""\
        #version 330 core

        out vec4 FragColor;
        uniform vec2 u_Resolution;
        uniform float u_Time;
        uniform float[{num_levels}] u_Microphone;
        #line 1
        """
    )

    FRAGMENT_SHADER = dedent("""\
        void main() {
            vec2 pos = gl_FragCoord.xy / u_Resolution;
            FragColor = vec4(pos.x, pos.y, 1.0 - (pos.x + pos.y) / 2.0, 1.0);
        }
        """
    )

    def __init__(self,
                 shader_paths: list[str],
                 width: int = 1280,
                 height: int = 720,
                 watch: bool = False,
                 midi: bool = False,
                 midi_mapping: dict = {},
                 microphone: bool = False):
        # need to be defined for __del__() before glfw/imgui init can fail
        self.renderer = None
        self.gui = None
        self._file_watcher = None
        self._midi_listener = None
        self._microphone = None
        self._glfw_imgui_renderer = None

        self._start_time = glfw.get_time()
        self._time_running = True
        self._frame_times = deque([1.0], maxlen=100)
        self._file_changed = Event()
        self._file_watcher_stop = Event()
        self._preset_index = 0
        self._new_preset_name = ""
        self._error = None

        self._window = self._init_window(self.NAME, width, height)
        self.gui = GUI(self)
        self._show_gui = True

        # move uniform handling into renderer, but leave shader_paths and _index out
        self.renderer = Renderer()
        self.uniforms: dict[str, Uniform] = {}
        self._system_uniforms: dict[str, Uniform] = {}
        self._midi_mapping: dict[int, str] = {}
        self._uniform_lock = Lock()
        self._shader_paths = shader_paths
        print("scenes:", [get_shader_title(s) for s in self._shader_paths])
        self._shader_index = 0  # initializes @property ._shader_path
        self.__shader_path = self._shader_path
        self._lineno_offset = \
            len(self.FRAGMENT_SHADER_PREAMBLE.splitlines()) + 1
        with open(self._shader_path) as f:
            shader_src = f.read()
        try:
            self.set_shader(shader_src, verbose=False)
        except ShaderCompileError as e:
            self._print_error(e)
            sys.exit(1)

        if watch:
            self._file_watcher = Thread(target=self._watch_file,
                                        name="VHSh.file_watcher",
                                        args=(self._shader_path,))
            self._file_watcher.start()

        if midi:
            self._midi_listener = MIDIManager(actions=self,
                                              system_mapping=midi_mapping,)
            self._midi_listener.start()

        if microphone:
            self._microphone = Microphone()
            self._microphone.start()

    @staticmethod
    def _init_window(name: str, width: int, height: int):
        if not glfw.init():
            RuntimeError("GLFW could not initialize OpenGL context")

        # Needed for restoring the window posision
        glfw.window_hint_string(glfw.COCOA_FRAME_NAME, name)

        # OS X supports only forward-compatible core profiles from 3.2
        glfw.window_hint(glfw.CONTEXT_VERSION_MAJOR, 3)
        glfw.window_hint(glfw.CONTEXT_VERSION_MINOR, 3)
        glfw.window_hint(glfw.OPENGL_PROFILE, glfw.OPENGL_CORE_PROFILE)

        glfw.window_hint(glfw.OPENGL_FORWARD_COMPAT, glfw.TRUE)

        window = glfw.create_window(int(width), int(height), name, None, None)
        glfw.make_context_current(window)

        if not window:
            glfw.terminate()
            raise RuntimeError("GLFW could not initialize Window")

        return window

    def _watch_file(self, filename: str):
        from watchfiles import watch
        print(f"Watching for changes in '{filename}'...")
        for _ in watch(filename, stop_event=self._file_watcher_stop):
            # print(f"'{filename}' changed!")
            self._file_changed.set()

    @property
    def _shader_path(self) -> str:
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

    def set_uniform_value(self,
                          name: str,
                          value: UniformValue,
                          normalized: bool = False):
        with self._uniform_lock:
            uniform = self.uniforms[name]

            if uniform.range and normalized:
                min_, max_ = uniform.range[:2]
                value = min_ + value * (max_ - min_)

            match uniform.type:
                case 'bool':
                    uniform.value = bool(value)
                case 'int':
                    uniform.value = int(value)
                case 'float':
                    uniform.value = float(value)
                case _:
                    raise NotImplementedError(
                        f"MIDI update not implemented for Uniform type '{uniform.type}'")

    def get_midi_mapping(self, cc: int) -> str:
        return self._midi_mapping[cc]

    def _update_system_uniforms(self):
        self.uniforms['u_Resolution'].value = [float(self.width),
                                               float(self.height)]

        # regular `time.time()` is too big for f32, so we just return
        # seconds from program start, glwf does this
        if self._time_running:
            self.uniforms['u_Time'].value = glfw.get_time()

        if self._microphone:
            self.uniforms['u_Microphone'].value = self._microphone.levels

    def _reload_shader(self):
            with open(self._shader_path) as f:
                shader_src = f.read()
            self._file_changed.clear()

            if self._shader_path != self.__shader_path:
                # clear instead of update uniforms if this is a
                # new file (vs just reload)
                with self._uniform_lock:
                    self.uniforms = {}
                self.__shader_path = self._shader_path

            try:
                self.set_shader(shader_src)
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
        with self._uniform_lock:
            if self._preset_index == 0:
                self.presets[0]['uniforms'] = self.uniforms

            self._preset_index = value % len(self.presets)
            print()
            print("current preset:", self.presets[self.preset_index]['name'])

            self._midi_mapping = {}
            for uniform in self.presets[self.preset_index]['uniforms'].values():
                if uniform.midi is not None:
                    self._midi_mapping[uniform.midi] = uniform.name
                if uniform.name not in self.FRAGMENT_SHADER_PREAMBLE:
                    print(" ", uniform)

            self.uniforms = {**self._system_uniforms,
                             **self.presets[self._preset_index]['uniforms']}

    def set_shader(self, shader_src: str, verbose: bool = True):
        preamble = self.FRAGMENT_SHADER_PREAMBLE
        num_levels = (len(self._microphone.levels) if self._microphone
                      else Microphone.NUM_LEVELS)
        preamble = preamble.format(num_levels=num_levels)
        shader_src = preamble + shader_src
        self.renderer.create_shader_program(shader_src)

        self.presets = [{'name': "<current>", 'uniforms': {}}]
        for n, line in enumerate(shader_src.split('\n')):
            line = line.strip()

            # <current> uniforms
            if line.startswith('uniform'):
                try:
                    uniform = Uniform.from_def(self.renderer.shader_program, line)
                    with self._uniform_lock:
                        if uniform.name in self.FRAGMENT_SHADER_PREAMBLE:
                            self._system_uniforms[uniform.name] = uniform
                        if (self.preset_index == 0
                                and uniform.name in self.uniforms):
                            uniform.value = self.uniforms[uniform.name].value
                        self.presets[0]['uniforms'][uniform.name] = uniform
                except UniformIntializationError as e:
                    lineno = n - self._lineno_offset
                    raise ShaderCompileError(f"ERROR 0:{lineno} {e}")
            # presets
            elif line.startswith('///'):
                line_content = line.lstrip('/ ').strip()
                if line.startswith('/// uniform'):
                    uniform = Uniform.from_def(self.renderer.shader_program, line_content)
                    if self.preset_index == len(self.presets) - 1:
                        with self._uniform_lock:
                            uniform.value = self.uniforms[uniform.name].value
                    self.presets[-1]['uniforms'][uniform.name] = uniform
                else:
                    index = len(self.presets)
                    self.presets.append({'name': line_content or str(index),
                                         'uniforms': {}})

        if verbose:
            print()
            print("scene:", get_shader_title(self._shader_paths[self._shader_index]))
            print("presets:", [p['name'] for p in self.presets])
            print("current preset:", self.presets[self.preset_index]['name'])

        with self._uniform_lock:
            self.uniforms = {**self._system_uniforms,
                             **self.presets[self.preset_index]['uniforms']}

            if verbose:
                print("uniforms:")
            self._midi_mapping = {}
            for uniform in self.uniforms.values():
                if verbose:
                    print(" ", uniform)

                if uniform.midi is not None:
                    self._midi_mapping[uniform.midi] = uniform.name

            if verbose and self._midi_listener:
                print("midi_mapping:")
                pprint(self._midi_mapping)

    def prev_preset(self, n: int = 1):
        self.preset_index = (self.preset_index - n) % len(self.presets)

    def next_preset(self, n: int = 1):
        self.preset_index = (self.preset_index + n) % len(self.presets)

    def write_file(self,
                   presets: bool = True,
                   uniforms: bool = False,
                   new_preset: str | None = None):
        with open(self._shader_path) as f:
            shader_src = f.read()

        if presets:
            with self._uniform_lock:
                if new_preset is not None:
                    self.presets.append({"name": new_preset,
                                         "uniforms": self.uniforms.copy()})
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
            if not self.renderer or not self.gui or self.gui._glfw_imgui_renderer is None:
                raise RuntimeError("glfw imgui renderer not initialized!")
            while not glfw.window_should_close(self._window):
                current_time = glfw.get_time()
                num_frames += 1
                if current_time - last_time >= 0.1:
                    self._frame_times.append(100/num_frames)
                    num_frames = 0
                    last_time += 0.1

                glfw.poll_events()
                self.gui.process_inputs()
                self.width, self.height = \
                    glfw.get_framebuffer_size(self._window)

                if self._file_changed.is_set():
                    self._reload_shader()
                self._update_system_uniforms()
                self.gui.update()

                self.renderer.render(self.uniforms.values())
                self.gui.render()
                glfw.swap_buffers(self._window)
        except KeyboardInterrupt:
            pass
        finally:
            self.shutdown()

    def shutdown(self):
        if self.renderer is not None:
            self.renderer.shutdown()
        if self.gui is not None:
            self.gui.shutdown()
        glfw.terminate()


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
