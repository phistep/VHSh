#!/usr/bin/env python3

import os
import sys
import argparse
import re
import time
import warnings
import tomllib
import signal
from datetime import datetime
from collections import defaultdict, deque
from array import array
from typing import (Optional, TypeVar, TypeAlias, Iterable, get_args, Literal,
                    Callable, Any, Self, TYPE_CHECKING)
from threading import Thread, Event, Lock
from pprint import pprint
from textwrap import dedent
from itertools import cycle
from contextlib import contextmanager
from time import sleep

from OpenGL.raw.GL.VERSION.GL_4_0 import glUniform1d
from imgui.integrations.glfw import GlfwRenderer
import OpenGL.GL as gl
import glfw
import numpy as np
import imgui

# `watchfiles` imported conditionally in VHShRenderer._watch_file()
# `mido` imported conditionally in VHShRenderer._midi_listen()
if TYPE_CHECKING:
    import pyaudio


VertexArrayObject: TypeAlias = np.uint32
VertexBufferObject: TypeAlias = np.uint32
Shader: TypeAlias = int
ShaderProgram: TypeAlias = int

GLSLBool: TypeAlias = bool
GLSLInt: TypeAlias = int
GLSLFloat: TypeAlias = float
GLSLVec2: TypeAlias = tuple[float, float]
GLSLVec3: TypeAlias = tuple[float, float, float]
GLSLVec4: TypeAlias = tuple[float, float, float, float]

T = TypeVar('T')
UniformT = TypeVar('UniformT', GLSLBool, GLSLInt, GLSLVec2, GLSLVec3, GLSLVec4)


class ShaderCompileError(RuntimeError):
    """shader compile error."""


class UniformIntializationError(ShaderCompileError):
    """custom uniform initialization error."""


class ProgramLinkError(RuntimeError):
    """program link error."""

class Microphone(Thread):

    NUM_LEVELS = 7

    def __init__(self,
                 rate: int = 44100,
                 chunk: int = 1024,
                 buffer_size_s: float = 5.0,
                 smooth: float = 0.05,
                 format: int | None = None,
                 intervals: list[int] = [0, 60, 250, 500, 2_000, 6_000, 8_000]):
        super().__init__()

        import pyaudio

        self.rate = rate
        self.chunk = chunk
        self.format = pyaudio.paInt16 if format is None else format
        self.channels = 1  # TODO configurable?

        self._bins = list(zip(intervals, intervals[1:]))
        self._levels = deque([np.zeros(len(intervals))],  # last interval is open
                             maxlen=int(rate/chunk*smooth))
        self._max_vol = chunk * (2**16-1)  # TODO sizeof(format)

        self._frame_buffer = deque(maxlen=int(rate / chunk * buffer_size_s))
        self._stop_stream = Event()
        self._output_lock = Lock()

    @property
    def levels(self) -> list[float]:
        with self._output_lock:
            level_history = np.array(self._levels)
        max_ = min(self._max_vol, level_history.ravel().max())
        levels = (level_history.mean(axis=0) / max_
                  if max_ != 0
                  else np.zeros(level_history.shape[-1]))
        return levels.tolist()

    def run(self):
        import pyaudio

        def _record(in_data, frame_count, time_info, status):
            self._frame_buffer.appendleft(in_data)
            return (in_data, pyaudio.paContinue)

        audio = pyaudio.PyAudio()
        stream = audio.open(input=True,
                            format=self.format,
                            channels=self.channels,
                            rate=self.rate,
                            frames_per_buffer=self.chunk,
                            stream_callback=_record)

        try:
            while stream.is_active() and not self._stop_stream.is_set():
                if not self._frame_buffer:
                    sleep(0.1)
                    continue

                raw_frame = self._frame_buffer.pop()
                frame = np.frombuffer(raw_frame, dtype=np.uint16)

                spect = np.fft.fft(frame)
                # Scale frequencies to Hz
                freq = np.fft.fftfreq(frame.shape[-1], 1/self.rate)
                # only positive frequencies (first half of the spectrum)
                positive_freqs = freq >= 0
                spectrum = abs(spect[positive_freqs])
                frequencies = freq[positive_freqs]

                levels = [np.sum(spectrum[(frequencies >= min_) & (frequencies < max_)])
                          for min_, max_ in self._bins]
                levels.append(np.sum(spectrum[frequencies >= self._bins[-1][1]]))
                with self._output_lock:
                    self._levels.append(np.array(levels))

        finally:
            stream.stop_stream()
            stream.close()
            audio.terminate()

    def stop(self):
        self._stop_stream.set()


class Uniform:

    def __init__(self,
                 program: ShaderProgram,
                 type_: str,
                 name: str,
                 value: Optional[UniformT] = None,
                 default: Optional[UniformT] = None,
                 range: Optional[list[float]] = None,  # TODO GLSLFloat?
                 widget: Optional[str] = None,
                 midi: Optional[int] = None):
        self._location: int = gl.glGetUniformLocation(program, name)
        self.type = type_
        self._type = None
        self.name = name
        self.default = default
        self.range = range
        self.widget = widget  # TODO enum
        self.midi = midi

        # TODO default step is dropped if not passed
        self._glUniform: Callable[[Any, ...], None]
        match self.type:
            case 'bool':
                self._type = GLSLBool
                self._glUniform = gl.glUniform1i
                self.default = self.default or True
                self.range = None
            case 'int':
                self._type = GLSLInt
                self._glUniform = gl.glUniform1i
                self.default = self.default or 1
                self.range = self.range or (0, 10, 1)
            case 'float':
                self._type = GLSLFloat
                self._glUniform = gl.glUniform1f
                self.default = self.default or 1.0
                self.range = self.range or (0.0, 1.0, 0.01)
            case str() as t if t.startswith('float['):
                try:
                    m = re.match(r'float\[(\d+)\]', type_)
                    len = int(m.group(1))
                except (AttributeError, ValueError) as e:
                    raise UniformIntializationError(
                        f"Unable to parse float array type '{type_}': {e}"
                    ) from e
                self._type = (float,) * len
                self._glUniform = gl.glUniform1fv
                self.default = self.default or (0.0,) * len
                self.range = None
            case 'vec2':
                self._type = GLSLVec2
                self._glUniform = gl.glUniform2f
                self.default = self.default or (1.,)*2
                self.range = self.range or (0.0, 1.0, 0.01)
            case 'vec3':
                self._type = GLSLVec3
                self._glUniform = gl.glUniform3f
                self.default = self.default or (1.,)*3
                self.range = self.range or (0.0, 1.0, 0.01)
            case 'vec4':
                self._type = GLSLVec4
                self._glUniform = gl.glUniform4f
                self.default = self.default or (1.,)*4
                self.range = self.range or (0.0, 1.0, 0.01)
            case _:
                raise NotImplementedError(
                    f"Uniform type '{self.type}' not implemented:"
                    f" {self.name} ({self.value})")

        uniform_type = get_args(self._type) or self._type
        value_type = (tuple(type(elem) for elem in self.default)
                      if isinstance(self.default, Iterable)
                      else type(self.default))
        if  value_type != uniform_type:
            raise UniformIntializationError(
                f"Uniform '{self.name}' defined as"
                f" '{self.type}' ({uniform_type}), but provided value"
                f" has type '{value_type}': {self.default!r}")

        self.value = value if value is not None else self.default

    def __str__(self):
        s = f"uniform {self.type} {self.name};  //"
        if self.widget is not None:
            s += f' <{self.widget}>'
        s += f" ={str(self.value).replace(' ', '')}"
        if self.range is not None:
            s += f" {str(list(self.range)).replace(' ', '')}"
        if self.midi is not None:
            s += f' #{self.midi}'
        return s

    def __repr__(self):
        return (f'<Uniform'
                f' type={self.type}'
                f' name="{self.name}"'
                f' value={self.value}'
                f' default={self.default}'
                f' range={self.range}'
                f' widget={self.widget}'
                f' midi={self.midi}'
                f' _type={self._type}'
                f' _glUniform={self._glUniform.__name__}'
                f' at 0x{self._location:04x}>')

    def update(self):
        args = self.value
        if not isinstance(args, Iterable):
            args = [args]
        if self._glUniform.__name__.endswith('v'):
            self._glUniform(self._location, len(args), args)
        else:
            self._glUniform(self._location, *args)

    def set_value_midi(self, value: int):
        assert 0 <= value <= 127
        if self.range:
            min_, max_ = self.range[:2]
            interpolated = min_ + (value / 127.0) * (max_ - min_)

        match self.type:
            case 'bool':
                self.value = bool(value)
            case 'int':
                self.value = int(interpolated)
            case 'float':
                self.value = float(interpolated)
            case _:
                raise NotImplementedError(
                    f"MIDI update not implemented for Uniform type '{self.type}'")

    @classmethod
    def from_def(cls,
                shader_program: ShaderProgram,
                definition: str) -> 'Uniform':
        try:
            matches = re.search(
                # TODO why ^(?!\/\/)\s* not working to ignore comments?
                (R'uniform\s+(?P<type>[\w\[\]]+)\s+(?P<name>\w+)\s*;'
                 R'(?:\s*//\s*(?:'
                 R'(?P<widget><\w+>)?\s*)?'
                 R'(?P<default>=(?:\S+|\([^\)]+\)))?'
                 R'\s*(?P<range>\[[^\]]+\])?'
                 R'\s*(?P<midi>#\d+)?'
                 R')?'),
                definition
            )
            type_, name, widget, default_s, range_s, midi = matches.groups()
        except Exception as e:
            raise UniformIntializationError(
                f"Syntax error in metadata defintion: {definition}") from e

        if widget is not None:
            widget = widget.strip('<>')

        try:
            default = (eval(default_s.removeprefix('='))
                        if default_s
                        else None)
        except SyntaxError as e:
            raise UniformIntializationError(
                f"Invalid 'default' metadata for uniform"
                f" '{name}': {e}: {default_s}"
            ) from e

        try:
            range = eval(range_s) if range_s else None
        except SyntaxError as e:
            raise UniformIntializationError(
                f"Invalid 'range' metadata for uniform"
                f" '{name}': {e}: {range_s!r}"
            ) from e

        if midi is not None:
            try:
                midi = int(midi.removeprefix('#'))
            except ValueError as e:
                raise UniformIntializationError(
                    f"Invalid 'midi' metadata for uniform"
                    f" '{name}': {e}: {midi!r}"
                ) from e

        return Uniform(program=shader_program,
                       type_=type_,
                       name=name,
                       default=default,
                       range=range,
                       widget=widget,
                       midi=midi)


class VHShRenderer:

    NAME = "Video Home Shader"

    VERTICES = np.array([[-1.0,  1.0, 0.0],
                         [-1.0, -1.0, 0.0],
                         [ 1.0,  1.0, 0.0],
                         [ 1.0, -1.0, 0.0]],
                        dtype=np.float32)

    VERTEX_SHADER = dedent("""\
        #version 330 core

        layout(location = 0) in vec3 VertexPos;

        void main() {
            gl_Position = vec4(VertexPos, 1.0);
        }
        """
    )

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
        self.vao = None
        self.vbo = None
        self.shader_program = None
        self._file_changed = Event()
        self._stop = Event()
        self._file_watcher = None
        self._midi_listener = None
        self._microphone = None
        self._glfw_imgui_renderer = None
        self._time_running = True
        self._preset_index = 0
        self._new_preset_name = ""
        self._show_gui = True
        self._error = None

        imgui.create_context()
        imgui_style = imgui.get_style()
        imgui.style_colors_dark(imgui_style)
        imgui_style.colors[imgui.COLOR_PLOT_HISTOGRAM] = \
            imgui_style.colors[imgui.COLOR_PLOT_LINES]
        imgui_style.colors[imgui.COLOR_PLOT_HISTOGRAM_HOVERED] = \
            imgui_style.colors[imgui.COLOR_BUTTON_HOVERED]
        self._window = self._init_window(self.NAME, width, height)
        self._glfw_imgui_renderer = GlfwRenderer(self._window)

        self._start_time = glfw.get_time()
        self._frame_times = deque([1.0], maxlen=100)

        self.vao, self.vbo = self._create_vertices(self.VERTICES)

        self.vertex_shader = self._create_shader(gl.GL_VERTEX_SHADER,
                                                 self.VERTEX_SHADER)

        self.uniforms: dict[str, Uniform] = {}
        self._system_uniforms: dict[str, Uniform] = {}
        self._midi_mapping: dict[int, str] = {}
        self._uniform_lock = Lock()
        self._shader_paths = shader_paths
        print("scenes:", [self._get_shader_title(s) for s in self._shader_paths])
        self._shader_index = 0
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
            self._midi_listener = Thread(target=self._midi_listen,
                                         name="VHSh.midi_listener",
                                         args=(midi_mapping,))
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
        for _ in watch(filename, stop_event=self._stop):
            # print(f"'{filename}' changed!")
            self._file_changed.set()

    def _midi_listen(self, system_mapping: dict):
        import mido

        print("midi system mapping:")
        pprint(system_mapping)
        system_mapping = defaultdict(dict, system_mapping)

        try:
            with mido.open_input() as inport:
                print(f"midi: listening for MIDI messages on '{inport.name}'...")

                while True:
                    if self._stop.is_set():
                        break
                    for msg in inport.iter_pending():
                        # print(f"Received MIDI message: {msg}")
                        # print(f"Received MIDI message: #{msg.control} = {msg.value}")
                        button_down = bool(msg.value)

                        if msg.control == system_mapping['scene'].get('prev'):
                            if button_down:
                                self.prev_shader()
                            continue
                        if msg.control == system_mapping['scene'].get('next'):
                            if button_down:
                                self.next_shader()
                            continue

                        if msg.control == system_mapping['preset'].get('prev'):
                            if button_down:
                                self.prev_preset()
                            continue
                        if msg.control == system_mapping['preset'].get('next'):
                            if button_down:
                                self.next_preset()
                            continue
                        if msg.control == system_mapping['preset'].get('save'):
                            if button_down:
                                self.write_file(uniforms=False, presets=True,
                                                new_preset=f"MIDI {datetime.now()}")
                            continue

                        if msg.control == system_mapping['preset'].get('next'):
                            if button_down:
                                self.next_preset()
                            continue

                        if msg.control == system_mapping['uniform'].get('time', {}).get('toggle'):
                            self._time_running = bool(msg.value)
                            continue

                        if msg.control == system_mapping['uniform'].get('toggle_ui'):
                            self._show_gui = bool(msg.value)
                            continue

                        try:
                            self._uniform_lock.acquire()
                            uniform = self.uniforms[self._midi_mapping[msg.control]]
                            uniform.set_value_midi(msg.value)
                        except KeyError as e:
                            print(f"MIDI mapping not found for: {msg.control}")
                            # print(msg)
                            pprint(self._midi_mapping)
                        except NotImplementedError as e:
                            self._print_error(f"ERROR setting uniform '{uniform.name}': {e}")
                        finally:
                            self._uniform_lock.release()
                    time.sleep(1e-6)
        except OSError:
            print("No MIDI devices found!")

    def _create_vertices(self, vertices: np.ndarray
                         ) -> tuple[VertexArrayObject, VertexBufferObject]:
        vao = gl.glGenVertexArrays(1)
        vbo = gl.glGenBuffers(1)

        gl.glBindVertexArray(vao)

        gl.glBindBuffer(gl.GL_ARRAY_BUFFER, vbo)
        gl.glBufferData(target=gl.GL_ARRAY_BUFFER,
                        size=vertices.nbytes,
                        data=vertices,
                        usage=gl.GL_STATIC_DRAW)

        # Specify the layout of the vertex data
        vertex_attrib_idx = 0
        gl.glVertexAttribPointer(index=vertex_attrib_idx,
                                size=3, # len(x, y, z)
                                type=gl.GL_FLOAT,
                                normalized=gl.GL_FALSE,
                                stride=3 * 4,  # (x, y, z) * sizeof(GL_FLOAT)  # TODO
                                pointer=gl.ctypes.c_void_p(0))
        gl.glEnableVertexAttribArray(vertex_attrib_idx)

        # Unbind the VAO
        gl.glBindVertexArray(vertex_attrib_idx)

        return vao, vbo

    def _create_shader(self, shader_type, shader_source: str) -> Shader:
        """creates a shader from its source & type."""
        shader = gl.glCreateShader(shader_type)
        gl.glShaderSource(shader, shader_source)
        gl.glCompileShader(shader)
        if gl.glGetShaderiv(shader, gl.GL_COMPILE_STATUS) != gl.GL_TRUE:
            raise ShaderCompileError(
                gl.glGetShaderInfoLog(shader).decode('utf-8'))
        return shader  # pyright: ignore [reportReturnType]

    def _create_program(self, *shaders) -> ShaderProgram:

        """creates a program from its vertex & fragment shader sources."""
        program = gl.glCreateProgram()
        for shader in shaders:
            gl.glAttachShader(program, shader)
        gl.glLinkProgram(program)
        if gl.glGetProgramiv(program, gl.GL_LINK_STATUS) != gl.GL_TRUE:
            raise ProgramLinkError(
                gl.glGetProgramInfoLog(program).decode('utf-8'))
        return program  # pyright: ignore [reportReturnType]

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

    def _get_shader_title(self, shader_path: str) -> str:
        return os.path.splitext(os.path.basename(shader_path))[0]

    def _update_gui(self):
        # TODO ctrl+tab? or ctrl+`
        # TODO not while in input
        if imgui.is_key_pressed(imgui.get_key_index(imgui.KEY_TAB)):
            self._show_gui = not self._show_gui

        imgui.new_frame()
        imgui.begin("Parameters", closable=False)

        if self._error is not None:
            imgui.open_popup("Error")
        with imgui.begin_popup_modal("Error",
            flags=imgui.WINDOW_NO_MOVE | imgui.WINDOW_NO_RESIZE
        ) as error_popup:
            if error_popup.opened:
                if self._error is None:
                    imgui.close_current_popup()
                else:
                    # TODO colored
                    imgui.text_wrapped(str(self._error))

        with imgui.begin_group():
            if imgui.begin_combo("##Scene",
                                 self._get_shader_title(self._shader_path)):
                for idx, item in enumerate(
                        map(self._get_shader_title, self._shader_paths)):
                    is_selected = (idx == self._shader_index)
                    if imgui.selectable(item, is_selected)[0]:
                        self._shader_index = idx
                    if is_selected:
                        imgui.set_item_default_focus()
                imgui.end_combo()
            imgui.same_line()
            if imgui.arrow_button("Prev Scene", imgui.DIRECTION_LEFT):
                self.prev_shader()
            imgui.same_line()
            if imgui.arrow_button("Next Scene", imgui.DIRECTION_RIGHT):
                self.next_shader()
            imgui.same_line()
            imgui.text("Scene")

        imgui.spacing()

        with imgui.begin_group():
            # TODO begin_list_box?
            if imgui.begin_combo(
                "##Preset", self.presets[self.preset_index]['name']
            ):
                for idx, item in  [(p['index'], p['name'])
                                   for p in self.presets]:
                    is_selected = (idx == self.preset_index)
                    if imgui.selectable(item, is_selected)[0]:
                        self.preset_index = idx
                    if is_selected:
                        imgui.set_item_default_focus()
                imgui.end_combo()
            imgui.same_line()
            if imgui.arrow_button("Prev Preset", imgui.DIRECTION_LEFT):
                self.prev_preset()
            imgui.same_line()
            if imgui.arrow_button("Next Preset", imgui.DIRECTION_RIGHT):
                self.next_preset()
            imgui.same_line()
            if imgui.button("Save"):
                self.write_file(uniforms=False, presets=True)
            imgui.same_line()
            imgui.text("Preset")

            _, self._new_preset_name = imgui.input_text_with_hint(
                "##Name", "New Preset Name", self._new_preset_name)
            imgui.same_line()
            if imgui.button("Save##Save New Preset"):
                self.write_file(uniforms=False, presets=True, new_preset=self._new_preset_name)
                self._new_preset_name = ""
            imgui.same_line()
            imgui.text("New Preset")

        imgui.spacing()

        with imgui.begin_group():
            frame_times = array('f', self._frame_times)
            imgui.plot_lines("Frame Time##Plot", frame_times,
                overlay_text=f"{frame_times[-1]:5.2f} ms"
                             f"  ({1000/frame_times[-1]:3.0f} fps)")
            imgui.same_line()

        imgui.spacing()
        imgui.separator()
        imgui.spacing()

        # TODO disabled https://github.com/ocornut/imgui/issues/211#issuecomment-1245221815
        with imgui.begin_group():
            imgui.drag_float("u_Time", self.uniforms['u_Time'].value)
            imgui.same_line()
            changed, self._time_running = imgui.checkbox(
                'playing' if self._time_running else 'paused',
                self._time_running
            )
            if changed:
                if self._time_running:
                    glfw.set_time(self._start_time)
                else:
                    self._start_time = glfw.get_time()

        imgui.drag_float2('u_Resolution', *self.uniforms['u_Resolution'].value)

        if self._microphone:
            imgui.plot_histogram("u_Microphone",
                                 array('f', self.uniforms['u_Microphone'].value))

        imgui.spacing()
        imgui.separator()
        imgui.spacing()

        # TODO move to Uniform @property range
        def get_range(value: Iterable[T],
                        min_default: T,
                        max_default: T,
                        step_default: T = None):
            match value:
                case (min_, max_):
                    return min_, max_, step_default
                case (min_, max_, step):
                    return min_, max_, step
                case _:
                    return min_default, max_default, step_default

        uniforms = list(self.uniforms.items())
        peaking_uniforms = zip(uniforms, uniforms[1:] + [(None, None)])
        for (name, uniform), (next_name, _) in peaking_uniforms:
            if name in self.FRAGMENT_SHADER_PREAMBLE:
                continue

            flags = 0
            if uniform.widget == 'log':
                flags |=  (imgui.SLIDER_FLAGS_LOGARITHMIC
                           | imgui.SLIDER_FLAGS_NO_ROUND_TO_FORMAT)

            # TODO move to Unifom.imgui??
            match uniform.value, uniform.widget:
                case bool(x), _:
                    _, uniform.value = imgui.checkbox(name, uniform.value)

                case int(x), 'drag':
                    min_, max_, step = get_range(uniform.range, 0, 100, 1)
                    _, uniform.value = imgui.drag_int(
                        name,
                        uniform.value,
                        min_value=min_,
                        max_value=max_,
                        change_speed=step
                    )
                case int(x), _:
                    min_, max_, step = get_range(uniform.range, 0, 100, 1)
                    _, uniform.value = imgui.slider_int(
                        name,
                        uniform.value,
                        min_value=min_,
                        max_value=max_,
                        flags=flags,
                    )

                case float(x), 'drag':
                    min_, max_, step = get_range(uniform.range, 0., 1., 0.01)
                    _, uniform.value = imgui.drag_float(
                        name,
                        uniform.value,
                        min_value=min_,
                        max_value=max_,
                        change_speed=step,
                        flags=flags,
                    )
                case float(x), _:
                    min_, max_, _ = get_range(uniform.range, 0., 1., 0.01)
                    _, uniform.value = imgui.slider_float(
                        name,
                        uniform.value,
                        min_value=min_,
                        max_value=max_,
                        flags=flags,
                    )

                case [float(x), float(y)], 'drag':
                    min_, max_, step = get_range(uniform.range, 0., 1., 0.01)
                    _, uniform.value = imgui.drag_float2(
                        name,
                        *uniform.value,
                        min_value=min_,
                        max_value=max_,
                        change_speed=step,
                        flags=flags
                    )
                case [float(x), float(y)], _:
                    min_, max_, step = get_range(uniform.range, 0., 1., 0.01)
                    _, uniform.value = imgui.slider_float2(
                        name,
                        *uniform.value,
                        min_value=min_,
                        max_value=max_,
                        flags=flags,
                    )

                case [float(x), float(y), float(z)], 'color':
                    _, uniform.value = imgui.color_edit3(name, *uniform.value,
                                                            imgui.COLOR_EDIT_FLOAT)  # pyright: ignore [reportCallIssue]
                case [float(x), float(y), float(z)], 'drag':
                    min_, max_, step = get_range(uniform.range, 0., 1., 0.01)
                    _, uniform.value = imgui.drag_float3(
                        name,
                        *uniform.value,
                        min_value=min_,
                        max_value=max_,
                        change_speed=step
                    )
                case [float(x), float(y), float(z)], _:
                    min_, max_, _ = get_range(uniform.range, 0., 1., 0.01)
                    _, uniform.value = imgui.slider_float3(
                        name,
                        *uniform.value,
                        min_value=min_,
                        max_value=max_,
                        flags=flags,
                    )

                case [float(x), float(y), float(z), float(w)], 'color':
                    _, uniform.value = imgui.color_edit4(name, *uniform.value,
                                                         imgui.COLOR_EDIT_FLOAT)  # pyright: ignore [reportCallIssue]
                case [float(x), float(y), float(z), float(w)], 'drag':
                    min_, max_, step = get_range(uniform.range, 0., 1., 0.01)
                    _, uniform.value = imgui.drag_float4(
                        name,
                        *uniform.value,
                        min_value=min_,
                        max_value=max_,
                        change_speed=step
                    )
                case [float(x), float(y), float(z), float(w)], _:
                    min_, max_, _ = get_range(uniform.range, 0., 1., 0.01)
                    _, uniform.value = imgui.slider_float4(
                        name,
                        *uniform.value,
                        min_value=min_,
                        max_value=max_,
                        flags=flags,
                    )

            # group prefixed uniforms
            if next_name is not None:
                if name.split('_')[0] != next_name.split('_')[0]:
                    imgui.spacing()

        imgui.end()
        imgui.end_frame()

    def _draw_gui(self):
        imgui.render()

    def _draw_shader(self):
        gl.glClear(gl.GL_COLOR_BUFFER_BIT)
        gl.glUseProgram(self.shader_program)

        self.uniforms['u_Resolution'].value = [float(self.width),
                                               float(self.height)]

        # regular `time.time()` is too big for f32, so we just return
        # seconds from program start, glwf does this
        if self._time_running:
            self.uniforms['u_Time'].value = glfw.get_time()

        if self._microphone:
            self.uniforms['u_Microphone'].value = self._microphone.levels

        for uniform in self.uniforms.values():
            uniform.update()

        gl.glBindVertexArray(self.vao)
        gl.glDrawArrays(gl.GL_TRIANGLE_STRIP, 0, len(self.VERTICES))

    def _reload_shader(self):
        if self._file_changed.is_set():
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
        fragment_shader = self._create_shader(gl.GL_FRAGMENT_SHADER,
                                              shader_src)

        self.shader_program = self._create_program(self.vertex_shader,
                                                   fragment_shader)
        gl.glDeleteShader(fragment_shader)

        self.presets = [{'name': "<current>", 'uniforms': {}}]
        for n, line in enumerate(shader_src.split('\n')):
            line = line.strip()

            # <current> uniforms
            if line.startswith('uniform'):
                try:
                    uniform = Uniform.from_def(self.shader_program, line)
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
                    uniform = Uniform.from_def(self.shader_program, line_content)
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
            print("scene:", self._get_shader_title(self._shader_paths[self._shader_index]))
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

    def _print_error(self, e: Exception):
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
            if self._glfw_imgui_renderer is None:
                raise RuntimeError("glfw imgui renderer not initialized!")
            while not glfw.window_should_close(self._window):
                current_time = glfw.get_time()
                num_frames += 1
                if current_time - last_time >= 0.1:
                    self._frame_times.append(100/num_frames)
                    num_frames = 0
                    last_time += 0.1

                glfw.poll_events()
                self._glfw_imgui_renderer.process_inputs()
                self.width, self.height = \
                    glfw.get_framebuffer_size(self._window)

                self._reload_shader()
                self._update_gui()
                self._draw_shader()

                if self._show_gui:
                    self._draw_gui()
                    self._glfw_imgui_renderer.render(imgui.get_draw_data())
                glfw.swap_buffers(self._window)
        except KeyboardInterrupt:
            pass
        finally:
            self.shutdown()

    def shutdown(self):
        # TODO this keeps crashing if not initialized correctly,
        #      do we really need to do this?
        # if self.vao is not None:
        #     gl.glDeleteVertexArrays(1, [self.vao])
        # if self.vbo is not None:
        #     gl.glDeleteBuffers(1, [self.vbo])
        # if self.shader_program is not None:
        #     gl.glDeleteProgram(self.shader_program)
        # if self._glfw_imgui_renderer is not None:
        #     self._glfw_imgui_renderer.shutdown()
        glfw.terminate()

        self._stop.set()

        if self._file_watcher is not None:
            if self._file_watcher.is_alive():
                self._file_watcher.join()

        if self._midi_listener is not None:
            if self._midi_listener.is_alive():
                self._midi_listener.join()

        if self._microphone is not None:
            if self._microphone.is_alive():
                self._microphone.stop()
                self._microphone.join()

    def __del__(self):
        self.shutdown()


def main(argv: Optional[list[str]] = None):
    parser = argparse.ArgumentParser()
    parser.add_argument('shader', nargs='+',
        help='Path to GLSL fragment shader')
    parser.add_argument('-w', '--watch', action='store_true',
        help="Watch for file changes and automatically reload shader")
    parser.add_argument('-m', '--midi', action='store_true',
        help="Listen to MIDI messages for uniform control")
    parser.add_argument('-M', '--midi-mapping',
        help="Path to TOML file with system MIDI mappings")
    # TODO support seelction the microphone
    parser.add_argument('-t', '--mic', action="store_true",
        help="Make microphone levels available as uniform.")
    args = parser.parse_args(argv)


    midi_mapping = {}
    if args.midi_mapping:
        with open(args.midi_mapping, 'rb') as f:
            midi_mapping = tomllib.load(f)

    vhsh_renderer = VHShRenderer(args.shader,
                                 watch=args.watch,
                                 midi=args.midi,
                                 midi_mapping=midi_mapping,
                                 microphone=args.mic)
    vhsh_renderer.run()


if __name__ == "__main__":
    main()
