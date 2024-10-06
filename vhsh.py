#!/usr/bin/env python

import sys
import argparse
import re
import time
import warnings
from typing import Optional, TypeVar, Iterable, get_args, Literal, Callable, Any, Self
from threading import Thread, Event, Lock
from pprint import pprint
from textwrap import dedent

from OpenGL.raw.GL.VERSION.GL_4_0 import glUniform1d
from imgui.integrations.glfw import GlfwRenderer
import OpenGL.GL as gl
import glfw
import numpy as np
import imgui

# `watchfiles` imported conditionally in VHShRenderer._watch_file()
# `mido` imported conditionally in VHShRenderer._midi_listen()


VertexArrayObject = np.uint32
VertexBufferObject = np.uint32
Shader = int
ShaderProgram = int

GLSLBool = bool
GLSLInt = int
GLSLFloat = float
GLSLVec2 = tuple[float, float]
GLSLVec3 = tuple[float, float, float]
GLSLVec4 = tuple[float, float, float, float]

T = TypeVar('T')
UniformT = TypeVar('UniformT', GLSLBool, GLSLInt, GLSLVec2, GLSLVec3, GLSLVec4)


class ShaderCompileError(RuntimeError):
    """shader compile error."""


class UniformIntializationError(ShaderCompileError):
    """custom uniform initialization error."""


class ProgramLinkError(RuntimeError):
    """program link error."""


class Uniform:

    def __init__(self,
                 program: ShaderProgram,
                 type_: str,
                 name: str,
                 value: Optional[UniformT] = None,
                 default: Optional[UniformT] = None,
                 range: Optional[list[UniformT]] = None,
                 widget: Optional[str] = None):
        self._location: int = gl.glGetUniformLocation(program, name)
        self.type = type_
        self._type = None
        self.name = name
        self.default = default
        self.range = range
        self.widget = widget

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
        return (f"uniform {self.type} {self.name};"
                f"  // ={self.value} {self.range} <{self.widget}> ")

    def __repr__(self):
        return (f'Uniform'
                f' type={self.type}'
                f' name="{self.name}"'
                f' value={self.value}'
                f' default={self.default}'
                f' range={self.range}'
                f' widget={self.widget}'
                f' _type={self._type}'
                f' _glUniform={self._glUniform.__name__}'
                f' at 0x{self._location:04x}>')

    def update(self):
        args = self.value
        if not isinstance(args, Iterable):
            args = [args]
        self._glUniform(self._location, *args)

    @classmethod
    def from_def(cls,
                shader_program: ShaderProgram,
                definition: str) -> 'Uniform':

        try:
            matches = re.search(
                # TODO why ^(?!\/\/)\s* not working to ignore comments?
                (r'uniform\s+(?P<type>\w+)\s+(?P<name>\w+)\s*;'
                 r'(?:\s*//\s*(?:'
                 r'(?P<widget><\w+>)?\s*)?'
                 r'(?P<default>=(?:\S+|\([^\)]+\)))?'
                 r'\s*(?P<range>\[[^\]]+\])?'
                 r')?'),
                definition
            )
            type_, name, widget, value_s, range_s = matches.groups()
        except Exception as e:
            raise UniformIntializationError(
                f"Syntax error in metadata defintion: {definition}") from e

        if widget is not None:
            widget = widget.strip('<>')
        try:
            default = (eval(value_s.lstrip('='))
                        if value_s
                        else None)
        except SyntaxError as e:
            raise UniformIntializationError(
                f"Invalid 'value' metadata for uniform"
                f" '{name}': {e}: {value_s}"
            ) from e
        try:
            range = eval(range_s) if range_s else None
        except SyntaxError as e:
            raise UniformIntializationError(
                f"Invalid 'range' metadata for uniform"
                f" '{name}': {e}: {range_s!r}"
            ) from e

        uniform = Uniform(program=shader_program,
                          type_=type_,
                          name=name,
                          default=default,
                          range=range,
                          widget=widget)


        return uniform

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
                 shader_path: str,
                 width: int = 1280,
                 height: int = 720,
                 watch: bool = False,
                 midi: bool = False):
        self.vao = None
        self.vbo = None
        self.shader_program = None
        self._shader_path = shader_path
        self._stop = Event()
        self._uniform_lock = Lock()
        self._file_watcher = None
        self._midi_listener = None
        self._glfw_imgui_renderer = None
        self._lineno_offset = \
            len(self.FRAGMENT_SHADER_PREAMBLE.splitlines()) + 1

        imgui.create_context()
        self._window = self._init_window(width, height, self.NAME)
        self._glfw_imgui_renderer = GlfwRenderer(self._window)

        self._start_time = time.time()

        self.vao, self.vbo = self._create_vertices(self.VERTICES)

        self.vertex_shader = self._create_shader(gl.GL_VERTEX_SHADER,
                                            self.VERTEX_SHADER)

        self.uniforms = {}
        with open(self._shader_path) as f:
            shader_src = f.read()
        try:
            self.set_shader(shader_src)
        except ShaderCompileError as e:
            self._print_error(e)
            sys.exit(1)

        self._file_changed = Event()
        if watch:
            self._file_watcher = Thread(target=self._watch_file,
                                        name="VHSh.file_watcher",
                                        args=(self._shader_path,))
            self._file_watcher.start()

        if midi:
            self._midi_listener = Thread(target=self._midi_listen,
                                         name="VHSh.midi_listener",
                                         )#args=(,))
            self._midi_listener.start()

    @staticmethod
    def _init_window(width, height, name):
        if not glfw.init():
            RuntimeError("GLFW could not initialize OpenGL context")

        # OS X supports only forward-compatible core profiles from 3.2
        glfw.window_hint(glfw.CONTEXT_VERSION_MAJOR, 3)
        glfw.window_hint(glfw.CONTEXT_VERSION_MINOR, 3)
        glfw.window_hint(glfw.OPENGL_PROFILE, glfw.OPENGL_CORE_PROFILE)

        glfw.window_hint(glfw.OPENGL_FORWARD_COMPAT, gl.GL_TRUE)

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

    def _midi_listen(self):
        import mido
        print("Listening for MIDI messages...")

        # TODO metadata parsing
        mapping = {
            0: "scale",
            16: "n_max"
        }

        with mido.open_input() as inport:
            while True:
                if self._stop.is_set():
                    break
                for msg in inport.iter_pending():
                    # print(f"Received MIDI message: {msg}")
                    # print(f"Received MIDI message: @{msg.control} = {msg.value}")

                    try:
                        # TODO range mapping
                        self._uniform_lock.acquire()
                        self.uniforms[mapping[msg.control]].value = float(msg.value / 127)
                    except KeyError as e:
                        print(f"MIDI mapping not found for {msg.control}: {e}")
                        print(msg)
                        pprint(mapping)
                    finally:
                        self._uniform_lock.release()

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

    def _update_gui(self):
        imgui.new_frame()
        imgui.begin("Paramters", True)

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

        for name, uniform in self.uniforms.items():
            if name in self.FRAGMENT_SHADER_PREAMBLE:
                continue

            match uniform.value, uniform.widget:
                case bool(x), _:
                    _, uniform.value = imgui.checkbox(name, uniform.value)
                case int(x), _:
                    min_, max_, step = get_range(uniform.range, 0, 100, 1)
                    _, uniform.value = imgui.drag_int(
                        name,
                        uniform.value,
                        min_value=min_,
                        max_value=max_,
                        change_speed=step
                    )
                case float(x), _:
                    min_, max_, step = get_range(uniform.range, 0., 1., 0.01)
                    _, uniform.value = imgui.drag_float(
                        name,
                        uniform.value,
                        min_value=min_,
                        max_value=max_,
                        change_speed=step
                    )
                case [float(x), float(y)], _:
                    min_, max_, step = get_range(uniform.range, 0., 1., 0.01)
                    _, uniform.value = imgui.drag_float2(
                        name,
                        *uniform.value,
                        min_value=min_,
                        max_value=max_,
                        change_speed=step
                    )
                case [float(x), float(y), float(z)], _:
                    min_, max_, step = get_range(uniform.range, 0., 1., 0.01)
                    _, uniform.value = imgui.drag_float3(
                        name,
                        *uniform.value,
                        min_value=min_,
                        max_value=max_,
                        change_speed=step
                    )
                case [float(x), float(y), float(z), float(w)], 'color':
                    _, uniform.value = imgui.color_edit4(name, *uniform.value,
                                                         imgui.COLOR_EDIT_FLOAT)  # pyright: ignore [reportCallIssue]
                case [float(x), float(y), float(z), float(w)], _:
                    min_, max_, step = get_range(uniform.range, 0., 1., 0.01)
                    _, uniform.value = imgui.drag_float4(
                        name,
                        *uniform.value,
                        min_value=min_,
                        max_value=max_,
                        change_speed=step
                    )

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
        # seconds from program start
        self.uniforms['u_Time'].value = time.time() - self._start_time
        for uniform in self.uniforms.values():
            uniform.update()

        gl.glBindVertexArray(self.vao)
        gl.glDrawArrays(gl.GL_TRIANGLE_STRIP, 0, len(self.VERTICES))

    def _reload_shader(self):
        if self._file_changed.is_set():
            with open(self._shader_path) as f:
                shader_src = f.read()
            self._file_changed.clear()
            try:
                self.set_shader(shader_src)
            except ShaderCompileError as e:
                self._print_error(e)
            else:
                print("\x1b[2;32mOK:"
                      f" \x1b[2;37m{self._shader_path}"
                      "\x1b[0;0m")

    def set_shader(self, shader_src: str):
        shader_src = self.FRAGMENT_SHADER_PREAMBLE + shader_src
        fragment_shader = self._create_shader(gl.GL_FRAGMENT_SHADER,
                                              shader_src)

        self.shader_program = self._create_program(self.vertex_shader,
                                                   fragment_shader)
        gl.glDeleteShader(fragment_shader)

        for n, line in enumerate(shader_src.split('\n')):
            if line.strip().startswith('uniform'):
                try:
                    uniform = Uniform.from_def(self.shader_program, line)
                    try:
                        self._uniform_lock.acquire()
                        self.uniforms[uniform.name] = uniform
                    finally:
                        self._uniform_lock.release()
                except UniformIntializationError as e:
                    lineno = n - self._lineno_offset
                    raise ShaderCompileError(f"ERROR 0:{lineno}: {e}")


        try:
            self._uniform_lock.acquire()
            for uniform in self.uniforms.values():
                print(uniform)
        finally:
            self._uniform_lock.release()

    def _print_error(self, e: Exception):
        try:
            lines = str(e).strip().split('\n')
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
        if self._glfw_imgui_renderer is None:
            raise RuntimeError("glfw imgui renderer not initialized!")
        while not glfw.window_should_close(self._window):
            glfw.poll_events()
            self._glfw_imgui_renderer.process_inputs()
            self.width, self.height = \
                glfw.get_framebuffer_size(self._window)

            self._reload_shader()
            self._update_gui()
            self._draw_shader()
            self._draw_gui()

            self._glfw_imgui_renderer.render(imgui.get_draw_data())
            glfw.swap_buffers(self._window)

    def __del__(self):
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


def main(argv: Optional[list[str]] = None):
    parser = argparse.ArgumentParser()
    parser.add_argument('shader', help='Path to GLSL fragment shader')
    parser.add_argument('-w', '--watch', action='store_true',
        help="Watch for file changes and automatically reload shader")
    parser.add_argument('-m', '--midi', action='store_true',
        help="Listen to MIDI messages for uniform control")
    args = parser.parse_args(argv)

    vhsh_renderer = VHShRenderer(args.shader, watch=args.watch, midi=args.midi)
    vhsh_renderer.run()


if __name__ == "__main__":
    main()
