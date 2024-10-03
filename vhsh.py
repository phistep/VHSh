#!/usr/bin/env python

import sys
import argparse
import re
import time
from typing import Optional
from threading import Thread, Event

from imgui.integrations.glfw import GlfwRenderer
import OpenGL.GL as gl
import glfw
import numpy as np
import imgui
from watchfiles import watch


VertexArrayObject = np.uint32
VertexBufferObject = np.uint32
Shader = int
ShaderProgram = int

class ShaderCompileError(RuntimeError):
    """shader compile error."""


class ProgramLinkError(RuntimeError):
    """program link error."""


class Uniform:
    def __init__(self, program, name, value):
        self.name = name
        self.location = gl.glGetUniformLocation(program, name)
        self.value = value

    def __get__(self):
        return self.value

    # TODO not working?
    def __set__(self, val):
        self.value = val

    def update(self):
        match self.value:
            case int(x):
                gl.glUniform1i(self.location, x)
            case float(x):
                gl.glUniform1f(self.location, x)
            case [float(x), float(y)]:
                gl.glUniform2f(self.location, x, y)
            case [float(x), float(y), float(z)]:
                gl.glUniform3f(self.location, x, y, z)
            case _:
                raise NotImplementedError(f"{self.name} {type(self.value)}: {self.value}")

    def __iter__(self):
        return iter(self.value)


class VHShRenderer:

    NAME = "Video Home Shader"

    VERTICES = np.array([[-1.0,  1.0, 0.0],
                         [-1.0, -1.0, 0.0],
                         [ 1.0,  1.0, 0.0],
                         [ 1.0, -1.0, 0.0]],
                        dtype=np.float32)

    VERTEX_SHADER = """
        #version 330 core

        layout(location = 0) in vec3 VertexPos;

        void main() {
            gl_Position = vec4(VertexPos, 1.0);
        }
    """

    FRAGMENT_SHADER_PREAMBLE = """
        #version 330 core

        out vec4 FragColor;
        uniform vec2 u_Resolution;
        uniform float u_Time;
        #line 1
    """

    FRAGMENT_SHADER = """
        void main() {
            vec2 pos = gl_FragCoord.xy / u_Resolution;
            FragColor = vec4(pos.x, pos.y, 1.0 - (pos.x + pos.y) / 2.0, 1.0);
        }
    """

    def __init__(self, shader_path: str, width=1280, height=720, watch=False):
        self.vao = None
        self.vbo = None
        self.shader_program = None
        self._file_watcher = None
        self._glfw_imgui_renderer = None

        imgui.create_context()  # pyright: ignore
        self._window = self._init_window(width, height, self.NAME)
        self._glfw_imgui_renderer = GlfwRenderer(self._window)

        self._start_time = time.time()

        self.vao, self.vbo = self._create_vertices(self.VERTICES)

        self.vertex_shader = self._create_shader(gl.GL_VERTEX_SHADER,
                                            self.VERTEX_SHADER)

        self._shader_path = shader_path
        self.uniforms = {}
        with open(self._shader_path) as f:
            shader_src = f.read()
        try:
            self.set_shader(shader_src)
        except ShaderCompileError as e:
            self._print_error(e)
            sys.exit(1)

        self._file_changed = Event()
        self._stop_watching = Event()
        if watch:
            self._file_watcher = Thread(target=self._watch_file,
                                        name="VHSh.file_watcher",
                                        args=(self._shader_path,))
            self._file_watcher.start()

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
        print(f"Watching for changes in '{filename}'")
        for _ in watch(filename, stop_event=self._stop_watching):
            # print(f"'{filename}' changed!")
            self._file_changed.set()

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
        imgui.new_frame()  # pyright: ignore
        imgui.begin("Settings", True)  # pyright: ignore [reportAttributeAccessIssue]

        for name, uniform in self.uniforms.items():
            if name in self.FRAGMENT_SHADER_PREAMBLE:
                continue

            match uniform.value:
                case int(x):
                    _, uniform.value = imgui.drag_int(  # pyright: ignore [reportAttributeAccessIssue]
                        name,
                        uniform.value,
                        min_value=0,
                        max_value=100,
                    )
                case float(x):
                    _, uniform.value = imgui.drag_float(  # pyright: ignore [reportAttributeAccessIssue]
                        name,
                        uniform.value,
                        min_value=0.,
                        max_value=1.,
                        change_speed=0.01
                    )
                case [float(x), float(y)]:
                    _, uniform.value = imgui.drag_float2(  # pyright: ignore [reportAttributeAccessIssue]
                        name,
                        *uniform.value,
                        min_value=0.,
                        max_value=1.,
                        change_speed=0.01
                    )
                case [float(x), float(y), float(z)]:
                    _, uniform.value = imgui.drag_float3(  # pyright: ignore [reportAttributeAccessIssue]
                        name,
                        *uniform.value,
                        min_value=0.,
                        max_value=1.,
                        change_speed=0.01
                    )

        imgui.end()  # pyright: ignore [reportAttributeAccessIssue]
        imgui.end_frame()  # pyright: ignore [reportAttributeAccessIssue]

    def _draw_gui(self):
        imgui.render()  # pyright: ignore [reportAttributeAccessIssue]

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

    def _render(self, width, height):
        if self._file_changed.is_set():
            with open(self._shader_path) as f:
                shader_src = f.read()
            # print(f"Read '{self._shader_path}'")
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

        uniform_defs = re.findall(
            # TODO why ^(?!\/\/)\s* not working to ignore comments?
            r'uniform\s+(?P<type>\w+)\s+(?P<name>\w+)\s*;',
            shader_src
        )
        for type_, name in uniform_defs:
            if name not in self.uniforms:
                # TODO if we change the type but not the name, it won't be
                # reloaded -> do store the type
                default = {'int': 0,
                        'float': 0.,
                        'vec2': [0.]*2,
                        'vec3': [0.]*3,
                        'vec4': [0.]*4}
                self.uniforms[name] = Uniform(self.shader_program,
                                            name,
                                            default[type_])

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

            self._update_gui()
            self._draw_shader()
            self._draw_gui()

            self._glfw_imgui_renderer.render(imgui.get_draw_data())  # pyright: ignore [reportAttributeAccessIssue]
            glfw.swap_buffers(self._window)

    def __del__(self):
        if self.vao is not None:
            gl.glDeleteVertexArrays(1, [self.vao])
        if self.vbo is not None:
            gl.glDeleteBuffers(1, [self.vbo])
        if self.shader_program is not None:
            gl.glDeleteProgram(self.shader_program)
        if self._glfw_imgui_renderer is not None:
            self._glfw_imgui_renderer.shutdown()
        glfw.terminate()

        if self._file_watcher is not None:
            self._stop_watching.set()
            if self._file_watcher.is_alive():
                self._file_watcher.join()


def main(argv: Optional[list[str]] = None):
    parser = argparse.ArgumentParser()
    parser.add_argument('shader', help='Path to GLSL fragment shader')
    parser.add_argument('-w', '--watch', action='store_true',
        help="Watch for file changes and automatically reload shader")
    args = parser.parse_args(argv)

    vhsh_renderer = VHShRenderer(args.shader, watch=args.watch)
    vhsh_renderer.run()


if __name__ == "__main__":
    main()
