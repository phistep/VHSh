#!/usr/bin/env python

import sys
import argparse
from typing import Optional

from OpenGL.raw.GL.VERSION.GL_2_0 import glUseProgram
from imgui.integrations.glfw import GlfwRenderer
import OpenGL.GL as gl
import glfw
import numpy as np
import imgui

# TODO
# - auto-generate imgui from uniforms
#      manual reges or https://github.com/anentropic/python-glsl-shaderinfo
# - imgui display shader compile errors
# - hot reload
#     https://watchfiles.helpmanual.io/api/watch/
# - select different shaders
# - save and load different presets (toml in the shader file?)
# - split into runtime and imgui viewer
#     maybe just have option to show or hide the controls as separate window
#       https://github.com/ocornut/imgui/wiki/Multi-Viewports
#       https://github.com/ocornut/imgui/blob/docking/examples/example_glfw_opengl2/main.cpp
# - uniforms
#   - time
#   - prev frame
#   - audio fft
#   - video in
# - raspberry pi midi or gpio support

VertexArrayObject = np.uint32
VertexBufferObject = np.uint32
Shader = int
ShaderProgram = int

WIDTH = 1280
HEIGHT = 720


class ShaderCompileError(RuntimeError):
    """shader compile error."""


class ProgramLinkError(RuntimeError):
    """program link error."""


def init_window(width, height, name):
    if not glfw.init():
        RuntimeError("GLFW could not initialize OpenGL context")

    # OS X supports only forward-compatible core profiles from 3.2
    glfw.window_hint(glfw.CONTEXT_VERSION_MAJOR, 3)
    glfw.window_hint(glfw.CONTEXT_VERSION_MINOR, 3)
    glfw.window_hint(glfw.OPENGL_PROFILE, glfw.OPENGL_CORE_PROFILE)

    glfw.window_hint(glfw.OPENGL_FORWARD_COMPAT, gl.GL_TRUE)

    # Create a windowed mode window and its OpenGL context
    window = glfw.create_window(int(width), int(height), name, None, None)
    glfw.make_context_current(window)

    if not window:
        glfw.terminate()
        raise RuntimeError("GLFW could not initialize Window")

    return window


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
            case [float(x), float(y)]:
                gl.glUniform2f(self.location, x, y)
            case [float(x), float(y), float(z)]:
                gl.glUniform3f(self.location, x, y, z)
            case _:
                raise NotImplementedError(f"{type(self.value)}: {self.value}")

    def __iter__(self):
        return iter(self.value)


class VHSRenderer:

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

    FRAGMENT_SHADER = """
        #version 330 core

        out vec4 FragColor;
        uniform vec2 u_Resolution;

        void main() {
            vec2 pos = gl_FragCoord.xy / u_Resolution;
            FragColor = vec4(pos.x, pos.y, 1.0 - (pos.x + pos.y) / 2.0, 1.0);
        }
    """

    def __init__(self):
        self.vao, self.vbo = self._create_vertices(self.VERTICES)

        self.vertex_shader = self._create_shader(gl.GL_VERTEX_SHADER,
                                            self.VERTEX_SHADER)
        fragment_shader = self._create_shader(gl.GL_FRAGMENT_SHADER,
                                              self.FRAGMENT_SHADER)
        self.shader_program = self._create_program(self.vertex_shader,
                                                   fragment_shader)

        # we keep the vertex shader object around, so we can re-use it when
        # updating the fragment shader
        gl.glDeleteShader(fragment_shader)

        self.uniforms = {
            'u_Resolution': Uniform(self.shader_program, 'u_Resolution',
                                    [float(WIDTH), float(HEIGHT)]),
            'u_color': Uniform(self.shader_program, 'u_color', [1., 1., 1.])
        }

    def __del__(self):
        gl.glDeleteVertexArrays(1, [self.vao])
        gl.glDeleteBuffers(1, [self.vbo])
        gl.glDeleteProgram(self.shader_program)

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
            raise ShaderCompileError(gl.glGetShaderInfoLog(shader).decode('utf-8'))
        return shader

    def _create_program(self, *shaders) -> ShaderProgram:
        """creates a program from its vertex & fragment shader sources."""
        program = gl.glCreateProgram()
        for shader in shaders:
            gl.glAttachShader(program, shader)
        gl.glLinkProgram(program)
        if gl.glGetProgramiv(program, gl.GL_LINK_STATUS) != gl.GL_TRUE:
            raise ProgramLinkError(gl.glGetProgramInfoLog(program).decode('utf-8'))
        return program

    def _update_gui(self):
        imgui.new_frame()
        imgui.begin("Settings", True)
        imgui.text("color")
        changed, color = imgui.drag_float3(
            "RGB",
            *self.uniforms['u_color'].value,
            min_value=0.,
            max_value=1.,
            change_speed=0.01
        )
        self.uniforms['u_color'].value = color
        imgui.end()
        imgui.end_frame()

    def _draw_gui(self):

        imgui.render()

    def _draw_shader(self, width: float, height: float):
        self.uniforms['u_Resolution'].value = [float(width), float(height)]
        gl.glClear(gl.GL_COLOR_BUFFER_BIT)
        gl.glUseProgram(self.shader_program)
        for uniform in self.uniforms.values():
            uniform.update()
        gl.glBindVertexArray(self.vao)
        gl.glDrawArrays(gl.GL_TRIANGLE_STRIP, 0, len(self.VERTICES))

    def render(self, width, height):
        self._update_gui()
        self._draw_shader(width, height)
        self._draw_gui()

    def set_shader(self, shader_src: str):
        fragment_shader = self._create_shader(gl.GL_FRAGMENT_SHADER,
                                              shader_src)
        self.shader_program = self._create_program(self.vertex_shader,
                                                   fragment_shader)
        gl.glDeleteShader(fragment_shader)


def main(argv: Optional[list[str]] = None):
    parser = argparse.ArgumentParser()
    parser.add_argument('shader', help='Path to GLSL fragment shader')
    args = parser.parse_args(argv)

    imgui.create_context()
    window = init_window(WIDTH, HEIGHT, "VHShaderboi")
    glfw_imgui_renderer = GlfwRenderer(window)

    vhs_renderer = VHSRenderer()

    if args.shader:
        with open(args.shader) as f:
            shader_src = f.read()
        vhs_renderer.set_shader(shader_src)

    while not glfw.window_should_close(window):
        glfw.poll_events()
        glfw_imgui_renderer.process_inputs()
        width, height = glfw.get_framebuffer_size(window)

        vhs_renderer.render(width, height)

        glfw_imgui_renderer.render(imgui.get_draw_data())
        glfw.swap_buffers(window)

    del vhs_renderer
    glfw_imgui_renderer.shutdown()
    glfw.terminate()


if __name__ == "__main__":
    main()
