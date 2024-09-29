#!/usr/bin/env python

import sys

from OpenGL.raw.GL.VERSION.GL_2_0 import glUseProgram
from imgui.integrations.glfw import GlfwRenderer
import OpenGL.GL as gl
import glfw
import numpy as np
import imgui

# TODO
# - full screen triangles
# - read shader src from file
# - classes for Shader, ShaderProgram, etc
# - hot reload
# - auto-generate imgui from uniforms


WIDTH = 1280
HEIGHT = 720

VERTICES = np.array([[-0.5, -0.5, 0.0],
                     [ 0.5, -0.5, 0.0],
                     [ 0.0,  0.5, 0.0]],
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
uniform vec3 u_tunable;

void main() {
    vec2 pos = gl_FragCoord.xy / u_Resolution;
    FragColor = vec4(
        pos.x * u_tunable.x,
        pos.y * u_tunable.y,
        u_tunable.z,
        1.0
    );
}
"""


class ShaderCompileError(RuntimeError):
	"""shader compile error."""


class ProgramLinkError(RuntimeError):
	"""program link error."""


def init_window(width, height, name):
    if not glfw.init():
        print("Could not initialize OpenGL context")
        sys.exit(1)

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
        print("Could not initialize Window")
        sys.exit(1)

    return window


def create_vertices(vertices: np.array):
    # Create Vertex Array Object and Vertex Buffer Object
    vao = gl.glGenVertexArrays(1)
    vbo = gl.glGenBuffers(1)

    gl.glBindVertexArray(vao)

    gl.glBindBuffer(gl.GL_ARRAY_BUFFER, vbo)
    gl.glBufferData(target=gl.GL_ARRAY_BUFFER,
                    size=vertices.nbytes,
                    data=vertices,
                    usage=gl.GL_STATIC_DRAW)

    # Specify the layout of the vertex data
    gl.glVertexAttribPointer(index=0,
                             size=3,
                             type=gl.GL_FLOAT,
                             normalized=gl.GL_FALSE,
                             stride=3 * 4,  # (x, y, z) * sizeof(GL_FLOAT)  # TODO
                             pointer=gl.ctypes.c_void_p(0))
    gl.glEnableVertexAttribArray(0)

    # Unbind the VAO
    gl.glBindVertexArray(0)

    return vao, vbo


def create_shader(shader_type, shader_source):
	"""creates a shader from its source & type."""
	shader = gl.glCreateShader(shader_type)
	gl.glShaderSource(shader, shader_source)
	gl.glCompileShader(shader)
	if gl.glGetShaderiv(shader, gl.GL_COMPILE_STATUS) != gl.GL_TRUE:
		raise ShaderCompileError(gl.glGetShaderInfoLog(shader).decode('utf-8'))
	return shader


def create_program(*shaders):
	"""creates a program from its vertex & fragment shader sources."""
	program = gl.glCreateProgram()
	for shader in shaders:
		gl.glAttachShader(program, shader)
	gl.glLinkProgram(program)
	if gl.glGetProgramiv(program, gl.GL_LINK_STATUS) != gl.GL_TRUE:
		raise ProgramLinkError(gl.glGetProgramInfoLog(program).decode('utf-8'))
	return program


def main():
    imgui.create_context()
    window = init_window(WIDTH, HEIGHT, "Triangle Demo")
    renderer = GlfwRenderer(window)

    tunable_values = (1., 1., 1.,)

    vao, vbo = create_vertices(VERTICES)
    vertex_shader = create_shader(gl.GL_VERTEX_SHADER, VERTEX_SHADER)
    fragment_shader = create_shader(gl.GL_FRAGMENT_SHADER, FRAGMENT_SHADER)
    shader_program = create_program(vertex_shader, fragment_shader)
    gl.glDeleteShader(vertex_shader)
    gl.glDeleteShader(fragment_shader)

    uniform_location = {
        'u_Resolution': gl.glGetUniformLocation(shader_program, "u_Resolution"),
        'u_tunable': gl.glGetUniformLocation(shader_program, "u_tunable"),
    }

    while not glfw.window_should_close(window):
        glfw.poll_events()
        renderer.process_inputs()
        width, height = glfw.get_framebuffer_size(window)

        # layout imgui
        imgui.new_frame()

        # open new window context
        imgui.begin("Settings", True)
        # draw text label inside of current window
        imgui.text("tunable_values")
        changed, tunable_values = imgui.drag_float3(
            "RGB",
            *tunable_values,
            min_value=0.,
            max_value=1.,
            change_speed=0.01
        )
        imgui.end()

        imgui.end_frame()

        gl.glClear(gl.GL_COLOR_BUFFER_BIT)


        # draw OpenGL
        gl.glUseProgram(shader_program)
        gl.glUniform2f(uniform_location["u_Resolution"], float(width), float(height))
        gl.glUniform3f(uniform_location["u_tunable"], *tunable_values)
        gl.glBindVertexArray(vao)
        gl.glDrawArrays(gl.GL_TRIANGLES, 0, len(VERTICES))


        # render imgui on top
        imgui.render()
        renderer.render(imgui.get_draw_data())
        glfw.swap_buffers(window)

    gl.glDeleteVertexArrays(1, [vao])
    gl.glDeleteBuffers(1, [vbo])
    gl.glDeleteProgram(shader_program)

    renderer.shutdown()
    glfw.terminate()


if __name__ == "__main__":
    main()
