#!/usr/bin/env python
"""https://github.com/pyimgui/pyimgui/blob/master/doc/examples/integrations_glfw3.py"""
import sys

from OpenGL.raw.GL.VERSION.GL_2_0 import glUseProgram
from imgui.integrations.glfw import GlfwRenderer
import OpenGL.GL as gl
import glfw
import imgui
import numpy as np

# Vertex Shader (GLSL)
vertex_shader_src = """
#version 330 core
layout(location = 0) in vec3 aPos;
void main() {
    gl_Position = vec4(aPos, 1.0);
}
"""

# Fragment Shader (GLSL)
fragment_shader_src = """
#version 330 core
out vec4 FragColor;
void main() {
    vec2 pos = gl_FragCoord.xy / vec2(1280., 720.);
    FragColor = vec4(pos.x, pos.y, 1., 1.0);
}
"""

def main():
    imgui.create_context()
    window = impl_glfw_init()
    impl = GlfwRenderer(window)

    show_custom_window = True

    values = (1., 1., 1.,)

    VAO, VBO = create_vertices()
    vertex_shader = create_shader(gl.GL_VERTEX_SHADER, vertex_shader_src)
    fragment_shader = create_shader(gl.GL_FRAGMENT_SHADER, fragment_shader_src)
    shader_program = create_program(vertex_shader, fragment_shader)
    # TODO delete shaders gl.glDeleteShader(fragment_shader)

    while not glfw.window_should_close(window):
        glfw.poll_events()
        # impl.process_inputs()

        # imgui.new_frame()

        ######

        # open new window context
        # imgui.begin("Your first window!", True)
        # # draw text label inside of current window
        # imgui.text("Hello world!")
        # changed, values = imgui.drag_float3("Drag Float", *values,
        #                                     min_value=0.,
        #                                     max_value=1.,
        #                                     change_speed=0.01)
        # imgui.end()

        ######

        # imgui.render()
        # imgui.end_frame()

        gl.glClear(gl.GL_COLOR_BUFFER_BIT)

        gl.glUseProgram(shader_program)
        gl.glBindVertexArray(VAO)
        gl.glDrawArrays(gl.GL_TRIANGLES, 0, 3)

        # imgui.render()
        # impl.render(imgui.get_draw_data())
        glfw.swap_buffers(window)

    # TODO
    # glDeleteVertexArrays(1, [VAO])
    # glDeleteBuffers(1, [VBO])
    # glDeleteProgram(shader_program)

    impl.shutdown()
    glfw.terminate()


def impl_glfw_init():
    width, height = 1280, 720
    window_name = "minimal ImGui/GLFW3 example"

    if not glfw.init():
        print("Could not initialize OpenGL context")
        sys.exit(1)

    # OS X supports only forward-compatible core profiles from 3.2
    glfw.window_hint(glfw.CONTEXT_VERSION_MAJOR, 3)
    glfw.window_hint(glfw.CONTEXT_VERSION_MINOR, 3)
    glfw.window_hint(glfw.OPENGL_PROFILE, glfw.OPENGL_CORE_PROFILE)

    glfw.window_hint(glfw.OPENGL_FORWARD_COMPAT, gl.GL_TRUE)

    # Create a windowed mode window and its OpenGL context
    window = glfw.create_window(int(width), int(height), window_name, None, None)
    glfw.make_context_current(window)

    if not window:
        glfw.terminate()
        print("Could not initialize Window")
        sys.exit(1)

    return window


class ShaderCompileError(RuntimeError):
	"""shader compile error."""

class ProgramLinkError(RuntimeError):

	"""program link error."""

def create_vertices():
    # Vertex data (triangle)
    vertices = np.array([
        [-0.5, -0.5, 0.0],
        [ 0.5, -0.5, 0.0],
        [ 0.0,  0.5, 0.0]
    ], dtype=np.float32)

    # Create Vertex Array Object and Vertex Buffer Object
    VAO = gl.glGenVertexArrays(1)
    VBO = gl.glGenBuffers(1)

    gl.glBindVertexArray(VAO)

    gl.glBindBuffer(gl.GL_ARRAY_BUFFER, VBO)
    gl.glBufferData(gl.GL_ARRAY_BUFFER, vertices.nbytes, vertices, gl.GL_STATIC_DRAW)

    # Specify the layout of the vertex data
    gl.glVertexAttribPointer(0, 3, gl.GL_FLOAT, gl.GL_FALSE, 3 * 4, gl.ctypes.c_void_p(0))
    gl.glEnableVertexAttribArray(0)

    # Unbind the VAO
    gl.glBindVertexArray(0)

    return VAO, VBO


def create_shader(shader_type, shader_source):
	"""creates a shader from its source & type."""
	shader = gl.glCreateShader(shader_type)
	gl.glShaderSource(shader, shader_source)
	gl.glCompileShader(shader)
	if gl.glGetShaderiv(shader, gl.GL_COMPILE_STATUS) != gl.GL_TRUE:
		raise ShaderCompileError(gl.glGetShaderInfoLog(shader))
	return shader


def create_program(*shaders):
	"""creates a program from its vertex & fragment shader sources."""
	program = gl.glCreateProgram()
	for shader in shaders:
		gl.glAttachShader(program, shader)
	gl.glLinkProgram(program)
	if gl.glGetProgramiv(program, gl.GL_LINK_STATUS) != gl.GL_TRUE:
		raise ProgramLinkError(gl.glGetProgramInfoLog(program))
	return program


if __name__ == "__main__":
    main()
