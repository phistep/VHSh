#!/usr/bin/env python
"""https://github.com/pyimgui/pyimgui/blob/master/doc/examples/integrations_glfw3.py"""
import sys

from imgui.integrations.glfw import GlfwRenderer
import OpenGL.GL as gl
import glfw
import imgui

frag_shader_source = """\
#version 410
out vec4 fragColor;
void main() {
	fragColor = vec4(1., 0.5, 0.1, 1.0);
}
"""


def main():
    imgui.create_context()
    window = impl_glfw_init()
    impl = GlfwRenderer(window)

    show_custom_window = True

    values = (1., 1., 1.,)

    init_opengl()

    while not glfw.window_should_close(window):
        glfw.poll_events()
        impl.process_inputs()

        imgui.new_frame()

        ######

        # open new window context
        imgui.begin("Your first window!", True)
        # draw text label inside of current window
        imgui.text("Hello world!")
        changed, values = imgui.drag_float3("Drag Float", *values,
                                            min_value=0.,
                                            max_value=1.,
                                            change_speed=0.01)
        imgui.end()

        ######

        imgui.render()
        imgui.end_frame()





        #gl.glClearColor(*values, 1)
        #gl.glClear(gl.GL_COLOR_BUFFER_BIT)

        gl.glClear(gl.GL_COLOR_BUFFER_BIT)
        gl.glBegin(gl.GL_TRIANGLE_STRIP)
        cx, cy = center = 0.5, 0.
        scale = 1
        for x in [-1, 1]:
            for y in [-1, 1]:
                gl.glTexCoord(x/scale-cx, y/scale-cy)
                gl.glVertex(x, y)
        #gl.glEnd()

        imgui.render()
        impl.render(imgui.get_draw_data())
        glfw.swap_buffers(window)

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


def init_opengl():
	frag_shader = create_shader(frag_shader_source, gl.GL_FRAGMENT_SHADER)
	program = create_program(frag_shader)
	gl.glUseProgram(program)

    #gl.glEnable(gl.GL_TEXTURE_1D)
	#gl.glActiveTexture(gl.GL_TEXTURE0+0)
	#gl.glBindTexture(gl.GL_TEXTURE_1D, gl.glGenTextures(1))
	#gl.glTexParameteri(gl.GL_TEXTURE_1D, gl.GL_TEXTURE_MIN_FILTER, gl.GL_NEAREST)
	#gl.glTexParameteri(gl.GL_TEXTURE_1D, gl.GL_TEXTURE_MAG_FILTER, gl.GL_NEAREST)

	#gl.glTexImage1D(GL_TEXTURE_1D, 0, GL_LUMINANCE, 16, 0,
	#             GL_LUMINANCE, GL_UNSIGNED_BYTE,
	#             "".join(chr(c*16) for c in range(16)))


class ShaderCompileError(RuntimeError):
	"""shader compile error."""
	
class ProgramLinkError(RuntimeError):
	"""program link error."""

def create_shader(shader_source, shader_type):
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
