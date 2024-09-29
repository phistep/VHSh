# ChatGPT

import glfw
from OpenGL.GL import *
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
    FragColor = vec4(0.4, 0.7, 0.2, 1.0);  // Greenish color
}
"""

# Function to compile shaders
def compile_shader(shader_type, source):
    shader = glCreateShader(shader_type)
    glShaderSource(shader, source)
    glCompileShader(shader)

    # Check for compilation errors
    if not glGetShaderiv(shader, GL_COMPILE_STATUS):
        raise RuntimeError(glGetShaderInfoLog(shader).decode('utf-8'))

    return shader

# Initialize GLFW
if not glfw.init():
    raise Exception("GLFW could not be initialized")

# Set OpenGL version to 3.3 and core profile
glfw.window_hint(glfw.CONTEXT_VERSION_MAJOR, 3)
glfw.window_hint(glfw.CONTEXT_VERSION_MINOR, 3)
glfw.window_hint(glfw.OPENGL_PROFILE, glfw.OPENGL_CORE_PROFILE)

# For MacOS, this is necessary
glfw.window_hint(glfw.OPENGL_FORWARD_COMPAT, GL_TRUE)

# Create a windowed mode window and its OpenGL context
window = glfw.create_window(640, 480, "Minimal OpenGL Example", None, None)
if not window:
    glfw.terminate()
    raise Exception("GLFW window could not be created")

# Make the window's context current
glfw.make_context_current(window)

# Vertex data (triangle)
vertices = np.array([
    [-0.5, -0.5, 0.0],
    [ 0.5, -0.5, 0.0],
    [ 0.0,  0.5, 0.0]
], dtype=np.float32)

# Create Vertex Array Object and Vertex Buffer Object
VAO = glGenVertexArrays(1)
VBO = glGenBuffers(1)

glBindVertexArray(VAO)

glBindBuffer(GL_ARRAY_BUFFER, VBO)
glBufferData(GL_ARRAY_BUFFER, vertices.nbytes, vertices, GL_STATIC_DRAW)

# Specify the layout of the vertex data
glVertexAttribPointer(0, 3, GL_FLOAT, GL_FALSE, 3 * 4, ctypes.c_void_p(0))
glEnableVertexAttribArray(0)

# Unbind the VAO
glBindVertexArray(0)

# Compile shaders and create the shader program
vertex_shader = compile_shader(GL_VERTEX_SHADER, vertex_shader_src)
fragment_shader = compile_shader(GL_FRAGMENT_SHADER, fragment_shader_src)

shader_program = glCreateProgram()
glAttachShader(shader_program, vertex_shader)
glAttachShader(shader_program, fragment_shader)
glLinkProgram(shader_program)

# Check for linking errors
if not glGetProgramiv(shader_program, GL_LINK_STATUS):
    raise RuntimeError(glGetProgramInfoLog(shader_program).decode('utf-8'))

# Delete the shaders as they're linked into our program now and no longer needed
glDeleteShader(vertex_shader)
glDeleteShader(fragment_shader)

# Main loop
while not glfw.window_should_close(window):
    # Poll for and process events
    glfw.poll_events()

    # Clear the screen
    glClear(GL_COLOR_BUFFER_BIT)

    # Use the shader program and draw the triangle
    glUseProgram(shader_program)
    glBindVertexArray(VAO)
    glDrawArrays(GL_TRIANGLES, 0, 3)

    # Swap front and back buffers
    glfw.swap_buffers(window)

# Cleanup
glDeleteVertexArrays(1, [VAO])
glDeleteBuffers(1, [VBO])
glDeleteProgram(shader_program)

glfw.terminate()
