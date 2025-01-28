import re
from typing import Optional, Any, Callable, Iterable, get_args
from textwrap import dedent

import OpenGL.GL as gl
import numpy as np

from .gl_types import (
    VertexArrayObject, VertexBufferObject,
    Shader, ShaderProgram,
    GLSLBool, GLSLInt, GLSLFloat, GLSLVec2, GLSLVec3, GLSLVec4,
    UniformT
)

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


class Renderer:

    # TODO use stdlib arrays, make np optional for mic input
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

    def __init__(self):
        self.vao, self.vbo = self._create_vertices(self.VERTICES)
        self.vertex_shader = self._create_shader(gl.GL_VERTEX_SHADER,
                                                 self.VERTEX_SHADER)

    @staticmethod
    def _create_vertices(vertices: np.ndarray) -> tuple[VertexArrayObject,
                                                        VertexBufferObject]:
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

    @staticmethod
    def _create_shader(shader_type, shader_source: str) -> Shader:
        """creates a shader from its source & type."""
        shader = gl.glCreateShader(shader_type)
        gl.glShaderSource(shader, shader_source)
        gl.glCompileShader(shader)
        if gl.glGetShaderiv(shader, gl.GL_COMPILE_STATUS) != gl.GL_TRUE:
            raise ShaderCompileError(
                gl.glGetShaderInfoLog(shader).decode('utf-8'))
        return shader  # pyright: ignore [reportReturnType]

    @staticmethod
    def _create_program(*shaders) -> ShaderProgram:
        """creates a program from its vertex & fragment shader sources."""
        program = gl.glCreateProgram()
        for shader in shaders:
            gl.glAttachShader(program, shader)
        gl.glLinkProgram(program)
        if gl.glGetProgramiv(program, gl.GL_LINK_STATUS) != gl.GL_TRUE:
            raise ProgramLinkError(
                gl.glGetProgramInfoLog(program).decode('utf-8'))
        return program  # pyright: ignore [reportReturnType]

    def create_shader_program(self, shader_src: str):
        fragment_shader = self._create_shader(gl.GL_FRAGMENT_SHADER,
                                              shader_src)
        self.shader_program = self._create_program(self.vertex_shader,
                                              fragment_shader)
        gl.glDeleteShader(fragment_shader)

    def render(self, uniforms: Iterable[Uniform]):
        gl.glUseProgram(self.shader_program)
        for uniform in uniforms:
            uniform.update()

        gl.glClear(gl.GL_COLOR_BUFFER_BIT)
        gl.glBindVertexArray(self.vao)
        gl.glDrawArrays(gl.GL_TRIANGLE_STRIP, 0, len(self.VERTICES))

    def shutdown(self):
        gl.glDeleteVertexArrays(1, [self.vao])
        gl.glDeleteBuffers(1, [self.vbo])
        gl.glDeleteProgram(self.shader_program)

# TODO
# Uniform.from_def should not be dependent on the program...
# maybe
# Renderer should own the uniforms, offer create_uniform  wrapper, lock management, ...
# clearer separation between Uniforms and Parameters?
#   - Uniforms: location/program, type, name, value
#   - Paramter: name, type, default, range, widget, midi_mapping
#   then, Renderer.render({name: value}) only takes values, from Parameters and
#   updates own Uniforms
