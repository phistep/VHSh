import re
import logging
from typing import (get_args, overload,
                    Literal, Sequence, Iterable, Callable)
from textwrap import dedent
from threading import Lock

import OpenGL.GL as gl
import numpy as np

from .types import (
    VertexArrayObject, VertexBufferObject,
    Shader, ShaderProgram,
    GLSLBool, GLSLInt, GLSLFloat, GLSLVec2, GLSLVec3, GLSLVec4,
    UniformValue, UniformLike,
    ProgramLinkError, ShaderCompileError, UniformIntializationError,
)

logger = logging.getLogger(__name__)


class Uniform(UniformLike):

    def __init__(self,
                 program: ShaderProgram,
                 type_: str,
                 name: str,
                 value: UniformValue | None = None):
        self._location: int = gl.glGetUniformLocation(program, name)
        self.type = type_
        self.name = name

        # TODO default step is dropped if not passed
        self._glUniform: Callable[..., None]
        match self.type:
            case 'bool':
                self._type = GLSLBool
                self._glUniform = gl.glUniform1i
                self.value = True if value is None else value
            case 'int':
                self._type = GLSLInt
                self._glUniform = gl.glUniform1i
                self.value = 1 if value is None else value
                self.range = self.range or (0, 10, 1)
            case 'float':
                self._type = GLSLFloat
                self._glUniform = gl.glUniform1f
                self.value = 1.0 if value is None else value
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
                self.value = (0.0,) * len if value is None else value
            case 'vec2':
                self._type = GLSLVec2
                self._glUniform = gl.glUniform2f
                self.value = (1.,)*2 if value is not None else value
            case 'vec3':
                self._type = GLSLVec3
                self._glUniform = gl.glUniform3f
                self.value = (1.,)*3 if value is not None else value
            case 'vec4':
                self._type = GLSLVec4
                self._glUniform = gl.glUniform4f
                self.value = (1.,)*4 if value is not None else value
            case _:
                raise NotImplementedError(
                    f"Uniform type '{self.type}' not implemented:"
                    f" {self.name} ({self.value})")

        self.value = value

        # TODO re-enable
        # uniform_type = get_args(self._type) or self._type
        # value_type = (tuple(type(elem) for elem in self.value)
        #               if isinstance(self.value, Iterable)
        #               else type(self.value))
        # if  value_type != uniform_type:
        #     raise UniformIntializationError(
        #         f"Uniform '{self.name}' defined as"
        #         f" '{self.type}' ({uniform_type}), but provided value"
        #         f" has type '{value_type}': {self.value!r}")

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

    FRAGMENT_SHADER_PREAMBLE = dedent("""\
        #version 330 core

        out vec4 FragColor;
        {system_uniforms}
        #line 1
        """
    )

    def __init__(self, system_uniforms: list[UniformLike]):
        self.preamble = self.FRAGMENT_SHADER_PREAMBLE.format(
            system_uniforms="\n".join(str(u) for u in system_uniforms)
        )
        logger.debug("preamble:\n%s", self.preamble)
        # TODO remove?
        self._lineno_offset = len(self.preamble.splitlines()) + 1

        self._uniform_lock = Lock()
        self.uniforms: dict[str, Uniform] = {}

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


    @overload
    def update_uniform(self,
                       name: str,
                       value: GLSLBool,
                       normalized: Literal[False] = False): ...
    def update_uniform(self,
                       name: str,
                       value: GLSLInt | GLSLFloat,
                       normalized: bool = False):
        with self._uniform_lock:
            uniform = self.uniforms[name]
            if not isinstance(value, uniform._type):
                raise ValueError(f"Argument 'value' needs to be of type"
                                 f" '{uniform._type}' (not '{type(value)}')")

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
                        f"Update not implemented for Uniform type '{uniform.type}'")

    def create_shader_program(self, shader_src: str):
        fragment_shader = self._create_shader(gl.GL_FRAGMENT_SHADER,
                                              shader_src)
        self.shader_program = self._create_program(self.vertex_shader,
                                              fragment_shader)
        gl.glDeleteShader(fragment_shader)

    # TODO this needs more cleanup: clear vs relead, re-use Uniform objects
    def set_shader(self,
                   source: str,
                   uniforms: Sequence[UniformLike],
                   clear: bool = False):
        # clear instead of update uniforms if this is a
        # new file (vs just reload)
        if clear:
            with self._uniform_lock:
                self.uniforms = {}

        self.create_shader_program(self.preamble + source)

        with self._uniform_lock:
            self.uniforms.update(
                **{parameter.name: Uniform(self.shader_program,
                                           type_=parameter.type,
                                           name=parameter.name,
                                           value=parameter.value)
                   for parameter in uniforms},
            )

    def update(self, uniforms: Sequence[UniformLike]):
        for uniform in uniforms:
            try:
                with self._uniform_lock:
                    self.uniforms[uniform.name].value = uniform.value
            except KeyError as e:
                logger.warning(f"'{e}' not in uniforms={self.uniforms}")

    def render(self):
        gl.glUseProgram(self.shader_program)
        for uniform in self.uniforms.values():
            uniform.update()

        gl.glClear(gl.GL_COLOR_BUFFER_BIT)
        gl.glBindVertexArray(self.vao)
        gl.glDrawArrays(gl.GL_TRIANGLE_STRIP, 0, len(self.VERTICES))

    def shutdown(self):
        gl.glDeleteVertexArrays(1, [self.vao])
        gl.glDeleteBuffers(1, [self.vbo])
        gl.glDeleteProgram(self.shader_program)
