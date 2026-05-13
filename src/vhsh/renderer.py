import logging
import re
from textwrap import dedent
from threading import Lock
from typing import TYPE_CHECKING, Iterable, Sequence

import numpy as np
import OpenGL.GL as gl

from .types import (
    Color,
    GLSLBool,
    GLSLFloat,
    GLSLInt,
    GLSLVec2,
    GLSLVec3,
    GLSLVec4,
    Shader,
    ShaderProgram,
    UniformLike,
    UniformValue,
    VertexArrayObject,
    VertexBufferObject,
)

if TYPE_CHECKING:
    from types import FunctionType

logger = logging.getLogger(__name__)


class ShaderCompileError(RuntimeError):
    def format(self) -> str:
        try:
            lines = str(self).strip().splitlines()
            if len(lines) == 2:
                flex, error = lines
            else:
                error = lines[0]
                flex = ""
            parts = error.split(":")
            title = parts[0].strip()  # noqa: F841
            col = parts[1].strip()
            line = parts[2].strip()
            offender = parts[3].strip()
            message = ":".join(parts[4:])

            return (
                f"{Color.Style.BOLD}{col}:{line}{Color.RESET}"
                f" {Color.Style.FAINT}({offender}){Color.RESET}"
                f":{message}"
                f" {Color.Style.FAINT}({flex}){Color.RESET}"
            )
        except IndexError:
            logger.debug(
                "Python error formatting GLSL error message: %s",
                str(self),
                exc_info=True,
            )
            return str(self)


class UniformIntializationError(ShaderCompileError): ...


class ProgramLinkError(RuntimeError): ...


class Uniform(UniformLike):
    def __init__(
        self,
        program: ShaderProgram,
        type_: str,
        name: str,
        value: UniformValue | None = None,
    ):
        self._location: int = gl.glGetUniformLocation(program, name)
        # TODO free on __del__?
        self.type = type_
        self.name = name

        # TODO default step is dropped if not passed
        self._glUniform: FunctionType[..., None]
        match self.type:
            case "bool":
                self._type = GLSLBool
                self._glUniform = gl.glUniform1i
                self.value = True if value is None else value
            case "int":
                self._type = GLSLInt
                self._glUniform = gl.glUniform1i
                self.value = 1 if value is None else value
            case "float":
                self._type = GLSLFloat
                self._glUniform = gl.glUniform1f
                self.value = 1.0 if value is None else value
            case str() as t if t.startswith("float["):
                try:
                    m = re.match(r"float\[(\d+)\]", type_)
                    if m is None:
                        raise ValueError("No matches")
                    len = int(m.group(1))
                except ValueError as e:
                    raise UniformIntializationError(
                        f"Unable to parse float array type '{type_}': {e}"
                    ) from e
                self._type = (float,) * len
                self._glUniform = gl.glUniform1fv
                self.value = (0.0,) * len if value is None else value
            case "vec2":
                self._type = GLSLVec2
                self._glUniform = gl.glUniform2f
                self.value = (1.0,) * 2 if value is not None else value
            case "vec3":
                self._type = GLSLVec3
                self._glUniform = gl.glUniform3f
                self.value = (1.0,) * 3 if value is not None else value
            case "vec4":
                self._type = GLSLVec4
                self._glUniform = gl.glUniform4f
                self.value = (1.0,) * 4 if value is not None else value
            case _:
                raise NotImplementedError(
                    f"Uniform type '{self.type}' not implemented:"
                    f" {self.name} ({self.value})"
                )

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
        return f"uniform {self.type} {self.name};"

    def __repr__(self):
        return (
            f"<Uniform"
            f" type={self.type}"
            f' name="{self.name}"'
            f" value={self.value}"
            f" _type={self._type}"
            f" _glUniform={self._glUniform.__name__}"
            f" at 0x{self._location:04x}>"
        )

    def update(self):
        args = self.value
        if not isinstance(args, Iterable):
            args = [args]
        if self._glUniform.__name__.endswith("v"):
            self._glUniform(self._location, len(args), args)
        else:
            self._glUniform(self._location, *args)


class Renderer:
    # TODO use stdlib arrays, make np optional for mic input
    VERTICES = np.array(
        [[-1.0,  1.0, 0.0],
         [-1.0, -1.0, 0.0],
         [ 1.0,  1.0, 0.0],
         [ 1.0, -1.0, 0.0]],
        dtype=np.float32
    )  # fmt: skip

    VERTEX_SHADER = dedent(  # glsl
        """\
        #version 330 core

        layout(location = 0) in vec3 VertexPos;

        void main() {
            gl_Position = vec4(VertexPos, 1.0);
        }
        """
    )

    FRAGMENT_SHADER_PREAMBLE = dedent(  # glsl
        """\
        #version 330 core

        out vec4 FragColor;
        {system_uniforms}
        #line 1
        """
    )

    DEFAULT_FRAGMENT_SHADER = dedent(  # glsl
        """\
        void main() {
            vec2 pos = gl_FragCoord.xy / Resolution;
            FragColor = vec4(pos.x, pos.y, 1.0 - (pos.x + pos.y) / 2.0, 1.0);
        }
    """
    )

    def __init__(self, system_uniforms: list[UniformLike]):
        self.preamble = self.FRAGMENT_SHADER_PREAMBLE.format(
            system_uniforms="\n".join(str(u) for u in system_uniforms)
        )
        logger.debug("preamble:\n%s", self.preamble)

        self._uniform_lock = Lock()
        self.uniforms: dict[str, Uniform] = {}

        self.vao, self.vbo = self._create_vertices(self.VERTICES)
        self.vertex_shader = self._create_shader(
            gl.GL_VERTEX_SHADER, self.VERTEX_SHADER
        )

        self.set_shader(
            self.DEFAULT_FRAGMENT_SHADER, uniforms=system_uniforms, clear=True
        )

    @staticmethod
    def _create_vertices(
        vertices: np.ndarray,
    ) -> tuple[VertexArrayObject, VertexBufferObject]:
        vao = gl.glGenVertexArrays(1)
        vbo = gl.glGenBuffers(1)

        gl.glBindVertexArray(vao)

        gl.glBindBuffer(gl.GL_ARRAY_BUFFER, vbo)
        gl.glBufferData(
            target=gl.GL_ARRAY_BUFFER,
            size=vertices.nbytes,
            data=vertices,
            usage=gl.GL_STATIC_DRAW,
        )

        # Specify the layout of the vertex data
        vertex_attrib_idx = 0
        gl.glVertexAttribPointer(
            index=vertex_attrib_idx,
            size=3,  # len(x, y, z)
            type=gl.GL_FLOAT,
            normalized=gl.GL_FALSE,
            stride=3 * 4,  # (x, y, z) * sizeof(GL_FLOAT)  # TODO
            pointer=gl.ctypes.c_void_p(0),
        )
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
            raise ShaderCompileError(gl.glGetShaderInfoLog(shader).decode("utf-8"))
        return shader  # pyright: ignore [reportReturnType]

    @staticmethod
    def _create_program(*shaders) -> ShaderProgram:
        """creates a program from its vertex & fragment shader sources."""
        program = gl.glCreateProgram()
        for shader in shaders:
            gl.glAttachShader(program, shader)
        gl.glLinkProgram(program)
        if gl.glGetProgramiv(program, gl.GL_LINK_STATUS) != gl.GL_TRUE:
            raise ProgramLinkError(gl.glGetProgramInfoLog(program).decode("utf-8"))
        return program  # pyright: ignore [reportReturnType]

    def create_shader_program(self, shader_src: str):
        fragment_shader = self._create_shader(gl.GL_FRAGMENT_SHADER, shader_src)
        self.shader_program = self._create_program(self.vertex_shader, fragment_shader)
        gl.glDeleteShader(fragment_shader)

    # TODO this needs more cleanup: clear vs relead, re-use Uniform objects
    def set_shader(
        self, source: str, uniforms: Sequence[UniformLike], clear: bool = False
    ):
        # clear instead of update uniforms if this is a
        # new file (vs just reload)
        if clear:
            with self._uniform_lock:
                self.uniforms = {}

        self.create_shader_program(self.preamble + source)

        with self._uniform_lock:
            self.uniforms.update(
                **{
                    parameter.name: Uniform(
                        self.shader_program,
                        type_=parameter.type,
                        name=parameter.name,
                        value=parameter.value,
                    )
                    for parameter in uniforms
                },
            )

    def update(self, uniforms: Sequence[UniformLike]):
        with self._uniform_lock:
            for uniform in uniforms:
                try:
                    self.uniforms[uniform.name].value = uniform.value
                except KeyError:
                    self.uniforms[uniform.name] = Uniform(
                        self.shader_program,
                        type_=uniform.type,
                        name=uniform.name,
                        value=uniform.value,
                    )
            for unused in set(self.uniforms) - set(u.name for u in uniforms):
                del self.uniforms[unused]

    def render(self, framebuffer_size: tuple[int, int]):
        """:param framebuffer_size: width, height"""
        # TODO do I need to do this every frame? also: glBindVertexArray
        gl.glViewport(0, 0, *framebuffer_size)
        gl.glClear(gl.GL_COLOR_BUFFER_BIT)

        gl.glUseProgram(self.shader_program)
        with self._uniform_lock:
            for uniform in self.uniforms.values():
                uniform.update()
        gl.glBindVertexArray(self.vao)
        gl.glDrawArrays(gl.GL_TRIANGLE_STRIP, 0, len(self.VERTICES))

    def shutdown(self):
        gl.glDeleteVertexArrays(1, [self.vao])
        gl.glDeleteBuffers(1, [self.vbo])
        gl.glDeleteProgram(self.shader_program)
