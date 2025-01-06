import re
from typing import Optional, Any, Callable, Iterable, get_args

import OpenGL.GL as gl

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

    def set_value_midi(self, value: int):
        assert 0 <= value <= 127
        if self.range:
            min_, max_ = self.range[:2]
            interpolated = min_ + (value / 127.0) * (max_ - min_)

        match self.type:
            case 'bool':
                self.value = bool(value)
            case 'int':
                self.value = int(interpolated)
            case 'float':
                self.value = float(interpolated)
            case _:
                raise NotImplementedError(
                    f"MIDI update not implemented for Uniform type '{self.type}'")

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
