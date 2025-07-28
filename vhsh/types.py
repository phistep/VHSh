import os
from typing import Protocol, TypeAlias, TypeVar, Union, Generic, Sequence
from dataclasses import dataclass

import numpy as np

### Exceptions

class ParameterParserError(ValueError): ...
class ShaderCompileError(RuntimeError): ...
class UniformIntializationError(ShaderCompileError): ...
class ProgramLinkError(RuntimeError): ...


### GLSL

VertexArrayObject: TypeAlias = np.uint32
VertexBufferObject: TypeAlias = np.uint32
Shader: TypeAlias = int
ShaderProgram: TypeAlias = int

GLSLBool: TypeAlias = bool
GLSLInt: TypeAlias = int
GLSLFloat: TypeAlias = float
GLSLVec2: TypeAlias = tuple[float, float]
GLSLVec3: TypeAlias = tuple[float, float, float]
GLSLVec4: TypeAlias = tuple[float, float, float, float]

_UniformValue: TypeAlias = Union[GLSLBool, GLSLInt, GLSLFloat,
                                 GLSLVec2, GLSLVec3, GLSLVec4]
UniformValue: TypeAlias = Union[_UniformValue, Sequence[_UniformValue]]
UniformT = TypeVar('UniformT', bound=UniformValue)

class UniformLike(Protocol, Generic[UniformT]):
    name: str
    type: str
    value: UniformT | list[UniformT]

    def __str__(self) -> str:
        return f"uniform {self.type} {self.name};"


# TODO Split into Actions and State
# factor out ShaderRenderer first, then decide on the interface =
# (separate Action classes for each interface?)
class App(Protocol):
    window: "Window"
    _error: ShaderCompileError | ParameterParserError | None
    @property
    def scene_index(self) -> int: ...
    @scene_index.setter
    def scene_index(self, value: int): ...
    scenes: list["Scene"]
    scene: "Scene"
    parameters: dict[str, "Parameter"]
    system_parameters: dict[str, "SystemParameter"]
    time: "Time"
    def prev_scene(self, n=1): ...
    def next_scene(self, n=1): ...
    _frame_times: list[float]
    _microphone: object
