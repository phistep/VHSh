import os
from typing import Protocol, TypeAlias, TypeVar, Union
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

UniformValue: TypeAlias = Union[GLSLBool, GLSLInt, GLSLFloat,
                                GLSLVec2, GLSLVec3, GLSLVec4,
                                list["UniformValue"]]
UniformT = TypeVar('UniformT', bound=UniformValue)

class UniformLike(Protocol):
    name: str
    type: str
    value: UniformValue

    def __str__(self) -> str:
        return f"uniform {self.type} {self.name};"


# TODO Split into Actions and State
# factor out ShaderRenderer first, then decide on the interface =
# (separate Action classes for each interface?)
class App(Protocol):
    window: "Window"
    _error: ShaderCompileError | ParameterParserError | None
    _shader_path: str
    _shader_paths: list[str]
    _shader_index: int
    scene: "Scene"
    parameters: dict[str, "Parameter"]
    system_parameters: dict[str, "SystemParameter"]
    time: "Time"
    def prev_shader(self, n=1): ...
    def next_shader(self, n=1): ...
    def prev_preset(self, n: int = 1): ...
    def next_preset(self, n: int = 1): ...
    def write_file(self,
                   presets: bool = True,
                   uniforms: bool = False,
                   new_preset: str | None = None): ...
    _new_preset_name: str
    _frame_times: list[float]
    _microphone: object


# TODO -> scene.name
def get_shader_title(shader_path: str) -> str:
    return os.path.splitext(os.path.basename(shader_path))[0]
