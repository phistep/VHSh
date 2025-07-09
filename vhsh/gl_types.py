from typing import TypeAlias, TypeVar, Union

import numpy as np

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

UniformValue: TypeAlias = Union[GLSLBool,
                                GLSLInt,
                                GLSLFloat,
                                GLSLVec2,
                                GLSLVec3,
                                GLSLVec4,
                                list["UniformValue"]]
UniformT = TypeVar('UniformT', bound=UniformValue)
