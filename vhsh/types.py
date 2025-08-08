from __future__ import annotations

from typing import (Protocol, TypeAlias, TypeVar, Union, Generic, Sequence,
                    TYPE_CHECKING, Callable)
from abc import ABC
from threading import Thread, Event
from collections import deque
from dataclasses import dataclass

import numpy as np

if TYPE_CHECKING:
    from .window import Window
    from .scene import Scene
    from .app import Time

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


# TODO protocols for all circular imports
class App(Protocol):
    window: Window
    error: ShaderCompileError | ParameterParserError | None
    frame_times: deque[float]
    system_parameters: dict[str, SystemParameter]
    controllers: dict[str, Controller]
    time: Time
    @property
    def scene_index(self) -> int: ...
    @scene_index.setter
    def scene_index(self, value: int): ...
    scenes: list[Scene]
    @property
    def scene(self) -> Scene: ...
    def prev_scene(self, n: int = 1): ...
    def next_scene(self, n: int = 1): ...
    def load(self, clear: bool = True): ...


class Controller(ABC, Thread):
    def __init__(self, *args, **kwargs):
        if not 'name' in kwargs:
            kwargs['name'] = self.__class__.__name__
        super().__init__(*args, **kwargs)

        self._stop_controller = Event()

    def update_pre(self):
        """Called in the main loop before graphics rendering."""
        pass

    def update_post(self):
        """Called in the main loop after graphics rendering."""
        pass

    def stop(self):
        self._stop_controller.set()


@dataclass
class SystemParameter(UniformLike, Generic[UniformT]):
    """Pass a function that returns a value to update the Parameter with.

    Pass None if value should be kept constant.
    """
    name: str
    type: str
    value: UniformT
    # TODO why are recursive types not working?
    update: Callable[[App], UniformT | None]

    def __post_init__(self):
        # wrap `.update()` so that it sets .value,
        # but can be passed as `update=`

        self._update = self.update

        def update(renderer: App):
            value = self._update(renderer)
            if value is not None:
                self.value = value
            return value

        self.update = update
