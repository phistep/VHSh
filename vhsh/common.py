import os
from typing import Protocol
from dataclasses import dataclass

from .shader import ShaderCompileError
from .gl_types import UniformValue


class Actions(Protocol):
    def prev_shader(self): ...
    def next_shader(self): ...
    def prev_preset(self, n: int = 1): ...
    def next_preset(self, n: int = 1): ...
    def write_file(self,
                   presets: bool = True,
                   uniforms: bool = False,
                   new_preset: str | None = None): ...
    # TODO @properties
    def set_time_running(self, value: bool): ...
    def set_show_gui(self, value: bool): ...
    def get_midi_mapping(self, cc: int) -> str: ...
    def set_uniform_value(self,
                          name: str,
                          value: UniformValue,
                          normalized: bool = False): ...
    # TODO -> fmt_error, also display in GUI
    def _print_error(self, e: Exception | str): ...


# TODO Split into Actions and State
# factor out ShaderRenderer first, then decide on the interface =
# (separate Action classes for each interface?)
class App(Protocol):
    _show_gui: bool
    _window: int  # TODO ?? GLFW Window
    _error: ShaderCompileError | None
    _shader_path: str
    _shader_paths: list[str]
    _shader_index: int
    opacity: float
    floating: bool
    def prev_shader(self, n=1): ...
    def next_shader(self, n=1): ...
    presets: list[dict]
    preset_index: int
    def prev_preset(self, n: int = 1): ...
    def next_preset(self, n: int = 1): ...
    def write_file(self,
                   presets: bool = True,
                   uniforms: bool = False,
                   new_preset: str | None = None): ...
    _new_preset_name: str
    _frame_times: list[float]
    uniforms: dict[str, "Uniform"]
    _time_running: float
    _microphone: object
    FRAGMENT_SHADER_PREAMBLE: str


def get_shader_title(shader_path: str) -> str:
    return os.path.splitext(os.path.basename(shader_path))[0]
