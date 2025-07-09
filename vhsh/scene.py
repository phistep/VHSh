import re
from pathlib import Path
from typing import Type
from enum import StrEnum
from dataclasses import dataclass

from .gl_types import UniformValue
from .common import SYSTEM_UNIFORMS


class ParameterParserError(ValueError):
    pass


class Widget(StrEnum):
    COLOR = "color"
    LOG = "log"
    DRAG = "drag"


@dataclass
class Parameter:
    name: str
    type: Type[UniformValue]
    value: UniformValue
    default: UniformValue
    range: tuple[float, float]
    widget: Widget
    midi: int

    @classmethod
    def from_def(cls, definition: str) -> "Parameter":
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
            raise ParameterParserError(
                f"Syntax error in metadata defintion: {definition}") from e

        if widget is not None:
            widget = widget.strip('<>')

        try:
            default = (eval(default_s.removeprefix('='))
                        if default_s
                        else None)
        except SyntaxError as e:
            raise ParameterParserError(
                f"Invalid 'default' metadata for uniform"
                f" '{name}': {e}: {default_s}"
            ) from e

        try:
            range = eval(range_s) if range_s else None
        except SyntaxError as e:
            raise ParameterParserError(
                f"Invalid 'range' metadata for uniform"
                f" '{name}': {e}: {range_s!r}"
            ) from e

        if midi is not None:
            try:
                midi = int(midi.removeprefix('#'))
            except ValueError as e:
                raise ParameterParserError(
                    f"Invalid 'midi' metadata for uniform"
                    f" '{name}': {e}: {midi!r}"
                ) from e

        return Parameter(name=name,
                         value=default,
                         type=type_,
                         default=default,
                         range=range,
                         widget=widget,
                         midi=midi)


@dataclass
class Preset:
    name: str
    index: int
    parameters: dict[str, Parameter]


class Scene:

    def __init__(self, path: Path):
        self.path = path

        self.source = self._read_file(path)
        self.presets = self._load_presets(self.source)

    def __str__(self) -> str:
        # TODO ext
        return self.path.name

    def __repr__(self) -> str:
        return f"<{self.__class__.__name__} {self.path}>"

    # TODO reload
    @staticmethod
    def _read_file(path: Path) -> str:
        with open(path) as f:
            return f.read()

    @staticmethod
    def _load_presets(source: str) -> list[Preset]:
        presets = [Preset(name="<current>", index=0, parameters={})]
        for n, line in enumerate(source.split('\n')):
            line = line.strip()

            # <current> uniforms
            if line.startswith('uniform'):
                parameter = Parameter.from_def(line)
                presets[0].parameters[parameter.name] = parameter

            # presets
            elif line.startswith('///'):
                line_content = line.lstrip('/ ').strip()
                if line.startswith('/// uniform'):
                    parameter = Parameter.from_def(line_content)
                    presets[-1].parameters[parameter.name] = parameter

                else:
                    index = len(presets)
                    presets.append(Preset(name=line_content or str(index),
                                          index=index,
                                          parameters={}))

        return presets
