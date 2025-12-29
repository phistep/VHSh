import logging
import math
import re
from ast import literal_eval
from dataclasses import dataclass
from enum import StrEnum
from pathlib import Path
from typing import Generic, Iterable, NotRequired, TypedDict, get_args

from .types import (
    Color,
    GLSLBool,
    GLSLFloat,
    GLSLInt,
    GLSLVec2,
    GLSLVec3,
    GLSLVec4,
    UniformLike,
    UniformT,
)

logger = logging.getLogger(__name__)


class ParameterParserError(ValueError): ...


class Widget(StrEnum):
    COLOR = "color"
    LOG = "log"
    DRAG = "drag"


def interpolate_log(
    t: float, v_min: float, v_max: float, type_: str = "float"
) -> float:
    # imgui_widgets.cpp:ImGui::DragBehaviourT()
    # When using logarithmic sliders, we need to clamp to avoid hitting zero, but our
    # choice of clamp value greatly affects slider precision. We attempt to use the
    # specified precision to estimate a good lower bound.
    decimal_precision = 1 if type_ == "int" else 3
    logarithmic_zero_epsilon = 0.1**decimal_precision

    # imgui_widgets.cpp:ImGui::SliderCalcValueFromRatioT()
    # We special-case the extents because otherwise our fudging can lead to
    # "mathematically correct" but non-intuitive behaviors like a fully-left slider not
    # actually reaching the minimum value
    if t <= 0:
        return v_min
    elif t >= 1:
        return v_max
    flipped = v_max < v_min  # Check if range is "backwards"
    v_min_fudged = (
        math.copysign(logarithmic_zero_epsilon, v_min)
        if math.fabs(v_min) < logarithmic_zero_epsilon
        else v_min
    )
    v_max_fudged = (
        math.copysign(logarithmic_zero_epsilon, v_max)
        if math.fabs(v_max) < logarithmic_zero_epsilon
        else v_max
    )
    if flipped:
        v_min_fudged, v_max_fudged = v_max_fudged, v_min_fudged

    # Awkward special case - we need ranges of the form (-100 .. 0)
    # to convert to (-100 .. -epsilon), not (-100 .. epsilon)
    if v_max == 0 and v_min < 0:
        v_max_fudged = -logarithmic_zero_epsilon

    t_with_flip = 1 - t if flipped else t

    if (v_min * v_max) < 0:  # Range crosses zero, so we have to do this in two parts
        zero_point = (-min(v_min, v_max)) / math.fabs(v_max - v_min)
        if t_with_flip == zero_point:
            # Special case to make getting exactly zero possible
            # (the epsilon prevents it otherwise)
            return 0
        elif t_with_flip < zero_point:
            return -(
                logarithmic_zero_epsilon
                * math.pow(
                    -v_min_fudged / logarithmic_zero_epsilon,
                    (1.0 - (t_with_flip / zero_point)),
                )
            )
        else:
            return (
                logarithmic_zero_epsilon
                * math.pow(
                    v_max_fudged / logarithmic_zero_epsilon,
                    ((t_with_flip - zero_point) / (1.0 - zero_point))
                )
            )  # fmt: skip
    elif v_min < 0 or v_max < 0:  # Entirely negative slider
        return (
            -abs(v_max_fudged)
            * math.pow(abs(v_min_fudged) / abs(v_max_fudged), 1 - t_with_flip)
        )  # fmt: skip
    else:
        return v_min_fudged * (v_max_fudged / v_min_fudged) ** t_with_flip


@dataclass
class Parameter(UniformLike, Generic[UniformT]):
    name: str
    type: str
    default: UniformT
    value: UniformT = None  # annotate optional param, but always set
    range: tuple[float, float, float] | None = None
    widget: Widget | None = None
    midi: int | None = None  # TODO -> controls: tuple[int]

    # delete everything but the default setting at the bottom? _type not needed
    def __post_init__(self):
        # TODO default step is dropped if not passed
        match self.type:
            case "bool":
                _type = GLSLBool
                if self.default is None:
                    self.default = True
                self.default = bool(self.default)
                self.range = None
                if self.value is not None:
                    self.value = bool(self.value)

            case "int":
                _type = GLSLInt
                if self.default is None:
                    self.default = 1
                self.default = int(self.default)
                if self.range is None:
                    self.range = (0, 100, 1)
                elif len(self.range) == 2:
                    self.range = (*self.range, 1)
                if self.value is not None:
                    self.value = int(self.value)

            case "float":
                _type = GLSLFloat
                if self.default is None:
                    self.default = 1.0
                self.default = float(self.default)
                if self.range is None:
                    self.range = (0.0, 1.0, 0.01)
                elif len(self.range) == 2:
                    self.range = (*self.range, 0.01)
                if self.value is not None:
                    self.value = float(self.value)

            case str() as t if t.startswith("float["):
                try:
                    m = re.match(r"float\[(\d+)\]", self.type)
                    length = int(m.group(1))  # pyright: ignore[reportOptionalMemberAccess]
                except (AttributeError, ValueError) as e:
                    raise ParameterParserError(
                        f"Unable to parse float array type '{self.type}': {e}"
                    ) from e
                _type = (float,) * length
                self.default = tuple(float(v) for v in self.default) or (0.0,) * length  # type: ignore
                self.range = None
                if self.value is not None:
                    self.value = tuple(float(v) for v in self.value)

            case "vec2":
                _type = GLSLVec2
                if self.default is None:
                    self.default = (1.0,) * 2  # type: ignore
                self.default = tuple(float(v) for v in self.default)
                if self.range is None:
                    self.range = (0.0, 1.0, 0.01)
                elif len(self.range) == 2:
                    self.range = (*self.range, 0.01)
                if self.value is not None:
                    self.value = tuple(float(v) for v in self.value)

            case "vec3":
                _type = GLSLVec3
                if self.default is None:
                    self.default = (1.0,) * 3  # type: ignore
                self.default = tuple(float(v) for v in self.default)
                if self.range is None:
                    self.range = (0.0, 1.0, 0.01)
                elif len(self.range) == 2:
                    self.range = (*self.range, 0.01)
                if self.value is not None:
                    self.value = tuple(float(v) for v in self.value)

            case "vec4":
                _type = GLSLVec4
                if self.default is None:
                    self.default = (1.0,) * 4  # type: ignore
                self.default = tuple(float(v) for v in self.default)
                if self.range is None:
                    self.range = (0.0, 1.0, 0.01)
                elif len(self.range) == 2:
                    self.range = (*self.range, 0.01)
                if self.value is not None:
                    self.value = tuple(float(v) for v in self.value)
            case _:
                raise NotImplementedError(
                    f"Uniform type '{self.type}' not implemented:"
                    f" {self.name} ({self.value})"
                )

        # TODO I think this is not working. in `set_value_normalized` I got
        # flaot values for int parameters.
        uniform_type = get_args(_type) or _type
        value_type = (
            tuple(type(elem) for elem in self.default)
            if isinstance(self.default, Iterable)
            else type(self.default)
        )
        if value_type != uniform_type:
            raise ParameterParserError(
                f"Uniform '{self.name}' defined as"
                f" '{self.type}' ({uniform_type}), but provided value"
                f" has type '{value_type}': {self.default!r}"
            )

        if self.value is None:
            self.value = self.default

    def __str__(self) -> str:
        s = f"uniform {self.type} {self.name};  //"
        if self.widget is not None:
            s += f" <{self.widget}>"
        s += f" ={str(self.value).replace(' ', '')}"
        if self.range is not None:
            s += f" {str(list(self.range)).replace(' ', '')}"
        if self.midi is not None:
            s += f" #{self.midi}"
        return s

    @classmethod
    def from_def(cls, definition: str) -> "Parameter":
        # TODO take line number, better error messages
        try:
            matches = re.search(
                # TODO why ^(?!\/\/)\s* not working to ignore comments?
                (
                    R"uniform\s+(?P<type>[\w\[\]]+)\s+(?P<name>\w+)\s*;"
                    R"(?:\s*//\s*(?:"
                    R"(?P<widget><\w+>)?\s*)?"
                    R"(?P<default>=(?:\S+|\([^\)]+\)))?"
                    R"\s*(?P<range>\[[^\]]+\])?"
                    R"\s*(?P<midi>#\d+)?"
                    R")?"
                ),
                definition,
            )
            type_, name, widget, default_s, range_s, midi = matches.groups()
        except Exception as e:
            raise ParameterParserError(
                f"Syntax error in metadata defintion: {definition}"
            ) from e

        if widget is not None:
            try:
                widget = Widget(widget.strip("<>"))
            except Exception as e:
                raise ParameterParserError(f"Unknown widget type '{widget}'") from e

        try:
            # TODO ast.literal_eval
            default = literal_eval(default_s.removeprefix("=")) if default_s else None
        except SyntaxError as e:
            raise ParameterParserError(
                f"Invalid 'default' metadata for uniform '{name}': {e}: {default_s}"
            ) from e

        try:
            range = literal_eval(range_s) if range_s else None
        except SyntaxError as e:
            raise ParameterParserError(
                f"Invalid 'range' metadata for uniform '{name}': {e}: {range_s!r}"
            ) from e

        if midi is not None:
            try:
                midi = int(midi.removeprefix("#"))
            except ValueError as e:
                raise ParameterParserError(
                    f"Invalid 'midi' metadata for uniform '{name}': {e}: {midi!r}"
                ) from e

        return Parameter(
            name=name,
            value=default,
            type=type_,
            default=default,
            range=range,
            widget=widget,
            midi=midi,
        )

    def set_value_normalized(self, value):
        if self.range is None:
            raise ValueError(
                f"Paramter '{self.name}' of type {self.type} cannot be set normalized"
            )
        min_, max_ = self.range[:2]

        logger.debug(
            "%s: min_=%s max_=%s type(value)=%s", self, min_, max_, type(value)
        )

        if self.widget is Widget.LOG:
            # new_value = math.exp(math.log(max_-min_+1)*value) + min_ - 1
            new_value = interpolate_log(value, min_, max_)
        else:  # linear
            new_value = min_ + value * (max_ - min_)

        if self.type == "int":
            new_value = int(round(new_value))

        self.value = new_value
        logger.debug("%f -> %f", value, self.value)


@dataclass
class Preset:
    name: str
    index: int
    parameters: dict[str, Parameter]

    def __str__(self) -> str:
        # to Preset.__str__
        presets = f"/// // {self.name}\n"
        presets += "\n".join(
            f"/// {parameter}" for parameter in self.parameters.values()
        )
        presets += "\n"
        return presets


class SceneMetadata(TypedDict):
    name: NotRequired[str]
    author: NotRequired[str]
    version: NotRequired[int]


class Scene:
    presets: list[Preset]

    def __init__(self, path: Path, required_version: int | None = None):
        self.path = path
        self.name = self.path.stem.replace("_", " ").replace("-", " ").title()
        self._required_version = required_version

        self.reload()

        if self._required_version is not None:
            scene_version = self.metadata.get("version")
            if scene_version is None:
                logger.warning(
                    "'%s': Undefined version."
                    " Might be incompatble with required version %i!",
                    self.path,
                    self._required_version,
                )
            elif scene_version != self._required_version:
                logger.warning(
                    "'%s': Incompatible version %i!"
                    " Might be incompatble with required version %i!",
                    self.path,
                    scene_version,
                    self._required_version,
                )
                logger.warning(
                    "HINT: Migrate using\n\n    vhsh migrate '%s'", path.absolute()
                )

    def __str__(self) -> str:
        return self.name

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
        for n, line in enumerate(source.split("\n")):
            line = line.strip()

            # <current> uniforms
            if line.startswith("uniform"):
                parameter = Parameter.from_def(line)
                presets[0].parameters[parameter.name] = parameter

            # presets
            elif line.startswith("///"):
                line_content = line.lstrip("/ ").strip()
                if line_content.startswith("@"):
                    # metadata line
                    continue
                if line.startswith("/// uniform"):
                    parameter = Parameter.from_def(line_content)
                    presets[-1].parameters[parameter.name] = parameter

                else:
                    index = len(presets)
                    presets.append(
                        Preset(
                            name=line_content or str(index), index=index, parameters={}
                        )
                    )

        return presets

    @staticmethod
    def _load_metadata(source: str) -> SceneMetadata:
        metadata = SceneMetadata()
        for line in source.splitlines():
            if line.startswith("/// @"):
                key, value = line.lstrip("/ @").strip().split(" ", 1)
                key = key.strip()
                value = value.strip()
                match key:
                    case "version":
                        value = int(value)
                metadata[key] = value

        return metadata

    def reload(self):
        self.source = self._read_file(self.path)
        self._preset_index = 0
        self.presets = self._load_presets(self.source)
        self.metadata = self._load_metadata(self.source)
        self.name = self.metadata.get("name", self.name)

    @property
    def parameters(self) -> dict[str, Parameter]:
        current_preset = self.presets[self.preset_index]
        return current_preset.parameters

    @property
    def preset_index(self) -> int:
        return self._preset_index

    @preset_index.setter
    def preset_index(self, value: int):
        self._preset_index = value % len(self.presets)
        logger.info(
            f"{Color.Style.BOLD}Current Preset:{Color.RESET}\n%s",
            "\n".join(
                f"  {p.removeprefix('/// uniform ')}"
                for p in str(self.presets[self.preset_index]).splitlines()
            ),
        )

    def prev_preset(self, n: int = 1):
        self.preset_index = (self.preset_index - n) % len(self.presets)

    def next_preset(self, n: int = 1):
        self.preset_index = (self.preset_index + n) % len(self.presets)

    def write_file(self, new_preset: str | None = None):
        if self._required_version is not None and "version" not in self.metadata:
            logger.warning(
                "'%s': Updating Scene version to %i", self.name, self._required_version
            )
            self.metadata["version"] = self._required_version
        metadata = (
            "\n".join(f"/// @{key} {value}" for key, value in self.metadata.items())
            + "\n"
        )

        if new_preset is not None:
            logger.info("New preset: '%s'", new_preset)
            self.presets.append(
                Preset(
                    name=new_preset,
                    parameters=self.parameters.copy(),
                    index=len(self.presets) - 1,
                )
            )

        presets = "\n".join(str(preset) for preset in self.presets[1:]) + "\n"

        lines = [
            line for line in self.source.splitlines() if not line.startswith("///")
        ]

        self.source = metadata + presets + "\n".join(lines) + "\n"

        if self._preset_index == 0:
            for parameter in self.parameters.values():
                parameter.default = parameter.value
                self.source = re.sub(
                    f"^uniform \\w+ {parameter.name}.*$",
                    str(parameter),
                    self.source,
                    flags=re.MULTILINE,
                )

        self.path.write_text(self.source)
        logger.info(f"Wrote presets to '{self.path}'")
