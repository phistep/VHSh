import logging
from textwrap import dedent

import pytest

from vhsh.scene import Parameter, Scene, SceneMetadata, Widget
from vhsh.types import UniformLike, UniformT


@pytest.mark.parametrize("midi", [None, 23])
@pytest.mark.parametrize("widget", [None, *list(Widget)])
@pytest.mark.parametrize(
    "type_, default, value, range",
    [
        ("bool", True, True, None),
        ("bool", False, False, None),
        ("int", 1, 1, (0, 1, 1)),
        ("int", 1, 1, (-1, 1, 1)),
        ("int", 1, 1, (0, 10, 2)),
        ("float", 1.0, 1.0, (0.0, 1.0, 0.1)),
        ("float", 1.0, 1.0, (-1.0, 1.0, 0.1)),
        ("float", 1.0, 1.0, (0.0, 10.0, 2.0)),
        ("vec2", (1.0, 0.0), (1.0, 0.0), (0.0, 1.0, 0.1)),
        ("vec3", (1.0, 0.0, 0.0), (1.0, 0.0, 0.0), (0.0, 1.0, 0.1)),
        ("vec4", (1.0, 0.0, 0.0, 0.0), (1.0, 0.0, 0.0, 0.0), (0.0, 1.0, 0.1)),
        (
            "float[10]",
            float_array_default := [float(n) for n in range(10)],
            float_array_default,
            None,
        ),
    ],
)
def test_parameter(
    type_: str,
    default: UniformT | None,
    value: UniformT | None,
    range: tuple[float, float, float] | None,
    widget: Widget | None,
    midi: int | None,
):
    parameter = Parameter(
        name="name",
        type=type_,
        default=default,
        value=value,
        range=range,
        widget=widget,
        midi=midi,
    )
    assert parameter.name == "name"
    assert parameter.type == type_
    assert parameter.default == default
    assert parameter.value == value
    assert parameter.range == range
    assert parameter.widget == widget
    assert parameter.midi == midi


@pytest.mark.xfail  # TODO
def test_paramter_type():
    # Test that invalid type/value combinations fail
    raise NotImplementedError


@pytest.mark.xfail  # TODO
def test_paramter_range():
    # Test that default bounds and steps are fallen back to
    raise NotImplementedError


# TODO
# def test_paramter_from_def(line: str, default: UniformLike):
# parameter = Parameter.from_def(line)
# assert parameter.default == default
# assert parameter.value == default


@pytest.mark.parametrize(
    "line,default",
    [
        # ruff: disable[E501]
        ("uniform bool bool_with_int_default; // =0", False),
        ("uniform bool bool_with_int_default; // =1", True),
        ("uniform int int_with_bool_default; // =False", 0),
        ("uniform int int_with_bool_default; // =True", 1),
        ("uniform float float_with_bool_default; // =False", 0.),
        ("uniform float float_with_bool_default; // =True", 1.),
        ("uniform vec2 vec2_with_bool_default; // =(True,False)", (1., 0.)),
        ("uniform vec3 vec3_with_bool_default; // =(True,False,False)", (1., 0., 0.)),
        ("uniform vec4 vec4_with_bool_default; // =(True,True,False,False)", (1., 1., 0., 0.)),  # noqa: E501
        # NOTE float array always defaults to just ones: [1., 1., ...]
        ("uniform float[2] float_array_with_bool_default; // =(True,False)", [1., 1.]),
        ("uniform vec3 vec3_with_no_default; //", (1., 1., 1.)),
        # ruff: enable[E501]
    ],
)  # fmt: skip
def test_paramter_value_type_coersion(line: str, default: UniformLike):
    parameter = Parameter.from_def(line)
    assert parameter.default == default
    assert parameter.value == default


@pytest.mark.parametrize("flipped", [True, False], ids=["regular", "flipped"])
@pytest.mark.parametrize(
    "min_,max_",
    [
        pytest.param(0, 1, id="zero_positive"),
        pytest.param(-1, 0, id="negative_zero"),
        pytest.param(1, 2, id="all_positive"),
        pytest.param(-2, -1, id="all_negative"),
        pytest.param(-1, 1, id="zero_crossing"),
    ],
)
@pytest.mark.parametrize("widget", [None, Widget.LOG], ids=["lin", "log"])
@pytest.mark.parametrize("type_", ["int", "float"])
def test_parameter_set_value_normalized(
    type_: str,
    widget: Widget | None,
    min_: float,
    max_: float,
    flipped: bool,
):
    if flipped:
        min_, max_ = max_, min_

    parameter = Parameter(
        name="test",
        type=type_,
        default=(default := (1 if type_ == "int" else 1.0)),
        value=default,
        range=(min_, max_, 1),
        widget=widget,
    )

    parameter.set_value_normalized(0)
    assert parameter.value == pytest.approx(min_)

    linear_result = min_ + (max_ - min_) / 2
    if type_ == "int":
        linear_result = int(round(linear_result))

    parameter.set_value_normalized(0.5)

    if widget == Widget.LOG:
        assert min(min_, max_) <= parameter.value <= max(min_, max_)
        if min_ != 0 and max_ != 0 and abs(min_ - max_) > 1 and (min_ * max_) > 0:
            assert parameter.value != linear_result
    else:
        assert parameter.value == linear_result

    parameter.set_value_normalized(1)
    assert parameter.value == pytest.approx(max_)


@pytest.mark.parametrize(
    "header, expected",
    [
        pytest.param("", {}, id="none"),
        pytest.param("/// @version 23", {"version": 23}, id="version"),
        pytest.param("/// @author Author", {"author": "Author"}, id="author"),
        pytest.param(
            "/// @name Unicode \N{VIDEOCASSETTE}",
            {"name": "Unicode \N{VIDEOCASSETTE}"},
            id="name",
        ),
        pytest.param("/// @name  \t   Name", {"name": "Name"}, id="whitespace"),
        pytest.param("/// @extra Extra", {"extra": "Extra"}, id="extra"),
        pytest.param(
            dedent(
                """\
            /// @name    Name
            /// @version 23
            /// @extra   Additional Info
            """
            ),
            {
                "name": "Name",
                "version": 23,
                "extra": "Additional Info",
            },
            id="all",
        ),
        pytest.param("// @version 23", {}, id="no_triple_slash"),
    ],
)
def test_scene__load_metadata(header: str, expected: SceneMetadata):
    assert Scene._load_metadata(header) == expected


def test_scene__load_metadata_warn(caplog: pytest.LogCaptureFixture):
    assert "name" in SceneMetadata.__annotations__
    with caplog.at_level(logging.WARNING):
        assert Scene._load_metadata("/// @name Name") == {"name": "Name"}
        assert not caplog.text

    assert "extra" not in SceneMetadata.__annotations__
    with caplog.at_level(logging.WARNING):
        assert Scene._load_metadata("/// @extra Extra") == {"extra": "Extra"}
        assert "Unknown Metadata Tag" in caplog.text
