import pytest

from vhsh.scene import Parameter, Widget
from vhsh.types import UniformLike


@pytest.mark.parametrize(
    "line,default",
    [("uniform bool bool_with_int_default; // =0", False),
     ("uniform bool bool_with_int_default; // =1", True),
     ("uniform int int_with_bool_default; // =False", 0),
     ("uniform int int_with_bool_default; // =True", 1),
     ("uniform float float_with_bool_default; // =False", 0.),
     ("uniform float float_with_bool_default; // =True", 1.),
     ("uniform vec2 vec2_with_bool_default; // =(True,False)", (1., 0.)),
     ("uniform vec3 vec3_with_bool_default; // =(True,False,False)", (1., 0., 0.)),
     ("uniform vec4 vec4_with_bool_default; // =(True,True,False,False)", (1., 1., 0., 0.)),
     ("uniform float[2] float_array_with_bool_default; // =(True,False)", (1., 0.)),
     ("uniform vec3 vec3_with_no_default; //", (1., 1., 1.)),
    ]
)
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

    parameter = Parameter("test", type_, default=1, range=(min_,max_), widget=widget)

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
