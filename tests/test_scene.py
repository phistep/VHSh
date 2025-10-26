import pytest

from vhsh.scene import Parameter
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
     ("uniform vec3 vec3_with_bool_default; // =(True,False,False)", (1., 0., 0)),
     ("uniform vec4 vec4_with_bool_default; // =(True,True,False,False)", (1., 1., 0., 0.)),
     ("uniform float[2] float_array_with_bool_default; // =(True,False)", (1., 0.))]
)
def test_paramter_value_type_coersion(line: str, default: UniformLike):
    parameter = Parameter.from_def(line)
    assert parameter.default == default
    assert parameter.value == default
