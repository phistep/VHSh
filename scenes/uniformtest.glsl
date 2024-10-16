/// 0
/// uniform float default_range;  // =1.0 [-1.0,1.0] #16
/// uniform float default_only;  // =1.0 [0.0,1.0,0.01]
/// uniform float range_only;  // =1.0 [0.0,1.0]
/// uniform vec4 widget_default_range;  // =(0.0,0.0,0.0,0.0) [0.0,1.0,0.01]
/// uniform vec4 widget_only;  // =(0.0,0.0,0.0,0.0) [0.0,0.5]
/// Widget Test
/// uniform int int_default;
/// uniform int int_log; // <log>
/// uniform int int_drag; // <drag>
/// uniform float float_default;
/// uniform float float_log;  // <log>
/// uniform float float_drag; // <drag>
/// uniform vec2 vec2_default;
/// uniform vec2 vec2_log;  // <log>
/// uniform vec2 vec2_drag; // <drag>
/// uniform vec3 vec3_default;
/// uniform vec3 vec3_log;  // <log>
/// uniform vec3 vec3_drag; // <drag>
/// uniform vec3 vec3_color; // <color>
/// uniform vec4 vec4_default;
/// uniform vec4 vec4_log;  // <log>
/// uniform vec4 vec4_drag; // <drag>
/// uniform vec4 vec4_color; // <color>

uniform float default_range; // =0.3499999940395355 [-1.0,1.0] #16
uniform float default_only; // =1.0 [0.0,1.0,0.01]
uniform float range_only; // =1.0 [0.0,1.0]
uniform vec4 widget_default_range; // =(0.0,0.0,0.0,0.0) [0.0,1.0,0.01]
uniform vec4 widget_only; // =(0.0,0.0,0.0,0.0) [0.0,0.5]

void main() {
    FragColor = vec4(default_range);
}
