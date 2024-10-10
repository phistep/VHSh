///
/// uniform float default_range; // =0. [-1.,1.]
/// uniform float default_only; // =0.
/// uniform float range_only; // [0.,1.]
/// uniform vec4 widget_default_range; // =(0.,0.,0.,0.)
/// uniform vec4 widget_only; // =(0.,0.,0.,0.) [0.,0.5]
///// foo
/// uniform float default_range; // =5.5 [-1.,1.]
/// uniform float default_only; // =5.5
/// uniform float range_only; // [-55.,55.]
/// uniform vec4 widget_default_range; // <color> =(1.,0.,0.,1.) [0.,0.5]
/// uniform vec4 widget_only; // <color> =(1.,0.,0.,1.) [0.,0.5]

uniform float default_range; // =0.5 [-1.,1.]
uniform float default_only; // =0.5
uniform float range_only; // [-1.,1.]
uniform vec4 widget_default_range; // <color> =(1.,0.,1.,1.) [0.,0.5]
uniform vec4 widget_only; // <color> =(1.,1.,0.,1.) [0.,0.5]

void main() {
    FragColor = vec4(default_range);
}
