uniform vec2 origin; // =(0.,0.) [-2.,-2.]
uniform float scale; // =1. [0.,2.]
uniform int n_max; // =10 [1,20]
uniform bool override_red; // =True
uniform vec4 base_color; // <color> =(1.,1.,0.,1.)

#define PI 3.14156

float f(vec2 c) {
    int it_max = n_max + int(5 * sin(2 * PI * u_Time));
    vec2 z = vec2(0.);
    for (int n = 0; n < it_max; n++) {
        z = vec2(z.x * z.x - z.y * z.y, 2. * z.x * z.y) + c;
        if (length(z) > 2.0) {
            return float(n) / it_max;
        }
    }
    return 0.;
}

void main() {
    vec2 pos = gl_FragCoord.xy / u_Resolution * 2. - vec2(1.5, 1.0);
    float n = f(scale * pos + origin);
    FragColor = base_color;
    if (override_red) {
        FragColor.r = n;
    }
    FragColor.g = n / 2 * (sin(2 * PI * u_Time) + 1.) / 2.;
}
