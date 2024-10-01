#version 330 core

out vec4 FragColor;
uniform vec2 u_Resolution;
uniform float u_Time;

uniform vec2 origin;
uniform float scale;
uniform int n_max;

float f(vec2 c) {
    vec2 z = vec2(0.);
    for (int n = 0; n < n_max; n++) {
        z = vec2(z.x * z.x - z.y * z.y, 2. * z.x * z.y) + c;
        if (length(z) > 2.0) {
            return float(n) / n_max;
        }
    }
    return 0.;
}

void main() {
    vec2 pos = gl_FragCoord.xy / u_Resolution * 2. - vec2(1.5, 1.0);
    float n = f(scale * pos + origin);
    FragColor.rgba = vec4(vec3(n), 1.0);
}
