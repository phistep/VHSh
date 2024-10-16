/// // Blue
/// uniform vec2 origin;  // =(-0.6299999952316284,-0.41999998688697815) [-2.0,-2.0]
/// uniform float scale;  // <log> =0.07774548977613449 [0.0,1000] #0
/// uniform int n_max;  // =400 [100,1000,100] #16
/// uniform vec4 base_color;  // <color> =(0.0,0.8529412746429443,1.0,1.0) [0.0,1.0,0.01]
/// uniform float intensity;  // =0.1899999976158142 [0.0,1.0,0.01]
/// uniform bool animate;  // =True
uniform vec2 origin; // =(0.0,0.0) [-2.0,-2.0]
uniform float scale; // <log> =1.0 [0.0,1000] #0
uniform int n_max; // =100 [1,100] #16
uniform vec4 base_color; // <color> =(1.0,1.0,0.0,1.0) [0.0,1.0,0.01]
uniform float intensity; // =0.5 [0.0,1.0,0.01]
uniform bool animate; // =True

#define PI 3.14156

float f(vec2 c) {
    int it_max = n_max;
    if (animate) {
        it_max += int(5 * sin(2 * PI * u_Time));
    }
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
    vec4 color = vec4(vec3(n), 1.);
    FragColor = mix(color, base_color, intensity);
}
