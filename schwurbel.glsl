uniform float zoom; // =.5 [0.,1.,0.01]
uniform vec2 focus; // =(0.5,0.5) [-1.,1.]
uniform vec4 bgcolor; // <color> =(1.,1.,0.,1.)
uniform float colormix;
uniform float noisiness; // =0.1
uniform float range; // =0. [-4.,4.]

float rand(vec2 co) {
    return fract(sin(dot(floor(co.xy), vec2(0, 1) + tan(u_Time * 0.0001))) * 500.0);
}

void main(void) {
    float z = zoom / 10000.;
    vec2 position = gl_FragCoord.xy - (focus * u_Resolution);

    float noise = rand(position.xy * (0.25 + asin(sin(z)) * 11.1));

    FragColor = vec4(
            bgcolor.r,
            mix(bgcolor.g, fract(position.x * position.y * z) + sin(range), colormix),
            bgcolor.b, //cos(color2.b + z * 3.0),
            1);

    FragColor = mix(FragColor, vec4(vec3(noise), 1.), noisiness);
}
