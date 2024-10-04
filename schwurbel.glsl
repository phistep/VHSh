uniform float zoom; // =.01 [0.,.01,0.0001]

float rand(vec2 co) {
    return fract(sin(dot(floor(co.xy), vec2(0, 1) + tan(u_Time * 0.0001))) * 500.0);
}

void main(void) {
    vec2 surfacePosition = vec2(0.5);
    vec2 mouse = vec2(0.5);

    //float zoom = atan(fract(u_Time * 2e-1) * 2.0 - 1.0);
    vec2 sp = surfacePosition;
    vec2 position = gl_FragCoord.xy - (mouse * u_Resolution);
    float color = rand(position.xy * (0.25 + asin(sin(zoom)) * 11.1));
    position *= sp;

    //if (color < 0.5) color = 0.0; else color = 1.0;
    color = (floor(color * 4.0) + 0.0) / 4.0;

    FragColor = vec4(color, sin(position.x * position.y * zoom * 2.0), cos(color + zoom * 3.0), 1);
}
