#version 330 core

out vec4 FragColor;
uniform vec2 u_Resolution;
uniform float u_Time;
uniform vec3 u_color;

void main() {
    vec2 pos = gl_FragCoord.xy / u_Resolution;
    FragColor = vec4(
            pos.x * u_color.r,
            pos.y * u_color.g,
            u_color.b * (0.5 + 0.5 * (sin(2 * 3.14 * u_Time * 0.5))),
            1.0
        );
}
