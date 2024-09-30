#version 330 core

out vec4 FragColor;
uniform vec2 u_Resolution;
uniform vec3 u_color;

void main() {
    vec2 pos = gl_FragCoord.xy / u_Resolution;
    FragColor = vec4(
            pos.x * u_color.x,
            pos.y * u_color.y,
            u_color.z,
            1.0
        );
}
