#define PI 3.1415

uniform vec2 freq; // =(5.,5.) [-10,10]
uniform vec2 speed; // =(1.,1.) [-1000,1000,1]
uniform vec2 shift; // =(0.,0.) [0,1]

void main(void) {
    vec2 pos = gl_FragCoord.xy / u_Resolution;
    vec2 sh = vec2(u_Time);
    pos = pos + sh; //ift; // mod(pos + shift, 1.);

    vec2 phase = 2 * PI * pos * freq;
    vec2 signal;
    //signal.x = (sin(phase.x + offs.x) + 1.) / 2.;
    //signal.y = (cos(phase.x + offs.y) + 1.) / 2.;
    signal.x = (sin(phase.x) + 1.) / 2.;
    signal.y = (cos(phase.y) + 1.) / 2.;

    float color = (signal.x + signal.y) / 2.;
    // float color = length(signal)
    FragColor = vec4(vec3(color), 1.);
}
