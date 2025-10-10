float rect_mask(vec2 p, vec2 center, vec2 dimensions) {
    return (1. - step(dimensions.x, length(p.x - center.x)))
        * (1. - step(dimensions.y, length(p.y - center.y)));
}

float circle_mask(vec2 p, vec2 center, float r) {
    return 1. - step(r, length(p - center));
}

void main(void) {
    vec2 p = gl_FragCoord.xy / Resolution;
    vec2 center = vec2(0.5);
    vec3 color = vec3(0.);

    // "circle"
    float the_cirle = circle_mask(p, center, 0.475);
    float not_the_circle = 1. - the_cirle;

    // background
    color += vec3(0.3) * not_the_circle;

    // backround grid + middle part
    color += (floor(pow(sin((p.x + .05) * 10. * 3.141), 2.) + 0.01)
            + floor(pow(sin((p.y + .05) * 10. * 3.141), 2.) + 0.01)
        ) * (1. - (the_cirle * (1. - rect_mask(p, vec2(.5, .5), vec2(.5, .05)))));

    // top black/white shit
    color += vec3(1.) * the_cirle * rect_mask(p, vec2(.15, .9), vec2(.2, .05));
    color += vec3(1.) * the_cirle * rect_mask(p, vec2(.85, .9), vec2(.2, .05));
    color += vec3(1.) * the_cirle * rect_mask(p, vec2(.5, 1.), vec2(.5, .05));

    // stripes
    color += .5 * mod(floor(p.x * 10.), 2.) * the_cirle * rect_mask(p, vec2(.5, .8), vec2(.5, .05));

    // color test
    // yellow
    color += vec3(.5, .5, 0.) * the_cirle * rect_mask(p, vec2(.0, .65), vec2(.1, .1));
    // turquise
    color += vec3(.0, .5, .5) * the_cirle * rect_mask(p, vec2(.2, .65), vec2(.1, .1));
    // green
    color += vec3(0., 1., 0.) * the_cirle * rect_mask(p, vec2(.4, .65), vec2(.1, .1));
    // magenta
    color += vec3(.5, .0, .5) * the_cirle * rect_mask(p, vec2(.6, .65), vec2(.1, .1));
    // red
    color += vec3(1., 0., 0.) * the_cirle * rect_mask(p, vec2(.8, .65), vec2(.1, .1));
    // blue
    color += vec3(0., 0., 1.) * the_cirle * rect_mask(p, vec2(1., .65), vec2(.1, .1));

    // middle part
    //color += vec3(0.) * the_cirle * rect_mask(p, vec2(.5,.5), vec2(.5,.05));

    // ever smaller stripes
    float c = p.x + .3;
    color += sin(c * 3.14 * 200. * log(c)) * the_cirle * rect_mask(p, vec2(.5, .35), vec2(0.5, 0.1));

    // gray scale test
    color += floor(p.x * 6.) / 6. * the_cirle * rect_mask(p, vec2(.5, .2), vec2(.5, .05));

    // bottom color shit
    color += vec3(.5, .5, 0.) * the_cirle * rect_mask(p, vec2(.25, .05), vec2(.2, .1));
    color += vec3(.5, .5, 0.) * the_cirle * rect_mask(p, vec2(.75, .05), vec2(.2, .1));
    color += vec3(1., 0., 0.) * the_cirle * rect_mask(p, vec2(.5, .05), vec2(.05, .1));

    FragColor = vec4(color, 0.);
}
