# VHSh

_Video Home Shader_: A demo tool for digitally assisted analog vjaying


## Setup

Create a virtual environmenet and install the dependencies

```bash
python3.10 -m venv .venv
source .venv/bin/activate
pip3 install -r requirements.txt
```

Then run `VHSh` fron that environment

```bash
source .venv/bin/activate
python3 vhsh.py mandelbrot.glsl
```


## TODO

- [x] render fragment shader over the whole screen
- [x] load shader from file
- [x] auto-generate tuning ui for uniforms
- [ ] imgui display shader compile errors
- [ ] define defaults and ranges in uniform definition as comment
- [ ] auto-define builtin uniforms / math library / preamble
- [ ] hot reload https://watchfiles.helpmanual.io/api/watch/
- [ ] select different shaders
- [ ] save and load different presets (toml in the shader file?)
- [ ] split into runtime and imgui viewer
      - maybe just have option to show or hide the controls as separate window
      - https://github.com/ocornut/imgui/wiki/Multi-Viewports
      - https://github.com/ocornut/imgui/blob/docking/examples/example_glfw_opengl2/main.cpp
- uniforms
  - [x] time
  - [ ] mouse
  - [ ] prev frame
  - [ ] audio fft
  - [ ] video in
- [ ] raspberry pi midi or gpio support
