# TODO

## 1.0

- preset change: new uniforms are not added. do we want that? def not error flood
- too many current preset logs on midi save presert
- fix midi/set_normalized log (i have a stash)
- pretty INFO logs
  use Colors in shader error format
- test image when started without any shader files
  dump into workdir?
  -> move scenes/ into package
    if no files passed: load default scenes
      ?? dump minimal scene + add that to scenes
- migration tool: version 0 -> version 1
  - system uniform rename
  - version number

release beta

- read through every file and polish, check for dead code
- test on the machine!
- cleanup diagnostics
- ruff format?
- docstrings
- dev docs: Secenes, Parameters, Controllers, scene_index
- docs for `import` command

release rc

- github actions (test rcs)

release 1.0.0 🥳


## Road Map

- 0.2: multi-midi
- 1.0: merge refactor, backport new features on main and branches
- 1.1: named mappings
- 1.2: scene dir support: Collection? Project?
- 1.3: sampler2D support: image, video
- 1.4: built-in midi drivers by name/id
- 1.5: kiosk
- ...
- 2.0: timeline support (playlist)
- 3.0: wgpu


## Features

- [ ] autosave and restore uniform values
      - `atexit` and `pickle`
      - app dirs
- [ ] `#include`s, or at least one stdlib in preamble, or pass libs
- [ ] simplify parser: split on `" "`, then `match` on first char
- [ ] mouse uniform
- [ ] change time speed
- [ ] debian package: install system deps, mime handlers, dekstop file,
      branch `package-linux` or flatpak

- [ ] sampler2d
  - [ ] prev frame
  - [ ] video in
  - [ ] image/video file in with `uniform sampler2D foo; // @assets/foo.mp4`
  - [ ] arbitrary data as buffer object
  -> chatgpt: opencv
- [ ] limit resolution and upscale
- [ ] record mp4
- [ ] render to buffer, make prev frame available as texture

- [ ] multiple midi devcies
  - [x] open all devices
  - [ ] one mapping per devcie
  - [ ] default mappings by device name
- [ ] make midoi logger not log same message twice in a row
- [ ] vec3 input method: have the user assign multiple `#1:#2:#3`
    - multiple mappings for one parameter with `,`
- [ ] built-in midi drivers by name/id
      - program ccs
      - toggle
      - render controller diagram with control names to image
        texture for imgui, use openCV, can do video as well
        https://github.com/pyimgui/pyimgui/issues/82#issuecomment-1658259821
      - write state to MIDI controler (uTime, UI toggle etc)
      - https://www.korg.com/us/support/download/manual/0/159/2710/
      - also: configure push-button/toggle on-the-fly <toggle>

- [ ] make named midi ccs in toml via #defines
     ```toml
     [uniform.inputs]
     slider = [1, 2, 3, 4]
     knob = [10, 11, 12, 13]
     button = [20, 21, 22, 23]
     master_button = 42
     ```
     ```glsl
     uniform float zoom; // #slider1
     uniform bool debug; // <toggle> #button1
     uniform bool flash; // #master_button
     ```
   - [ ] view midi mappings in imgui

- [ ] kiosk startup mode: no gui and fullscreen (not possible in glfw, need sdl)
      maybe `glfw.get_cocoa_window` https://github.com/glfw/glfw/issues/1216
      - `--kiosk`: no gui, credits, auto cycle scenes after inactivity, morph presets
      - @author metadata field

- [ ] scene dir format
  - [ ] support reading from zip: `myscene.vhsh`
  - ```
    vhsh init NAME            run with `vhsh run NAME`, can also be run from zip
    --project
    -> NAME/
    -> NAME.glsl
    --raymarching             basic raymarching renderl loop
    -> raymarching.glsl
    --midi-controller=NAME    mapping and config file for controller NAME, if known else template
    -> midi.toml
    -> vhsh.nktrl2_data
    --lib LIB                 explicit hard copy of the stdlib
    -> iq.glsl                can be included via #include <iq>
    ```

- [ ] Gamma Correction
    - [_Monitor Guide: Gamma ramp_](https://www.glfw.org/docs/latest/monitor_guide.html)
    - [`GLFW_SRGB_CAPABLE`](https://www.glfw.org/docs/latest/window_guide.html#GLFW_SRGB_CAPABLE)
    - [`GLFWgammarramp`](https://www.glfw.org/docs/latest/group__monitor.html#ga939cf093cb0af0498b7b54dc2e181404)

- [-] audio fft
  - [x] listen
  - [x] fft
  - [x] array uniforms
  - [ ] normalization
  - [x] gui bar plot
  - [ ] selecting microphone
  - [ ] use sampler2d like shadertoy?
  - [ ] dome's advice


## Bugs

- fix `t` as uniform name doesn't generate ui
- bug uniform parsing when float `=0.0`
- fix dropdown crashes when no presets available
      ```
      File "/Users/phistep/Projects/vhsh/vhsh.py", line 563, in _update_gui
      for idx, item in  [(p['index'], p['name'])
                        ~^^^^^^^^^
      ```
- ugly crashes on `vhsh /*` when directories are passed, better input sanitation


## Ideas
- switch to SDL?
  - native macos fullscreen
  - mic input https://www.lazyfoo.net/tutorials/SDL/34_audio_recording/index.php
- "touchpad" widget for `vec2`
- move to wgpu (macos deprecates opengl)
