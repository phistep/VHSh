# TODO

## 1.0

release alpha

- fix midi/set_normalized log
- pretty INFO logs
  use colors in shader error format
- subcommand `run`
- rename uniforms to just capitalized: `Time`, etc.
- test image when started without any shader files
  dump into workdir?
  -> move scenes/ into package
    if no files passed: load default scenes
      ?? dump minimal scene + add that to scenes
- version number in shaders?

release beta

- test on the machine!
- cleanup diagnostics
- ruff format
- docstrings
- dev docs: Secenes, Parameters, Controllers, scene_index

release rc

- github actions (test rcs)

release 1.0.0 🥳


## Road Map

- 0.2: multi-midi
- 1.0: merge refactor, backport new features on main and branches
- 1.1: named mappings
- 1.2: scene dir support: Collection? Project?
        midi_mapping, scenes
- 1.3: sampler2D support: image, video
- 1.4: built-in midi drivers by name/id
        program ccs
        toggle
- 1.5: kiosk
- ...
- 2.0: timeline support (playlist)
- 3.0: wgpu


## Features

- [ ] multiple midi devcies
  - [x] open all devices
  - [ ] one mapping per devcie
  - [ ] default mappings by device name
- [ ] limit resolution and upscale
- [ ] write state to MIDI controler (uTime, UI toggle etc)
    - https://www.korg.com/us/support/download/manual/0/159/2710/
    - also: configure push-button/toggle on-the-fly <toggle>
- [ ] autosave and restore uniform values
      - `atexit` and `pickle`
      - app dirs
- [ ] `#include`s, or at least one stdlib in preamble, or pass libs
- [ ] vec3 input method: have the user assign multiple `#1:#2:#3`
    - multiple mappings for one parameter with `,`
- [ ] record mp4
- [ ] simplify parser: split on `" "`, then `match` on first char
- [ ] mouse uniform
- [ ] debian package: install system deps, mime handlers, dekstop file,
      branch `package-linux`
- [ ] make midoi logger not log same message twice in a row

- [ ] sampler2d
  - [ ] prev frame
  - [ ] video in
  - [ ] image/video file in with `uniform sampler2D foo; // @assets/foo.mp4`
  - [ ] arbitrary data as buffer object
  -> chatgpt: opencv

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

- [ ] subcommands
    - run
      - `--kiosk`: no gui, credits, auto cycle scenes after inactivity, morph presets
    - init
    - import (or run url directly?)
- [ ] shadertoy import

- [ ] kiosk startup mode: no gui and fullscreen (not possible in glfw, need sdl)
      maybe `glfw.get_cocoa_window` https://github.com/glfw/glfw/issues/1216

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
