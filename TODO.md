# TODO

## 1.0

- fix package can't find the default scenes!
  uv run --with  dist/vhsh-1.0.0b0-py3-none-any.whl vhsh -v  run

- git tag `__version__`
- proper pyproject.toml metadata
- license -> AGPL3.0?

release b1

- read through every file and polish, check for dead code
- test on the machine!

release rc

- dev docs: Scenes, Parameters, Controllers, scene_index
- docstrings
- github actions (test, lint, tag+publish on merge)

release 1.0.0 🥳


## Road Map

- 0.2: multi-midi
- 1.0: merge refactor, backport new features on main and branches
- 1.1: midi
  - named mappings `#btn1`
  - vec2/3/4 mappings `#1:2`
  - multiple ccs per parameter `#2,sl3`
  - "shift/alt" for more mappings
  - joystick support `<rotation>`, `<translation>`
  - gui show mappings
  - version + pydantic model midi mapping? 
  - specialized parameter classes
- 1.2: scene dir support: Collection? Project?
- 1.3: kiosk
- 1.4: sampler2D support: image, video
- 1.5: built-in midi drivers by name/id
- ...
- 2.0: timeline support (playlist)
- 3.0: wgpu + run as wasm in browser


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

- included scenes
  - testcard
    - slider for saturation
    - mic fft display
    - correct streching
    - improved sinus

  - system integration
    - linux/macos
    - [ ] open file picker
    - [ ] file type default app
    - [ ] app icon
    - [ ] imgui ini location: XDG_STATE_HOME
          https://pyimgui.readthedocs.io/en/latest/reference/imgui.core.html#imgui.core._IO.ini_file_name
    - [ ] autosave uniform values, restore
    - [ ] save window position, monitor, transparency, floating, ...
        - flag to start afreash
    - [ ] config file? XDG_CONFIG_HOME, just cli flag file?
        - import api key,
        - microphone
        - keybindings (time, scene, preset, ...)
    - [ ] write default midi mappings to XDG_DATA_HOME
    - https://www.glfw.org/docs/3.3/window_guide.html#window_hints_osx
      - `GLFW_COCOA_GRAPHICS_SWITCHING, False`
      - `GLFW_COCOA_RETINA_FRAMEBUFFER, True`

- [ ] joystick support:
      `vec3 <position>`, `vec4 <orientation>` (quat?)
      keep internal state, treat inputs as delta/velocity
- [ ] arrow keys for <position>, mouse for <orientation>

- [ ] sampler2d
  - [ ] prev frame
  - [ ] video in
  - [ ] image/video file in with `uniform sampler2D foo; // @assets/foo.mp4`
  - [ ] arbitrary data as buffer object
  -> chatgpt: opencv
- [ ] limit resolution and upscale
  - `glfw.window_hint(glfw.COCOA_RETINA_FRAMEBUFFER, glfw.TRUE)`

- [ ] record mp4
- [ ] render to buffer, make prev frame available as texture

- [ ] multiple midi devcies
  - [x] open all devices
  - [ ] one mapping per devcie
  - [ ] default mappings by device name
- [ ] make midoi logger not log same message twice in a row
- [ ] vec3 input method: have the user assign multiple `#1:#2:#3`
    - multiple mappings for one parameter with `,`
    - set_value_normalized takes optional `entry:
- [ ] built-in midi drivers by name/id
      - program ccs
      - toggle
      - render controller diagram with control names to image
        texture for imgui, use openCV, can do video as well
        https://github.com/pyimgui/pyimgui/issues/82#issuecomment-1658259821
      - write state to MIDI controler (uTime, UI toggle etc)
      - https://www.korg.com/us/support/download/manual/0/159/2710/
      - also: configure push-button/toggle on-the-fly <toggle>


- [ ] gui refactor: functions for windows,sections
- [ ] sidebar?
- [ ] imgui debug windows when `-v` passed
      ```python
      imgui.begin("Guide", closable=False)
      imgui.show_user_guide()
      imgui.end()
      imgui.show_metrics_window()
      imgui.begin("Style", closable=False)
      imgui.show_style_editor()
      imgui.end()
      ```


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
- embed strudel.cc
- switch to SDL?
  - native macos fullscreen
  - mic input https://www.lazyfoo.net/tutorials/SDL/34_audio_recording/index.php
- switch to https://github.com/pthom/imgui_bundle
  - markdown rendering for docs/news
  - text editor
  - node editor
- "touchpad" widget for `vec2`
- move to wgpu (macos deprecates opengl)


## Upstream
### pyimgui-stubs
- [ ] [#3: `plot_historgram`](https://github.com/denballakh/pyimgui-stubs/pull/3)
- [ ] [#4: build-system in pyproject.toml](https://github.com/denballakh/pyimgui-stubs/pull/4)
- [ ] [#5: ignore .zed editor config dir](zed:https://github.com/denballakh/pyimgui-stubs/pull/5)
### mido
- [ ] [#641: Missing Attributes on Message and Backend when Type Checking v1.3.4.dev](https://github.com/mido/mido/issues/641)
### ty
- [ ] [#2250: Incomplete function argument inlay hint for unpacked iterable](https://github.com/astral-sh/ty/issues/2250)
- [ ] [#2251: Emit error when unpacking a not-iterable argument in a call](https://github.com/astral-sh/ty/issues/2251)
- [ ] [#2253: Unable to infer zip type](https://github.com/astral-sh/ty/issues/2253)
- [ ] [#2282: Support dynamic type guards with `hasattr` in a loop](https://github.com/astral-sh/ty/issues/2282)


```py
# FIXME: `uv` cannot catch the dynamic type guard
# for attr in [...]: raise RuntimeError(f"...{attr}")
if not hasattr(glfw, attr := "window_hint_string"):
    raise RuntimeError(f"GLFW is missing required attribute: {attr}")
if not hasattr(glfw, attr := "set_window_opacity"):
    raise RuntimeError(f"GLFW is missing required attribute: {attr}")
if not hasattr(glfw, attr := "set_window_attrib"):
    raise RuntimeError(f"GLFW is missing required attribute: {attr}")
```
