# VHSh

_Video Home Shader_: A demo tool for digitally assisted analog vjaying

![Screenshot of VHSh in action](https://raw.githubusercontent.com/phistep/vhsh/0.1.1/screenshot.png)

## Setup

- macOS
  ```sh
  brew install uv
  ```
- ubuntu
  ```sh
  curl -LsSf https://astral.sh/uv/install.sh | sh
  # for audio support
  sudo apt install portaudio19-dev
  ```

```sh
uv tool update-shell
uv tool install vhsh[all]
vhsh -h
```

- MIDI support: `vhsh[midi]`
- audio support: `vhsh[audio]`
- everyting: `vhsh[all]`


### Development

Create a virtual environmenet and install the dependencies

```sh
uv venv
source .venv/bin/activate
uv install -e '.[all,dev]'
```

Building and publishing a package

```sh
uv build

source .env  # export UV_PUBLISH_TOKEN=
uv publish
```


## Usage

Then run `VHSh` fron that environment

```bash
source .venv/bin/activate
python3 vhsh.py mandelbrot.glsl
```

If you pass multiple shader files, you can switch between them in the tool.
To open all files in a given folder, use `my_shader_folder/*`.

You can pass `--watch` to automatically reload the shader upon file change.

You can pass `--mic` to enable microphone input. See
[_Builin Parameters_](#builtin-parameters).

To toggle the UI, press `<tab>`.

If you're seeing a message like

> 2024-10-02 22:10:15.567 Python\[75271:1828570\] ApplePersistenceIgnoreState:
> Existing state will not be touched. New state will be written to
> /var/folders/2b/gfpmffr15n9cwdy6_44mhy8r0000gn/T/org.python.python.savedState

run the following to get rid of it:

```bash
defaults write org.python.python ApplePersistenceIgnoreState NO
```


### MIDI Support

When using the `--midi` flag, VHSh will listen to incoming MIDI messages and allow
you to map parameters to MIDI controls. How to assign uniform mappings is described
in [Custom Parameters](#Custom_Parameters).

There are also a couple of system controls, like switching scenes, that can be
mapped to buttons as well. Such a mapping is defined as a [TOML][toml] file and
passed via `--midi-mapping`.

```toml
[scene]
prev = 58  # switch to next scene
next = 59  # switch to previous scene

[preset]
prev = 61  # switch to next preset
next = 62  # switch to previous preset
save = 60  # save current parameter values to a new preset

[uniform]
toggle_ui = 45  # toggle paramter tweaking window

[uniform.time]
toggle = 41  # toggle u_Time running
```

Sensible mappings for various controls are supplied in
[`midi_mappings/`](./midi_mappings).


## Writing Shaders for _Video Home Shader_

_Video Home Shader_ supplies you with a 2D canvas to draw into using an OpenGL
_fragment shader_. It is run once for every pixel on the screen and determines
it's color.

Berfore your shader file is run, a preamble is prepended to the source code.
It defines

> OpenGL Version 330 core

so you just need to supply a `main` function and set the output color `FragColor`
as an RGBA `vec4` with floats between 0 and 1 (`0., 0., 0., 1.)` being black).

```glsl
void main() {
    FragColor = vec4(0.8, 0.2, 0.2, 1.0);
}
```

### Builtin Parameters

You can use the following built-in parameters, that are pre-defined in the
preamble:

- `vec2 u_Resolution`: width and height of the window in pixels. This can
  be used to calculate normalized screen space coordinates like
  ```glsl
  vec2 pos = gl_FragCoord.xy / u_Resolution;
  ```
  where `pos.xy` will now have the current pixel's coordinates between
  `[-1, 1]^2`
- `float u_Time`: Seconds since the program start. This can be used to animate
  things. For example
  ```glsl
  vec4 color = vec4((sin(2. * 3.14 * u_Time * ) + 1.) / 2., 0., 0., 1.);
  ```
  will create a red pulsing effect with one pulse per second.
- `float[7] u_Microphone`: If started with `--mic`, this is a float
  array that gives you volume per frequency band normalized over the last 5s.

  | Index             | Range   |        | Description |
  | ----------------- | ------- | ------ | ----------- |
  | `u_Microphone[0]` | 0 Hz    | 60 Hz  | Rumble      |
  | `u_Microphone[1]` | 60 Hz   | 250 Hz | Low End     |
  | `u_Microphone[2]` | 250 Hz  | 500 Hz | Low Mids    |
  | `u_Microphone[3]` | 500 Hz  | 2 kHz  | Mids        |
  | `u_Microphone[4]` | 2 KHz   | 6 kHz  | High Mids   |
  | `u_Microphone[5]` | 6 kHz   | 8 kHz  | Highs       |
  | `u_Microphone[6]` | > 8 KHz |        | Air         |

### Custom Parameters

You can define custom parameters to vary directly in the code, and the user
interface to manipulate them will be generated automatically. Use the `uniform`
keyword followed by a type (`bool`, `int`, `float`, `vec2`, `vec3`, `vec4`) and
a name.
```glsl
uniform bool override_red; // =False
uniform int n_max; // =10 [1,200] #0
uniform float scale; // =1. [0.,2.] #16
uniform vec2 origin; // =(0.,0.) [-2.,-2.]
uniform vec3 dir; // =(1.,0.,0.) [-1.,-1.]
uniform vec4 base_color; // <color> =(1.,1.,0.,1.)
```

Using a special syntax in a comment on the same line, you can define the the
following uniform control properties:

- `=VALUE` default value
- `[MIN,MAX,STEP]` range (where `STEP` is optional).
- `<WIDGET>` special UI widget
  - `<color>` on `vec3` for a RGB and on `vec4` for a RGBA color picker
  - `<log>` for a logarithmic scale
  - `<drag>` for controling the UI widget with dragging (instead of slider)
- `#MIDI` MIDI control ID. To bind a MIDI control to a uniform,
  for example: `#16`.

The have to be defined in the order

```glsl
uniform type name; // <WIDGET> =VALUE [MIN,MAX,STEP] #MIDI
```

and the values (and ranges) may not contain whitespace. Each individual part
(widget, value, range) is optional and they can be mixed and matched as
desired, as long as the order of appearance is correct. As generally with
GLSL, it is also important to strictly match the types. Supplying a `float`
as default value for an `int` will not work. There may be no other text in
the comment. All vector types have to be supplied as a comma-separated list
of floats, enclosed by parentheses `(1.,2.,3.)`. One can only supply a
scalar range that applies along all dimensions.

```glsl
// syntax error
uniform float scale; // =1
// OK
uniform float scale; // =1.

// syntax error
uniform vec2 origin; // =[0.,0.]
// OK
uniform vec2 origin; // =(0.,0.)

// syntax error
uniform vec3 dir; // [[0.,1.],[0.,1.],[0.,5.]]
// OK
uniform vec3 dir; // [0.,1.]
dir.z *= 5;
```

By using `ctrl+click`, one can directly edit the values with keyboard
input.

If two consecutive uniforms share a common prefix in their name (like
`box_size` and `box_color`), they will be grouped together.


## Presets

You can save the current uniform values as the new `=DEFAULT` parameter in your
loaded shader source file by clicking `Save` when the currently selected preset.
is `<current>`.

Furthermore, you can store multiple sets of parameters (including different)
default values, ranges, MIDI mappings etc.) as _presets_. To save a new preset,
enter the name in the `Name` field and click `New Preset`. The shader source file
will be modified by prepending the unform and metadata defintions with a special
comment prefix (`/// `). Since all those lines will deleted and rewritten on save,
be sure to not use triple-slashes for other reasons. Each preset is preceeded by
its name.

```glsl
/// // My New Preset
/// uniform bool override_red; // =False
/// uniform int n_max; // =10 [1,200] #0
/// uniform float scale; // =1. [0.,2.] #16
/// uniform vec2 origin; // =(0.,0.) [-2.,-2.]
/// uniform vec3 dir; // =(1.,0.,0.) [-1.,-1.]
/// uniform vec4 base_color; // <color> =(1.,1.,0.,1.)
```

You can add, modify and delete these comment blocks with you're text editor as
well.

To update an existing preset, select it, adjust the parameter values and click
`Save`. The current paremeter values will be written to the default values of the
currently selected preset.


## TODO

- [x] render fragment shader over the whole screen
- [x] load shader from file
- [x] auto-generate tuning ui for uniforms
- [x] auto-define builtin uniforms / math library / preamble
- [x] hot reload https://watchfiles.helpmanual.io/api/watch/
- [x] define defaults and ranges in uniform definition as comment
- [x] MIDI controller support
- [x] select different shaders
- [x] save and load different presets
- [x] write current values to file
- [x] 60fps cap / fps counter
- [x] show or hide the controls
- [x] imgui display shader compile errors
- [x] widget size and close button
- [x] re-parse metadata on reload
- [x] remember window position
- [ ] fix dropdown crashes when no presets available
      ```
      File "/Users/phistep/Projects/vhsh/vhsh.py", line 563, in _update_gui
      for idx, item in  [(p['index'], p['name'])
                        ~^^^^^^^^^
      ```
- [ ] fix `t` as uniform name doesn't generate ui
- [ ] bug uniform parsing when float `=0.0`
- [ ] limit resolution and upscale
- [ ] write state to MIDI controler (uTime, UI toggle etc)
- [ ] autosave and restore uniform values
      `atexit` and `pickle`
- [ ] `#include`s, or at least one stdlib in preamble, or pass libs
- [ ] vec3 input method:
      - select dim with S/M/R buttons, then use the slider
      - auto assign n sucessor ids as well
      - have the user assign multiple `#1,#2,#3`
- [ ] "touchpad" widget for `vec2`
- [ ] test image when started without any shader files
- [ ] record mp4
- [ ] startup mode: no gui and fullscreen (not possible in glfw, need sdl)
      maybe `glfw.get_cocoa_window` https://github.com/glfw/glfw/issues/1216
- [ ] TODO.md
- [ ] pypi
- [ ] pass scene dir with scenes, midi mapping and other assets
- [ ] shadertoy import
- [ ] rename uniforms to just capitalized: `Time`, etc.
- [ ] simplify parser: split on `" "`, then `match` on first char
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
- [ ] widgets
  - [x] `<log>`
  - [x] `<drag>` drag input, others sliders (for slider flags)
  - [x] ~~~`<hsv` and `<rgb>`~~~
  - [ ] MIDI vector control with button triplet
- [ ] uniforms
  - [x] time
  - [ ] mouse
  - [ ] prev frame
  - [-] audio fft
    - [x] listen
    - [x] fft
    - [x] array uniforms
    - [ ] normalization
    - [ ] gui bar plot
    - [ ] docs, demo scene
    - [ ] selecting microphone
  - [ ] video in
  - [ ] image/video file in with `uniform sampler2D foo; // @assets/foo.mp4`
  - [ ] arbitrary data as buffer object
- [ ] Gamma Correctio
    - [_Monitor Guide: Gamma ramp_](https://www.glfw.org/docs/latest/monitor_guide.html)
    - [`GLFW_SRGB_CAPABLE`](https://www.glfw.org/docs/latest/window_guide.html#GLFW_SRGB_CAPABLE)
    - [`GLFWgammarramp`](https://www.glfw.org/docs/latest/group__monitor.html#ga939cf093cb0af0498b7b54dc2e181404)
- [ ] big refactoring
  - one file? or full package with exe in PATH?
  - docstrings
  - ```
    VideoHomeShader
        context: all variables to consider
      MIDIManager
        Thread
        needs uniforms, system commands
      GUI
        needs uniforms, system commands
      ShaderRenderer
        scenes
        needs system commands
      FileWatcher
        talks to Shader Renderer
      PresetManager
        presets
    ```
- switch to SDL?
  - native macos fullscreen
  - mic input https://www.lazyfoo.net/tutorials/SDL/34_audio_recording/index.php

## Resources

- https://pyopengl.sourceforge.net/documentation/manual-3.0/
- https://regex101.com
- https://github.com/pyimgui/pyimgui/blob/master/doc/examples/testwindow.py
- https://pthom.github.io/imgui_manual_online/manual/imgui_manual.html
- https://iquilezles.org/articles/
- https://docs.gl/sl4/
- https://www.youtube.com/watch?v=f4s1h2YETNY
- http://dev.thi.ng/gradients/
- https://mido.readthedocs.io/en/stable/intro.html


[imgui-issue-stubs]: https://github.com/pyimgui/pyimgui/issues/364
[imgui.pyi]: https://raw.githubusercontent.com/denballakh/pyimgui-stubs/refs/heads/master/imgui.pyi
[toml]: https://toml.io
