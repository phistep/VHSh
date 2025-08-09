from typing import Type, Protocol
from array import array

import imgui

from .types import App
from .microphone import Microphone
from .scene import Widget


class ImguiRenderer(Protocol):
    def render(self, draw_data) -> None: ...
    def process_inputs(self) -> None: ...
    def shutdown(self) -> None: ...




class GUI:
    def __init__(self,
                 app: App,
                 renderer: Type[ImguiRenderer],
                 *args,
                 **kwargs):
        self._app = app
        self.visible = True

        imgui.create_context()
        imgui_style = imgui.get_style()
        imgui.style_colors_dark(imgui_style)
        imgui_style.colors[imgui.COLOR_PLOT_HISTOGRAM] = \
            imgui_style.colors[imgui.COLOR_PLOT_LINES]
        imgui_style.colors[imgui.COLOR_PLOT_HISTOGRAM_HOVERED] = \
            imgui_style.colors[imgui.COLOR_BUTTON_HOVERED]

        self._renderer = renderer(*args, **kwargs)

        self._new_preset_name = ""

    def update(self):
        app = self._app

        # TODO ctrl+tab? or ctrl+`
        # TODO not while in input
        if imgui.is_key_pressed(imgui.get_key_index(imgui.KEY_TAB)):
            self.visible = not self.visible

        imgui.new_frame()
        imgui.begin("Parameters", closable=False)

        if app.error is not None:
            imgui.open_popup("Error")
        with imgui.begin_popup_modal("Error",
            flags=imgui.WINDOW_NO_MOVE | imgui.WINDOW_NO_RESIZE
        ) as error_popup:
            if error_popup.opened:
                if app.error is None:
                    imgui.close_current_popup()
                else:
                    # TODO colored
                    imgui.text_wrapped(str(app.error))

        with imgui.begin_group():
            _, app.window.opacity = \
                imgui.slider_float("Opacity",
                                   app.window.opacity,
                                   min_value=0.,
                                   max_value=1.)

            imgui.same_line()

            _, app.window.floating = \
                imgui.checkbox('Floating', app.window.floating)

        imgui.spacing()
        imgui.separator()
        imgui.spacing()

        with imgui.begin_group():
            if imgui.begin_combo("##Scene", app.scene.name):
                for idx, item in enumerate([scene.name for scene in app.scenes]):
                    is_selected = (idx == app.scene_index)
                    if imgui.selectable(item, is_selected)[0]:
                        app.scene_index = idx
                    if is_selected:
                        imgui.set_item_default_focus()
                imgui.end_combo()
            imgui.same_line()
            if imgui.arrow_button("Prev Scene", imgui.DIRECTION_LEFT):
                app.prev_scene()
            imgui.same_line()
            if imgui.arrow_button("Next Scene", imgui.DIRECTION_RIGHT):
                app.next_scene()
            imgui.same_line()
            imgui.text("Scene")

        imgui.spacing()

        with imgui.begin_group():
            # TODO begin_list_box?
            if imgui.begin_combo(
                "##Preset", app.scene.presets[app.scene.preset_index].name
            ):
                for idx, item in  [(p.index, p.name)
                                   for p in app.scene.presets]:
                    is_selected = (idx == app.scene.preset_index)
                    if imgui.selectable(item, is_selected)[0]:
                        app.scene.preset_index = idx
                    if is_selected:
                        imgui.set_item_default_focus()
                imgui.end_combo()
            imgui.same_line()
            if imgui.arrow_button("Prev Preset", imgui.DIRECTION_LEFT):
                app.scene.prev_preset()
            imgui.same_line()
            if imgui.arrow_button("Next Preset", imgui.DIRECTION_RIGHT):
                app.scene.next_preset()
            imgui.same_line()
            if imgui.button("Save"):
                app.scene.write_file()
            imgui.same_line()
            imgui.text("Preset")

            # TODO should live in GUI
            _, self._new_preset_name = imgui.input_text_with_hint(
                "##Name", "New Preset Name", self._new_preset_name)
            imgui.same_line()
            if imgui.button("Save##Save New Preset"):
                app.scene.write_file(new_preset=self._new_preset_name)
                self._new_preset_name = ""
            imgui.same_line()
            imgui.text("New Preset")

        imgui.spacing()

        with imgui.begin_group():
            frame_times = array('f', app.frame_times)
            imgui.plot_lines("Frame Time##Plot", frame_times,
                overlay_text=f"{frame_times[-1]:5.2f} ms"
                             f"  ({1000/frame_times[-1]:3.0f} fps)")
            imgui.same_line()

        imgui.spacing()
        imgui.separator()
        imgui.spacing()

        # TODO disabled https://github.com/ocornut/imgui/issues/211#issuecomment-1245221815
        with imgui.begin_group():
            imgui.drag_float("u_Time", app.system_parameters['u_Time'].value)
            imgui.same_line()
            _, app.time.running = imgui.checkbox(
                'playing' if app.time.running else 'paused',
                app.time.running
            )

        imgui.drag_float2('u_Resolution',
                           *app.system_parameters['u_Resolution'].value,
                           format="%.0f")

        if "Microphone" in app.controllers:
            imgui.plot_histogram(
                Microphone.UNIFORM_NAME,
                array('f', app.system_parameters[Microphone.UNIFORM_NAME].value)
            )

        imgui.spacing()
        imgui.separator()
        imgui.spacing()

        current_preset = app.scene.presets[app.scene.preset_index]
        parameters = list(current_preset.parameters.items())
        peaking_parameters = zip(parameters, parameters[1:] + [(None, None)])
        for (name, parameter), (next_name, _) in peaking_parameters:
            flags = 0
            if parameter.widget == Widget.LOG:
                flags |=  (imgui.SLIDER_FLAGS_LOGARITHMIC
                           | imgui.SLIDER_FLAGS_NO_ROUND_TO_FORMAT)

            match parameter.value, parameter.widget:
                case bool(x), _:
                    _, parameter.value = imgui.checkbox(name, parameter.value)

                case int(x), Widget.DRAG:
                    min_, max_, step = parameter.range
                    _, parameter.value = imgui.drag_int(
                        name,
                        parameter.value,
                        min_value=min_,
                        max_value=max_,
                        change_speed=step
                    )
                case int(x), _:
                    min_, max_, step = parameter.range
                    _, parameter.value = imgui.slider_int(
                        name,
                        parameter.value,
                        min_value=min_,
                        max_value=max_,
                        flags=flags,
                    )

                case float(x), Widget.DRAG:
                    min_, max_, step = parameter.range
                    _, parameter.value = imgui.drag_float(
                        name,
                        parameter.value,
                        min_value=min_,
                        max_value=max_,
                        change_speed=step,
                        flags=flags,
                    )
                case float(x), _:
                    min_, max_, _ = parameter.range
                    _, parameter.value = imgui.slider_float(
                        name,
                        parameter.value,
                        min_value=min_,
                        max_value=max_,
                        flags=flags,
                    )

                case [float(x), float(y)], Widget.DRAG:
                    min_, max_, step = parameter.range
                    _, parameter.value = imgui.drag_float2(
                        name,
                        *parameter.value,
                        min_value=min_,
                        max_value=max_,
                        change_speed=step,
                        flags=flags
                    )
                case [float(x), float(y)], _:
                    min_, max_, step = parameter.range
                    _, parameter.value = imgui.slider_float2(
                        name,
                        *parameter.value,
                        min_value=min_,
                        max_value=max_,
                        flags=flags,
                    )

                case [float(x), float(y), float(z)], Widget.COLOR:
                    _, parameter.value = imgui.color_edit3(name, *parameter.value,
                                                            imgui.COLOR_EDIT_FLOAT)  # pyright: ignore [reportCallIssue]
                case [float(x), float(y), float(z)], Widget.DRAG:
                    min_, max_, step = parameter.range
                    _, parameter.value = imgui.drag_float3(
                        name,
                        *parameter.value,
                        min_value=min_,
                        max_value=max_,
                        change_speed=step
                    )
                case [float(x), float(y), float(z)], _:
                    min_, max_, _ = parameter.range
                    _, parameter.value = imgui.slider_float3(
                        name,
                        *parameter.value,
                        min_value=min_,
                        max_value=max_,
                        flags=flags,
                    )

                case [float(x), float(y), float(z), float(w)], Widget.COLOR:
                    _, parameter.value = imgui.color_edit4(name, *parameter.value,
                                                         imgui.COLOR_EDIT_FLOAT)  # pyright: ignore [reportCallIssue]
                case [float(x), float(y), float(z), float(w)], Widget.DRAG:
                    min_, max_, step = parameter.range
                    _, parameter.value = imgui.drag_float4(
                        name,
                        *parameter.value,
                        min_value=min_,
                        max_value=max_,
                        change_speed=step
                    )
                case [float(x), float(y), float(z), float(w)], _:
                    min_, max_, _ = parameter.range
                    _, parameter.value = imgui.slider_float4(
                        name,
                        *parameter.value,
                        min_value=min_,
                        max_value=max_,
                        flags=flags,
                    )

            # group prefixed uniforms
            if next_name is not None:
                if name.split('_')[0] != next_name.split('_')[0]:
                    imgui.spacing()

        imgui.end()
        imgui.end_frame()

    def process_inputs(self):
        self._renderer.process_inputs()

    def render(self):
        if not self.visible:
            return

        imgui.render()
        self._renderer.render(imgui.get_draw_data())

    def shutdown(self):
        self._renderer.shutdown()
