from typing import Iterable, TypeVar
from array import array

import imgui
from imgui.integrations.glfw import GlfwRenderer

from .common import App, get_shader_title


T = TypeVar('T')


# TODO move to Uniform @property range
def _get_range(value: Iterable[T],
                min_default: T,
                max_default: T,
                step_default: T = None):
    match value:
        case (min_, max_):
            return min_, max_, step_default
        case (min_, max_, step):
            return min_, max_, step
        case _:
            return min_default, max_default, step_default

class GUI:
    def __init__(self, app: App):
        self._app = app

        imgui.create_context()
        imgui_style = imgui.get_style()
        imgui.style_colors_dark(imgui_style)
        imgui_style.colors[imgui.COLOR_PLOT_HISTOGRAM] = \
            imgui_style.colors[imgui.COLOR_PLOT_LINES]
        imgui_style.colors[imgui.COLOR_PLOT_HISTOGRAM_HOVERED] = \
            imgui_style.colors[imgui.COLOR_BUTTON_HOVERED]

        self._glfw_imgui_renderer = GlfwRenderer(self._app._window)


    def update(self):
        # TODO ctrl+tab? or ctrl+`
        # TODO not while in input
        if imgui.is_key_pressed(imgui.get_key_index(imgui.KEY_TAB)):
            self._app._show_gui = not self._app._show_gui

        imgui.new_frame()
        imgui.begin("Parameters", closable=False)

        if self._app._error is not None:
            imgui.open_popup("Error")
        with imgui.begin_popup_modal("Error",
            flags=imgui.WINDOW_NO_MOVE | imgui.WINDOW_NO_RESIZE
        ) as error_popup:
            if error_popup.opened:
                if self._app._error is None:
                    imgui.close_current_popup()
                else:
                    # TODO colored
                    imgui.text_wrapped(str(self._app._error))

        with imgui.begin_group():
            _, self._app.floating = imgui.checkbox('Floating' ,
                                                              self._app.floating)

            imgui.same_line()
            _, self._app.opacity = imgui.slider_float("Opacity",
                                                      self._app.opacity,
                                                      min_value=0.,
                                                      max_value=1.)

        imgui.spacing()
        imgui.separator()
        imgui.spacing()

        with imgui.begin_group():
            if imgui.begin_combo("##Scene",
                                 get_shader_title(self._app._shader_path)):
                for idx, item in enumerate(
                        map(get_shader_title, self._app._shader_paths)):
                    is_selected = (idx == self._app._shader_index)
                    if imgui.selectable(item, is_selected)[0]:
                        self._app._shader_index = idx
                    if is_selected:
                        imgui.set_item_default_focus()
                imgui.end_combo()
            imgui.same_line()
            if imgui.arrow_button("Prev Scene", imgui.DIRECTION_LEFT):
                self._app.prev_shader()
            imgui.same_line()
            if imgui.arrow_button("Next Scene", imgui.DIRECTION_RIGHT):
                self._app.next_shader()
            imgui.same_line()
            imgui.text("Scene")

        imgui.spacing()

        with imgui.begin_group():
            # TODO begin_list_box?
            if imgui.begin_combo(
                "##Preset", self._app.presets[self._app.preset_index].name
            ):
                for idx, item in  [(p.index, p.name)
                                   for p in self._app.presets]:
                    is_selected = (idx == self.preset_index)
                    if imgui.selectable(item, is_selected)[0]:
                        self._app.preset_index = idx
                    if is_selected:
                        imgui.set_item_default_focus()
                imgui.end_combo()
            imgui.same_line()
            if imgui.arrow_button("Prev Preset", imgui.DIRECTION_LEFT):
                self._app.prev_preset()
            imgui.same_line()
            if imgui.arrow_button("Next Preset", imgui.DIRECTION_RIGHT):
                self._app.next_preset()
            imgui.same_line()
            if imgui.button("Save"):
                self._app.write_file(uniforms=False, presets=True)
            imgui.same_line()
            imgui.text("Preset")

            _, self._app._new_preset_name = imgui.input_text_with_hint(
                "##Name", "New Preset Name", self._app._new_preset_name)
            imgui.same_line()
            if imgui.button("Save##Save New Preset"):
                self.write_file(uniforms=False, presets=True, new_preset=self._app._new_preset_name)
                self._app._new_preset_name = ""
            imgui.same_line()
            imgui.text("New Preset")

        imgui.spacing()

        with imgui.begin_group():
            frame_times = array('f', self._app._frame_times)
            imgui.plot_lines("Frame Time##Plot", frame_times,
                overlay_text=f"{frame_times[-1]:5.2f} ms"
                             f"  ({1000/frame_times[-1]:3.0f} fps)")
            imgui.same_line()

        imgui.spacing()
        imgui.separator()
        imgui.spacing()

        # TODO disabled https://github.com/ocornut/imgui/issues/211#issuecomment-1245221815
        with imgui.begin_group():
            imgui.drag_float("u_Time", self._app.uniforms['u_Time'].value)
            imgui.same_line()
            changed, self._app._time_running = imgui.checkbox(
                'playing' if self._app._time_running else 'paused',
                self._app._time_running
            )
            if changed:
                if self._app._time_running:
                    # TODO api in App
                    # glfw.set_time(self._app._start_time)
                    pass
                else:
                    # self._start_time = glfw.get_time()
                    pass

        imgui.drag_float2('u_Resolution', *self._app.uniforms['u_Resolution'].value)

        if self._app._microphone:
            imgui.plot_histogram("u_Microphone",
                                 array('f', self._app.uniforms['u_Microphone'].value))

        imgui.spacing()
        imgui.separator()
        imgui.spacing()


        uniforms = list(self._app.uniforms.items())
        peaking_uniforms = zip(uniforms, uniforms[1:] + [(None, None)])
        for (name, uniform), (next_name, _) in peaking_uniforms:
            # TODO system_uniforms?
            if name in self._app.FRAGMENT_SHADER_PREAMBLE:
                continue

            flags = 0
            if uniform.widget == 'log':
                flags |=  (imgui.SLIDER_FLAGS_LOGARITHMIC
                           | imgui.SLIDER_FLAGS_NO_ROUND_TO_FORMAT)

            match uniform.value, uniform.widget:
                case bool(x), _:
                    _, uniform.value = imgui.checkbox(name, uniform.value)

                case int(x), 'drag':
                    min_, max_, step = _get_range(uniform.range, 0, 100, 1)
                    _, uniform.value = imgui.drag_int(
                        name,
                        uniform.value,
                        min_value=min_,
                        max_value=max_,
                        change_speed=step
                    )
                case int(x), _:
                    min_, max_, step = _get_range(uniform.range, 0, 100, 1)
                    _, uniform.value = imgui.slider_int(
                        name,
                        uniform.value,
                        min_value=min_,
                        max_value=max_,
                        flags=flags,
                    )

                case float(x), 'drag':
                    min_, max_, step = _get_range(uniform.range, 0., 1., 0.01)
                    _, uniform.value = imgui.drag_float(
                        name,
                        uniform.value,
                        min_value=min_,
                        max_value=max_,
                        change_speed=step,
                        flags=flags,
                    )
                case float(x), _:
                    min_, max_, _ = _get_range(uniform.range, 0., 1., 0.01)
                    _, uniform.value = imgui.slider_float(
                        name,
                        uniform.value,
                        min_value=min_,
                        max_value=max_,
                        flags=flags,
                    )

                case [float(x), float(y)], 'drag':
                    min_, max_, step = _get_range(uniform.range, 0., 1., 0.01)
                    _, uniform.value = imgui.drag_float2(
                        name,
                        *uniform.value,
                        min_value=min_,
                        max_value=max_,
                        change_speed=step,
                        flags=flags
                    )
                case [float(x), float(y)], _:
                    min_, max_, step = _get_range(uniform.range, 0., 1., 0.01)
                    _, uniform.value = imgui.slider_float2(
                        name,
                        *uniform.value,
                        min_value=min_,
                        max_value=max_,
                        flags=flags,
                    )

                case [float(x), float(y), float(z)], 'color':
                    _, uniform.value = imgui.color_edit3(name, *uniform.value,
                                                            imgui.COLOR_EDIT_FLOAT)  # pyright: ignore [reportCallIssue]
                case [float(x), float(y), float(z)], 'drag':
                    min_, max_, step = _get_range(uniform.range, 0., 1., 0.01)
                    _, uniform.value = imgui.drag_float3(
                        name,
                        *uniform.value,
                        min_value=min_,
                        max_value=max_,
                        change_speed=step
                    )
                case [float(x), float(y), float(z)], _:
                    min_, max_, _ = _get_range(uniform.range, 0., 1., 0.01)
                    _, uniform.value = imgui.slider_float3(
                        name,
                        *uniform.value,
                        min_value=min_,
                        max_value=max_,
                        flags=flags,
                    )

                case [float(x), float(y), float(z), float(w)], 'color':
                    _, uniform.value = imgui.color_edit4(name, *uniform.value,
                                                         imgui.COLOR_EDIT_FLOAT)  # pyright: ignore [reportCallIssue]
                case [float(x), float(y), float(z), float(w)], 'drag':
                    min_, max_, step = get_range(uniform.range, 0., 1., 0.01)
                    _, uniform.value = imgui.drag_float4(
                        name,
                        *uniform.value,
                        min_value=min_,
                        max_value=max_,
                        change_speed=step
                    )
                case [float(x), float(y), float(z), float(w)], _:
                    min_, max_, _ = _get_range(uniform.range, 0., 1., 0.01)
                    _, uniform.value = imgui.slider_float4(
                        name,
                        *uniform.value,
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
        self._glfw_imgui_renderer.process_inputs()

    def render(self):
        if not self._app._show_gui:
            return

        imgui.render()
        self._glfw_imgui_renderer.render(imgui.get_draw_data())

    def shutdown(self):
        self._glfw_imgui_renderer.shutdown()
