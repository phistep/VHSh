
import glfw

class Window:

    def __init__(self,
                 title: str,
                 width: int,
                 height: int,
                 opacity: float = 1.0,
                 floating: bool = False):
        # TODO @property
        self.title = title
        self._opacity = opacity
        self._floating = floating

        if not glfw.init():
            RuntimeError("GLFW could not initialize OpenGL context")

        # needed for restoring the window posision
        glfw.window_hint_string(glfw.COCOA_FRAME_NAME, title)

        # macOS supports only forward-compatible core profiles from 3.2
        glfw.window_hint(glfw.CONTEXT_VERSION_MAJOR, 3)
        glfw.window_hint(glfw.CONTEXT_VERSION_MINOR, 3)
        glfw.window_hint(glfw.OPENGL_PROFILE, glfw.OPENGL_CORE_PROFILE)
        glfw.window_hint(glfw.OPENGL_FORWARD_COMPAT, glfw.TRUE)

        self._window = glfw.create_window(width=width,
                                          height=height,
                                          title=self.title,
                                          monitor=None,
                                          share=None)
        glfw.make_context_current(self._window)
        if not self._window:
            glfw.terminate()
            raise RuntimeError("GLFW could not initialize Window")

    @property
    def size(self) -> tuple[int, int]:
        ":returns: width, height"
        return glfw.get_framebuffer_size(self._window)

    @property
    def opacity(self) -> float:
        return self._opacity

    @opacity.setter
    def opacity(self, opacity: float):
        if self._opacity == opacity:
            return
        self._opacity = opacity
        glfw.set_window_opacity(self._window, opacity)

    @property
    def floating(self) -> bool:
        return self._floating

    @floating.setter
    def floating(self, floating: bool):
        if self._floating == floating:
            return
        self._floating = floating
        glfw.set_window_attrib(self._window,
                               glfw.FLOATING,
                               glfw.TRUE if floating else glfw.FALSE)

    # TODO as contetxt manager?
    # with window.update():
    #    # poll
    #    ...
    #    # swap buffers
    def update(self):
        glfw.poll_events()
        # TODO properties?

    def should_close(self) -> bool:
        return glfw.window_should_close(self._window)

    def swap_buffers(self):
        glfw.swap_buffers(self._window)

    def close(self):
        glfw.terminate()

    def __del__(self):
        self.close()
