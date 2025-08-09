import logging
from threading import Event
from pathlib import Path

from .types import Controller, App


logger = logging.getLogger(__name__)


class FileWatcher(Controller):

    def __init__(self, app: App):
        super().__init__()
        self._app = app
        self.file_changed = Event()
        self.filenames = [scene.path for scene in app.scenes]

    def run(self):
        try:
            from watchfiles import watch
        except ImportError:
            logger.warning(
                "Not watching for file changes!"
                " 'watchfiles' not installed, install with 'vhsh[watch]'")
            self.stop()
            return

        logger.info(f"Watching for changes in %s...",
                    [str(f) for f in self.filenames])

        for changes in watch(*self.filenames, stop_event=self._stop_controller):
            for _, filename in changes:
                logger.debug(filename)
                if Path(filename).absolute() == self._app.scene.path.absolute():
                    logger.debug("'%s' changed!", filename)
                    self.file_changed.set()

    def update_pre(self):
        if self.file_changed.is_set():
            try:
                self._app.load(clear=False)
            finally:
                self.file_changed.clear()
