import logging
from threading import Thread, Event, Lock
from pathlib import Path


logger = logging.getLogger(__name__)


class FileWatcher(Thread):

    def __init__(self, filenames: list[Path], file_changed: Event = Event()):
        super().__init__()
        self._file_watcher_stop = Event()
        self.file_changed = file_changed
        self.filenames = filenames
        self.current = Path(filenames[0])

    def run(self):
        from watchfiles import watch

        logger.info(f"Watching for changes in %s...", self.filenames)

        for changes in watch(*self.filenames, stop_event=self._file_watcher_stop):
            for _, filename in changes:
                if Path(filename).absolute() == self.current.absolute():
                    logger.debug("'%s' changed!", filename)
                    self.file_changed.set()

    def stop(self):
        self._file_watcher_stop.set()
