import logging
from threading import Event, Lock
from collections import deque
from time import sleep

import numpy as np

from .types import App, Controller, SystemParameter


logger = logging.getLogger(__name__)


class Microphone(Controller):

    UNIFORM_NAME = 'u_Microphone'

    def __init__(self,
                 app: App,
                 enabled: bool = True,
                 rate: int = 44100,
                 chunk: int = 1024,
                 buffer_size_s: float = 5.0,
                 smooth: float = 0.05,
                 format: int | None = None,
                 intervals: list[int] = [0, 60, 250, 500, 2_000, 6_000, 8_000]):
        super().__init__()

        try:
            import pyaudio
            logger.debug("imported 'pyaudio'")
            if format is None:
                format = pyaudio.paInt16
        except ImportError:
            logger.debug("error importing 'pyaudio'")
            if enabled:
                logger.error(
                    "Microphone input requested, but unable to initialize."
                    " 'pyaudio' not installed, install with 'vhsh[audio]'")
            self.stop()

        self.rate = rate
        self.chunk = chunk
        self.format = format
        self.channels = 1  # TODO configurable?

        self._bins = list(zip(intervals, intervals[1:]))
        self._levels = deque([np.zeros(len(intervals))],  # last interval is open
                             maxlen=int(rate/chunk*smooth))
        self._max_vol = chunk * (2**16-1)  # TODO sizeof(format)

        self._frame_buffer = deque(maxlen=int(rate / chunk * buffer_size_s))
        self._output_lock = Lock()

        if self.UNIFORM_NAME in app.system_parameters:
            raise RuntimeError(
                f"Microphone uniform '{self.UNIFORM_NAME}' already exists")
        num_levels = len(self._levels)
        app.system_parameters[self.UNIFORM_NAME] = SystemParameter(
            self.UNIFORM_NAME,
            type=f"float[{num_levels}]",
            value=(0.) * num_levels,
            update=lambda app: app.controllers[self.__class__.__name__].levels  # type: ignore
        )

        if not enabled:
            logger.debug("microphone disabled, shutting down...")
            self.stop()

    @property
    def levels(self) -> list[float]:
        with self._output_lock:
            level_history = np.array(self._levels)
        max_ = min(self._max_vol, level_history.ravel().max())
        levels = (level_history.mean(axis=0) / max_
                  if max_ != 0
                  else np.zeros(level_history.shape[-1]))
        return levels.tolist()

    def run(self):
        # we need this if `mido` is not installed and class is not initialized
        # due to early return.
        if self._stop_controller.is_set():
            return

        import pyaudio

        def _record(in_data, frame_count, time_info, status):
            self._frame_buffer.appendleft(in_data)
            return (in_data, pyaudio.paContinue)

        audio = pyaudio.PyAudio()
        stream = audio.open(input=True,
                            format=self.format,
                            channels=self.channels,
                            rate=self.rate,
                            frames_per_buffer=self.chunk,
                            stream_callback=_record)

        try:
            while stream.is_active() and not self._stop_controller.is_set():
                if not self._frame_buffer:
                    sleep(0.1)
                    continue

                raw_frame = self._frame_buffer.pop()
                frame = np.frombuffer(raw_frame, dtype=np.uint16)

                spect = np.fft.fft(frame)
                # Scale frequencies to Hz
                freq = np.fft.fftfreq(frame.shape[-1], 1/self.rate)
                # only positive frequencies (first half of the spectrum)
                positive_freqs = freq >= 0
                spectrum = abs(spect[positive_freqs])
                frequencies = freq[positive_freqs]

                levels = [np.sum(spectrum[(frequencies >= min_) & (frequencies < max_)])
                          for min_, max_ in self._bins]
                levels.append(np.sum(spectrum[frequencies >= self._bins[-1][1]]))
                with self._output_lock:
                    self._levels.append(np.array(levels))

        finally:
            stream.stop_stream()
            stream.close()
            audio.terminate()
