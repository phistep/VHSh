from threading import Thread, Event, Lock
from collections import deque
from time import sleep

import numpy as np


class Microphone(Thread):

    NUM_LEVELS = 7

    def __init__(self,
                 rate: int = 44100,
                 chunk: int = 1024,
                 buffer_size_s: float = 5.0,
                 smooth: float = 0.05,
                 format: int | None = None,
                 intervals: list[int] = [0, 60, 250, 500, 2_000, 6_000, 8_000]):
        super().__init__()

        import pyaudio

        self.rate = rate
        self.chunk = chunk
        self.format = pyaudio.paInt16 if format is None else format
        self.channels = 1  # TODO configurable?

        self._bins = list(zip(intervals, intervals[1:]))
        self._levels = deque([np.zeros(len(intervals))],  # last interval is open
                             maxlen=int(rate/chunk*smooth))
        self._max_vol = chunk * (2**16-1)  # TODO sizeof(format)

        self._frame_buffer = deque(maxlen=int(rate / chunk * buffer_size_s))
        self._stop_stream = Event()
        self._output_lock = Lock()

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
            while stream.is_active() and not self._stop_stream.is_set():
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

    def stop(self):
        self._stop_stream.set()
