# TODO
# normalize correctly, maybe use float audio input type
# running average normalization
# fix crash on shutdown/thread not stopping
# fix frame drops

import pyaudio
import numpy as np
from time import sleep
import threading
from queue import Queue, Full as QueueFull

import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation

FORMAT = pyaudio.paInt16
CHANNELS = 1
RATE = 44100
CHUNK = 1024

def record(audio_frames: Queue, stop: threading.Event, format: int = pyaudio.paInt16, channels: int = 1, rate: int = 44100, chunk: int = 1024):

    def _record(in_data, frame_count, time_info, status):
            try:
                audio_frames.put(in_data, block=False)
            except QueueFull:
                print(f"[{time_info['input_buffer_adc_time']:8f}] {frame_count} frames dropped")

            # If len(data) is less than requested frame_count, PyAudio automatically
            # assumes the stream is finished, and the stream stops.
            return (in_data, pyaudio.paContinue)

    audio = pyaudio.PyAudio()
    stream = audio.open(format=FORMAT, channels=CHANNELS,
                        rate=RATE, input=True,
                        frames_per_buffer=CHUNK,
                        stream_callback=_record)


    try:
        while stream.is_active() and not stop.is_set():
            sleep(0.1)
    finally:
        stream.stop_stream()
        stream.close()
        audio.terminate()


audio_frames = Queue(maxsize=int(RATE / CHUNK * 5))

stop_recording = threading.Event()
recording_thread = threading.Thread(name="record", target=record,
                             args=(audio_frames, stop_recording, FORMAT, CHANNELS, RATE, CHUNK))
recording_thread.start()


plt.ion()  # Enable interactive mode
fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 8))

# Spectrum plot
line, = ax1.plot([], [])
ax1.set_xlabel('Frequency (Hz)')
ax1.set_ylabel('Magnitude')
ax1.set_title('Frequency Spectrum')
ax1.set_xlim(0, 8000)
#ax1.set_xlim(0, RATE/2)  # Nyquist frequency
ax1.grid(True)

# Histogram plot
labels = ['Rumble\n0-60', 'Low End\n60-250', 'Low Mids\n250-500',
          'Mids\n500-2k', 'High Mids\n2k-6k', 'Highs\n6k-8k', 'Air\n8k+']
x_pos = np.arange(len(labels))
bars = ax2.bar(x_pos, np.zeros(len(labels)))
ax2.set_xticks(x_pos)
ax2.set_xticklabels(labels)
ax2.set_ylabel('Magnitude')
ax2.set_title('Frequency Bands')
ax2.set_ylim(0, 1)

plt.tight_layout()


try:
    while True:
        raw_frame = audio_frames.get()
        frame = np.frombuffer(raw_frame, dtype=np.uint16)
        #vol = data.sum() / ( CHUNK * (2**16-1) )
        #print(f"{vol:1.2f} ", int(vol*80)*'#' + int((1-vol)*80)*'.', end="\r")

        sp = np.fft.fft(frame) / ( CHUNK * (2**16-1) )
        freq = np.fft.fftfreq(frame.shape[-1], 1/RATE)  # Scale frequencies to Hz

        # Plot only positive frequencies (first half of the spectrum)
        positive_freq_mask = freq >= 0
        spectrum = abs(sp[positive_freq_mask])
        frequencies = freq[positive_freq_mask]

        # Update spectrum plot
        line.set_data(frequencies, spectrum)
        ax1.set_ylim(0, max(ax1.get_ylim()[1], np.max(spectrum) * 1.1))

        # Calculate and update histogram
        edges = [0, 60, 250, 500, 2_000, 6_000, 8_000]
        levels = [np.sum(spectrum[(frequencies >= min_) & (frequencies < max_)])
                 for min_, max_ in zip(edges, edges[1:])]
        levels.append(np.sum(spectrum[frequencies >= edges[-1]]))
        levels = np.array(levels)

        # Update histogram bars
        for bar, level in zip(bars, levels):
            bar.set_height(level)
        #ax2.set_ylim(0, max(levels) * 1.1)

        fig.canvas.draw()
        fig.canvas.flush_events()

        audio_frames.task_done()
except KeyboardInterrupt:
    plt.close()
    stop_recording.set()
    recording_thread.join()
