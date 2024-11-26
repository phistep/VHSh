import pyaudio
import numpy as np
from time import sleep
import threading
from queue import Queue, Full as QueueFull

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


import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation

plt.ion()  # Enable interactive mode
fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 8))

# Spectrum plot
line, = ax1.plot([], [])
ax1.set_xlabel('Frequency (Hz)')
ax1.set_ylabel('Magnitude')
ax1.set_title('Frequency Spectrum')
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
        ax1.set_xlim(0, RATE/2)  # Nyquist frequency
        ax1.set_ylim(0, max(ax1.get_ylim()[1], np.max(spectrum) * 1.1))

        # Calculate and update histogram
        edges = [0, 60, 250, 500, 2_000, 6_000, 8_000]
        levels = [np.sum(spectrum[(frequencies >= min_) & (frequencies < max_)])
                 for min_, max_ in zip(edges, edges[1:])]
        levels.append(np.sum(spectrum[frequencies >= edges[-1]]))
        levels = np.array(levels) / (CHUNK * (2**16-1))

        # Update histogram bars
        for bar, level in zip(bars, levels):
            bar.set_height(level)
        ax2.set_ylim(0, max(levels) * 1.1)

        fig.canvas.draw()
        fig.canvas.flush_events()

        audio_frames.task_done()
except KeyboardInterrupt:
    stop_recording.set()
    recording_thread.join()



# TODO
# normalize correctly, maybe use float audio input type
# matplotlib interactive
# fixed bins for wide audio range -> rescale input

# RUMBLE – 0-60HZ: The frequencies in this range represent sound that is below human hearing, or are perceived as low-end rumble. You typically would not boost these frequencies, but you can cut them with a high pass filter if you need to reduce low-end rumble on any given track.
# LOW END – 60-250HZ: These frequencies are primarily where your kick drum, bass, and the low end of instruments will live. If a track needs more thump, you can boost this range. If you need to reduce muddiness, cut out some of this range.
# LOW MIDS – 250-500HZ: Low mids provide warmth and body to instruments and vocals. Boosting in this range can add thickness, while cutting can reduce boxiness or muddiness.
# MIDS – 500-2KHZ: The midrange is where most of the musical content resides. Vocals, guitars, and many other instruments are prominent here. Boost for clarity and presence, cut to reduce harshness or to make room for other elements.
# HIGH MIDS – 2-6KHZ: High mids add definition and attack to instruments and vocals. Boost to make elements cut through the mix, cut to reduce sharpness or to create space for other sounds.
# HIGHS – 6-8KHZ: The high frequencies provide brilliance and sparkle. Boosting in this range can add airiness and detail to vocals and instruments. Cutting can reduce sibilance or excessive brightness.
# AIR – 8KHZ AND ABOVE: Air frequencies are the highest range of audible frequencies. Boosting here can add a sense of openness and sheen to your mix, especially on vocals and cymbals. Be subtle with boosts in this range to avoid harshness.

"""

import matplotlib.pyplot as plt
for frame in frames[-1:]:
    sp = np.fft.fft(frame)
    freq = np.fft.fftfreq(frame.shape[-1])

    print(f"{sp.shape=}")
    # Reduce sp to 10 frequency bins
    num_bins = 10
    bin_size = len(sp) // num_bins
    sp_reduced = np.array([np.mean(abs(sp[i:i+bin_size])) for i in range(0, len(sp), bin_size)])
    sp_reduced = sp_reduced[:num_bins]  # Ensure we have exactly 10 bins
    freq_reduced = np.linspace(freq.min(), freq.max(), num_bins)

    plt.figure()
    plt.plot(freq, sp.real, label="real")
    plt.plot(freq, sp.imag, label="imag")
    plt.plot(freq, abs(sp), label="abs")
    plt.legend()

    plt.figure()
    plt.hist(abs(sp))

    plt.figure()
    print(f"{sp_reduced/(8**2 *bin_size)}")
    print(freq_reduced)
    plt.plot(freq_reduced/(8**2 *bin_size), sp_reduced)

    # Create a bar chart that can be updated
    fig, ax = plt.subplots()
    bars = ax.bar(freq_reduced, sp_reduced)
    ax.set_xticks(range(num_bins))
    ax.set_xticklabels([f'{f:.2f}' for f in freq_reduced])
    ax.set_xlabel('Frequency')
    ax.set_ylabel('Magnitude')
    ax.set_title('Frequency Spectrum')
    plt.tight_layout()

    bars.datavalues

    def update_bars(new_data):
        for bar, height in zip(bars, new_data):
            bar.set_height(height)
        fig.canvas.draw()


    def update():
        while True:
            for bar in bars:
                bar.set_height(bar.get_height()*1.1)
                print(bar, bar.get_height())
                sleep(0.5)
            fig.canvas.draw()
    th = threading.Thread(name="update", target=update)
    th.start()

    plt.show()
    while True:
        plt.clf()
        fig.canvas.draw()
        sleep(0.01)

    print()
"""
