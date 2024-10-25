import pyaudio
import numpy as np
from time import sleep
import threading

FORMAT = pyaudio.paInt16
CHANNELS = 1
RATE = 44100
CHUNK = 1024

audio = pyaudio.PyAudio()
stream = audio.open(format=FORMAT, channels=CHANNELS,
                    rate=RATE, input=True,
                    frames_per_buffer=CHUNK)

frames = []
for _ in range(int(RATE / CHUNK * 5)):
    data = np.frombuffer(stream.read(CHUNK), dtype=np.uint16)
    frames.append(data)

    #vol = data.sum() / ( CHUNK * (2**16-1) )
    #print(f"{vol:1.2f} ")# + int(vol*80)*'#' + int((1-vol)*80)*'.', end="\r")

stream.stop_stream()
stream.close()
audio.terminate()


# TODO
# normalize correctly, maybe use float audio input type
# matplotlib interactive
# fixed bins for wide audio range -> rescale input


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

    ax.
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

    print()
