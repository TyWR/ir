import matplotlib.pyplot as plt
import argparse
import os
import numpy as np
from scipy.fftpack import fft
from scipy.signal import blackman
from scipy.io import wavfile


if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument("fname", metavar="file", type=str)
    parser.add_argument("--save", action="store_true")
    parser.add_argument("--stereo", action="store_true")
    parser.add_argument("--n-bits", type=int, default=24)
    args = parser.parse_args()

    fs, data = wavfile.read(args.fname)
    s = data.T[0] if args.stereo else data.T
    rf = 2 ** (args.n_bits + 7)
    ts = s / rf

    N = len(ts)
    T = 1 / fs
    yf = fft(ts)
    xf = np.linspace(0.0, 1.0 / (2.0 * T), N // 2)

    plt.figure(1, figsize=(12, 7))
    plt.title(os.path.basename(args.fname))
    plt.semilogx(xf, 20 * np.log10(np.abs(yf[: N // 2])))
    plt.grid()
    frequencies = [20, 50, 100, 250, 500, 1500, 3000, 5000, 10_000, 20_000]
    plt.xticks(frequencies, [f"{f}Hz" for f in frequencies], rotation=40)
    plt.ylim([-30, 30])
    plt.xlim([20, 2e4])

    if not args.save:
        plt.show()
    else:
        plt.savefig(args.fname.replace(".wav", ".fresponse.png"), dpi="figure")
