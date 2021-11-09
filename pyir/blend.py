from typing import Optional
import matplotlib.pyplot as plt
import argparse
import numpy as np
import os
from scipy.io import wavfile
from scipy import signal

"""
blend
----------
Compute a dry/wet impulse response based on a given .wav impulse response wav
file
"""


def blend_signal(
    s: np.array,
    fs: float,
    n_bits: int,
    ratio: int = 50,
    high_pass: Optional[int] = None,
    low_pass: Optional[int] = None,
    filter_order: int = 2,
):
    # Rescale wet signal with adequat number of bits
    rf = 2 ** (args.n_bits + 7)
    wet = 1 / rf * s

    # Init dry signal with a dirac
    dry = np.zeros_like(wet)
    dry[0] = 1

    if high_pass:
        sos_hp = signal.butter(
            filter_order, high_pass, "hp", fs=fs, output="sos"
        )
        wet = signal.sosfilt(sos_hp, wet)

    if low_pass:
        sos_lp = signal.butter(
            filter_order, low_pass, "lp", fs=fs, output="sos"
        )
        dry = signal.sosfilt(sos_lp, dry)

    r = ratio / 100
    blended = wet * r + (1 - r) * dry
    blended_wav = rf * blended
    return blended_wav


if __name__ == "__main__":

    parser = argparse.ArgumentParser(
        description="""
        Compute a dry/wet impulse response based on a given .wav impulse
        response wav file
        """
    )
    parser.add_argument(
        "input", metavar="input", type=str, help="Input .wav file"
    )
    parser.add_argument(
        "--ratio", type=int, default=50, help="Wet/Dry ratio in %%"
    )
    parser.add_argument(
        "--high-pass",
        type=int,
        default=0,
        help="Frequency on which to apply high-pass filtering on wet signal",
    )
    parser.add_argument(
        "--low-pass",
        type=int,
        default=0,
        help="Frequency on which to apply low-pass filtering on dry signal",
    )
    parser.add_argument(
        "--order",
        type=int,
        default=2,
        help="Order of high/low-pass filters (default 2)",
    )
    parser.add_argument(
        "--n-bits", type=int, default=24, help="Number of bits of the signal"
    )
    parser.add_argument(
        "--stereo",
        action="store_true",
        help="Flag if you want to deal with stereo value",
    )
    parser.add_argument(
        "-o", "--output", type=str, default="output", help="Output directory"
    )
    args = parser.parse_args()

    os.makedirs(args.output, exist_ok=True)

    fs, data = wavfile.read(args.input)
    s = data.T[0] if args.stereo else data.T
    blend = blend_signal(
        s,
        fs,
        args.n_bits,
        low_pass=args.low_pass,
        high_pass=args.high_pass,
        filter_order=args.order,
    )

    input_name = os.path.basename(args.input).replace(".wav", "")
    name = f"{input_name}-{args.ratio}"
    if args.high_pass:
        name = name + f"-HP{args.high_pass}"
    if args.low_pass:
        name = name + f"-LP{args.low_pass}"
    if args.high_pass or args.low_pass:
        name = name + f"-ORDER{args.order}"

    wavfile.write(os.path.join(args.output, name) + ".wav", fs, blend)
