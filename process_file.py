#!/usr/bin/env python3

import os
import scipy.signal as sgn
import numpy as np
import xarray as xr
import argparse
import logging


def normalize_complex_arr(a):
    a_oo = a - a.real.min() - 1j * a.imag.min()  # origin offsetted
    return a_oo / np.abs(a_oo).max()


def make_spectrogram(filename, fs, fc, nfft):

    data = np.memmap(
        filename, dtype=np.dtype([("i", np.int16), ("q", np.int16)]), mode="r"
    )
    cdata = data["i"] + data["q"] * 1j
    cdata = normalize_complex_arr(cdata)

    f, t, Sxx = sgn.spectrogram(
        cdata, fs=fs, noverlap=None, nfft=nfft, return_onesided=False
    )
    da = xr.DataArray(
        data=Sxx,
        dims=["Frequency", "Time"],
        coords={"Frequency": f, "Time": t}
    )
    da.attrs = {"nfft": nfft, "fs": fs, "fc": fc}
    da["Frequency"].attrs = {"Units": "Hz"}
    da["Time"].attrs = {"Units": "s"}


if __name__ == "__main__":

    parser = argparse.ArgumentParser(description="Process some integers.")
    parser.add_argument("-l", "--log_file", metavar="N", type=str, help="logging file")
    parser.add_argument("-f", "--file", metavar="N", type=str, help="recording file")
    parser.add_argument(
        "-v", "--verbose", help="increase output verbosity", action="store_true"
    )

    args = parser.parse_args()

    if args.verbose:
        verbose = True

    log_fmt = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    logging.basicConfig(
        level=logging.INFO, format=log_fmt, filename="groundstation.log"
    )

    filename = args.file
    file_params = par
    make_spectrogram(filename, fs, fc, nfft)

    sys.exit(main())
