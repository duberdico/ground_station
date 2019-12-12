#!/usr/bin/env python3

import os
import scipy.signal as sgn
import numpy as np
import xarray as xr
import argparse
import logging
import datetime as dt


def parse_filename(fname):
    fname = os.path.split(fname)[-1].replace(".CS16", "")
    parts = fname.split("_")
    for i, s in enumerate(parts):
        if (len(s) == 8) and (len(parts[i + 1]) == 6):
            time = dt.datetime.strptime(s + parts[i + 1], "%Y%m%d%H%M%S")
        elif "fs" in s:
            fs = int(s.replace("fs", ""))
        elif "Hz" in s:
            f0 = int(s.replace("Hz", ""))

    return (time, fs, f0)


def normalize_complex_arr(a):
    a_oo = a - a.real.min() - 1j * a.imag.min()  # origin offsetted
    return a_oo / np.abs(a_oo).max()


def make_spectrogram(filename, fs, fc, nfft):

    xa = []
    with open(filename, "rb") as fid:
        while True:
            data = np.fromfile(
                fid,
                dtype=np.dtype([("i", np.int16), ("q", np.int16)]),
                count=100 * nfft,
            )
            if len(data) == 0:
                break
            cdata = data["i"] + data["q"] * 1j
            cdata = normalize_complex_arr(cdata)
            Sxx, f, t, im = plt.specgram(cdata, NFFT=nfft, Fs=fs, Fc=fc, noverlap=0)
            t_step = t[1] - t[0]
            da.append(
                xr.DataArray(
                    data=Sxx,
                    dims=["Frequency", "Time"],
                    coords={"Frequency": f, "Time": lt + t_step * 0.5 + t},
                )
            )
            t_step = t[1] - t[0]
            lt = t[-1]

    da = xr.concat(da, dim="Time")
    da.attrs = {"nfft": nfft, "fs": fs, "fc": fc}
    da["Frequency"].attrs = {"Units": "Hz"}
    da["Time"].attrs = {"Units": "s"}

    return da


if __name__ == "__main__":

    parser = argparse.ArgumentParser(description="Process some integers.")
    parser.add_argument("-l", "--log_file", metavar="N", type=str, help="logging file")
    parser.add_argument("-f", "--file", metavar="N", type=str, help="recording file")
    parser.add_argument("--nfft", metavar="N", type=int, help="nfft", default=2048)
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
    nfft = args.nfft
    time, fs, fc = parse_filename(filename)
    da = make_spectrogram(filename, fs, fc, nfft)

    sys.exit(main())
