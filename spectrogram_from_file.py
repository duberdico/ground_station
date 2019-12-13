#!/usr/bin/env python3
import sys
import os
import scipy.signal as sgn
import numpy as np
import xarray as xr
import argparse
import logging
import datetime as dt
import matplotlib.pyplot as plt


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


def make_spectrogram_from_file(filename, fs, fc, nfft, normalize = False):

    da = []
    lt = 0
    with open(filename, "rb") as fid:
        while True:
            data = np.fromfile(
                fid,
                dtype=np.dtype([("i", np.int16), ("q", np.int16)]),
                count=500 * nfft,
            )
            if len(data) == 0:
                break
            cdata = data["i"] + data["q"] * 1j
            if normalize:
                cdata = normalize_complex_arr(cdata)
            Sxx, f, t, im = plt.specgram(
                cdata, NFFT=nfft, Fs=fs, Fc=fc, noverlap=0, mode="psd", scale="linear"
            )

            t_step = t[1] - t[0]
            t = lt + t_step * 0.5 + t
            da.append(
                xr.DataArray(
                    data=10 * np.log10(Sxx),
                    dims=["Frequency", "Time"],
                    coords={"Frequency": f, "Time": t},
                )
            )
            lt = np.max(t)

    da = xr.concat(da, dim="Time")

    return da


def main(args):
    if args.verbose:
        verbose = True

    log_fmt = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    logging.basicConfig(
        level=logging.INFO, format=log_fmt, filename="groundstation.log"
    )

    filename = args.file
    nfft = args.nfft
    normalize = args.normalize
    time, fs, fc = parse_filename(filename)
    da = make_spectrogram_from_file(filename, fs, fc, nfft, normalize)
    da.name = "spectrogram"
    da.attrs = {"nfft": nfft, "fs": fs, "fc": fc, "file": filename, "Timestamp": time.strftime("%Y-%m-%dT%H:%M:%S")}
    da["Frequency"].attrs = {"Units": "Hz"}
    da["Time"].attrs = {"Units": "s"}

    output_filename = filename.replace(".CS16", ".nc")
    da.to_netcdf(output_filename)


if __name__ == "__main__":

    parser = argparse.ArgumentParser(description="Calculate spectrogram")
    parser.add_argument("-l", "--log_file", metavar="N", type=str, help="logging file")
    parser.add_argument("-f", "--file", metavar="N", type=str, help="recording file")
    parser.add_argument("--nfft", metavar="N", type=int, help="nfft", default=2048)
    parser.add_argument("-n", "--normalize", help="normalize", action="store_true", default=False)
    parser.add_argument(
        "-v", "--verbose", help="increase output verbosity", action="store_true"
    )

    args = parser.parse_args()

    sys.exit(main(args))
