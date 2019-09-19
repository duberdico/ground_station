import os
import sys
import argparse
import struct
import logging
import skyfield.api as sky
import pandas as pd
import json
import numpy as np
import pathlib
import matplotlib
import psutil
import subprocess

from pandas.plotting import register_matplotlib_converters
register_matplotlib_converters()

matplotlib.use("Agg")
import matplotlib.pyplot as plt


def read_TLE(TLE_dir):
    logger = logging.getLogger(__name__)
    logger.info("reading TLE data from {0} UTC".format(TLE_dir))
    satellites = {}
    if os.path.isdir(TLE_dir):
        satellites = {}
        files = os.listdir(TLE_dir)
        for file in files:
            if file.endswith(".txt"):
                satellites.update(sky.load.tle(os.path.join(TLE_dir, file)))
    return satellites


def rx_sdr_cmd(cfg, filename=None, verbose=False):
    """
    Wrapper funtion to call rx_sdr

    Parameters
        ----------
        cfg: dict
            dataframe with the data to saved.
        filename: str
            path to filename to be saved

    rx_sdr usage is copied here for reference:
     -f frequency_to_tune_to [Hz]
    [-s samplerate (default: 2048000 Hz)]
    [-d device key/value query (ex: 0, 1, driver=rtlsdr, driver=hackrf)]
    [-g tuner gain(s) (ex: 20, 40, LNA=40,VGA=20,AMP=0)]
    [-c channel number (ex: 0)]
    [-a antenna (ex: 'Tuner 1 50 ohm')]
    [-p ppm_error (default: 0)]
    [-b output_block_size (default: 16 * 16384)]
    [-n number of samples to read (default: 0, infinite)]
    [-I input format, CU8|CS8|CS12|CS16|CF32 (default: CS16)]
    [-F output format, CU8|CS8|CS12|CS16|CF32 (default: CU8)]
    [-S force sync output (default: async)]
    [-D direct_sampling_mode, 0 (default/off), 1 (I), 2 (Q), 3 (no-mod)]
    [-t SDR settings (ex: rfnotch_ctrl=false,dabnotch_ctrlb=true)]
    filename (a '-' dumps samples to stdout)

    """

    if isinstance(cfg, dict):
        cmd = ["rx_sdr"]

        for key, item in cfg.items():
            if key in ["freq", "f", "frequency"]:
                cmd.extend(("-f", str(int(item))))
            if key in ["sr", "s", "samplerate", "fs"]:
                cmd.extend(("-s", str(int(item))))
            if key in ["dev", "d", "device"]:
                cmd.extend(("-d", str(item)))
            if key in ["gain", "g"]:
                cmd.extend(("-g", str(item)))
            if key in ["nsamples", "n"]:
                cmd.extend(("-n", str(int(item))))
            if key in ["output_format", "F", "output format"]:
                cmd.extend(("-F", item))

        if not filename:
            # no filename given. Create one.
            filename = (
                datetime.now().strftime("%d%m%Y%H%M%S") + "".join(cmd[1:]) + ".rawiq"
            )

        cmd.extend((filename,))
        if verbose:
            print(" ".join(cmd))
        try:
            return_code = subprocess.call(cmd)
        except:
            pass

        return return_code
    else:
        print("error: config not dict")


def next_pass(config_json, verbose=False):
    c = 299792458  # speed of light m/s
    logger = logging.getLogger(__name__)
    TLEs = read_TLE(config_json["TLE_dir"])
    station = sky.Topos(
        config_json["Location"]["Latitude"], config_json["Location"]["Longitude"]
    )
    satellites = config_json["Satellites"]
    ts = sky.load.timescale()
    t = ts.now()
    logger.info("now time is {0} UTC".format(t.utc_datetime()))
    d = ts.utc(t.utc[0], t.utc[1], t.utc[2] + 1) - t
    step_seconds = 10
    T = ts.tt_jd(
        t.tt
        + np.array(range(0, round(86400 / step_seconds))).astype(np.float)
        * (step_seconds / 86400)
    )
    last_duration = 0
    last_start_time = ts.tt_jd(t.tt + 10)
    cur_df = pd.DataFrame()
    for satellite in satellites:
        if satellite["Name"] in TLEs.keys():
            if verbose:
                print("looking for {0} passes".format(satellite["Name"]))
            logger.info("looking for {0} passes".format(satellite["Name"]))
            freq = satellite["Frequency_kHz"] * 1e3
            geocentric = TLEs[satellite["Name"]].at(T)
            subpoint = geocentric.subpoint()
            loc_difference = TLEs[satellite["Name"]] - station
            topocentric = loc_difference.at(T)
            alt, az, distance = topocentric.altaz()
            # separate periods
            j = (alt.degrees >= 0) * 1
            k = j[1:] - j[0:-1]
            s = np.argwhere(k == 1).reshape(-1)
            e = np.argwhere(k == -1).reshape(-1)
            for si in s:
                h = e[e > si].reshape(-1).min()
                if h > 0:
                    if (
                        alt.degrees[si:h] >= config_json["Altitude_threshold_degrees"]
                    ).any():
                        cur_duration = T[h] - T[si]
                        if last_start_time - T[si] > 0:
                            last_duration = cur_duration
                            last_start_time = T[si]
                            cur_df = pd.DataFrame(data=None)
                            delta_t = np.diff(T[si - 1 : h]) * 86400  # seconds
                            cur_df["Azimuth_degrees"] = az.degrees[si:h]
                            cur_df["Distance_km"] = distance.km[si:h]
                            cur_df["Altitude_degrees"] = alt.degrees[si:h]
                            cur_df["Latitude"] = subpoint.latitude.degrees[si:h]
                            cur_df["Longitude"] = subpoint.longitude.degrees[si:h]
                            cur_df["UTC_time"] = T.utc_datetime()[si:h]
                            delta_distance_meter = (
                                np.diff(distance.km[si - 1 : h]) * 1e3
                            )
                            range_rate = delta_distance_meter / delta_t
                            cur_df["doppler_shift"] = (1 - (range_rate / c)).astype(
                                np.float
                            )
                            cur_df["f0"] = np.round(freq)
                            cur_df["freq"] = np.round(freq * cur_df["doppler_shift"])
                            cur_df["Satellite"] = satellite["Name"]
                            break
    return cur_df


def which(program):
    def is_exe(fpath):
        return os.path.isfile(fpath) and os.access(fpath, os.X_OK)

    fpath, fname = os.path.split(program)
    if fpath:
        if is_exe(program):
            return program
    else:
        for path in os.environ["PATH"].split(os.pathsep):
            exe_file = os.path.join(path, program)
            if is_exe(exe_file):
                return exe_file
    return None


def read_config(config_json):
    if os.path.isfile(config_json):
        with open(config_json, "r") as f:
            config_data = json.load(f)
    return config_data


def main():
    logger = logging.getLogger(__name__)
    logger.info("++++++++++++++++++++++++++++++++++++++++")
    logger.info("running pandas v" + pd.__version__)
    logger.info("++++++++++++++++++++++++++++++++++++++++")

    project_dir = pathlib.Path(__file__).resolve().parents[0]
    logger.info("running in directory {0}".format(project_dir))

    # check if necessary command line tools are available
    if not which("rx_sdr"):
        logger.error("Did not find rx_sdr !")
        sys.exit()

    config_json = read_config("config.json")

    if "TLE_dir" in config_json.keys():
        logger.info("looking for TLE_dir ({0})".format(config_json["TLE_dir"]))
        if not os.path.isdir(config_json["TLE_dir"]):
            print(
                "could not find TLE_dir ({0}). Defaulting to {1} ".format(
                    config_json["TLE_dir"], project_dir
                )
            )
            logger.warning(
                "could not find TLE_dir ({0}). Defaulting to {1} ".format(
                    config_json["TLE_dir"], project_dir
                )
            )
            config_json["TLE_dir"] = str(project_dir)
    else:
        logger.info(
            "TLE_dir not found in configuration. Defaulting to {}".format(project_dir)
        )
        config_json["TLE_dir"] = str(project_dir)

    if "log_dir" in config_json.keys():
        if os.path.isdir(config_json["log_dir"]):
            logfilename = os.path.join(config_json["log_dir"], "groundstation.log")
        else:
            print(
                "couldn't find log_dir ({0}). Defaulting to {1} ".format(
                    config_json["log_dir"], "./log"
                )
            )
            logger.warning(
                "couldn't find log_dir ({0}). Defaulting to {1} ".format(
                    config_json["log_dir"], "./log"
                )
            )
            config_json["log_dir"] = "./log"
            logfilename = os.path.join(".", "log", "groundstation.log")
    else:
        config_json["log_dir"] = "./log"
        logfilename = os.path.join(".", "log", "groundstation.log")

    logging.basicConfig(level=logging.INFO, format=log_fmt, filename=logfilename)

    if config_json:
        ts = sky.load.timescale()
        logger.info(str(psutil.disk_usage("/")).replace(", ", ",\n"))
        du = psutil.disk_usage("/")

        while du[3] < 95:  # run while at least 5% of disk space available
            config_json = read_config("config.json")
            logger.info(str(psutil.virtual_memory()).replace(", ", ",\n"))
            pass_df = next_pass(config_json, verbose=verbose)
            sys.stderr.write(
                "next pass is of {0} starting at UTC {1} lasting {2} seconds\n".format(
                    pass_df.iloc[0]["Satellite"],
                    pass_df.iloc[0]["UTC_time"],
                    (
                        pass_df.iloc[-1]["UTC_time"] - pass_df.iloc[0]["UTC_time"]
                    ).seconds,
                )
            )
            logger.info(
                "next pass is of {0} starting at UTC {1} lasting {2} seconds".format(
                    pass_df.iloc[0]["Satellite"],
                    pass_df.iloc[0]["UTC_time"],
                    (
                        pass_df.iloc[-1]["UTC_time"] - pass_df.iloc[0]["UTC_time"]
                    ).seconds,
                )
            )


            # wait until next pass
            t = ts.now().utc_datetime()
            # wait until defined time
            logger.info("waiting until {0} UTC".format(pass_df.iloc[0]["UTC_time"]))
            st = pass_df.iloc[0]["UTC_time"]
            while t  < st:
                t = ts.now().utc_datetime()

            freq = int(pass_df.iloc[0]["f0"])
            duration = (
                pass_df.iloc[-1]["UTC_time"] - pass_df.iloc[0]["UTC_time"]
            ).seconds
            fs = int(config_json["sample_rate"])

            rec_cfg = {
                "freq": freq,
                "gain": 0,
                "fs": fs,
                "F": "CS16",
                "nsamples": duration * fs}


            if "dev_driver" in config_json.keys():
                rec_cfg["dev"] = config_json["dev_driver"]

            if "output_format" not in config_json.keys():
                config_json["output_format"] = "CS16"

            rec_cfg["dev"] = config_json["output_format"]

            filename = pass_df.iloc[0]["Satellite"].split("[")[0].replace(
                " ", "_"
            ) + ts.now().utc_datetime().strftime("%Y%m%d_%H%M%S")

            rec_file = os.path.join(config_json["Recording_dir"], filename + f'_{freq}Hz_.{config_json["dev_driver"]}')
            fig_file = os.path.join(config_json["Recording_dir"], filename + ".png")
            csv_file = os.path.join(config_json["Recording_dir"], filename + ".csv")

            # record pass
            logger.info("starting recording at {0}".format(ts.now().utc_datetime()))
            logger.info(f'recording {duration * fs} samples at {fs} Hz sample rate. Tunning to {freq} Hz')

            rx_sdr_cmd(cfg = rec_cfg, filename=rec_file)

            # TODO: check output file exist and is correct size and log result

            # plot
            plt.figure()
            ax = plt.subplot(122, projection="polar")
            plt.plot(
                pass_df["Azimuth_degrees"] * np.pi / 180,
                90 - pass_df["Altitude_degrees"],
                "b-",
            )
            ax.set_ylim(bottom=0, top=90)
            ax.set_theta_zero_location("N")
            ax.set_theta_direction(-1)
            ax.set_yticklabels([])
            plt.title(pass_df.iloc[0]["Satellite"])
            ax = plt.subplot(121)
            ax.plot_date(pass_df["UTC_time"], pass_df["freq"] * 1e-6, "b-")
            plt.xticks(rotation="vertical")
            plt.ylabel("Freq [MHz]")
            ax.grid()
            plt.savefig(fig_file)
            # save last recorded pass data to csv
            pass_df.to_csv(csv_file)
            del pass_df
            du = psutil.disk_usage("/")


if __name__ == "__main__":

    verbose = False
    doppler = True

    parser = argparse.ArgumentParser(description="Process some integers.")
    parser.add_argument("--log_dir", metavar="N", type=str, help="logging directory")
    parser.add_argument("--rec_dir", metavar="N", type=str, help="recording directory")
    parser.add_argument(
        "-v", "--verbose", help="increase output verbosity", action="store_true"
    )
    parser.add_argument("-d", "--doppler", help="correct for doppler shift")

    args = parser.parse_args()

    if args.verbose:
        verbose = True

    log_fmt = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    logging.basicConfig(
        level=logging.INFO, format=log_fmt, filename="groundstation.log"
    )

    sys.exit(main())
