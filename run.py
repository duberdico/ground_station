#!/usr/bin/env python3

"""A simple python script template.
"""
import os
import sys
import argparse
import logging
import skyfield.api as sf 
import os 
import pandas as pd 
import json 
import numpy as np
import matplotlib.pyplot as plt
import hid
import wave
import sounddevice as sd


def load_TLEs (TLE_dir):
    if os.path.isdir(TLE_dir):
        for file in os.listdir(TLE_dir):
            if file.endswith(".txt"):
                print(os.path.join(TLE_dir, file))
    else:
        pass

def record_pass(pass_df,rec_file,sox_found):
    print('recording pass')
    pass_duration = pass_df.iloc[-1]['UTC_time'] - pass_df.iloc[0]['UTC_time']
    if sox_found:
        print(pass_duration.seconds)
        s = ['sox', '-b 16', '-r 192k',rec_file,'trim 0', str(pass_duration.seconds)]
        p = subprocess.Popen(s)
        
        print(p.pid)
        print(p.returncode)
        print(p.errors)
        print(p.poll())
    for row in pass_df.iterrows():
        pass

def next_pass (config_json):
    c = 299792458
    TLEs = read_TLE(TLE_dir) 
    station = sf.Topos(config_json['Location']['Latitude'], config_json['Location']['Longitude']) 
    satellites = config_json["Satellites"]
    ts = sf.load.timescale() 
    t = ts.now() 
    print('now time is {0} UTC'.format(t.utc_datetime()))
    d = ts.utc(t.utc[0], t.utc[1], t.utc[2]+1) - t
    T = ts.tt_jd(t.tt + np.array(range(0,8640)) * (1/8640))
    last_duration = 0
    for satellite in satellites: 
        if satellite['Name'] in TLEs.keys(): 
            geocentric = TLEs[satellite['Name']].at(T)
            subpoint = geocentric.subpoint()
            loc_difference = TLEs[satellite['Name']] - station
            topocentric = loc_difference.at(T)
            alt, az, distance = topocentric.altaz()
            
            # separate periods
            j = (alt.degrees >= 0) * 1
            k = j[1:] - j[0:-1]
            s = np.argwhere(k == 1).reshape(-1)
            e = np.argwhere(k == -1).reshape(-1)
            for si in s:
                h = e[e>si].reshape(-1).min()
                if h > 0:
                    if (alt.degrees[si:h] >= config_json["Altitude_threshold_degrees"]).any():
                        cur_duration = T[h] - T[si]
                        if cur_duration > last_duration:
                            last_duration =  cur_duration
                            cur_df = pd.DataFrame(data=None)
                            delta_t = np.diff(T[si-1:h]) * 86400 # seconds
                            cur_df['Azimuth_degrees'] = az.degrees[si:h]
                            cur_df['Distance_km'] = distance.km[si:h]
                            cur_df['Altitude_degrees'] = alt.degrees[si:h]
                            cur_df['Latitude'] = subpoint.latitude.degrees[si:h]
                            cur_df['Longitude'] = subpoint.longitude.degrees[si:h]
                            cur_df['UTC_time'] = T.utc_datetime()[si:h]
                            delta_distance_meter = np.diff(distance.km[si-1:h]) * 1e3
                            range_rate = delta_distance_meter / delta_t
                            cur_df['doppler_shift'] =  (1-(range_rate / c)) 
                            cur_df['Satellite'] = satellite['Name']
                            break
    return cur_df


def main(arguments):

    # not used in this stub but often useful for finding various files
    file_dir = Path(__file__).resolve().parents[2]

    print(file_dir)

    #TLEdir = '\\Users\\ricardo.antunes\\Source\\Repos\\sat_station'

    #log_fmt = '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    #logging.basicConfig(level=logging.INFO, format=log_fmt)

    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('infile', help="Input file", type=argparse.FileType('r'))
    parser.add_argument('-d', '--TLEdir', help="TLE directory",
                        default=sys.stdout, type=argparse.FileType('w'))

    args = parser.parse_args(arguments)

    # check for FCDPro+
    #dongle_dev = []
    #devs = hid.enumerate()
    #for dev in devs:
    #    if 'FUNcube Dongle V2.0' in dev['product_string']:
    #        dongle_dev = dev

    #sdevs = sd.query_devices(device=None, kind=None)
    #for sdev in sdevs:
    #    if 'FUNcube Dongle V2.0' in sdev['name']:
    #        dongle_sdev = sdev


if __name__ == '__main__':
    
    sys.exit(main(sys.argv[1:]))