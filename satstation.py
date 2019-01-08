#!/usr/bin/env python3

"""A simple python script template.
"""
import os
import sys
#import argparse
import logging
import skyfield.api as sky
import pandas as pd 
import json 
import numpy as np
import hid
#import soundfile as sf
#import wave
import sounddevice as sd
import pathlib


def read_TLE(TLE_dir):
    logger = logging.getLogger(__name__)
    logger.info('reading TLE data from {0} UTC'.format(TLE_dir))
    if os.path.isdir(TLE_dir): 
        satellites = {} 
        files = os.listdir(TLE_dir) 
        for file in files: 
            if file.endswith(".txt"): 
                satellites.update(sky.load.tle(os.path.join(TLE_dir,file))) 
    return satellites 


def record_pass(sdev,pass_df,rec_file,fs):
    logger = logging.getLogger(__name__)
    logger.info('recording pass of {0} @ {1}'.format(pass_df.iloc[0]['Satellite'],''))
    sd.default.samplerate = fs
    sd.default.device = sdev['name']
    pass_duration = pass_df.iloc[-1]['UTC_time'] - pass_df.iloc[0]['UTC_time']
    duration = pass_duration.seconds
    logger.info('satellite pass duration {0} seconds'.format(duration))
     
    satrec = sd.rec(int(duration * fs), samplerate=fs, channels=2)
    for r in pass_df.iterrows():
        wait_until(r['UTC_time'], True)
        logger.info('doppler step at {0}'.format(r['UTC_time']))
    sd.wait()
    logger.info('recorded {0} samples @ {1} kHz'.format(satrec.shape[0], fs/1e3))
    logger.info('saving samples to {0}'.format(rec_file))
    #fid = wave.open(rec_file, mode='w')
    #fid.setnchannels(2)
    #fid.setsampwidth(2)
    #fid.setframerate(fs)
    #fid.writeframes(satrec)
    logger.info('finished saving')

def next_pass (config_json,verbose = False):
    c = 299792458 # speed of light m/s
    logger = logging.getLogger(__name__)
    TLEs = read_TLE(config_json['TLE_dir']) 
    station = sky.Topos(config_json['Location']['Latitude'], config_json['Location']['Longitude']) 
    satellites = config_json["Satellites"]
    ts = sky.load.timescale() 
    t = ts.now() 
    logger.info('now time is {0} UTC'.format(t.utc_datetime()))
    d = ts.utc(t.utc[0], t.utc[1], t.utc[2]+1) - t
    #T = ts.tt_jd(t.tt + np.array(range(0,int(d * 8640))) * (1/8640))
    T = ts.tt_jd(t.tt + np.array(range(0,8640)) * (1/8640))
    last_duration = 0

    for satellite in satellites: 
        if satellite['Name'] in TLEs.keys(): 
            if verbose:
                print('looking for {0} passes'.format(satellite['Name']))
            logger.info('looking for {0} passes'.format(satellite['Name']))
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

def wait_until(target_utc, verbose=False):
    logger = logging.getLogger(__name__)
    ts = sky.load.timescale()
    t = ts.now() 
    # wait until defined time
    logger.info('waiting until {0}'.format(target_utc))
    if verbose:
        print('waiting until {0}'.format(target_utc))
    while target_utc >t.utc_datetime() :
        t = ts.now()

def read_config(config_json): 
    if os.path.isfile(config_json): 
        with open(config_json, 'r') as f: 
            config_data = json.load(f) 
    return(config_data) 

def main():

    verbose = True

    
    logger = logging.getLogger(__name__)
    
    logger.info('running pandas v' + pd.__version__)
    #logger.info('running skyfield v' + sky.__version__)
    logger.info('running sounddevice v' + sd.__version__)


    project_dir = pathlib.Path(__file__).resolve().parents[0]
    logger.info('running in directory {0}'.format(project_dir))

    config_json = read_config('config.json') 
    if 'TLE_dir' in config_json.keys():
        if os.path.isdir(config_json['TLE_dir']):
            print("couldn't find TLE_dir ({0}). Defaulting to {1} ".format(config_json['TLE_dir'],project_dir))
            logger.warning("couldn't find TLE_dir ({0}). Defaulting to {1} ".format(config_json['TLE_dir'],project_dir))
            config_json['TLE_dir'] = project_dir
    else:
        config_json['TLE_dir'] = project_dir

    #check if recording directory exists and create if necessary:
    #if not os.path.isdir(config_json['Recording_dir']):
    #    os.mkdir(config_json['Recording_dir'])

    #check if recording directory exists and create if necessary:
    if not os.path.isdir('./log'):
        os.mkdir('./log')
    if not os.path.isdir('./report'):
        os.mkdir('./report')
    
    
    # check for FCDPro+
    dongle_dev = None
    devs = hid.enumerate()
    for dev in devs:
        if 'FUNcube Dongle V2.0' in dev['product_string']:
            dongle_dev = dev

    sdevs = sd.query_devices(device=None, kind=None)
    dongle_sdev = None
    for sdev in sdevs:
        if 'FUNcube Dongle V2.0' in sdev['name']:
            dongle_sdev = sdev
            if verbose:
                print('found FUNcube Dongle V2.0')
                logger.info('found FUNcube Dongle V2.0')

    if config_json and dongle_sdev:
        ts = sky.load.timescale()
        while 1:
            pass_df = next_pass(config_json,verbose=verbose)
            print('next pass is of {0} starting at UTC {1} lasting {2} seconds'.format(pass_df.iloc[0]['Satellite'],pass_df.iloc[0]['UTC_time'], (pass_df.iloc[-1]['UTC_time'] - pass_df.iloc[0]['UTC_time']).seconds))
            logger.info('next pass is of {0} starting at UTC {1} lasting {2} seconds'.format(pass_df.iloc[0]['Satellite'],pass_df.iloc[0]['UTC_time'], (pass_df.iloc[-1]['UTC_time'] - pass_df.iloc[0]['UTC_time']).seconds))

            #plt.figure()
            #pylab.polar(pass_df['Azimuth_degrees']*np.pi/180, 90-pass_df['Altitude_degrees'],'b-')
            #ax.set_ylim(bottom = 0,top = 90)
            #ax.set_theta_zero_location("N")
            #ax.set_theta_direction(-1)
            #ax.set_yticklabels([])
            #ax = plt.subplot(121 )
            #ax.plot_date(pass_df['UTC_time'], 100* pass_df['doppler_shift'] ,'b-')
            #ax.grid()
            #plt.title(pass_df.iloc[0]['Satellite'])
            #plt.show()

            # wait until next pass
            wait_until(pass_df.iloc[0]['UTC_time'])
            logger.info('starting recording at {0}'.format(ts.now().utc_datetime()))

            # record pass
            rec_file = os.path.join(config_json['Recording_dir'], 'rec.wav')
            record_pass(dongle_sdev,pass_df,rec_file,96000)
     

if __name__ == '__main__':

    log_fmt = '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    logging.basicConfig(level=logging.INFO, format=log_fmt)
    
    sys.exit(main())