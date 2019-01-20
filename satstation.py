#!/usr/bin/python3

"""A simple python script template.
"""
import os
import sys
import argparse
import struct
import logging
import skyfield.api as sky
import pandas as pd 
import json 
import numpy as np
import hid
import wave
import sounddevice as sd
import soundfile as sf
import pathlib
import queue
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from scipy.io import wavfile



class FCDProPlus(object):
    # modified from https://github.com/bazuchan/ghpsdr3-fcdproplus-server/blob/master/fcdpp-server.py
    def __init__(self, dev=None, swapiq=None, lna_gain=True, mixer_gain=True, if_gain=0, init_freq=7000000, ppm_offset=0.0):
        self.dev = dev
        if not self.dev:
            self.dev = self.autodetect_dev()
            if not self.dev:
                raise IOError('FCDPro+ device not found')
        self.swapiq = swapiq
        self.ppm_offset = ppm_offset
        self.ver = self.get_fw_ver()
        self.set_lna_gain(lna_gain)
        self.set_mixer_gain(mixer_gain)
        self.set_if_gain(if_gain)
        self.set_freq(init_freq)


    def autodetect_dev(self):
        dongle_dev = None
        devs = hid.enumerate()
        for dev in devs:
            if 'FUNcube Dongle V2.0' in dev['product_string']:
                dongle_dev = dev
                break
        return dongle_dev

    def get_fw_ver(self):
        d = hid.device()
        d.open_path(self.dev['path'])
        d.write([0,1])
        ver = d.read(65)[2:15]
        d.close()
        return ver

    def set_lna_gain(self, lna_gain):
        d = hid.device()
        d.open_path(self.dev['path'])
        d.write([0, 110, int(bool(lna_gain))])
        if d.read(65)[0]!=110:
            raise IOError ('Cant set lna gain')
        d.close()

    def set_mixer_gain(self, mixer_gain):
        d = hid.device()
        d.open_path(self.dev['path'])
        d.write([0, 114, int(bool(mixer_gain))])
        if d.read(65)[0]!=114:
            raise ('Cant set mixer gain')
        d.close()

    def set_if_gain(self, if_gain):
        d = hid.device()
        d.open_path(self.dev['path'])
        d.write([0, 117, if_gain])
        if d.read(65)[0]!=117:
            raise IOError ('Cant set if gain')
        d.close()

    def set_freq(self, freq):
        d = hid.device()
        d.open_path(self.dev['path'])
        corrected_freq = int(np.round(freq + (float(freq)/1000000.0)*float(self.ppm_offset)))
        d.write([0, 101] + list( struct.pack('I', corrected_freq)))
        if d.read(65)[0]!=101:
            raise IOError ('Cant set freq')
        d.close()

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

def callback(indata, frames, time, status):
    q.put(indata.copy())

def record_pass(sdev,pass_df,rec_file,fs, doppler_switch = False):
    logger = logging.getLogger(__name__)
    logger.info('recording pass of {0} starting @ {1}'.format(pass_df.iloc[0]['Satellite'],pass_df.iloc[0]['UTC_time']))
    sd.default.samplerate = fs
    pass_duration = pass_df.iloc[-1]['UTC_time'] - pass_df.iloc[0]['UTC_time']
    duration = pass_duration.seconds
    logger.info('satellite pass duration: {0} seconds'.format(duration))
    logger.info('maximum elevation: {0} degrees'.format(pass_df['Altitude_degrees'].max()))
    logger.info('f0: {0} kHz'.format(1e-3* pass_df.iloc[0]['f0']))
    ts = sky.load.timescale()
    fcd = FCDProPlus()
    fcd.set_freq(pass_df.iloc[0]['f0'] )
    fcd_set_if_gain(True)
    # Make sure the file is opened before recording anything:
    with sf.SoundFile(rec_file, mode='x', samplerate=int(fs), channels=2, subtype='PCM_16') as file:
        with sd.InputStream(samplerate=int(fs), device=0, channels=2, callback=callback):
            for i,r in pass_df.iterrows():
                t = ts.now() 
                while t.utc_datetime() < r['UTC_time']:
                    file.write(q.get())
                    t = ts.now() 
                if doppler_switch:
                    fcd.set_freq(r['freq'] )
                    logger.info('doppler step at {0}: {1} Hz'.format(r['UTC_time'],r['freq']))
    logger.info('finished recording {0}'.format(rec_file))

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
    step_seconds = 10
    T = ts.tt_jd(t.tt + np.array(range(0,round(86400/step_seconds))).astype(np.float) * (step_seconds/86400) )
    last_duration = 0
    last_start_time =  ts.tt_jd(t.tt + 10)

    for satellite in satellites: 
        if satellite['Name'] in TLEs.keys(): 
            if verbose:
                print('looking for {0} passes'.format(satellite['Name']))
            logger.info('looking for {0} passes'.format(satellite['Name']))
            freq = (satellite['Frequency_kHz'] * 1e3)
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
                        if last_start_time  - T[si] > 0:
                            last_duration =  cur_duration
                            last_start_time = T[si]
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
                            cur_df['doppler_shift'] =  (1-(range_rate / c)).astype(np.float)
                            cur_df['f0'] =  np.round(freq)
                            cur_df['freq'] =  np.round( freq * cur_df['doppler_shift'])
                            cur_df['Satellite'] = satellite['Name']
                            break
    return cur_df

def read_config(config_json): 
    if os.path.isfile(config_json): 
        with open(config_json, 'r') as f: 
            config_data = json.load(f) 
    return(config_data) 


q = queue.Queue()

def main():
    logger = logging.getLogger(__name__)
    logger.info('++++++++++++++++++++++++++++++++++++++++')
    logger.info('running pandas v' + pd.__version__)
    #logger.info('running skyfield v' + sky.__version__)
    logger.info('running sounddevice v' + sd.__version__)
    logger.info('++++++++++++++++++++++++++++++++++++++++')


    project_dir = pathlib.Path(__file__).resolve().parents[0]
    logger.info('running in directory {0}'.format(project_dir))

    config_json = read_config('config.json') 
    if 'TLE_dir' in config_json.keys():
        if os.path.isdir(config_json['TLE_dir']):
            print("couldn't find TLE_dir ({0}). Defaulting to {1} ".format(config_json['TLE_dir'],project_dir))
            logger.warning("couldn't find TLE_dir ({0}). Defaulting to {1} ".format(config_json['TLE_dir'],project_dir))
            config_json['TLE_dir'] = str(project_dir)
    else:
        config_json['TLE_dir'] = str(project_dir)


    if 'log_dir' in config_json.keys():
        if os.path.isdir(config_json['log_dir']):
            print("couldn't find log_dir ({0}). Defaulting to {1} ".format(config_json['TLE_dir'],'./log'))
            logger.warning("couldn't find TLE_dir ({0}). Defaulting to {1} ".format(config_json['TLE_dir'],'./log'))
            config_json['log_dir'] = './log'
    else:
        config_json['log_dir'] = './log'


    if 'log_dir' in config_json.keys():
         if os.path.isdir(config_json['log_dir']):
            #logging.basicConfig(level=logging.INFO, format=log_fmt, filename = 'satstation.log')
            pass
         else:
            print("couldn't find log_dir ({0}). Defaulting to {1} ".format(config_json['TLE_dir'],'./log'))
            logger.warning("couldn't find TLE_dir ({0}). Defaulting to {1} ".format(config_json['TLE_dir'],'./log'))
            config_json['log_dir'] = './log'
    else:
        config_json['log_dir'] = './log'


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
            break
    dongle_sdev = None
    for i, sdev in enumerate(sd.query_devices(device=None, kind=None)):
        if 'FUNcube Dongle V2.0' in sdev['name']:
            dongle_sdev = sdev
            dongle_sdev['dev'] = i
            logger.info('found FUNcube Dongle V2.0')
            break
    if not dongle_sdev:
        logger.error('Did not find FUNcube Dongle V2.0 !')

    if config_json and dongle_sdev:
        
        tmp = sd.default.device
        tmp[0] = dongle_sdev['dev']
        sd.default.device = tmp
        
        fcd = FCDProPlus()
        fcd.set_if_gain(True)
        fcd.set_mixer_gain(True)
        fcd.set_freq(137 * 1e6)
        
        ts = sky.load.timescale()
        while 1:
            pass_df = next_pass(config_json,verbose=verbose)
            sys.stderr.write('next pass is of {0} starting at UTC {1} lasting {2} seconds\n'.format(pass_df.iloc[0]['Satellite'],pass_df.iloc[0]['UTC_time'], (pass_df.iloc[-1]['UTC_time'] - pass_df.iloc[0]['UTC_time']).seconds))
            logger.info('next pass is of {0} starting at UTC {1} lasting {2} seconds'.format(pass_df.iloc[0]['Satellite'],pass_df.iloc[0]['UTC_time'], (pass_df.iloc[-1]['UTC_time'] - pass_df.iloc[0]['UTC_time']).seconds))
            filename = pass_df.iloc[0]['Satellite'].split('[')[0].replace(' ','_') +  ts.now().utc_datetime().strftime("%Y%m%d_%H%M%S")
            rec_file = os.path.join(config_json['Recording_dir'],filename + '.wav')
            fig_file = os.path.join(config_json['Recording_dir'],filename + '.png')
            csv_file = os.path.join(config_json['Recording_dir'],filename + '.csv')
            # wait until next pass
            t = ts.now() 
            # wait until defined time
            logger.info('waiting until {0}'.format(pass_df.iloc[0]['UTC_time']))
            while t.utc_datetime() < pass_df.iloc[0]['UTC_time']:
                t = ts.now()

            logger.info('starting recording at {0}'.format(ts.now().utc_datetime()))
            # record pass
           
            record_pass(dongle_sdev,pass_df,rec_file,192e3, doppler_switch= doppler)

             # plot
            plt.figure()
            ax = plt.subplot(122, projection='polar' )
            plt.plot(pass_df['Azimuth_degrees']*np.pi/180, 90-pass_df['Altitude_degrees'],'b-')
            ax.set_ylim(bottom = 0,top = 90)
            ax.set_theta_zero_location("N")
            ax.set_theta_direction(-1)
            ax.set_yticklabels([])
            plt.title(pass_df.iloc[0]['Satellite'])
            ax = plt.subplot(121 )
            ax.plot_date(pass_df['UTC_time'], pass_df['freq'] *1e-6,'b-')
            plt.xticks(rotation='vertical')
            plt.ylabel('Freq [MHz]')
            ax.grid()
            plt.savefic(fig_file)
            # save last recorded pass data to csv
            pass_df.to_csv(csv_file)
            
            
            





#def callback(indata, frames, time, status):    """This is called (from a separate thread) for each audio block."""
#    q.put(indata.copy())



if __name__ == '__main__':

    
    verbose = False
    doppler = True

    parser = argparse.ArgumentParser(description='Process some integers.')
    parser.add_argument('--log_dir', metavar='N', type=str, help='logging directory')
    parser.add_argument('--rec_dir', metavar='N', type=str, help='recording directory')
    parser.add_argument("-v","--verbose", help="increase output verbosity",action="store_true")
    parser.add_argument("-d","--doppler", help="correct for doppler shift")
    
    args = parser.parse_args()

    if args.verbose:
        verbose = True

    
    
    log_fmt = '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    logging.basicConfig(level=logging.INFO, format=log_fmt, filename = 'satstation.log')
   

    sys.exit(main())
