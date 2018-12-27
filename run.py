#!/usr/bin/python3
import skyfield.api as sf
import os
import pandas as pd
import json



def read_config(config_json):
    if os.path.isfile(config_json):
       with open(config_json, 'r') as f:
           config_data = json.load(f)
    return(config_data)

def read_TLE(TLE_dir):
    if os.path.isdir(TLE_dir):
        satellites = {}
        files = os.listdir(TLE_dir)
        for file in files:
           if file.endswith(".txt"):
              satellites.update(sf.load.tle(os.path.join(TLE_dir,file)))
    return satellites

def main():
    TLE_dir = '/usr/local/etc/TLE'                      # Define the main function
    config_json = read_config('config.json')
    if config_json:
          sf.Topos(config_json['Location']['Latitude'], config_json['Location']['Longitude'])
          TLEs = read_TLE(TLE_dir)
          print(TLEs)
          for satellite in config_json["Satellites"]:
              if satellite in TLEs.keys():
                 # predict next satellite passes for today
                  ts = sf.load.timescale()
                 t = ts.now()
                 #cur_sat = sf.EarthSatellite(TLEs[satellite])
                 print(satellite)
                 print(TLEs[satellite])
                 print(type(TLEs[satellite]))
             
if __name__ == "__main__":
    main()

