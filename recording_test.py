#!/usr/bin/python3

"""A simple python script template.
"""
import os
import sys
import argparse
import subprocess



def rx_sdr_cmd(freq = 137000000, fs = 2000000, duration = 10, rec_file = 'test_file.iq'):

    if rec_file == "":
        cur_time_str = time.strftime("%Y%m%d%H%M%S", time.localtime())
        rec_file = f"{cur_time_str}_{freq}_{fs}_{duration}.raw"
    command_str = ['rx_sdr',
                   '-f', str(int(freq)),
                   '-g', '50',
                         '-s', str(int(fs)),
                   '-n', str(int(duration * fs)),
                   rec_file
                   ]
    sdr_output = subprocess.run(command_str)
    return sdr_output


def main():
    print(rx_sdr_cmd())

            
if __name__ == '__main__':

    sys.exit(main())
