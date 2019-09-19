# ground_station
Python code to run an automatic satellite ground station using software defined radio.
Used to predict next pass of one in a list of satellites specified in config.json, wait until time of pass and initiate raw IQ samples recording using a call to rx_sdr for the duration of the pass. Recording parameters (e.g. tunned frequency, gain, etc.) are also specified in the config file. 

Not meant to be a general tool but something that it is useful for my setup. 

Initially meant to be ran on a Raspberry Pi 3 but currently running on a Fujitsu RX200 S4 server (2x Quad Core Xenon, 32Gb RAM, 2x 146GB SAS) in my loft. SDR hardware connected is one of what I currently have available - SDRPlay RSPduo, Nooelec NESDR SMArTee v2 SDR or FUNcube Dongle Pro+. RF hardware currently in use, targetting 137 MHz APT transmissions is 137.5MHz SAW Filtered Preamp and 137 MHz RHCP turnstile antenna. 




