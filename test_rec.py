
# coding: utf-8

# In[1]:


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
import psutil
import time
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from multiprocessing import Process
from scipy.io import wavfile


# In[2]:


def callback(indata, frames, time, status):
    q.put(indata.copy())


# In[3]:


print(sd.query_devices())


# In[4]:


q = queue.Queue()


# In[5]:


def record (rec_file,fs,dev):
    with sf.SoundFile(rec_file, mode='x', samplerate=int(fs), channels=2, subtype='PCM_16') as file:
        with sd.InputStream(samplerate=int(fs), dtype='int16', channels=2, callback=callback):
            while 1:
                file.write(q.get())
    


# In[6]:


p = Process(target=record, args=('test_file.wav',96000,0))
p.start()
time.sleep(10)
p.terminate()
q.queue.clear()
print('finished')


