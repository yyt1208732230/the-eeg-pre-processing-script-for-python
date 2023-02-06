import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import mne
import copy
from mne.preprocessing import ICA
from mne_icalabel import label_components
from mne.time_frequency import tfr_morlet


# Initial
# File Name
FILENAME = 'Reading_Yueteng'
FILE_PATH = './preprocessedFiles/' + FILENAME + '-tfr.csv'

#  Read Power Data File 
df_power_overall = pd.read_csv(FILE_PATH)


for idx, chal in enumerate(df_power_overall.columns):
    if(idx > 7):
        plt.plot(df_power_overall[1], df_power_overall[idx])
pass
