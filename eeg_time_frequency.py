# Script for EEG signals pre-processing
# 2023 @ Yan Zhang, Laurence Yu

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
# Data Loading
RAW_VISULIZATION = False
# Preprocessing
RAW_PREPROCESSING_VISULIZATION = False
DURATION_PLOT=10
FILTERED_CHANNELS = ['HEO', 'Trigger', 'CZ', 'CB1', 'CB2']
SECONDS_CROP_FROM = 5.0
RANGE_FREQUENCE_HIGHEST = 40
RANGE_FREQUENCE_LOWEST = 0
HZ_RESAMPLE = 200
REF_CHANNELS=['M1', 'M2']
# ICA
ICA_ACCECPTED = ["brain", "other"]

# Split raw data
ANNOTATION_DESCRIPTION_TRAINING_STRAT = '1'
ANNOTATION_DESCRIPTION_EXPERIMENTAL_STRAT = '2'

# Experimental Configuration
training_prase = [
    'engage', 'disengage', #Training 第1组
]
experimental_prase = [
    'engage', 'disengage', #Reading 第1组
    'disengage', 'engage', #Reading 第2组
    'engage', 'disengage', #Reading 第3组
    'disengage', 'engage', #Reading 第4组
    'engage', 'disengage', #Reading 第5组
]

# Output File Name
path_power_engaged_alpha = './preprocessedFiles/' + FILENAME + '_engaged_alpha-tfr.csv'
path_power_engaged_beta = './preprocessedFiles/' + FILENAME + '_engaged_beta-tfr.csv'
path_power_engaged_theta = './preprocessedFiles/' + FILENAME + '_engaged_theta-tfr.csv'
path_power_disengaged_alpha = './preprocessedFiles/' + FILENAME + '_disengaged_alpha-tfr.csv'
path_power_disengaged_beta = './preprocessedFiles/' + FILENAME + '_disengaged_beta-tfr.csv'
path_power_disengaged_theta = './preprocessedFiles/' + FILENAME + '_disengaged_theta-tfr.csv'

path_power_engaged = './preprocessedFiles/' + FILENAME + '_engaged_-tfr.csv'
path_power_disengaged = './preprocessedFiles/' + FILENAME + '_engaged_-tfr.csv'
path_power_tf_overall = './preprocessedFiles/' + FILENAME + '-tfr.csv'

#α，β，θ Frequency Configuration
freqs_alpha = np.arange(8, 13, 0.5)
freqs_beta = np.arange(13, 30, 0.5)
freqs_theta = np.arange(4, 8, 0.5)

# --------------------------------------
# 1. Data loading 数据导入
sample_data_raw_file = ('./data/yuedurenwu01-12 Data 202301291643.edf')
raw = mne.io.read_raw_edf(sample_data_raw_file, preload=True)
channel_count = len(raw.ch_names)

# 导入电极位置配置文件
locs_info_path = ('./data/64_ch_montage.loc')
chan_types_dict = {"HEO":"eog", "VEO":"eog"}
montage = mne.channels.read_custom_montage(locs_info_path)
raw.set_montage(montage)
raw.set_channel_types(chan_types_dict)

if(RAW_VISULIZATION is True):
    # 可视化Raw data
    raw.plot_psd(fmax=50) #只绘制低于50赫兹的频率
    raw.plot(duration=5, n_channels=channel_count)

# Load preprocessed Raw file
path_raw_ica_preprocessed = ('./preprocessedFiles/raw_ica_Reading_Yueteng1675365526.fif')
raw_ica_preprocessed = mne.io.read_raw(path_raw_ica_preprocessed, preload=True)
# raw_ica_preprocessed.drop_channels(FILTERED_CHANNELS)

# Reset Events for Raw
events,events_id = mne.events_from_annotations(raw_ica_preprocessed)
raw_ica_preprocessed.add_events(events, stim_channel=None, replace=True)
events = mne.find_events(raw_ica_preprocessed)

# Epoch Raw with Events
epochs = mne.Epochs(raw_ica_preprocessed.drop_channels(FILTERED_CHANNELS), events, events_id, tmin=-0.5, tmax=50.0)
epochs.apply_baseline((None,0))

target_engage = epochs['engage'].average()
target_disengage = epochs['disengage'].average()

# epochs['engage'].plot()
# epochs['disengage'].plot()

target_engage.plot_joint();
target_disengage.plot_joint();

# epochs.plot_joint()

pass
alpha = None
beta = None
theta = None
# # Engaged
# power_engaged_alpha=tfr_morlet(epochs['engage'],freqs_alpha,2,return_itc=False,average=False)
# power_engaged_beta=tfr_morlet(epochs['engage'],freqs_beta,2,return_itc=False,average=False)
# power_engaged_theta=tfr_morlet(epochs['engage'],freqs_theta,1,return_itc=False,average=False)

# # CSV File Save
# alpha = power_engaged_alpha.average(method='mean', dim='freqs', copy=False).to_data_frame()
# beta = power_engaged_beta.average(method='mean', dim='freqs', copy=False).to_data_frame()
# theta = power_engaged_theta.average(method='mean', dim='freqs', copy=False).to_data_frame()

# alpha.to_csv(path_power_engaged_alpha)
# beta.to_csv(path_power_engaged_beta)
# theta.to_csv(path_power_engaged_theta)

# Disengaged
# # Engaged
# power_disengaged_alpha=tfr_morlet(epochs['disengage'],freqs_alpha,2,return_itc=False)
# power_disengaged_beta=tfr_morlet(epochs['disengage'],freqs_beta,2,return_itc=False)
# power_disengaged_theta=tfr_morlet(epochs['disengage'],freqs_theta,1,return_itc=False)

# # CSV File Save
# power_disengaged_alpha.to_data_frame().to_csv(path_power_disengaged_alpha)
# power_disengaged_beta.to_data_frame().to_csv(path_power_disengaged_beta)
# power_disengaged_theta.to_data_frame().to_csv(path_power_disengaged_theta)

# Time Frequency Overall
power_alpha=tfr_morlet(epochs,freqs_alpha,2,return_itc=False,average=False)
power_beta=tfr_morlet(epochs,freqs_beta,2,return_itc=False,average=False)
power_theta=tfr_morlet(epochs,freqs_theta,1,return_itc=False,average=False)

# CSV File Save
alpha = power_alpha.average(method='mean', dim='freqs', copy=False).to_data_frame()
beta = power_beta.average(method='mean', dim='freqs', copy=False).to_data_frame()
theta = power_theta.average(method='mean', dim='freqs', copy=False).to_data_frame()

engaged_cal = beta.loc[:, ~beta.columns.isin(['condition', 'time', 'epoch', 'freq'])] / (alpha.loc[:, ~alpha.columns.isin(['condition', 'time', 'epoch', 'freq'])] + theta.loc[:, ~theta.columns.isin(['condition', 'time', 'epoch', 'freq'])])
engaged_cal = pd.concat([alpha[['condition', 'time', 'epoch']], engaged_cal], axis=1, join="inner")
engaged_cal.to_csv(path_power_tf_overall)

pass