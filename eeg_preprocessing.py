# Script for EEG signals pre-processing
# 2023 @ Yan Zhang, Laurence Yu

import os
import numpy as np
import matplotlib.pyplot as plt
import mne
import copy
import calendar
import time
from mne.preprocessing import ICA
from mne_icalabel import label_components


# Initial
# File Name
FILENAME = 'Reading_Cao'
# Data Loading
RAW_VISULIZATION = False
# Preprocessing
RAW_PREPROCESSING_VISULIZATION = False
DURATION_PLOT=10
FILTERED_CHANNELS = ['HEO', 'Trigger', 'CZ', 'CB1', 'CB2']
SECONDS_CROP_FROM = 5.0
RANGE_FREQUENCE_HIGHEST = 40
RANGE_FREQUENCE_LOWEST = 0.5
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
    # 'engage', 'disengage', #Reading 第3组
    # 'disengage', 'engage', #Reading 第4组
    # 'engage', 'disengage', #Reading 第5组
]

# Utils
def get_timestamp():
    # gmt stores current gmtime
    localtime = time.localtime()
    # ts stores timestamp
    ts = calendar.timegm(localtime)
    return ts

# --------------------------------------
# 1. Data loading 数据导入
sample_data_raw_file = ('./data/Reading Cao Acq 2023_02_24_1803 Data.edf')
raw = mne.io.read_raw_edf(sample_data_raw_file, preload=True)
channel_count = len(raw.ch_names)

raw = raw.drop_channels(['EKG', 'EMG'])

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


# --------------------------------------
# 2. Data  Preprocessing 数据预处理

# Load functions
def raw_preprocessing(raw_process_temp):
    '''Data preprocessing for raw

    Args:
        raw_process_temp (Raw): The raw data that require slice by annotations

    Returns:
        raw_preprocessed(Raw): The preprocessed Raw object.
    '''
    pass

    # 去掉无用电极
    raw_process_temp = raw_process_temp.drop_channels(FILTERED_CHANNELS)
    if(RAW_PREPROCESSING_VISULIZATION): raw_process_temp.plot(duration=DURATION_PLOT)
    # 留下有效时间段
    raw_process_temp = raw.crop(tmin=SECONDS_CROP_FROM, tmax=None)
    if(RAW_PREPROCESSING_VISULIZATION): raw_process_temp.plot(duration=DURATION_PLOT)
    # 去掉50Hz频段影响
    raw_process_temp = raw_process_temp.filter(RANGE_FREQUENCE_LOWEST,RANGE_FREQUENCE_HIGHEST)
    if(RAW_PREPROCESSING_VISULIZATION): raw_process_temp.plot(duration=DURATION_PLOT)
    # 降采样
    raw_process_temp = raw_process_temp.resample(sfreq=HZ_RESAMPLE)
    if(RAW_PREPROCESSING_VISULIZATION): raw_process_temp.plot(duration=DURATION_PLOT)
    # Re-reference
    raw_process_temp = raw_process_temp.set_eeg_reference(ref_channels = REF_CHANNELS)
    # raw_process_temp = raw_process_temp.set_eeg_reference("average")
    raw_preprocessed = raw_process_temp.copy()
    return raw_preprocessed

def raw_ica(raw_process_temp):
    '''Data ICA processing for raw

    Args:
        raw_process_temp (Raw): The raw data that require slice by annotations

    Returns:
        ica_preprocessed(ICA): The preprocessed ICA object.
        exclude_idx(list): The excluded ica list.
    '''
    pass

    # ICA                                                         
    ica_preprocessed = ICA(
        # n_components=len(raw_process_temp.ch_names)-3,
        n_components=7,
        max_iter="auto",
        method="infomax",
        random_state=97,
        fit_params=dict(extended=True),
    )
    ica_preprocessed.fit(raw_process_temp.copy())
    ic_labels = label_components(raw_process_temp.copy(), ica_preprocessed, method="iclabel")
    labels = ic_labels["labels"]
    exclude_idx = [idx for idx, label in enumerate(labels) if label not in ["brain", "other"]]

    return ica_preprocessed, exclude_idx

# Data Process
raw_preprocessed = raw_preprocessing(raw.copy())
ica_preprocessed, exclude_idx = raw_ica(raw_preprocessed.copy())
raw_ica_preprocessed = ica_preprocessed.apply(raw_preprocessed.copy(), exclude=exclude_idx)

# Plot
raw_preprocessed.plot(duration=10)
ica_preprocessed.apply(raw_preprocessed.copy()).plot(duration=10)
# ica_preprocessed.plot_components(outlines="head")

"""# X. Epoching 将点标注(原标注)，改为定长的段标注"""

#  ------------------------------------
# 3. Epoching + Re-Annotation 将点标注(原标注)，改为定长的段标注

def get_experimental_raw_list_from_annotations(raw_temp, training_prase, experimental_prase):
    '''Get Experimental Raw List / New Annotations from annotations

    Args:
        raw_temp (mne.Raw): The raw data that require slice by annotations
        training_prase (list): The description list (flags) for each training segement
        experimental_prase (list): The description list (flags) for each training experimental

    Returns:
        annotations_experimental: Modified Annotations.
    '''
    pass

    raw_training = None
    raw_experimental = None
    duration = []
    duration_training = []
    duration_experimental = []
    description = []
    onset_training = []
    onset_experimental = []
    fixed_duration = 100.0
    fixed_description = ''
    annotation_count = 0

    onset = raw_temp.annotations.onset
    description = raw_temp.annotations.description
    len_onset = len(onset)
    len_description = len(description)
    for idx, _desc in enumerate(description):
        if(_desc == ANNOTATION_DESCRIPTION_TRAINING_STRAT):
            onset_training.append(onset[idx])
        if(_desc == ANNOTATION_DESCRIPTION_EXPERIMENTAL_STRAT and idx > 4): #Temp 本次demo标注用错的特殊处理
        # if(_desc == ANNOTATION_DESCRIPTION_EXPERIMENTAL_STRAT): #Origin 正常版本
            onset_experimental.append(onset[idx] - 5.0)

    if(len(onset_experimental) != len(experimental_prase)):
        raise Exception("Sth wrong with the length from the onset and description of annotations.")


    duration_experimental.extend([fixed_duration for i in range(len(onset_experimental))])
    annotations_experimental = mne.Annotations(onset_experimental, duration_experimental, experimental_prase)
    # raws_experimental = raw_temp.crop_by_annotations(annotations_experimental, verbose = 'debug')
    # raws_experimental = raw_temp.crop_by_annotations(annotations_experimental)
    # return raws_experimental, annotations_experimental
    return annotations_experimental

# Reset Annotations for a raw file
annotations_experimental = get_experimental_raw_list_from_annotations(raw_ica_preprocessed, training_prase, experimental_prase)

# Reset Annotation
raw_ica_preprocessed.set_annotations(annotations_experimental)

# Save preprocessed raw and ica files
ica_preprocessed.save('./preprocessedFiles/ica_' + FILENAME + str(get_timestamp()) + '-ica.fif')
raw_ica_preprocessed.save('./preprocessedFiles/raw_ica_' + FILENAME + str(get_timestamp()) + '.fif')

pass