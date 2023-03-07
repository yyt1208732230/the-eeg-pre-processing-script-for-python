# Script for EEG signals pre-processing
# 2023 @ Yan Zhang, Laurence Yu

import os
import ast
import numpy as np
import matplotlib.pyplot as plt
import mne
import copy
import calendar
import time
import argparse
import json
import logging
from mne.preprocessing import ICA
from mne_icalabel import label_components
from dotenv import load_dotenv
from envload import *
from util import *

# Initial

# logging.basicConfig(filename=LOGS_PATH,
#                     filemode='a',
#                     format='%(asctime)s,%(msecs)d %(name)s %(levelname)s %(message)s',
#                     datefmt='%H:%M:%S',
#                     level=logging.INFO)


# --------------------------------------
# 1. Data loading 数据导入
def data_loading(raw_file_path, custom_montage_path, raw_visulization_path='./'):
    '''Load raw curry raw file as raw data and customize montage settings with mne

    Args:
        raw_file_path (str): The raw data path.
        custom_montage_path (str): The montage file path.
        raw_visulization_path (str): The path saving the raw figures if the env config is True.

    Returns:
        raw(Raw): The Raw object.
        montage(montage): The montage object.
    '''
    pass
    succeed = False
    # sample_data_raw_file = ('./data/yuedurenwu01-12 Data 202301291643.edf')
    sample_data_raw_file = (raw_file_path)
    raw = mne.io.read_raw_edf(sample_data_raw_file, preload=True)
    # raw = mne.io.read_raw_curry(sample_data_raw_file, preload=True)
    channel_count = len(raw.ch_names)
    logging.info('Raw Loaded:' + raw_file_path)

    # 导入电极位置配置文件
    # locs_info_path = ('./data/64_ch_montage.loc')
    locs_info_path = (custom_montage_path)
    chan_types_dict = {"HEO":"eog", "VEO":"eog"}
    montage = mne.channels.read_custom_montage(locs_info_path)
    raw.drop_channels(['EKG', 'EMG']) #temp2023/02/26
    raw.set_montage(montage)
    raw.set_channel_types(chan_types_dict)
    logging.info('Montage Loaded:' + custom_montage_path)

    if(RAW_VISULIZATION is True):
        # 可视化Raw data
        raw.plot_psd(fmax=50, show=False).savefig(raw_visulization_path + '-raw_psd.png') #只绘制低于50赫兹的频率
        # raw.plot(duration=50, n_channels=channel_count).savefig(raw_visulization_path + '-raw.png')
        raw.plot(n_channels=channel_count, show=False).savefig(raw_visulization_path + '-raw.png')
        logging.info('Raw figures visualized')

    return raw, montage

# --------------------------------------
# 2. Data  Preprocessing 数据预处理

# Load functions
def raw_preprocessing(raw, subjectName, raw_visulization_path='./'):
    '''Data preprocessing for raw data

    Args:
        raw (Raw): The raw data that require slice by annotations.

    Returns:
        raw_preprocessed(Raw): The preprocessed Raw object.
    '''
    pass

    raw_process_temp = None
    raw_process_temp = raw.copy()

    # 去掉无用电极
    raw_process_temp = raw_process_temp.drop_channels(FILTERED_CHANNELS)
    logging.info('Channels dropped: '+ subjectName)
    if(RAW_PREPROCESSING_VISULIZATION): 
        raw_process_temp.plot(duration=DURATION_PLOT, show=False).savefig(raw_visulization_path + '-raw-drop_channels.png')
    # 留下有效时间段
    raw_process_temp = raw_process_temp.crop(tmin=SECONDS_CROP_FROM, tmax=None)
    logging.info('Time cropped: '+ subjectName)
    if(RAW_PREPROCESSING_VISULIZATION): 
        raw_process_temp.plot(duration=DURATION_PLOT, show=False).savefig(raw_visulization_path + '-raw-crop.png')
    # 去掉50Hz频段影响
    raw_process_temp = raw_process_temp.filter(RANGE_FREQUENCE_LOWEST,RANGE_FREQUENCE_HIGHEST)
    logging.info('Frequency filtered: '+ subjectName)
    if(RAW_PREPROCESSING_VISULIZATION): 
        raw_process_temp.plot(duration=DURATION_PLOT, show=False).savefig(raw_visulization_path + '-raw-filter.png')
    # 降采样
    raw_process_temp = raw_process_temp.resample(sfreq=HZ_RESAMPLE)
    logging.info('Data resampled: '+ subjectName)
    if(RAW_PREPROCESSING_VISULIZATION): 
        raw_process_temp.plot(duration=DURATION_PLOT, show=False).savefig(raw_visulization_path + '-raw-resample.png')
    # Re-reference
    raw_process_temp = raw_process_temp.set_eeg_reference(ref_channels = REF_CHANNELS)
    logging.info('Raw referenced: '+ subjectName)
    # raw_process_temp = raw_process_temp.set_eeg_reference("average")
    raw_preprocessed = raw_process_temp.copy()

    return raw_preprocessed

# ICA  
def raw_ica(raw, raw_visulization_path='./'):
    '''Data ICA processing for raw

    Args:
        raw (Raw): The raw data that require slice by annotations.
        raw_visulization_path (str): The path for saving the ica figures.

    Returns:
        raw_icafit(Raw): Return the Raw data after ICA decomposition.
        ica_preprocessed(ICA): The preprocessed ICA object.
        exclude_idx(list): The excluded ica list.
    '''
    pass

    raw_process_temp = None
    raw_process_temp = raw.copy()

    ica_preprocessed = ICA(
        n_components=len(raw_process_temp.ch_names)-len(FILTERED_CHANNELS),
        # n_components=7, For test
        max_iter="auto",
        method="infomax",
        random_state=97,
        fit_params=dict(extended=True),
    )
    ica_preprocessed.fit(raw_process_temp)
    ic_labels = label_components(raw_process_temp, ica_preprocessed, method="iclabel")
    labels = ic_labels["labels"]
    exclude_idx = [idx for idx, label in enumerate(labels) if label not in ICA_ACCECPTED]

    ica_preprocessed.apply(raw_process_temp)

    # Plot
    if(ICA_VISULIZATION): 
        raw_process_temp.plot(duration=DURATION_PLOT, show=False).savefig(raw_visulization_path + '-ica.png')
    # raw_preprocessed.plot(duration=10)
    # ica_preprocessed.plot_components(outlines="head")

    raw_icafit = raw_process_temp.copy()

    return raw_icafit, ica_preprocessed, exclude_idx

#  ------------------------------------
# 3. Epoching & Re-Annotation 将点标注(原标注)，改为定长的段标注 (改方法需根据实验修改或重写)
# This function needs to be modified or rewritten to suit the experiment

def get_experimental_raw_list_from_annotations(raw, training_prase, experimental_prase):
    '''Get Experimental Raw List / New Annotations from annotations

    Args:
        raw (mne.Raw): The raw data that require slice by annotations
        training_prase (list) [deprecated] :The description list (flags) for each training segement
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
    fixed_duration = FIXED_DURATION
    fixed_description = ''
    annotation_count = 0
    raw_temp = None
    raw_temp = raw.copy()

    onset = raw_temp.annotations.onset
    description = raw_temp.annotations.description
    len_onset = len(onset)
    len_description = len(description)
    for idx, _desc in enumerate(description):
        if(_desc == ANNOTATION_DESCRIPTION_TRAINING_STRAT):
            onset_training.append(onset[idx])
        # if(_desc == ANNOTATION_DESCRIPTION_EXPERIMENTAL_STRAT and idx > 4): #Temp 本次demo标注用错的特殊处理
        if(_desc == ANNOTATION_DESCRIPTION_EXPERIMENTAL_STRAT): #Origin 正常版本
            onset_experimental.append(onset[idx] - 5.0)

    if(len(onset_experimental) != len(experimental_prase)):
        logging.error("Sth wrong with the length from the onset and description of annotations. " + len(onset_experimental) + " to " + len(experimental_prase))
        raise Exception("Sth wrong with the length from the onset and description of annotations.")


    duration_experimental.extend([fixed_duration for i in range(len(onset_experimental))])
    annotations_experimental = mne.Annotations(onset_experimental, duration_experimental, experimental_prase)
    # raws_experimental = raw_temp.crop_by_annotations(annotations_experimental, verbose = 'debug')
    # raws_experimental = raw_temp.crop_by_annotations(annotations_experimental)
    # return raws_experimental, annotations_experimental
    return annotations_experimental

#  ------------------------------------
# System Args
def parse_args():
    parse = argparse.ArgumentParser(description='Load and Preprocess the raw curry file for one subject.')
    parse.add_argument('-n', '--subjectName', metavar='', default='Cao Driving04', required=False, help='the Name of the Subject')
    # parse.add_argument('-p', '--raw_file_path', metavar='', default='./', required=True, help='the path of the raw file')
    sysArgs = parse.parse_args()
    return sysArgs

# Execute preprocessing 
def run_preprocessing(args):
    preprocessing_succeed = False
    ica_succeed = False
    annotation_reset_succeed = False
    subjectName = args["subjectName"]
    raw_file_path = args["raw_file_path"]
    custom_montage_path = args["custom_montage_path"]
    raw_visulization_path = args["raw_visulization_path"]
    preprocess_status = args["preprocess_status"]
    annotation_reset_status = args["annotation_reset_status"]

    try:
        # 1. Data loading 数据导入
        raw, montage = data_loading(raw_file_path, custom_montage_path, raw_visulization_path)
        
        if(preprocess_status != True):
            # 2.1 Data  Preprocessing 数据预处理
            raw_preprocessed = raw_preprocessing(raw, subjectName, raw_visulization_path)
            # Status update
            preprocessing_succeed = True
            logging.info('--Preprocessing completed: '+ subjectName)

            # 2.2 ICA 独立因子分析
            raw_ica_preprocessed, ica_preprocessed, exclude_idx = raw_ica(raw_preprocessed, raw_visulization_path)
            logging.info('--ICA completed: '+ subjectName)

            # The ICA files save before annotation reset
            # raw_ica_preprocessed.save('./preprocessedFiles/raw_ica_' + subjectName + str(get_timestamp()) + '.fif')
            raw_ica_preprocessed.save('./preprocessedFiles/raw_ica_' + subjectName + '.fif', overwrite=True)
            logging.info('--ICA fif files saved: '+ subjectName)

            # Status update
            ica_succeed = True
        else:
            preprocess_status = True
            ica_succeed = True
            
        if(annotation_reset_status != True):
            # 3. Reset Annotations for a raw file
            annotations_experimental = get_experimental_raw_list_from_annotations(raw_ica_preprocessed, TRAINING_LABELS, EXPERIMENTAL_LABELS)
            raw_ica_preprocessed.set_annotations(annotations_experimental)
            raw_ica_preprocessed.save('./preprocessedFiles/raw_ica_' + subjectName + '.fif', overwrite=True)
            logging.info('Reset annotations completed: '+ subjectName)

            # Status update
            annotation_reset_succeed = True
        else:
            annotation_reset_succeed = True

    except Exception as e:
        logging.error(subjectName + ": " + str(e))
    
    return preprocessing_succeed, ica_succeed, annotation_reset_succeed

if __name__ == "__main__":
    sysArgs = parse_args()
    subjectName = sysArgs.subjectName
    # raw_file_path = './data/yuedurenwu01-12 Data 202301291643.edf'
    raw_file_path = './data/Driving Test 04 Acq 2023_02_24_1956 Data.edf'
    custom_montage_path = MONTAGE_PATH
    raw_visulization_path = VISUALIZATION_FIGURE_PATH + subjectName
    args = {
        "subjectName": subjectName,
        "raw_file_path": raw_file_path,
        "custom_montage_path": custom_montage_path,
        "raw_visulization_path": raw_visulization_path
    }
    preprocessing_succeed, ica_succeed, annotation_reset_succeed = run_preprocessing(args)
    pass