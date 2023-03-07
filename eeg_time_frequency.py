# Script for EEG signals pre-processing
# 2023 @ Yan Zhang, Laurence Yu

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import mne
import copy
import argparse
import json
import logging
from mne.preprocessing import ICA
from mne_icalabel import label_components
from mne.time_frequency import tfr_morlet
from dotenv import load_dotenv
from envload import *
from eeg_visulization import *
from util import *


# Initial

# logging.basicConfig(filename=LOGS_PATH,
#                     filemode='a',
#                     format='%(asctime)s,%(msecs)d %(name)s %(levelname)s %(message)s',
#                     datefmt='%H:%M:%S',
#                     level=logging.INFO)

# File Name
FILENAME = 'Reading_Yueteng'

def getResultPathList(fileName):
    pass

    # Output File Name
    resultPathList = {
    #     "path_power_engaged_alpha": './preprocessedFiles/' + fileName + '_engaged_alpha-tfr.csv',
    #     "path_power_engaged_beta": './preprocessedFiles/' + fileName + '_engaged_beta-tfr.csv',
    #     "path_power_engaged_theta": './preprocessedFiles/' + fileName + '_engaged_theta-tfr.csv',
    #     "path_power_disengaged_alpha": './preprocessedFiles/' + fileName + '_disengaged_alpha-tfr.csv',
    #     "path_power_disengaged_beta": './preprocessedFiles/' + fileName + '_disengaged_beta-tfr.csv',
    #     "path_power_disengaged_theta": './preprocessedFiles/' + fileName + '_disengaged_theta-tfr.csv',
        "path_power_engaged": './preprocessedFiles/' + fileName + '_engaged_-tfr.csv', # for test
        "path_power_disengaged": './preprocessedFiles/' + fileName + '_disengaged_-tfr.csv', # for test
        "path_power_tf_overall": './preprocessedFiles/' + fileName + '-tfr.csv',
    }

    return resultPathList

#α，β，θ Frequency Configuration
def configFrequency():
    pass

    frqList = {
        "freqs_alpha": np.arange(8, 13, 0.5),
        "freqs_beta": np.arange(13, 30, 0.5),
        "freqs_theta": np.arange(4, 8, 0.5)
    }

    return frqList


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
    raw.set_montage(montage)
    raw.set_channel_types(chan_types_dict)
    logging.info('Montage Loaded:' + custom_montage_path)


    return raw, montage


def getEpochs(path_raw_ica_preprocessed):
    pass

    # Load preprocessed Raw file
    # path_raw_ica_preprocessed = ('./preprocessedFiles/raw_ica_Reading_Yueteng1675365526.fif')
    raw_ica_preprocessed = mne.io.read_raw(path_raw_ica_preprocessed, preload=True)
    # raw_ica_preprocessed.drop_channels(FILTERED_CHANNELS)

    # Reset Events for Raw
    events,events_id = mne.events_from_annotations(raw_ica_preprocessed)
    # raw_ica_preprocessed.add_events(events, stim_channel=None, replace=True)
    # events = mne.find_events(raw_ica_preprocessed)

    # Epoch Raw with Events
    # epochs = mne.Epochs(raw_ica_preprocessed.drop_channels(FILTERED_CHANNELS), events, events_id, tmin=EPOCH_TMIN, tmax=EPOCH_TMAX)
    epochs = mne.Epochs(raw_ica_preprocessed, events, events_id, tmin=EPOCH_TMIN, tmax=EPOCH_TMAX)
    epochs.apply_baseline((None,0))

    return epochs


def getPower(epochs, frqList, resultPathList):
    pass

    _epochs = None
    _epochs = epochs.copy()

    # target_engage = _epochs['engage'].average()
    # target_disengage = _epochs['disengage'].average()

    # _epoch['engage'].plot()
    # _epoch['disengage'].plot()

    # target_engage.plot_joint();
    # target_disengage.plot_joint();
    # _epoch.plot_joint()

    #get power
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
    power_alpha=tfr_morlet(_epochs, frqList['freqs_alpha'], 2, return_itc=False,average=False)
    power_beta=tfr_morlet(_epochs, frqList['freqs_beta'], 2, return_itc=False,average=False)
    power_theta=tfr_morlet(_epochs, frqList['freqs_theta'], 1, return_itc=False,average=False)

    # CSV File Save
    alpha = power_alpha.average(method='mean', dim='freqs', copy=False).to_data_frame()
    beta = power_beta.average(method='mean', dim='freqs', copy=False).to_data_frame()
    theta = power_theta.average(method='mean', dim='freqs', copy=False).to_data_frame()

    engaged_cal = beta.loc[:, ~beta.columns.isin(['condition', 'time', 'epoch', 'freq'])] / (alpha.loc[:, ~alpha.columns.isin(['condition', 'time', 'epoch', 'freq'])] + theta.loc[:, ~theta.columns.isin(['condition', 'time', 'epoch', 'freq'])])
    engaged_cal = pd.concat([alpha[['condition', 'time', 'epoch']], engaged_cal], axis=1, join="inner")

    # Visulization
    # pass

    return engaged_cal.copy()

#  ------------------------------------
# System Args
def parse_args():
    parse = argparse.ArgumentParser(description='Load and Preprocess the raw curry file for one subject.')
    parse.add_argument('-n', '--subjectName', metavar='', default='test05', required=False, help='the Name of the Subject')
    parse.add_argument('-ip', '--icaRawFilePath', metavar='', default='./temp.fif', required=False, help='the file path of the ica Raw file')
    sysArgs = parse.parse_args()
    return sysArgs

# Execute preprocessing 
def run_trf_power(args):
    trf_power_succeed = False
    subjectName = args["subjectName"]
    # raw_file_path = args["raw_file_path"]
    # custom_montage_path = args["custom_montage_path"]
    raw_visulization_path = args["raw_visulization_path"]
    path_raw_ica_preprocessed = args["path_raw_ica_preprocessed"]
    trf_status = args["trf_status"]

    # 1. Data loading 数据导入
    # raw, montage = data_loading(raw_file_path, custom_montage_path, raw_visulization_path)
    
    # 2. Get Configuration 获取时域分析配置规则和保存路径
    resultPathList = getResultPathList(subjectName)
    frqList = configFrequency()

    # 3. Get Epoch 获取Epochs
    try:
        epochs = getEpochs(path_raw_ica_preprocessed)
    except Exception as e:
        logging.error('Failed to get epoches from file path: '+ str(e))
    
    # 4. Time frequency processsing 获取时域分析power计算结果并保存
    try:
        engaged_cal = getPower(epochs, frqList, resultPathList)
        engaged_cal.to_csv(resultPathList['path_power_tf_overall'])
        trf_power_succeed = True
        logging.info('Power generated on:' + resultPathList['path_power_tf_overall'])
        logging.info('Time frequency (power) processsing completed: '+ subjectName)
        power_visulization(engaged_cal, raw_visulization_path)
    except Exception as e:
        logging.error('Failed to get power values: '+ str(e))


    # 5. ERP plotting
    try:
        engaged_cal = getPower(epochs['engage'].average(), frqList, resultPathList)
        engaged_cal.to_csv(resultPathList['path_power_engaged'])
        logging.info('Engaged Power generated on:' + resultPathList['path_power_engaged'])
        power_visulization(engaged_cal, raw_visulization_path)

        disengaged_cal = getPower(epochs['disengage'].average(), frqList, resultPathList)
        disengaged_cal.to_csv(resultPathList['path_power_disengaged'])
        logging.info('Engaged Power generated on:' + resultPathList['path_power_disengaged'])
        power_visulization(disengaged_cal, raw_visulization_path)
    except Exception as e:
        logging.error('Failed to ERP processing: '+ str(e))

    return trf_power_succeed

if __name__ == "__main__":
    # sysArgs = parse_args()
    # subjectName = sysArgs.subjectName
    # path_raw_ica_preprocessed = sysArgs.icaRawFilePath

    # raw_file_path = './data/yuedurenwu01-12 Data 202301291643.edf'
    subjectName = 'ERP_test022101'
    path_raw_ica_preprocessed = './preprocessedFiles/raw_ica_Reading_Yueteng1675365526.fif'
    custom_montage_path = MONTAGE_PATH
    raw_visulization_path = VISUALIZATION_FIGURE_PATH + subjectName
    preprocess_status = True

    args = {
        "subjectName": subjectName,
        # "raw_file_path": raw_file_path,
        # "custom_montage_path": custom_montage_path,
        "raw_visulization_path": raw_visulization_path,
        "path_raw_ica_preprocessed": path_raw_ica_preprocessed,
        "preprocess_status": preprocess_status
    }

    trf_power_succeed = run_trf_power(args)