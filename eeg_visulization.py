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
from util import *


# Initial

# logging.basicConfig(filename=LOGS_PATH,
#                     filemode='a',
#                     format='%(asctime)s,%(msecs)d %(name)s %(levelname)s %(message)s',
#                     datefmt='%H:%M:%S',
#                     level=logging.INFO)

# visual_file_path = './preprocessedFiles/yuetengTest01-tfr.csv'

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

def power_visulization(df_power, raw_visulization_path = None):

    figureSaved = False
    df_power_overall = None
    df_power_overall = df_power.copy()

    #  Read Power Data File 
    # df_power_overall = pd.read_csv(visual_file_path)

    conditions = list(df_power_overall.loc[:, ~df_power_overall.columns.isin(['Unnamed: 0','condition', 'time', 'epoch', 'freq'])])
    ch = len(conditions)

    fig, ax = plt.subplots(figsize=(10, 5))
    ax.set_xlabel('Time Instances')
    ax.set_ylabel('Power')

    # ax.plot(df_power_overall['C2'].astype(float), label='C2')
    for idx, chal in enumerate(conditions):
        ax.plot(df_power_overall[chal].astype(float), label=chal)


    legend = ax.legend(loc='upper right', shadow=True, fontsize='medium')
    plt.title('TRF')
    # plt.show()
    if (raw_visulization_path is not None):
        plt.savefig(raw_visulization_path + 'figure-'+ str(get_timestamp()) + '.png')
        figureSaved = True

    return figureSaved

def ica_visulization(raw, show=True, raw_visulization_path = None):
    fig = None
    fig = raw.copy().plot(duration=DURATION_PLOT, show=show)
    if (raw_visulization_path is not None):
        fig.savefig(raw_visulization_path + '-ica.png')
    return fig

#  ------------------------------------
# System Args
def parse_args():
    parse = argparse.ArgumentParser(description='Load and Preprocess the raw curry file for one subject.')
    parse.add_argument('-n', '--subjectName', metavar='', default='test05', required=False, help='the Name of the Subject')
    parse.add_argument('-ip', '--icaRawFilePath', metavar='', default='./temp.fif', required=False, help='the file path of the ica Raw file')
    sysArgs = parse.parse_args()
    return sysArgs

# df_power_overall.loc[:, ~df_power_overall.columns.isin(['condition', 'time', 'epoch', 'freq'])]
# list(df_power_overall.loc[:, ~df_power_overall.columns.isin(['condition', 'time', 'epoch', 'freq'])])
# df_power_overall.iloc[:, :1]
# df_power_overall.iloc[:, 0]
# list(df_power_overall.loc[:, ~df_power_overall.columns.isin(['Unnamed: 0','condition', 'time', 'epoch', 'freq'])])

if __name__ == "__main__":
    # sysArgs = parse_args()
    # subjectName = sysArgs.subjectName
    # path_raw_ica_preprocessed = sysArgs.icaRawFilePath

    # raw_file_path = './data/yuedurenwu01-12 Data 202301291643.edf'
    subjectName = 'ERP_test022101'
    # path_raw_ica_preprocessed = './preprocessedFiles/raw_ica_Reading_Yueteng1675365526.fif'
    path_raw_ica_preprocessed = './preprocessedFiles/raw_ica_Cao Driving041677423595.fif'
    custom_montage_path = MONTAGE_PATH
    raw_visulization_path = VISUALIZATION_FIGURE_PATH + subjectName
    prestep_succeed = True

    args = {
        "subjectName": subjectName,
        # "raw_file_path": raw_file_path,
        # "custom_montage_path": custom_montage_path,
        "raw_visulization_path": raw_visulization_path,
        "path_raw_ica_preprocessed": path_raw_ica_preprocessed,
        "prestep_succeed": prestep_succeed
    }

    raw_ica_preprocessed = mne.io.read_raw(path_raw_ica_preprocessed, preload=True)
    ica_visulization(raw_ica_preprocessed, show=True, raw_visulization_path = None)
