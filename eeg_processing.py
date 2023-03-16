# Script for EEG signals pre-processing
# 2023 @ Yan Zhang, Laurence Yu

# The eeg_processing.py is used for batch processing eeg raw files from a csv spread sheet includes multiple eeg raw signals of participants. 
# CSV spread sheet includes parameters of:
#   subject_name: The name of participant. It names each processing results(e.g. figures, edf files).
#   raw_file_path: The raw file path of each participant.
#   preprocess_status: The status of preprocessing including resize, resimple, ICA, drop bad channels etc,.
#   annotation_reset_status: The status of whether reset annotation succeed (before trf).
#   trf_status: The status of whether complete the trf processing.
#   bad_channels: Developing at the moment...


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

from eeg_preprocessing import * 
from eeg_time_frequency import * 
from eeg_visulization import * 

RAW_PREPROCESSING_VISULIZATION = os.getenv('RAW_PREPROCESSING_VISULIZATION')

# Initial
load_dotenv()


def read_metrics(path):
    '''Load metric csv file information.

    Args:
        path (str): the metric file path

    Returns:
        df_meta_metric(npy): Data from csv file
    '''
    pass

    df_meta_metric = pd.read_csv(path, dtype={'A':str, 'B': str, 'C': bool, 'D': bool, 'E': bool, 'F': str})
    return df_meta_metric

def verfication(df_meta_metric):
    '''Verify whether the metric match the template.

    Args:
        df_meta_metric (npy): the metrics for batch processing.

    Returns:
        len(int): Row count of the metric.
    '''
    pass

    _pd = df_meta_metric.copy()
    len=(_pd)
    headers = _pd.columns
    # header verification
    fixed_headers = ['subject_name', 'raw_file_path', 'preprocess_status', 'annotation_reset_status', 'trf_status', 'bad_channels']
    if(fixed_headers != headers.tolist()):
        len = 0
        raise Exception("The metric file and template not match.")
    
    return len


if __name__ == "__main__":

    df_meta_metric = read_metrics(META_METRIC_PATH)
    len = verfication(df_meta_metric)

    # 遍历metrics
    for index, row in df_meta_metric.iterrows():
        subject_name = row['subject_name']
        raw_file_path = row['raw_file_path']
        preprocess_status = row['preprocess_status']
        annotation_reset_status = row['annotation_reset_status']
        trf_status = row['trf_status']
        bad_channels = row['bad_channels']
        args = {
            "subjectName": subject_name,
            "raw_file_path": raw_file_path,
            "custom_montage_path": MONTAGE_PATH,
            "raw_visulization_path": VISUALIZATION_FIGURE_PATH + subject_name,
            "path_raw_ica_preprocessed": './preprocessedFiles/raw_ica_' + subject_name + '.fif',
            "preprocess_status": preprocess_status,
            "annotation_reset_status": annotation_reset_status,
            "trf_status": trf_status,
        }
        preprocessing_succeed = False
        ica_succeed = False
        annotation_reset_succeed = False
        trf_power_succeed = False
        logging.info('Starting processing participant: ' + subject_name + ', located on: ' + raw_file_path)

        try:
            # 预处理、ICA 和 标注修正，并往csV记录运行状态
            preprocessing_succeed, ica_succeed, annotation_reset_succeed = run_preprocessing(args)
            df_meta_metric.iloc[index, 2] = ica_succeed
            df_meta_metric.iloc[index, 3] = annotation_reset_succeed
            df_meta_metric.to_csv(META_METRIC_PATH ,index=False)

            # 时频分析，并往csV记录运行状态
            if(trf_status != True):
                trf_power_succeed = run_trf_power(args)
                df_meta_metric.iloc[index, 4] = trf_power_succeed
                df_meta_metric.to_csv(META_METRIC_PATH ,index=False)

            logging.info(subject_name + 'preprocessing_succeed: ' + str(preprocessing_succeed), ",ica_succeed: " + str(ica_succeed),
                         ",annotation_reset_succeed: " + str(annotation_reset_succeed),
                         ",trf_power_succeed: " + str(trf_power_succeed))
        except Exception as e:
            logging.error(subject_name + ": " + str(e))
            # 记录未运行状态
            df_meta_metric.iloc[index, 2] = preprocessing_succeed
            df_meta_metric.iloc[index, 3] = trf_power_succeed
            df_meta_metric.iloc[index, 4] = trf_power_succeed
            df_meta_metric.to_csv(META_METRIC_PATH ,index=False)
            logging.info(subject_name + 'preprocessing_succeed: ' + str(preprocessing_succeed), ",ica_succeed: " + str(ica_succeed),
                         ",annotation_reset_succeed: " + str(annotation_reset_succeed),
                         ",trf_power_succeed: " + str(trf_power_succeed))

        logging.info('Ending processing participant: ' + subject_name)