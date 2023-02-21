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

logging.basicConfig(filename=LOGS_PATH,
                    filemode='a',
                    format='%(asctime)s,%(msecs)d %(name)s %(levelname)s %(message)s',
                    datefmt='%H:%M:%S',
                    level=logging.INFO)

# visual_file_path = './preprocessedFiles/yuetengTest01-tfr.csv'

def visulization_power(df_power):

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
    plt.savefig('figure-'+ str(get_timestamp()) + '.png')

    return figureSaved

# df_power_overall.loc[:, ~df_power_overall.columns.isin(['condition', 'time', 'epoch', 'freq'])]
# list(df_power_overall.loc[:, ~df_power_overall.columns.isin(['condition', 'time', 'epoch', 'freq'])])
# df_power_overall.iloc[:, :1]
# df_power_overall.iloc[:, 0]
# list(df_power_overall.loc[:, ~df_power_overall.columns.isin(['Unnamed: 0','condition', 'time', 'epoch', 'freq'])])