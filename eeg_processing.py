import os
import numpy as np
import matplotlib.pyplot as plt
import mne
import copy
import calendar
import time
import argparse
from mne.preprocessing import ICA
from mne_icalabel import label_components
from dotenv import load_dotenv

RAW_PREPROCESSING_VISULIZATION = os.getenv('RAW_PREPROCESSING_VISULIZATION')

# Initial
load_dotenv()

def parse_args():
    parse = argparse.ArgumentParser(description='Calculate cylinder volume') 
    parse.add_argument('-r', '--radius', metavar='', required=True, help='Radius of Cylinder')  # 3、往参数对象添加参数
    parse.add_argument('-H', '--height', metavar='', required=True, help='height of Cylinder')
    args = parse.parse_args()
    return args

def cal_vol(radius, height):
    print(radius)
    print(height)
    print(RAW_PREPROCESSING_VISULIZATION)
    return 0

if __name__ == "__main__":
    args = parse_args()
    cal_vol(args.radius, args.height)