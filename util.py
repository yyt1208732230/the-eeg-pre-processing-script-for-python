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

# Utils
def get_timestamp():
    # stores current local time
    localtime = time.localtime()
    # ts stores timestamp
    ts = calendar.timegm(localtime)
    return ts