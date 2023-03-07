import os
import ast
import numpy as np
import logging
from dotenv import load_dotenv

# Initial
load_dotenv()
RAW_VISULIZATION = bool(os.getenv('RAW_VISULIZATION') == 'True')
ICA_VISULIZATION = bool(os.getenv('ICA_VISULIZATION') == 'True')
FILTERED_CHANNELS = ast.literal_eval(os.getenv('FILTERED_CHANNELS'))
DURATION_PLOT = float(os.getenv('DURATION_PLOT'))
RAW_PREPROCESSING_VISULIZATION = bool(os.getenv('RAW_PREPROCESSING_VISULIZATION') == 'True')
SECONDS_CROP_FROM = float(os.getenv('SECONDS_CROP_FROM'))
RANGE_FREQUENCE_LOWEST = float(os.getenv('RANGE_FREQUENCE_LOWEST'))
RANGE_FREQUENCE_HIGHEST = float(os.getenv('RANGE_FREQUENCE_HIGHEST'))
HZ_RESAMPLE = int(os.getenv('HZ_RESAMPLE'))
REF_CHANNELS = ast.literal_eval(os.getenv('REF_CHANNELS'))
ICA_ACCECPTED = ast.literal_eval(os.getenv('ICA_ACCECPTED'))
ANNOTATION_DESCRIPTION_TRAINING_STRAT = os.getenv('ANNOTATION_DESCRIPTION_TRAINING_STRAT') # deprecated
ANNOTATION_DESCRIPTION_EXPERIMENTAL_STRAT = os.getenv('ANNOTATION_DESCRIPTION_EXPERIMENTAL_STRAT')
FIXED_DURATION = float(os.getenv('FIXED_DURATION'))
TRAINING_LABELS = ast.literal_eval(os.getenv('TRAINING_LABELS'))
EXPERIMENTAL_LABELS = ast.literal_eval(os.getenv('EXPERIMENTAL_LABELS'))
LOGS_PATH = os.getenv('LOGS_PATH')
MONTAGE_PATH = os.getenv('MONTAGE_PATH')
VISUALIZATION_FIGURE_PATH = os.getenv('VISUALIZATION_FIGURE_PATH')
META_METRIC_PATH = os.getenv('META_METRIC_PATH')
EPOCH_TMIN = float(os.getenv('EPOCH_TMIN'))
EPOCH_TMAX = float(os.getenv('EPOCH_TMAX'))

logging.basicConfig(filename=LOGS_PATH,
                    filemode='a',
                    format='%(asctime)s,%(msecs)03d %(levelname)-8s [%(filename)s:%(lineno)d] %(message)s',
                    datefmt='%H:%M:%S',
                    level=logging.INFO)