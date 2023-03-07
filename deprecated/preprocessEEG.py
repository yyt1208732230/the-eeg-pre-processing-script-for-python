# Script for EEG signals pre-processing
# 2023 @ Yan Zhang, Laurence Yu

import os
import numpy as np
import matplotlib.pyplot as plt
import copy
import mne
from mne.preprocessing import ICA
from mne_icalabel import label_components

# Initial
# Data Loading
RAW_VISULIZATION = False
# Preprocessing
FILTERED_CHANNELS = ['HEO', 'Trigger', 'CZ', 'CB1', 'CB2']
SECONDS_CROP_FROM = 5.0
RANGE_FREQUENCE_HIGHEST = 40
RANGE_FREQUENCE_LOWEST = 0
HZ_RESAMPLE = 200
REF_CHANNELS=['M1', 'M2']

# Split raw data
ANNOTATION_DESCRIPTION_TRAINING_STRAT = '1'
ANNOTATION_DESCRIPTION_EXPERIMENTAL_STRAT = '2'


# --------------------------------------
# 1. Data loading 数据导入
sample_data_raw_file = ('./data/yuedurenwu01-12 Data 202301291643.edf')
raw = mne.io.read_raw_edf(sample_data_raw_file, preload=True)
channel_count = len(raw.ch_names)

# 导入电极位置配置文件
locs_info_path = ('./data/64_ch_montage.loc')
chan_types_dict = {"HEO":"eog", "VEO":"eog", }
montage = mne.channels.read_custom_montage(locs_info_path)
raw.set_montage(montage)
raw.set_channel_types(chan_types_dict)

if(RAW_VISULIZATION is True):
    # 可视化Raw data
    raw.plot_psd(fmax=50) #只绘制低于50赫兹的频率
    raw.plot(duration=5, n_channels=channel_count)

#  ------------------------------------
# 2.Epoching 将点标注(原标注)，改为定长的段标注
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

def get_experimental_raw_list_from_annotations(raw_temp, training_prase, experimental_prase):
    '''Get Experimental Raw List from annotations

    Args:
        raw_temp (mne.Raw): The raw data that require slice by annotations
        training_prase (list): The description list (flags) for each training segement
        experimental_prase (list): The description list (flags) for each training experimental

    Returns:
        raws_experimental(list): The experimental raw objects.
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
    fixed_duration = 50.0
    fixed_description = ''
    annotation_count = 0

    onset = raw_temp.annotations.onset
    description = raw_temp.annotations.description
    len_onset = len(onset)
    len_description = len(description)
    for idx, _desc in enumerate(description):
        if(_desc == ANNOTATION_DESCRIPTION_TRAINING_STRAT):
            onset_training.append(onset[idx])
        if(_desc == ANNOTATION_DESCRIPTION_EXPERIMENTAL_STRAT and idx > 4): #Temp 标注用错的特殊处理
        # if(_desc == ANNOTATION_DESCRIPTION_EXPERIMENTAL_STRAT): #Origin
            onset_experimental.append(onset[idx] - 5.0)

    if(len(onset_experimental) != len(experimental_prase)):
        raise Exception("Sth wrong with the length from the onset and description of annotations.")


    duration_experimental.extend([fixed_duration for i in range(len(onset_experimental))])
    annotations_experimental = mne.Annotations(onset_experimental, duration_experimental, experimental_prase)
    raws_experimental = raw_temp.crop_by_annotations(annotations_experimental)
    return raws_experimental

raw_temp = raw.copy()
raws_experimental = get_experimental_raw_list_from_annotations(raw_temp, training_prase, experimental_prase)

# --------------------------------------
# 3. Data  Preprocessing 数据预处理
def raw_preprocessing(raw_process_temp):
    '''Data preprocessing for raw

    Args:
        raw_process_temp (Raw): The raw data that require slice by annotations

    Returns:
        raw_preprocessed(Raw): The preprocessed Raw object.
        ica_preprocessed(ICA): The preprocessed ICA object.
        exclude_idx(list): The excluded ica list.
    '''
    pass

    # 去掉无用电极
    raw_process_temp = raw_process_temp.copy().drop_channels(FILTERED_CHANNELS)
    # raw_process_temp.plot(duration=60)
    # 留下有效时间段
    raw_process_temp = raw_process_temp.copy().crop(tmin=SECONDS_CROP_FROM, tmax=None)
    # raw_process_temp.plot(duration=60)
    # 去掉50Hz频段影响
    raw_process_temp = raw_process_temp.copy().filter(RANGE_FREQUENCE_LOWEST,RANGE_FREQUENCE_HIGHEST)
    # raw_process_temp.plot(duration=60)
    # 降采样
    raw_process_temp = raw_process_temp.copy().resample(sfreq=HZ_RESAMPLE)
    # raw_process_temp.plot(duration=60)
    # Re-reference
    raw_process_temp = raw_process_temp.copy().set_eeg_reference(ref_channels = REF_CHANNELS)
    # raw_process_temp = raw_process_temp.set_eeg_reference("average")
    # ICA                                                         
    ica_preprocessed = ICA(
        n_components=len(raw_process_temp.ch_names)-3,
        max_iter="auto",
        method="infomax",
        random_state=97,
        fit_params=dict(extended=True),
    )
    ica_preprocessed.fit(raw_process_temp)
    ic_labels = label_components(raw_process_temp, ica_preprocessed, method="iclabel")
    labels = ic_labels["labels"]
    exclude_idx = [idx for idx, label in enumerate(labels) if label not in ["brain", "other"]]
    raw_preprocessed = raw_process_temp.copy()
    return raw_preprocessed, ica_preprocessed, exclude_idx

raw_test = raws_experimental[1]
# raw_test = copy.deepcopy(raws_experimental[1])
raw_preprocessed, ica_preprocessed, exclude_idx = raw_preprocessing(raw_test.copy())
ica_preprocessed.plot_components(outlines="head")

# orig_raw = raw_preprocessed.copy()
# orig_raw.load_data()
# orig_raw.plot()
# ica_preprocessed.apply(orig_raw)
# raw.plot()

raw_preprocessed.plot(duration=60)
ica_preprocessed.apply(raw_preprocessed.copy()).plot(duration=60)
pass                                                                                                                                        # ica_preprocessed.exclude = [1, 2]  # details on how we picked these are omitted here
# ica_preprocessed.plot_properties(raw_preprocessed, picks=ica_preprocessed.exclude)
# ica_preprocessed.plot_properties(raw_preprocessed)


plt.savefig('python_pretty_plot.png')
writer = pd.ExcelWriter('python_plot.xlsx', engine = 'xlsxwriter')
global_num.to_excel(writer, sheet_name='Sheet1')
worksheet = writer.sheets['Sheet1']
worksheet.insert_image('C2','python_pretty_plot.png')
writer.save()
