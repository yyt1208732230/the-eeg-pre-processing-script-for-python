from dotenv import load_dotenv
import matplotlib.pyplot as plt
from eeg_processing import *
import pandas as pd
import numpy as np

AVE_STEP = 500  # 降采样取平均的点数
DEG = 8  # 做多项式拟合的次数
LAYOUT = 'hor'  # 'ver'=纵向布局，'hor'=横向布局


def read_tfr(args):
    # 读取已经处理好的 -tfr.csv
    subjectName = args["subjectName"]
    path = "./preprocessedFiles/"+subjectName+"-tfr.csv"
    df_tfr_metric = pd.read_csv(path, dtype={'A': int, 'B': str, 'C': float, 'D': int, 'E': float, 'F': float,
                                             'G': float, 'H': float, 'I': float, 'J': float, 'K': float, 'L': float,
                                             'M': float, 'N': float, 'O': float, 'P': float, 'Q': float, 'R': float,
                                             'S': float, 'T': float, 'U': float, 'V': float, 'W': float, 'X': float,
                                             'Y': float, 'Z': float, 'AA': float, 'AB': float, 'AC': float,
                                             'AD': float, 'AE': float, 'AF': float, 'AG': float, 'AH': float,
                                             'AI': float, 'AJ': float, 'AK': float, 'AL': float, 'AM': float,
                                             'AN': float, 'AO': float, 'AP': float, 'AQ': float, 'AR': float,
                                             'AS': float, 'AT': float, 'AU': float, 'AV': float, 'AW': float,
                                             'AX': float, 'AY': float, 'AZ': float, 'BA': float, 'BB': float,
                                             'BC': float, 'BD': float, 'BE': float, 'BF': float, 'BG': float,
                                             'BH': float, 'BI': float, 'BJ': float, 'BK': float, 'BL': float,
                                             'BM': float})
    return df_tfr_metric


def cal_channel_ave(args):
    # 对每个时间点的所有channel的power进行平均
    df_tfr_metric = read_tfr(args)
    mean_channel = df_tfr_metric.iloc[:, 4:].mean(axis=1)
    df_tfr_metric['mean_channel'] = mean_channel
    return df_tfr_metric


def downsample_average(args):
    df_tfr_metric = cal_channel_ave(args)
    grouped_power_list = df_tfr_metric.iloc[:, -1].rolling(window=AVE_STEP).mean().iloc[AVE_STEP-1::AVE_STEP].tolist()
    grouped_timestamp_list = df_tfr_metric.iloc[::AVE_STEP, 0].tolist()
    del grouped_timestamp_list[0]
    grouped_time_list = df_tfr_metric.iloc[::AVE_STEP, 2].tolist()
    del grouped_time_list[0]

    flag = []
    last_time = -1
    for i, num in enumerate(grouped_time_list):
        if num-last_time < 0:
            flag.append(i)
        last_time = num

    grouped_time_list_zoo = []
    grouped_timestamp_list_zoo = []
    grouped_power_list_zoo = []
    last_flag = 0
    for i, num in enumerate(flag):
        if i != flag.__len__()-1:
            # not the last section
            temp1 = grouped_time_list[last_flag:flag[i]]
            grouped_time_list_zoo.append(temp1)
            temp2 = grouped_timestamp_list[last_flag:flag[i]]
            grouped_timestamp_list_zoo.append(temp2)
            temp3 = grouped_power_list[last_flag:flag[i]]
            grouped_power_list_zoo.append(temp3)
            last_flag = num
        else:
            # the last section
            temp1 = grouped_time_list[last_flag:flag[i]]
            grouped_time_list_zoo.append(temp1)
            temp1 = grouped_time_list[flag[i]:]
            grouped_time_list_zoo.append(temp1)
            temp2 = grouped_timestamp_list[last_flag:flag[i]]
            grouped_timestamp_list_zoo.append(temp2)
            temp2 = grouped_timestamp_list[flag[i]:]
            grouped_timestamp_list_zoo.append(temp2)
            temp3 = grouped_power_list[last_flag:flag[i]]
            grouped_power_list_zoo.append(temp3)
            temp3 = grouped_power_list[flag[i]:]
            grouped_power_list_zoo.append(temp3)
    result_list_zoo = [grouped_time_list_zoo, grouped_timestamp_list_zoo, grouped_power_list_zoo]

# --------分两段
#     flag = -1
#     last_time = -1
#     for i, num in enumerate(grouped_time_list):
#         if num - last_time < 0:
#             flag = i
#             pass
#         last_time = num
#
#     grouped_timestamp_list1 = []
#     grouped_timestamp_list2 = []
#     grouped_power_list1 = []
#     grouped_power_list2 = []
#     grouped_time_list1 = []
#     grouped_time_list2 = []
#
#     if flag != -1:
#         grouped_timestamp_list1 = grouped_timestamp_list[:flag]
#         grouped_timestamp_list2 = grouped_timestamp_list[flag:]
#         grouped_power_list1 = grouped_power_list[:flag]
#         grouped_power_list2 = grouped_power_list[flag:]
#         grouped_time_list1 = grouped_time_list[:flag]
#         grouped_time_list2 = grouped_time_list[flag:]
#     else:
#         print("One group")
#
#     result_list_zoo = [grouped_power_list, grouped_power_list1, grouped_power_list2,
#                        grouped_time_list, grouped_time_list1, grouped_time_list2,
#                        grouped_timestamp_list, grouped_timestamp_list1, grouped_timestamp_list2]
# -------
    return result_list_zoo


def get_visual(args):
    result_list_zoo = downsample_average(args)
    if LAYOUT == 'ver':
        plt.figure(figsize=(5, 15))
        for i in range(result_list_zoo[0].__len__()):
            num = i
            plt.subplot(result_list_zoo[0].__len__(), 1, i + 1)
            # plt.subplot(1, result_list_zoo[0].__len__(), i + 1)
            curve_fitting(result_list_zoo[0][i], result_list_zoo[2][i], num)
        plt.show()
    elif LAYOUT == 'hor':
        plt.figure(figsize=(15, 5))
        for i in range(result_list_zoo[0].__len__()):
            num = i
            plt.subplot(1, result_list_zoo[0].__len__(), i + 1)
            curve_fitting(result_list_zoo[0][i], result_list_zoo[2][i], num)
        plt.show()
    # curve_fitting(result_list_zoo[3], result_list_zoo[0], 7, fig) # 分两段出图


def curve_fitting(x, y, num):
    parameter = np.polyfit(x, y, DEG)    #拟合deg次多项式
    p = np.poly1d(parameter)             #拟合deg次多项式
    aa = ''                              #方程拼接  ——————————————————
    for i in range(DEG+1):
        bb = float('%.3g' % parameter[i])
        # bb = round(parameter[i], 4)
        if bb > 0:
            if i == 0:
                bb = str(bb)
            else:
                bb = '+'+str(bb)
        else:
            bb = str(bb)
        if DEG == i:
            aa = aa+bb
        else:
            aa = aa+bb+'x^'+str(DEG-i) + str( )    #方程拼接  ——————————————————
    plt.scatter(x, y, s=10)     #原始数据散点图
    plt.plot(x, p(x), color='g')  # 画拟合曲线
    print("figure "+str(num+1)+" ploy fit curve is "+ aa)
    # plt.text(0, 0.9, aa, fontdict={'size':'5','color':'k'})
    # plt.legend([aa, round(np.corrcoef(y, p(x))[0, 1]**2, 2)])   #拼接好的方程和R方放到图例
    plt.ylim(0.2, 1.2)
    # plt.show()


if __name__ == "__main__":
    df_meta_metric = read_metrics(META_METRIC_PATH)
    len = verfication(df_meta_metric)

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
    print("Processing " + subject_name)
    get_visual(args)
