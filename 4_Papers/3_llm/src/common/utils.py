import time
from datetime import datetime
import scipy
import numpy as np


def zscore(np_arr):
    """
    Mean normalization, also known as z-score normalization
    """
    mean = np.mean(np_arr)
    std = np.std(np_arr)
    return np.zeros(len(np_arr), dtype=np_arr.dtype) if std == 0 else ((np_arr - mean) / std)


def min_max(np_arr, min_val, max_val):
    """
    Min-max normalization
    """
    return (np_arr - min_val) / (max_val - min_val)


def datetime_str():
    """
    返回当前时间的字符串表示，按照 “年月日时分秒” 格式
    """
    return datetime.now().strftime("%Y%m%d%H%M%S")


def timestamp():
    """
    返回当前时间以毫秒为单位的时间戳
    """
    return int(time.time() * 1000)


def calc_cover_radius(angle, height):
    """
    计算无人机的覆盖半径
    - angle: 无人机的覆盖角度，角度制
    - height: 无人机的高度
    """
    return np.tan(angle * np.pi / 360) * height


def calc_cover_distance(angle, height):
    return height / np.cos(angle * np.pi / 360)


def discount_cumulative_sum(x, discount):
    return scipy.signal.lfilter([1], [1, float(-discount)], x[::-1], axis=0)[::-1]
