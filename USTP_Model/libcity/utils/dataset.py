"""Tool functions related to the data preprocessing stage"""
import numpy as np
import time
from datetime import datetime, timedelta
from collections import defaultdict


def parse_time(time_in, timezone_offset_in_minute=0):
    """Convert time in time_format format in json to local datatime"""
    date = datetime.strptime(time_in, '%Y-%m-%dT%H:%M:%SZ')  # This is UTC time
    return date + timedelta(minutes=timezone_offset_in_minute)


def cal_basetime(start_time, base_zero):
    """Used to split the trajectory into a session,
    The idea is: given a start_time, find a base time base_time,
    The points in the range from base_time to base_time + time_length are divided into one session,
    The reason for choosing base_time is: This can ensure that the same small period is always encoded into the same number."""
    if base_zero:
        return start_time - timedelta(hours=start_time.hour,
                                      minutes=start_time.minute,
                                      seconds=start_time.second,
                                      microseconds=start_time.microsecond)
    else:
        # time length = 12
        if start_time.hour < 12:
            return start_time - timedelta(hours=start_time.hour,
                                          minutes=start_time.minute,
                                          seconds=start_time.second,
                                          microseconds=start_time.microsecond)
        else:
            return start_time - timedelta(hours=start_time.hour - 12,
                                          minutes=start_time.minute,
                                          seconds=start_time.second,
                                          microseconds=start_time.microsecond)


def cal_timeoff(now_time, base_time):
    """Calculates the difference between two times, returning the value in hours"""
    # First align now by hour
    delta = now_time - base_time
    return delta.days * 24 + delta.seconds / 3600


def caculate_time_sim(data):
    time_checkin_set = defaultdict(set)
    tim_size = data['tim_size']
    data_neural = data['data']
    for uid in data_neural:
        uid_sessions = data_neural[uid]
        for session in uid_sessions:
            for checkin in session:
                timid = checkin[1]
                locid = checkin[0]
                if timid not in time_checkin_set:
                    time_checkin_set[timid] = set()
                time_checkin_set[timid].add(locid)
    sim_matrix = np.zeros((tim_size, tim_size))
    for i in range(tim_size):
        for j in range(tim_size):
            set_i = time_checkin_set[i]
            set_j = time_checkin_set[j]
            if len(set_i | set_j) != 0:
                jaccard_ij = len(set_i & set_j) / len(set_i | set_j)
                sim_matrix[i][j] = jaccard_ij
    return sim_matrix


def parse_coordinate(coordinate):
    items = coordinate[1:-1].split(',')
    return float(items[0]), float(items[1])


def string2timestamp(strings, offset_frame):
    ts = []
    for t in strings:
        dtstr = '-'.join([t[:4].decode(), t[4:6].decode(), t[6:8].decode()])
        slot = int(t[8:]) - 1
        ts.append(np.datetime64(dtstr, 'm') + slot * offset_frame)
    return ts  # [numpy.datetime64('2014-01-01T00:00'), ...]


def timestamp2array(timestamps, t):
    """Convert each timestamp in the sequence of timestamps into a feature array, taking into account the week and hour,
    Timestamp: numpy.datetime64('2013-07-01T00:00:00.000000000')

    Args:
        timestamps: timestamp sequence
        t: how many time steps there are in a day

    Returns:
        np.ndarray: feature array, shape: (len(timestamps), ext_dim)"""
    vec_wday = [time.strptime(
        str(t)[:10], '%Y-%m-%d').tm_wday for t in timestamps]
    vec_hour = [time.strptime(str(t)[11:13], '%H').tm_hour for t in timestamps]
    vec_minu = [time.strptime(str(t)[14:16], '%M').tm_min for t in timestamps]
    ret = []
    for idx, wday in enumerate(vec_wday):
        # day
        v = [0 for _ in range(7)]
        v[wday] = 1
        if wday >= 5:  # 0 is Monday, 6 is Sunday
            v.append(0)  # weekend
        else:
            v.append(1)  # weekday len(v)=8
        # hour
        v += [0 for _ in range(t)]  # len(v)=8+T
        hour = vec_hour[idx]
        minu = vec_minu[idx]
        # 24*60/T represents how many minutes a time step is
        # hour * 60 + minu is the number of minutes from 0:0 to now, and what time step is the division calculation?
        # print(hour, minu, T, (hour * 60 + minu) / (24 * 60 / T))
        v[int((hour * 60 + minu) / (24 * 60 / t))] = 1
        # +8 because there are 8 digits in front of v indicating the day of the week
        if hour >= 18 or hour < 6:
            v.append(0)  # night
        else:
            v.append(1)  # day
        ret.append(v)  # len(v)=7+1+T+1=T+9
    return np.asarray(ret)


def timestamp2vec_origin(timestamps):
    """Convert each timestamp in the timestamp sequence into a feature array, only considering the day of the week,
    Timestamp: numpy.datetime64('2013-07-01T00:00:00.000000000')

    Args:
        timestamps: timestamp sequence

    Returns:
        np.ndarray: feature array, shape: (len(timestamps), 8)"""
    vec = [time.strptime(str(t)[:10], '%Y-%m-%d').tm_wday for t in timestamps]
    ret = []
    for i in vec:
        v = [0 for _ in range(7)]
        v[i] = 1
        if i >= 5:
            v.append(0)  # weekend
        else:
            v.append(1)  # weekday
        ret.append(v)
    return np.asarray(ret)
