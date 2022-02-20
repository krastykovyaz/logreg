import numpy as np
import pandas as pd
import math

def q5(x):
    if any(x):
        return x.quantile(0.05)

def q10(x):
    if any(x):
        return x.quantile(0.1)

def q50(x):
    if any(x):
        return x.quantile(0.5)

def q25(x):
    if any(x):
        return x.quantile(0.25)

def q75(x):
    if any(x):
        return x.quantile(0.75)

def ft_check(val):
    try:
        val = int(val)
        return True
    except:
        return False

def ft_count(arr):
    count = 0
    for val in arr:
        if ft_check(val):
            count += 1
    return count


def ft_sum(arr):
    sum_ = 0
    for val in arr:
        if ft_check(val):
            sum_ += val
            
    return sum_


def ft_mean(arr):
    count = ft_count(arr)
    if count == 0:
        return np.nan
    sum_ = ft_sum(arr)
    return sum_ / count


def ft_min(arr):
    if ft_count(arr) == 0:
        return np.nan
    min_ = arr[0]
    for val in arr:
        if ft_check(val) and val < min_:
            min_ = val
    return min_


def ft_max(arr):
    if ft_count(arr) == 0:
        return np.nan
    max_ = arr[0]
    for val in arr:
        if ft_check(val) and val > max_:
            max_ = val
            
    return max_


def ft_std(arr):
    count = ft_count(arr) - 1
    if count < 1:
        return np.nan
    mean_ = ft_mean(arr)
    sum_diff_sq = 0
    for val in arr:
        if ft_check(val):
            sum_diff_sq += (val - mean_)**2

    return (sum_diff_sq / count)**0.5


def ft_percent(arr, p):
    arr = [var for var in arr if ft_check(var)]
    count = ft_count(arr)
    if count == 0:
        return np.nan
    
    arr.sort()
    k = (count - 1) * p
    f = np.floor(k)
    c = np.ceil(k)
    if f == c:
        return arr[int(k)]
    
    d0 = arr[int(f)] * (c - k)
    d1 = arr[int(c)] * (k - f)
    
    return d0 + d1
    
def describe(df):
    df = df.select_dtypes(include=['int64', 'float64'])
    df.agg({col :['count', 'mean', 'std','min', q5, q10, q25,q50,q75, 'max'] \
        for col in df.columns}).rename(index={'q25':'25%', 'q5':'5%', 'q10':'10%', 'q50':'50%', 'q75':'75%'})
    df = df.select_dtypes(include=['int64', 'float64'])
    stat_dict = {n: [ft_count(df[n]), \
                    ft_mean(df[n]), \
                    ft_std(df[n]), \
                    ft_min(df[n]), \
                     ft_percent(df[n], 0.05),
                     ft_percent(df[n], 0.10),
                    ft_percent(df[n], 0.25),\
                    ft_percent(df[n], 0.5),\
                    ft_percent(df[n], 0.75),\
                    ft_max(df[n])
                   ] for n in df.columns}
    return pd.DataFrame(stat_dict, index=['count', \
                               'mean', \
                               'std', \
                               'min', \
                               '5%', '10%', '25%', '50%', '75%', \
                               'max'])


if __name__ == '__main__':
    df = pd.read_csv('datasets/dataset_train.csv')
