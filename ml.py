import pandas as pd
import datetime
import numpy as np
import talib as ta
import matplotlib.pyplot as plt
from numpy import log, polyfit, sqrt, std, subtract
from statsmodels.tsa.stattools import adfuller


def rolling_adf(series, window=100):
    def adf_calc(x):
        result = adfuller(x)
        return result[0]  # 返回ADF统计量

    return series.rolling(window).apply(adf_calc, raw=True)

print("adf calc finsh")
print("End")

print("sig1")

