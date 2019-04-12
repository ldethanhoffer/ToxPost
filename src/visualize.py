"""
author: Louis de thanhoffer de Volcsey / git:ldethanhoffer

the necessary functionalities to visualize and analyze datasets

"""


# return the descriptive statistics of a Pandas series with some added info:


def desc_stats(ser):

    # the standard frequentist stats dictionary:
    freq_stats = ser.describe()

    # the interquartile range:
    IQR = freq_stats.loc["75%"] - freq_stats.loc["25%"]
    freq_stats["IQR"] = IQR

    # the outlier range (following Tukey's rule):
    outlier_range = [freq_stats.loc["25%"] - 1.5 * IQR, freq_stats.loc["75%"] + 1.5 * IQR]
    freq_stats["outlier range"] = outlier_range

    # the number of outliers:
    nr_outliers = [len(ser.loc[ser < outlier_range[0]]), len(ser.loc[ser < outlier_range[1]])]
    freq_stats["nr of outliers"] = nr_outliers

    return freq_stats
