import pandas as pd
import numpy as np


def create_results(df , column, futur):
    """
    :param df: the dataframe that has the data
    :param column: String : the column to compute the results on
    :param futur: int : How much in the futur will we need to see for the result

    :return: The result in an numpy array where 1 means the price moved up and 0 means it moved down
    """
    
    result = np.empty((df.shape[0]))
    result[:] = np.nan
    for i in range(df.shape[0] - futur):
        if (df[column][i] <= df[column][i+ futur]): # Price moved up or stood still
            result[i] = 1
        else: # Price moved down
            result[i] = 0
    return result


# Price Averages
def moving_averages(prices, periods):
    """

    :param prices: OHLC data
    :param periods: periods for which to compute the averages
    :return: Averages over the given periods
            each key points to a dataframe that has the moving averages corresponding to that period
    """

    avs = {}

    for i in range(len(periods)):
        avs[periods[i]] = pd.DataFrame(prices[['open', 'high', 'low', 'close']].rolling(periods[i]).mean())
        avs[periods[i]].columns = ['MA' + str(periods[i]) + ' open', 'MA' + str(periods[i]) + ' high',
                                   'MA' + str(periods[i]) + ' low', 'MA' + str(periods[i]) + ' close']

    return avs


# Heikenashi
def heinkenashi(prices, periods):
    """

    :param prices: dataframe of OHLC & volume data
    :param periods: periods for which to create the candles
    :return: Heiken ashi OHLC candles

    """
    dict = {}

    HAclose = prices[['open', 'high', 'close', 'low']].sum(axis=1) / 4

    HAopen = HAclose.copy()

    HAopen.iloc[0] = HAclose.iloc[0]

    HAhigh = HAclose.copy()

    HAlow = HAclose.copy()
    
    for j in periods:
        for i in range(1, len(prices)):
            HAopen.iloc[i] = (HAopen.iloc[i - 1] + HAclose.iloc[i - 1]) / 2
            HAhigh.iloc[i] = np.array([prices.high.iloc[i], HAopen.iloc[i], HAclose.iloc[i]]).max()
            HAlow.iloc[i] = np.array([prices.low.iloc[i], HAopen.iloc[i], HAclose.iloc[i]]).min()

        df = pd.concat((HAopen, HAhigh, HAlow, HAclose), axis=1)
        df.columns = ['HA' + str(j) +' open', 'HA' + str(j) +' high', 'HA' + str(j) +' close', 'HA' + str(j) +' low']

        dict[j] = df
  

    return dict