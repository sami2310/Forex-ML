import pandas as pd
import numpy as np

from scipy import stats
import scipy.optimize
from scipy.optimize import OptimizeWarning
import warnings
from sklearn.linear_model import LinearRegression


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

    :return: Averages over the given periods in a dataframe
    """

    results = pd.DataFrame(index=prices.index)

    for i in range(len(periods)):
        df = pd.DataFrame(prices[['open', 'high', 'low', 'close']].rolling(periods[i]).mean())
        df.columns = ['MA' + str(periods[i]) + ' open', 'MA' + str(periods[i]) + ' high',
                      'MA' + str(periods[i]) + ' low', 'MA' + str(periods[i]) + ' close']
        results = pd.concat([results, df], axis=1)

    return results


# Exponential Price Averages
def exponential_averages(prices, periods):
    """

    :param prices: OHLC data
    :param periods: periods for which to compute the exponential moving averages

    :return: Exponential moving averages over the given periods in a dataframe
    """

    results = pd.DataFrame(index=prices.index)

    for i in range(len(periods)):
        df = pd.DataFrame(prices[['open', 'high', 'low', 'close']].ewm(span=periods[i], adjust=False).mean())
        df.columns = ['EMA' + str(periods[i]) + ' open', 'EMA' + str(periods[i]) + ' high',
                      'EMA' + str(periods[i]) + ' low', 'EMA' + str(periods[i]) + ' close']
        results = pd.concat([results, df], axis=1)

    return results


# Heikenashi
def heinkenashi(prices, periods):
    """

    :param prices: dataframe of OHLC & volume data
    :param periods: periods for which to create the candles
    
    :return: Heiken ashi OHLC candles in a dataframe

    """
    results = pd.DataFrame(index=prices.index)

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

        results = pd.concat([results, df], axis=1)
  

    return results


# Momentum Function
def momentum(prices, periods):
    """

    :param prices: DataFrame of OHLC data
    :param periods: List of periods to calculate function value
    
    :return: momentum indicator in a dataframe
    """

    results = pd.DataFrame(index=prices.index)

    for i in range(len(periods)):
        results['MomOpen ' + str(periods[i])] = pd.DataFrame(
            prices.open.iloc[periods[i]:] - prices.open.iloc[:-periods[i]].values,
            index=prices.iloc[periods[i]:].index)
        results['MomClose ' + str(periods[i])] = pd.DataFrame(
            prices.close.iloc[periods[i]:] - prices.close.iloc[:-periods[i]].values,
            index=prices.iloc[periods[i]:].index)

    return results


# Detrender
def detrend(prices, method='difference'):
    """
    :param prices: dataframe of OHLC currency data
    :param method: method by which to detrend 'linear' or 'difference'
    
    :return: the detrended price series
    """

    if method == 'difference':
        detrended = prices.close[1:] - prices.close[:-1].values
    elif method == 'linear':
        x = np.arange(0, len(prices))
        y = prices.close.values

        model = LinearRegression()

        model.fit(x.reshape(-1, 1), y.reshape(-1, 1))

        trend = model.predict(x.reshape(-1, 1))
        trend = trend.reshape((len(prices),))
        detrended = prices.close - trend

    else:
        print('You did not input a valid method fot detrending')

    return detrended


# Fourier Series expansion fit function
def fseries(x, a0, a1, b1, w):
    '''

    :param x: The hours (independent variable)
    :param a0: First Fourier Series coefficient
    :param a1: Second Fourier Series coefficient
    :param b1: Third Fourier Series coefficient
    :param w: Fourier Series Frequency
    
    :return: The Value of the Fourier function
    '''

    f = a0 + a1 * np.cos(w * x) + b1 * np.sin(w * x)
    return f


# Sine Series expansion fit function
def sseries(x, a0, b1, w):
    '''

    :param x: The hours (independent variable)
    :param a0: First Sine Series coefficient
    :param a1: Second Sine Series coefficient
    :param b1: Third Sine Series coefficient
    :param w: Sine Series Frequency
    
    :return: The Value of the Sine function
    '''

    f = a0 + b1 * np.sin(w * x)
    return f



# Fourier Series Coefficients Calculator
def fourier(prices, periods, method='difference'):
    '''
    :param prices: OHLC dataframe
    :param periods: list of periods for which to compute coefficients (3,5,10 ...)
    :param method: method by which to detrend the data
    
    :return: datafame containing coefficients for said periods
    '''

    results = pd.DataFrame(index=prices.index)

    # Compute the coefficients of the Series
    detrended = detrend(prices, method)

    for i in range(len(periods)):
        coeffs = []

        for j in range(periods[i], len(prices) - periods[i]):
            x = np.arange(0, periods[i])
            y = detrended.iloc[j - periods[i]:j]

            with warnings.catch_warnings():
                warnings.simplefilter('error', OptimizeWarning)

                try:
                    res = scipy.optimize.curve_fit(fseries, x, y)
                except(RuntimeError, OptimizeWarning):
                    res = np.empty((1, 4))
                    res[0, :] = np.NAN

            coeffs = np.append(coeffs, res[0], axis=0)

        warnings.filterwarnings('ignore', category=np.VisibleDeprecationWarning)

        coeffs = np.array(coeffs).reshape(((int(len(coeffs) / 4), 4)))

        df = pd.DataFrame(coeffs, index=prices.index[periods[i]:-periods[i]])

        df.columns = ['fourier ' + str(periods[i]) + ' a0', 'fourier ' + str(periods[i]) + ' a1',
                      'fourier ' + str(periods[i]) + ' b1', 'fourier ' + str(periods[i]) + ' w']
        df = df.fillna(method='bfill')
        results = pd.concat([results, df], axis=1)

    return results



# Sine Series Coefficients Calculator
def sine(prices, periods, method='difference'):
    '''
    :param prices: OHLC dataframe
    :param periods: list of periods for which to compute coefficients (3,5,10 ...)
    :param method: method by which to detrend the data

    :return: datafames containing coefficients for said periods
    '''

    results = pd.DataFrame(index=prices.index)

    # Compute the coefficients of the Series
    detrended = detrend(prices, method)

    for i in range(0, len(periods)):
        coeffs = []

        for j in range(periods[i], len(prices) - periods[i]):
            x = np.arange(0, periods[i])
            y = detrended.iloc[j - periods[i]:j]

            with warnings.catch_warnings():
                warnings.simplefilter('error', OptimizeWarning)

                try:
                    res = scipy.optimize.curve_fit(sseries, x, y)
                except(RuntimeError, OptimizeWarning):
                    res = np.empty((1, 3))
                    res[0, :] = np.NAN

            coeffs = np.append(coeffs, res[0], axis=0)

        warnings.filterwarnings('ignore', category=np.VisibleDeprecationWarning)

        coeffs = np.array(coeffs).reshape(((int(len(coeffs) / 3), 3)))

        df = pd.DataFrame(coeffs, index=prices.index[periods[i]:-periods[i]])
        # To edit if error remove outside brackets
        df.columns = ['sine ' + str(periods[i]) + ' a0', 'sine ' + str(periods[i]) + ' b1', 'sine ' + str(periods[i]) + ' w']
        df = df.fillna(method='bfill')
        results = pd.concat([results, df], axis=1)

    return results


def wadl(prices, periods):
    '''
    :param prices::param prices: OHLC dataframe
    :param periods: list of periods for which to compute the function
    
    :return: Williams accumulation Distribution lines for each period
    '''
    
    results = pd.DataFrame(index=prices.index)

    for i in range(0, len(periods)):
        WAD = []

        for j in range(periods[i], len(prices) - periods[i]):
            TRH = np.array([prices.high.iloc[j], prices.close.iloc[j - 1]]).max()
            TRL = np.array([prices.low.iloc[j], prices.close.iloc[j - 1]]).min()

            if prices.close.iloc[j] > prices.close.iloc[j - 1]:
                PM = prices.close.iloc[j] - TRL
            elif prices.close.iloc[j] < prices.close.iloc[j - 1]:
                PM = prices.close.iloc[j] - TRH
            else:
                PM = 0

            AD = PM * prices.volume.iloc[j]
            WAD = np.append(WAD, AD)

        WAD = WAD.cumsum()
        WAD = pd.DataFrame(WAD, index=prices.iloc[periods[i]:-periods[i]].index)
        WAD.columns = ['WAD'+str(periods[i])+' close']

        results = pd.concat([results, WAD], axis=1)

    return results


# Stochastic oscillator function
def stochastic(prices, periods):
    """

    :param prices: DataFrame of OHLC data
    :param periods: List of periods to calculate function value
    
    :return: oscillator function values
    """
    results = pd.DataFrame(index=prices.index)

    for i in range(0, len(periods)):
        Ks = []

        for j in range(periods[i], len(prices) - periods[i]):
            C = prices.close.iloc[j + 1]
            H = prices.high.iloc[j - periods[i]:j].max()
            L = prices.low.iloc[j - periods[i]:j].min()

            if H == L:
                K = 0
            else:
                K = 100 * (C - L) / (H - L)
            Ks = np.append(Ks, K)

        df = pd.DataFrame(Ks, index=prices.iloc[periods[i] + 1:-periods[i] + 1].index)
        df.columns = ['stochastics ' + str(periods[i]) + ' K']
        df['stochastics ' + str(periods[i]) + ' D'] = df['stochastics ' + str(periods[i]) + ' K'].rolling(3).mean()
        df = df.dropna()

        results = pd.concat([results, df], axis=1)

    return results

