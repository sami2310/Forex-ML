import pandas as pd

# Price Averages
def paverages(prices, periods):
    """

    :param prices: OHLC data
    :param periods: periods for which to compute the averages
    :return: Averages over the given periods
            each key points to a dataframe that has the moving averages corresponding to that period
    """

    avs = {}

    for i in range(0, len(periods)):
        avs[periods[i]] = pd.DataFrame(prices[['open', 'high', 'low', 'close']].rolling(periods[i]).mean())
        avs[periods[i]].columns = ['MA' + str(periods[i]) + ' open', 'MA' + str(periods[i]) + ' high',
                                   'MA' + str(periods[i]) + ' low', 'MA' + str(periods[i]) + ' close']

    return avs