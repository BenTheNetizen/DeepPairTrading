"""
    Date Created: 12/29/21
    About:
        This is an exploration of the effectiveness of a pairs trading strategy between a 
        crypto-related stock (MSTR) and a cryptocurrency (BTC). We first explore pairs trading 
        between BTC and ETH.
"""

import pdb
import pandas as pd
from matplotlib import pyplot as plt
import numpy as np


"""
    Purpose: Records a singular trade's position data 
"""
class Position():
    def __init__(self, is_asset_1_shorted, entry_time, entry_price_1, entry_price_2, balance, bet_size):      
        self.is_asset_1_shorted = is_asset_1_shorted  # True if asset_1 is shorted, False if longed 
        self.trade_return = None                # holding period return 
        self.entry_time = entry_time 
        self.exit_time = None 
        self.entry_balance = balance 
        self.exit_balance = None 
        self.entry_price_1 = entry_price_1 
        self.entry_price_2 = entry_price_2
        self.exit_price_1 = None
        self.exit_price_2 = None 
        self.max_drawdown = None 
        self.max_drawup = None 
        self.bet_size = bet_size 
    def close_position(self, price_1, price_2, time):
        self.exit_time = time 
        self.exit_price_1 = price_1 
        self.exit_price_2 = price_2

        payoff = 0
        if self.is_asset_1_shorted:
            payoff += self.entry_price_1 - price_1 
            payoff += price_2 - self.entry_price_2
        else:
            payoff += price_1 - self.entry_price_1
            payoff += self.entry_price_2 - price_2
        print("TRADE PAYOFF: " + str(payoff))
        self.trade_return = payoff / (self.entry_price_1 + self.entry_price_2)
        self.exit_balance = self.entry_balance + (self.bet_size * self.trade_return)


# Import and format data 
btc_data = pd.read_csv('./data/btcusd.csv')
btc_data['Date'] = pd.to_datetime(btc_data['Date'])
btc_data = btc_data.set_index('Date')
btc_data = btc_data[(btc_data.index > '2021-05-01') & (btc_data.index <= '2021-10-01')]
btc_data = btc_data.iloc[::3]

eth_data = pd.read_csv('./data/ethusd.csv')
eth_data['Date'] = pd.to_datetime(eth_data['Date'])
eth_data = eth_data.set_index('Date')
eth_data = eth_data[(eth_data.index > '2021-05-01') & (eth_data.index <= '2021-10-01')]
eth_data = eth_data.iloc[::3]

# Strategy Parameters
lookback = 100

"""
price = np.array(btc_data['Open'])
time = np.array(btc_data.index)
"""

"""
    Purpose: Returns a Standard Deviation DataFrame based on the input data

    Inputs:
        data: A Pandas DataFrame Series
        period: length of the Standard Deviation
"""
def get_stdev(data, period):
    data_series = pd.Series(data)
    window = data_series.rolling(period)
    sd = window.std()

    sd_list = sd.tolist()

    df_sd = pd.DataFrame(sd_list, columns=['stdev'])
    df_sd['stdev'] = df_sd['stdev'].fillna(0)
    df_sd.index = data.index

    return df_sd

"""
    Purpose: Returns a Simple Moving Average DataFrame based on the input data

    Inputs:
        data: A Pandas DataFrame Series
        period: length of the Simple Moving Average
"""
def sma(data, period):

    data_series = pd.Series(data)
    window = data_series.rolling(period)
    moving_averages = window.mean()

    moving_averages_list = moving_averages.tolist()

    df_sma = pd.DataFrame(moving_averages_list, columns=['sma'])
    df_sma['sma'] = df_sma['sma'].fillna(0)
    df_sma.index = data.index
    return df_sma


def backtest(starting_balance: int, data_1, data_2, bet_size: int, lookback: int):
    
    trades = []
    
    # backtesting variables
    has_position = False 

    sd = get_stdev(abs(data_1['Open'] - data_2['Open']), lookback)
    sd = sd['stdev'].iloc[lookback:]
    sma_ = sma(abs(data_1['Open']-data_2['Open']), lookback)
    sma_ = sma_['sma'].iloc[lookback:]
    data_1 = data_1.iloc[lookback:]
    data_2 = data_2.iloc[lookback:]

    prices_1 = np.array(data_1['Open'])
    prices_2 = np.array(data_2['Open'])
    times_1 = np.array(data_1.index)
    times_2 = np.array(data_2.index) 
    delta = abs(prices_1 - prices_2)
    z_scores = np.array(abs(delta - sma_) / sd)
    balance = starting_balance 

    # Plotting
    # Plot Both Charts
    fig = plt.figure(1)
    ax = fig.add_subplot(111)
    ax.plot(times_1, prices_1, color = 'blue')
    ax2 = ax.twinx()
    ax2.plot(times_2, prices_2, color = 'green')
    plt.title("Bitcoin and Ethereum")

    for i in range(len(prices_1)):
        price_1 = prices_1[i]
        price_2 = prices_2[i]
        time = times_1[i]
        z_score = z_scores[i]

        if has_position:
            # condition to close position
            if z_score < 0.25:
                print(f"Closed position at {time} with prices {price_1} and {price_2}")
                pos.close_position(price_1, price_2, time)
                balance = pos.exit_balance
                trades.append(pos)

                ax.plot(time, price_1, 'rD', zorder=3)
                has_position = False 
        else:
            if z_score > 1.5:
                # open position
                print(f"Opened position at {time} with prices {price_1} and {price_2}")
                bool_ = True if price_1 > price_2 else False 
                pos = Position(bool_, time, price_1, price_2, balance, bet_size)
                has_position = True 
                ax.plot(time, price_1, 'cD', zorder=3)

    
    # Plotting the NAV
    nav_time = []
    nav = []
    nav_time.append(times_1[0])
    nav.append(starting_balance)
    for trade in trades:
        nav_time.append(trade.exit_time)
        nav.append(trade.exit_balance)
    plt.figure(2)
    plt.plot(nav_time, nav)
    plt.title("NAV")
    plt.show()



backtest(1000, btc_data, eth_data, 100, lookback)

"""
# Plot Both Charts
fig = plt.figure()
ax = fig.add_subplot(111)
ax.plot(btc_data.index, btc_data['Open'])
ax2 = ax.twinx()
ax2.plot(eth_data.index, eth_data['Open'], color = 'green')
plt.show()
"""

