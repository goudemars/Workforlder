
#I. THE ONLY THING YOU HAVE TO CHANGE TO MAKE THE BOT WORK

#SuperAI Trading Bot - 1. mandatory cell to run
KEY_ID = "your own KEY_ID" #replace it with your own KEY_ID from Alpaca: https://alpaca.markets/
SECRET_KEY = "your own SECRET_KEY" #replace it with your own SECRET_KEY from Alpaca

#II. THINGS YOU DON'T NEED TO CHANGE, BUT MIGHT TO EASILY MAKE THE BOT BETTER
#INSTALL (optional) AND IMPORT EXTERNAL TOOLS (LIBRARIES, PACKAGES, AND MODULES)

#SuperAI Trading Bot - optional cell
#You need to run it only once when you create new environment
#install and upgrade TA-Lib library

#1. On Windows and if you are using Anaconda,  
#open Anaconda Prompt and
#write in Anaconda Prompt: conda install -c conda-forge ta-lib

#2. If it doesn't work or if you work on Linux or Mac you may unahsh pip line and check this way 
#!pip install TA-Lib --upgrade

#3. You could also check this site to learn how else you could do install it
#https://pypi.org/project/TA-Lib/
#SuperAI Trading Bot - optional cell
#You need to run it only once when you create new environment
#install and upgrade all the libraries, packages, and modules which you don't have



import alpaca_trade_api as tradeapi
from alpaca_trade_api.rest import TimeFrame
import datetime
from datetime import timedelta
import math
import numpy as np
import pandas as pd
import sys
import ta
import talib as ta_lib
import time
import vectorbt as vbt
import warnings
warnings.filterwarnings('ignore')

print("Python version: {}".format(sys.version))
print("alpaca trade api version: {}".format(tradeapi.__version__))
print("numpy version: {}".format(np.__version__))
print("pandas version: {}".format(pd.__version__))
print("ta_lib version: {}".format(ta_lib.__version__))
print("vectorbt version: {}".format(vbt.__version__))


# III CHOOSE AN ASSET TO TRADE
#You can choose any asset that can be traded with Alpaca API and paste the ticker of the asset in here.

#SuperAI Trading Bot - 3. mandatory cell to run

(asset, asset_type, rounding) = ("BTCUSD", "crypto", 0)

#if you want to trade crypto check: https://alpaca.markets/support/what-cryptocurrencies-does-alpaca-currently-support/
#rounding declares the no of numbers after comma for orders
#read more about minimum qty and qty increment at https://alpaca.markets/docs/trading/crypto-trading/

#(asset, asset_type, data_source) = ("AAPL", "stock", "Yahoo") #("AAPL", "stock", "Alpaca")

#if you want to trade stocks replace it with the ticker of the company you prefer: https://www.nyse.com/listings_directory/stock
#you can also use "Alpaca" as a data_source
#Alpaca gives you free access to more historical data, but in a free plan doesn't allow you to access data from last 15 minutes
#Yahoo gives you access to data from last 15 minutes, but gives you only 7 days of historical data with 1-min interval at a time
#from last month


# IV SHAPE THE BRAIN OF THE BOT
#Here you can declare which technical indicators and candlestick patterns you want to use as signals to buy and sell stocks and how the signals should be created (what should the logic for buying and selling be).
"""
You have 6 indicators from 4 categories to use:

TREND INDICATORS: EMA (Exponential Moving Average), MACD (Moving Average Convergence Divergence)
MOMENTUM INDICATORS: RSI (Relative Strength Index), Stoch (Stochastic)
VOLATILITY: BB (Bollinger Bands)
VOLUME: MFI (Money Flow Index)
You have 8 candlestick patterns from 3 categories to use:

BUY CANDLE PATTERNS: from 1 candle: Hammer, from 3 candles: Morning Star, 3 White Soldiers
SELL CANDLE PATTERNS: from 1 candle: Shooting Star, from 3 candles: Evening Star, 3 Black Crows
BUY - SELL CANDLE PATTERNS: from 2 candles: Engulfing Bullish / Bearish, from 3 candles: 3 Outside Up / Down
Declare which indicators and how should be used in a function for creating a SuperAI trading signal (buy / sell / do nothing)
1st Sample Strategy - with moving average: BUY when CLOSE PRICE crosses over MA ==> SELL when CLOSE PRICE crosses under MA

crossing.png

#SuperAI Trading Bot - 4. mandatory cell to run
# 1st SAMPLE STRATEGY - WITH MOVING AVERAGE
# BUY WHEN CLOSE PRICE CROSSES OVER THE MOVING AVERAGE 
# SELL WHEN CLOSE PRICE CROSSES UNDER THE MOVING AVERAGE
"""

def create_signal(open_prices, high_prices, low_prices, close_prices, volume,
                  buylesstime, selltime, #Time to abstain from buying and forced selling
                  ma, ma_fast, ma_slow, #Moving Average
                  macd, macd_diff, macd_sign, #Moving Average Convergence Divergence
                  rsi, rsi_entry, rsi_exit, #Relative Strength Index
                  stoch, stoch_signal, stoch_entry, stoch_exit, #Stochastic
                  bb_low, bb_high,  #Bollinger Bands
                  mfi, mfi_entry, mfi_exit, #Money Flow Index
                  candle_buy_signal_1, candle_buy_signal_2, candle_buy_signal_3, #Candle signals to buy
                  candle_sell_signal_1, candle_sell_signal_2, candle_sell_signal_3, #Candle signals to sell
                  candle_buy_sell_signal_1, candle_buy_sell_signal_2): #Candle signals to buy or sell
    
    
    SuperAI_signal_buy = np.where( 
                                  (buylesstime != 1) 
                                  &
                                  ((close_prices > ma) & (shifted(close_prices, 1) <= shifted(ma, 1)))
                                  , 1, 0) #1 is buy, -1 is sell, 0 is do nothing
    SuperAI_signal = np.where( 
                                  (selltime == 1)
                                  |
                                  ((close_prices < ma) & (shifted(close_prices, 1) >= shifted(ma, 1)))
                                  , -1, SuperAI_signal_buy) #1 is buy, -1 is sell, 0 is do nothing
    
    global cs_name
    cs_name = '1st Sample Strategy'
    
    return SuperAI_signal

#V DECLARE PARAMETERS AND FUNCTIONS FOR BACKTESTING AND LIVE TRADING
#SuperAI Trading Bot - 5. mandatory cell to run

# PARAMETERS FOR CONNECTING TO ALPACA

#The URLs we use here are for paper trading. If you want to use your bot with your live account, you should change these URLs to
#those dedicated to live account at Alpaca. Just remember, that with live account you are using real money, 
#so be sure that your bot works as you want it to work. Test your bot before you give it real money to trade.
APCA_API_BASE_URL = "https://paper-api.alpaca.markets"

#Connecting to API
api = tradeapi.REST(KEY_ID, SECRET_KEY, APCA_API_BASE_URL, "v2")

#Keys to use Alpaca with vectorbt backtesting library
vbt.settings.data['alpaca']['key_id'] = KEY_ID
vbt.settings.data['alpaca']['secret_key'] = SECRET_KEY


# PARAMETERS USED BY SUPERAI TRADING BOT WITH AND WITHOUT OPTIMIZATION

funds_percentage = 90 #replace it with the percentage of the amount of money you have to use for trading

take_profit_percent = 0.5 # 50%
stop_loss_percent = 0.05 # 5%

#Parameters for live trading (during paper trading take profit and stop loss are always on, but you can change it in the code)
take_profit_automatically = True #change to False if you don't want to use take_profit function during live trading
stop_loss_automatically = True #change to False if you don't want to use stop_loss function during live trading

#Funds to invest
account = api.get_account()
cash = float(account.cash)
buying_power = float(account.buying_power)
funds = cash * funds_percentage / 100

#Parameters for downloading the data
data_timeframe = '1m' #replace with preferable time between data: 1m, 5m, 15m, 30m, 1h, 1d
data_limit = None #replace with the limit of the data to download to speed up the process (500, 1000, None)

crypto_data_timeframe = TimeFrame.Minute
preferred_exchange = "CBSE"

#Basic parameters to use for indicators before/without optimization
ma_window_max = ma_window = ma_timeframe = 28
ma_fast_window_max = ma_fast_window = ma_fast_timeframe = 14 
ma_slow_window_max = ma_slow_window = ma_slow_timeframe = 50

macd_slow_window_max = macd_slow_window = macd_slow_timeframe = 26
macd_fast_window_max = macd_fast_window = macd_fast_timeframe = 12
macd_sign_window_max = macd_sign_window = macd_signal_timeframe = 9

rsi_window_max = rsi_window = rsi_timeframe = 14
rsi_entry_max = rsi_entry = rsi_oversold_threshold = 30
rsi_exit_max = rsi_exit = rsi_overbought_threshold = 70

stoch_window_max = stoch_window = stoch_timeframe = 14
stoch_smooth_window_max = stoch_smooth_window = stoch_smooth_timeframe = 3
stoch_entry_max = stoch_entry = stoch_oversold_threshold = 20
stoch_exit_max = stoch_exit = stoch_overbought_threshold = 80

bb_window_max = bb_window = bb_timeframe = 10
bb_dev_max = bb_dev = bb_dev = 2

mfi_window_max = mfi_window = mfi_timeframe = 14
mfi_entry_max = mfi_entry = mfi_oversold_threshold = 20
mfi_exit_max = mfi_exit = mfi_overbought_threshold = 80

#Parameters to optimize during backtesting

ma_window_opt = np.arange(14, 30, step=14, dtype=int) #[14, 28]
ma_fast_window_opt = np.arange(14, 22, step=7, dtype=int) #[14, 21]
ma_slow_window_opt = np.arange(30, 51, step=20, dtype=int) #[30, 50]

macd_slow_window_opt = np.arange(26, 27, step=100, dtype=int) #[26]
macd_fast_window_opt = np.arange(12, 13, step=100, dtype=int) #[12]
macd_sign_window_opt = np.arange(9, 10, step=100, dtype=int) #[9]

rsi_window_opt = np.arange(14, 22, step=7, dtype=int) #[14, 21]
rsi_entry_opt = np.arange(20, 31, step=10, dtype=int) #[20, 30]
rsi_exit_opt = np.arange(70, 81, step=10, dtype=int) #[70, 80]

stoch_window_opt = np.arange(14, 15, step=100, dtype=int) #[14]
stoch_smooth_window_opt = np.arange(3, 4, step=100, dtype=int) #[3]
stoch_entry_opt = np.arange(20, 21, step=100, dtype=int) #[20]
stoch_exit_opt = np.arange(80, 81, step=100, dtype=int) #[80]

bb_window_opt = np.arange(10, 21, step=10, dtype=int) #[10, 20]
bb_dev_opt = np.arange(2, 3, step=100, dtype=int) #[2]

mfi_window_opt = np.arange(14, 22, step=7, dtype=int) #[14, 21]
mfi_entry_opt = np.arange(10, 21, step=10, dtype=int) #[10, 20]
mfi_exit_opt = np.arange(80, 91, step=10, dtype=int) #[80, 90]


# PARAMETERS FOR DECISIONS REGARDING OPTIONS AND TIME OF BACKTESTING AND LIVE TRADING

optimization = True #True or False
validation = True #True or False

#Dates to download data for backtesting training phase
data_start = '2022-07-01' #replace with the starting point for collecting data
data_end = '2022-07-31' #replace with the ending point for collecting the data

#Dates to download data for backtesting validation phase
valid_start = '2022-08-01'
valid_end = '2022-08-15'

#The function for declaring trading hours
def trading_buy_sell_time():
    if asset_type == 'stock':
        #more about trading hours at: https://alpaca.markets/docs/trading/orders/#extended-hours-trading
        trading_hour_start = "09:30"
        trading_hour_stop = "16:00" 
        #time when you don't want to buy at the beginning of the day 
        buyless_time_start_1 = "09:30"
        buyless_time_end_1 = "09:45"
        buyless_time_start_2 = "15:55"
        buyless_time_end_2 = "16:00"
        #time when you want to sell by the end of the day
        selltime_start = "15:55"
        selltime_end = "16:00"

    elif asset_type == 'crypto':
        trading_hour_start = "00:00"
        trading_hour_stop = "23:59" 
        #time when you don't want to buy at the beginning of the day 
        buyless_time_start_1 = "23:59"
        buyless_time_end_1 = "00:01"
        buyless_time_start_2 = "23:58"
        buyless_time_end_2 = "23:59"
        #time when you want to sell by the end of the day
        selltime_start = "23:59"
        selltime_end = "00:00"
        
    return (trading_hour_start, trading_hour_stop, 
            buyless_time_start_1, buyless_time_end_1, buyless_time_start_2, buyless_time_end_2,
            selltime_start, selltime_end)

#DataFrame for testing various strategies
columns = ('Strategy', 'Basic Returns', 'Returns After Optimization', 'Returns From Validation')
strategies = pd.DataFrame([], columns=columns).set_index('Strategy')
strategy_returns = ['0',0,0,0]

#III. THINGS YOU DON'T NEED TO CHANGE, BUT MIGHT IF YOU WANT TO MAKE MORE ADVANCED CHANGES TO THE BOT
#All the other necessary functions and variables of the bot used during backtesting and live paper trading
#There is no need to change them.

#SuperAI Trading Bot - 6. mandatory cell to run

# 1. ALL THE OTHER NECESSARY FUNCTIONS AND VARIABLES OF THE BOT USED DURING BACKTESTING

#helper function to shift data in order to test differences between data from x min and data from x-time min
def shifted(data, shift_window):
    data_shifted = np.roll(data, shift_window)
    if shift_window >= 0:
        data_shifted[:shift_window] = np.NaN
    elif shift_window < 0:
        data_shifted[shift_window:] = np.NaN
    return data_shifted

#preparing data in one function
def prepare_data(start_date, end_date):
    data_start = start_date
    data_end = end_date
    
    if asset_type == "stock" and data_source == "Alpaca":
        full_data = vbt.AlpacaData.download(asset, start=data_start, end=data_end, 
                                            timeframe=data_timeframe, limit=data_limit).get()
        
    elif asset_type == "stock" and data_source == "Yahoo":
        try:
            full_data = vbt.YFData.download(asset, start = data_start, end= data_end, 
                                        interval=data_timeframe).get().drop(["Dividends", "Stock Splits"], axis=1)
        except:
            full_data = vbt.AlpacaData.download(asset, start=data_start, end=data_end, 
                                            timeframe=data_timeframe, limit=data_limit).get()
            print("""\nI tried downloading data with Yahoo, but something went wrong so I downloaded data with Alpaca.
                  That means than the data might not look the same as the data from Yahoo.\n\n""")
        
    elif asset_type == "crypto":
        crypto_data = api.get_crypto_bars(asset, crypto_data_timeframe, start = data_start, end=data_end).df
        full_crypto_data = crypto_data[crypto_data['exchange'] == preferred_exchange]
        full_data = full_crypto_data.rename(str.capitalize, axis=1).drop(["Exchange", "Trade_count", "Vwap"], axis=1)
        
    else:
        print("You have to declare asset type as crypto or stock for me to work properly.")
        
    full_data.index = full_data.index.tz_convert('America/New_York')
    
    (trading_hour_start, trading_hour_stop, 
            buyless_time_start_1, buyless_time_end_1, buyless_time_start_2, buyless_time_end_2,
            selltime_start, selltime_end) = trading_buy_sell_time()
    
    data = full_data.copy()
    data = data.between_time(trading_hour_start, trading_hour_stop)
    
    not_time_to_buy_1 = data.index.indexer_between_time(buyless_time_start_1, buyless_time_end_1) 
    not_time_to_buy_2 = data.index.indexer_between_time(buyless_time_start_2, buyless_time_end_2) 
    not_time_to_buy = np.concatenate((not_time_to_buy_1, not_time_to_buy_2), axis=0)
    not_time_to_buy = np.unique(not_time_to_buy)
    data["NotTimeToBuy"] = 1
    data["BuylessTime"] = data.iloc[not_time_to_buy, 5]
    data["BuylessTime"] = np.where(np.isnan(data["BuylessTime"]), 0, data["BuylessTime"])
    data = data.drop(["NotTimeToBuy"], axis=1)

    time_to_sell = data.index.indexer_between_time(selltime_start, selltime_end) 
    
    data["TimeToSell"] = 1
    data["SellTime"] = data.iloc[time_to_sell, 6]
    data["SellTime"] = np.where(np.isnan(data["SellTime"]), 0, data["SellTime"])
    data = data.drop(["TimeToSell"], axis=1)
    
    open_prices = data["Open"]
    high_prices = data["High"]
    low_prices = data["Low"]
    close_prices = data["Close"]
    volume = data["Volume"]
    buylesstime = data["BuylessTime"]
    selltime = data["SellTime"]
    
    return open_prices, high_prices, low_prices, close_prices, volume, buylesstime, selltime

def starting_max_parameters():
    #assigning the starting parameters as previously declared
    
    global ma_window_max, ma_fast_window_max, ma_slow_window_max 
    global macd_slow_window_max, macd_fast_window_max, macd_sign_window_max
    global rsi_window_max, rsi_entry_max, rsi_exit_max 
    global stoch_window_max, stoch_smooth_window_max, stoch_entry_max, stoch_exit_max
    global bb_window_max, bb_dev_max, mfi_window_max
    global mfi_entry_max, mfi_exit_max

    ma_window_max = ma_window
    ma_fast_window_max = ma_fast_window 
    ma_slow_window_max = ma_slow_window

    macd_slow_window_max = macd_slow_window
    macd_fast_window_max = macd_fast_window
    macd_sign_window_max = macd_sign_window
    
    rsi_window_max = rsi_window
    rsi_entry_max = rsi_entry
    rsi_exit_max = rsi_exit

    stoch_window_max = stoch_window
    stoch_smooth_window_max = stoch_smooth_window
    stoch_entry_max = stoch_entry
    stoch_exit_max = stoch_exit

    bb_window_max = bb_window
    bb_dev_max = bb_dev
    
    mfi_window_max = mfi_window
    mfi_entry_max = mfi_entry
    mfi_exit_max = mfi_exit

# Custom SuperAI Indicator
# Signals Function
def superai_signals (open_prices, high_prices, low_prices, close_prices, volume, buylesstime, selltime,
                     ma_window = ma_timeframe, 
                     ma_fast_window = ma_fast_timeframe,
                     ma_slow_window = ma_slow_timeframe,
                     macd_slow_window = macd_slow_timeframe, 
                     macd_fast_window = macd_fast_timeframe, 
                     macd_sign_window = macd_signal_timeframe,
                     rsi_window = rsi_timeframe,
                     rsi_entry = rsi_oversold_threshold, 
                     rsi_exit = rsi_overbought_threshold, 
                     stoch_window = stoch_timeframe,
                     stoch_smooth_window = stoch_smooth_timeframe,
                     stoch_entry = stoch_oversold_threshold, 
                     stoch_exit = stoch_overbought_threshold,
                     bb_window = bb_timeframe,
                     bb_dev = bb_dev,
                     mfi_window = mfi_timeframe,
                     mfi_entry = mfi_oversold_threshold,
                     mfi_exit = mfi_overbought_threshold):
    
    rsi = vbt.IndicatorFactory.from_ta('RSIIndicator').run(close_prices, window = rsi_window).rsi.to_numpy()
    
    stoch = vbt.IndicatorFactory.from_ta('StochasticOscillator').run(
        high_prices, low_prices, close_prices, window = stoch_window, smooth_window = stoch_smooth_window).stoch.to_numpy()
    stoch_signal = vbt.IndicatorFactory.from_ta('StochasticOscillator').run(
        high_prices, low_prices, close_prices, window = stoch_window, 
        smooth_window = stoch_smooth_window).stoch_signal.to_numpy()
    
    ma = vbt.IndicatorFactory.from_ta('EMAIndicator').run(close_prices, window = ma_window).ema_indicator.to_numpy()
    ma_fast = vbt.IndicatorFactory.from_ta('EMAIndicator').run(close_prices, window = ma_fast_window).ema_indicator.to_numpy()
    ma_slow = vbt.IndicatorFactory.from_ta('EMAIndicator').run(close_prices, window = ma_slow_window).ema_indicator.to_numpy()
    
    macd = vbt.IndicatorFactory.from_ta('MACD').run(
        close_prices, window_slow = macd_slow_window, window_fast = macd_fast_window, 
        window_sign = macd_sign_window).macd.to_numpy()
    macd_diff = vbt.IndicatorFactory.from_ta('MACD').run(
        close_prices, macd_slow_window, window_fast = macd_fast_window, 
        window_sign = macd_sign_window).macd_diff.to_numpy()
    macd_sign = vbt.IndicatorFactory.from_ta('MACD').run(
        close_prices, macd_slow_window, window_fast = macd_fast_window, 
        window_sign = macd_sign_window).macd_signal.to_numpy()

    bb_low = vbt.IndicatorFactory.from_ta('BollingerBands').run(
        close_prices, window = bb_window, window_dev = bb_dev).bollinger_lband.to_numpy()
    bb_high = vbt.IndicatorFactory.from_ta('BollingerBands').run(
        close_prices, window = bb_window, window_dev = bb_dev).bollinger_hband.to_numpy()
    
    mfi = vbt.IndicatorFactory.from_ta('MFIIndicator').run(
        high_prices, low_prices, close_prices, volume, window = mfi_timeframe).money_flow_index.to_numpy()
    
    candle_buy_signal_1 = vbt.IndicatorFactory.from_talib('CDLHAMMER').run(
        open_prices, high_prices, low_prices, close_prices).integer.to_numpy() # 'Hammer'
    candle_buy_signal_2 = vbt.IndicatorFactory.from_talib('CDLMORNINGSTAR').run(
        open_prices, high_prices, low_prices, close_prices).integer.to_numpy() # 'Morning star'
    candle_buy_signal_3 = vbt.IndicatorFactory.from_talib('CDL3WHITESOLDIERS').run(
        open_prices, high_prices, low_prices, close_prices).integer.to_numpy() # 'Three White Soldiers'
    candle_sell_signal_1 = vbt.IndicatorFactory.from_talib('CDLSHOOTINGSTAR').run(
        open_prices, high_prices, low_prices, close_prices).integer.to_numpy() # 'Shooting star' 
    candle_sell_signal_2 = vbt.IndicatorFactory.from_talib('CDLEVENINGSTAR').run(
        open_prices, high_prices, low_prices, close_prices).integer.to_numpy() # 'Evening star'
    candle_sell_signal_3 = vbt.IndicatorFactory.from_talib('CDL3BLACKCROWS').run(
        open_prices, high_prices, low_prices, close_prices).integer.to_numpy() # '3 Black Crows'
    candle_buy_sell_signal_1 = vbt.IndicatorFactory.from_talib('CDLENGULFING').run(
        open_prices, high_prices, low_prices, close_prices).integer.to_numpy() # 'Engulfing: Bullish (buy) / Bearish (sell)'
    candle_buy_sell_signal_2 = vbt.IndicatorFactory.from_talib('CDL3OUTSIDE').run(
        open_prices, high_prices, low_prices, close_prices).integer.to_numpy() # 'Three Outside: Up (buy) / Down (sell)'
    
    SuperAI_signal = create_signal(open_prices, high_prices, low_prices, close_prices, volume, 
                                   buylesstime, selltime, 
                                   ma, ma_fast, ma_slow,
                                   macd, macd_diff, macd_sign,
                                   rsi, rsi_entry, rsi_exit, 
                                   stoch, stoch_signal, stoch_entry, stoch_exit,
                                   bb_low, bb_high, 
                                   mfi, mfi_entry, mfi_exit,
                                   candle_buy_signal_1, candle_buy_signal_2, candle_buy_signal_3,
                                   candle_sell_signal_1, candle_sell_signal_2, candle_sell_signal_3,
                                   candle_buy_sell_signal_1, candle_buy_sell_signal_2)
    return SuperAI_signal

#Parameters to optimize
parameters_names = ["ma_window", "ma_fast_window", "ma_slow_window", 
                    "macd_slow_window", "macd_fast_window", "macd_sign_window",  
                    "rsi_window", "rsi_entry", "rsi_exit",
                    "stoch_window", "stoch_smooth_window", "stoch_entry", "stoch_exit",
                    "bb_window", "bb_dev", 
                    "mfi_window", "mfi_entry", "mfi_exit"]

#Indicator
SuperAI_Ind = vbt.IndicatorFactory(
    class_name = "SuperAI_Ind",
    short_name = "SuperInd",
    input_names = ["open", "high", "low", "close", "volume", "buylesstime", "selltime"],
    param_names = parameters_names,
    output_names = ["output"]).from_apply_func(superai_signals,
                                              ma_window = ma_timeframe,
                                              ma_fast_window = ma_fast_timeframe,
                                              ma_slow_window = ma_slow_timeframe, 
                                              
                                              macd_slow_window = macd_slow_timeframe, 
                                              macd_fast_window = macd_fast_timeframe,
                                              macd_sign_window = macd_signal_timeframe,
                                              
                                              rsi_window = rsi_timeframe,
                                              rsi_entry = rsi_oversold_threshold, 
                                              rsi_exit = rsi_overbought_threshold,
                                              
                                              stoch_window = stoch_timeframe,
                                              stoch_smooth_window = stoch_smooth_timeframe,
                                              stoch_entry = stoch_oversold_threshold, 
                                              stoch_exit = stoch_overbought_threshold,
                                               
                                              bb_window = bb_timeframe,
                                              bb_dev = bb_dev,
                                              
                                              mfi_window = mfi_timeframe,
                                              mfi_entry = mfi_oversold_threshold, 
                                              mfi_exit = mfi_overbought_threshold)

def SuperAI_Backtester():
    #BACKTESTING WITH TRAINING AND VALIDATION SET
    print("\nI start the backtesting. The declared parameters at the moment are:\n")
    
    #Resetting the parameters to the declared ones
    starting_max_parameters()
       
    global ma_window_max, ma_fast_window_max, ma_slow_window_max 
    global macd_slow_window_max, macd_fast_window_max, macd_sign_window_max
    global rsi_window_max, rsi_entry_max, rsi_exit_max 
    global stoch_window_max, stoch_smooth_window_max, stoch_entry_max, stoch_exit_max
    global bb_window_max, bb_dev_max, mfi_window_max
    global mfi_entry_max, mfi_exit_max
    
    basic_returns = np.NaN
    optimized_returns = np.NaN
    validated_returns = np.NaN

    print("""     ma_window_max: {}, ma_fast_window_max: {}, ma_slow_window_max: {}, 
     macd_slow_window_max: {}, macd_fast_window_max: {}, macd_sign_window_max: {},
     rsi_window_max: {}, rsi_entry_max: {}, rsi_exit_max: {}, 
     stoch_window_max: {}, stoch_smooth_window_max: {}, stoch_entry_max: {}, stoch_exit_max: {},
     bb_window_max: {}, bb_dev_max: {}, 
     mfi_window_max: {}, mfi_entry_max: {}, mfi_exit_max: {}\n""".format(
     ma_window_max, ma_fast_window_max, ma_slow_window_max, 
     macd_slow_window_max, macd_fast_window_max, macd_sign_window_max,
     rsi_window_max, rsi_entry_max, rsi_exit_max, 
     stoch_window_max, stoch_smooth_window_max, stoch_entry_max, stoch_exit_max,
     bb_window_max, bb_dev_max, mfi_window_max, 
     mfi_entry_max, mfi_exit_max))

    print("\nI'm downloading the data for {} from {} to {}.\n".format(asset, data_start, data_end))
    
    #Remembering the backtested asset
    global backtested_asset
    backtested_asset = asset
    
    #Preparing data
    open_prices, high_prices, low_prices, close_prices, volume, buylesstime, selltime = prepare_data(data_start, data_end)

    print("I have the data. I start doing the calculations.\n")
    
    #TESTING THE PROTOTYPE
    trading_signals = SuperAI_Ind.run(open_prices, high_prices, low_prices, close_prices, volume, buylesstime, selltime,

                            ma_window = ma_timeframe,
                            ma_fast_window = ma_fast_timeframe,
                            ma_slow_window = ma_slow_timeframe,

                            macd_slow_window = macd_slow_timeframe,
                            macd_fast_window = macd_fast_timeframe,
                            macd_sign_window = macd_signal_timeframe,

                            rsi_window = rsi_timeframe,
                            rsi_entry = rsi_oversold_threshold,
                            rsi_exit= rsi_overbought_threshold,

                            stoch_window = stoch_timeframe,
                            stoch_smooth_window = stoch_smooth_timeframe,
                            stoch_entry = stoch_oversold_threshold, 
                            stoch_exit = stoch_overbought_threshold,

                            bb_window = bb_timeframe,
                            bb_dev = bb_dev,

                            mfi_window = mfi_timeframe,
                            mfi_entry = mfi_oversold_threshold,
                            mfi_exit= mfi_overbought_threshold,

                            param_product = True)

    entries = trading_signals.output == 1.0
    exits = trading_signals.output == -1.0

    SuperAI_portfolio = vbt.Portfolio.from_signals(close_prices, 
                                                   entries, 
                                                   exits, 
                                                   init_cash = 100000,   
                                                   tp_stop = take_profit_percent,
                                                   sl_stop = stop_loss_percent,
                                                   fees = 0.00)

    returns = SuperAI_portfolio.total_return() * 100
    basic_returns = returns
    stats = SuperAI_portfolio.stats()
    
    print("I did the backtest with previously declared indicators and parameters.\n")
    print("Returns before optimization: ", returns, "\n")
    print("Stats before optimization:")
    print(stats, "\n")

    #OPTIMIZING THE BOT WITH A GRID OF DIFFERENT POSSIBILITIES FOR PREFERRED PARAMETER
    if optimization == True:
        print("I start the optimization. The parameters before optimization are:\n")
                
        print("""         ma_window_max: {}, ma_fast_window_max: {}, ma_slow_window_max: {}, 
         macd_slow_window_max: {}, macd_fast_window_max: {}, macd_sign_window_max: {},
         rsi_window_max: {}, rsi_entry_max: {}, rsi_exit_max: {}, 
         stoch_window_max: {}, stoch_smooth_window_max: {}, stoch_entry_max: {}, stoch_exit_max: {},
         bb_window_max: {}, bb_dev_max: {}, 
         mfi_window_max: {}, mfi_entry_max: {}, mfi_exit_max: {}\n""".format(
         ma_window_max, ma_fast_window_max, ma_slow_window_max, 
         macd_slow_window_max, macd_fast_window_max, macd_sign_window_max,
         rsi_window_max, rsi_entry_max, rsi_exit_max, 
         stoch_window_max, stoch_smooth_window_max, stoch_entry_max, stoch_exit_max,
         bb_window_max, bb_dev_max, mfi_window_max, 
         mfi_entry_max, mfi_exit_max))
        
        print("        Now I'm doing the calculations. It may take me some time (it depends on the power of your computer)\n\
        and the amount of data I have to analyze. So, be patient and relax for now.\n")
        
        trading_signals = SuperAI_Ind.run(open_prices, high_prices, low_prices, close_prices, volume, buylesstime, selltime,

                            ma_window = ma_window_opt,
                            ma_fast_window = ma_fast_window_opt,
                            ma_slow_window = ma_slow_window_opt,

                            macd_slow_window = macd_slow_window_opt,
                            macd_fast_window = macd_fast_window_opt,
                            macd_sign_window = macd_sign_window_opt,

                            rsi_window = rsi_window_opt,
                            rsi_entry = rsi_entry_opt,
                            rsi_exit = rsi_exit_opt,

                            stoch_window = stoch_window_opt,
                            stoch_smooth_window = stoch_smooth_window_opt,
                            stoch_entry = stoch_entry_opt,
                            stoch_exit = stoch_exit_opt,

                            bb_window = bb_window_opt,
                            bb_dev = bb_dev_opt,

                            mfi_window = mfi_window_opt,
                            mfi_entry = mfi_entry_opt,
                            mfi_exit = mfi_exit_opt,

                            param_product = True)

        entries = trading_signals.output == 1.0
        exits = trading_signals.output == -1.0

        SuperAI_portfolio = vbt.Portfolio.from_signals(close_prices, 
                                                   entries, 
                                                   exits, 
                                                   init_cash = 100000,   
                                                   tp_stop = take_profit_percent,
                                                   sl_stop = stop_loss_percent,
                                                   fees = 0.00)

        stats_all = SuperAI_portfolio.stats()

        returns = SuperAI_portfolio.total_return() * 100
        
        max_dd = SuperAI_portfolio.max_drawdown()

        sharpe_ratio = SuperAI_portfolio.sharpe_ratio(freq='m')

        #APPLYING THE BEST PARAMETERS TO THE MODEL
        
        (ma_window_max, ma_fast_window_max, ma_slow_window_max, 
         macd_slow_window_max, macd_fast_window_max, macd_sign_window_max,
         rsi_window_max, rsi_entry_max, rsi_exit_max, 
         stoch_window_max, stoch_smooth_window_max, stoch_entry_max, stoch_exit_max,
         bb_window_max, bb_dev_max, mfi_window_max, 
         mfi_entry_max, mfi_exit_max) = returns.idxmax() #max_dd.idxmax() #sharpe_ratio.idxmax()
                
        trading_signals = SuperAI_Ind.run(open_prices, high_prices, low_prices, close_prices, volume, buylesstime, selltime,

                            ma_window = ma_window_max,
                            ma_fast_window = ma_fast_window_max,
                            ma_slow_window = ma_slow_window_max,

                            macd_slow_window = macd_slow_window_max,
                            macd_fast_window = macd_fast_window_max,
                            macd_sign_window = macd_sign_window_max,

                            rsi_window = rsi_window_max,
                            rsi_entry = rsi_entry_max,
                            rsi_exit= rsi_exit_max,

                            stoch_window = stoch_window_max,
                            stoch_smooth_window = stoch_smooth_window_max,
                            stoch_entry = stoch_entry_max, 
                            stoch_exit = stoch_exit_max,

                            bb_window = bb_window_max,
                            bb_dev = bb_dev_max,

                            mfi_window = mfi_window_max,
                            mfi_entry = mfi_entry_max,
                            mfi_exit= mfi_exit_max,

                            param_product = True)

        entries = trading_signals.output == 1.0
        exits = trading_signals.output == -1.0

        SuperAI_portfolio = vbt.Portfolio.from_signals(close_prices, 
                                                   entries, 
                                                   exits, 
                                                   init_cash = 100000,   
                                                   tp_stop = take_profit_percent,
                                                   sl_stop = stop_loss_percent,
                                                   fees = 0.00)

        opt_returns = SuperAI_portfolio.total_return() * 100
        optimized_returns = opt_returns

        opt_stats = SuperAI_portfolio.stats()

        print("I optimized the parameters.")
        print("The parameters after optimization are:\n")
        print("""         ma_window_max: {}, ma_fast_window_max: {}, ma_slow_window_max: {}, 
         macd_slow_window_max: {}, macd_fast_window_max: {}, macd_sign_window_max: {},
         rsi_window_max: {}, rsi_entry_max: {}, rsi_exit_max: {}, 
         stoch_window_max: {}, stoch_smooth_window_max: {}, stoch_entry_max: {}, stoch_exit_max: {},
         bb_window_max: {}, bb_dev_max: {}, 
         mfi_window_max: {}, mfi_entry_max: {}, mfi_exit_max: {}\n""".format(
         ma_window_max, ma_fast_window_max, ma_slow_window_max, 
         macd_slow_window_max, macd_fast_window_max, macd_sign_window_max,
         rsi_window_max, rsi_entry_max, rsi_exit_max, 
         stoch_window_max, stoch_smooth_window_max, stoch_entry_max, stoch_exit_max,
         bb_window_max, bb_dev_max, mfi_window_max, 
         mfi_entry_max, mfi_exit_max))
        
        print("\nMax returns after optimization: ", opt_returns, "\n")
        print("Stats after optimization:")
        print(opt_stats, "\n")
    else:
        print("You declared You don't want any optimization. I respect that and I'm not going to do any optimization.\n")
        
    #VALIDATION OF THE MODEL
    if validation == True:
        print("I start the validation. I'm downloading the data for {} from {} to {}.".format(asset, valid_start, valid_end))
        print("For previous calculations I used data for {} from {} to {}.".format(asset, data_start, data_end))
        
        open_prices, high_prices, low_prices, close_prices, volume, buylesstime, selltime = prepare_data(valid_start, valid_end)

        print("I've got the data. I'm starting the calculations.\n")
        trading_signals = SuperAI_Ind.run(open_prices, high_prices, low_prices, close_prices, volume, buylesstime, selltime,

                            ma_window = ma_window_max,
                            ma_fast_window = ma_fast_window_max,
                            ma_slow_window = ma_slow_window_max,

                            macd_slow_window = macd_slow_window_max,
                            macd_fast_window = macd_fast_window_max,
                            macd_sign_window = macd_sign_window_max,

                            rsi_window = rsi_window_max,
                            rsi_entry = rsi_entry_max,
                            rsi_exit= rsi_exit_max,

                            stoch_window = stoch_window_max,
                            stoch_smooth_window = stoch_smooth_window_max,
                            stoch_entry = stoch_entry_max, 
                            stoch_exit = stoch_exit_max,

                            bb_window = bb_window_max,
                            bb_dev = bb_dev_max,

                            mfi_window = mfi_window_max,
                            mfi_entry = mfi_entry_max,
                            mfi_exit= mfi_exit_max,

                            param_product = True)

        entries = trading_signals.output == 1.0
        exits = trading_signals.output == -1.0

        SuperAI_portfolio = vbt.Portfolio.from_signals(close_prices, 
                                                   entries, 
                                                   exits, 
                                                   init_cash = 100000,   
                                                   tp_stop = take_profit_percent,
                                                   sl_stop = stop_loss_percent,
                                                   fees = 0.00)

        val_returns = SuperAI_portfolio.total_return() * 100
        validated_returns = val_returns
        val_stats = SuperAI_portfolio.stats()

        print("I've finished the validation.\n")
        print("Returns from validation: ", val_returns, "\n")
        print("Stats from validation:")
        print(val_stats, "\n")
    
    else:
        print("You declared You don't want to do any validation. Ok, I won't do any.\n")
        
    strategy_returns[0] = cs_name
    strategy_returns[1] = basic_returns
    strategy_returns[2] = optimized_returns
    strategy_returns[3] = validated_returns
    strategy_to_df = pd.DataFrame([strategy_returns], columns=columns)
    
    global strategies
    strategies = pd.concat([strategies, strategy_to_df])

    print("""    Now You have to decide whether we go with this strategy to live paper trading 
    or you want to try another strategy. Whatever your decision is, I'm here for you.\n\n""")     
    

# 2. ALL THE OTHER FUNCTIONS AND VARIABLES OF THE BOT NEEDED FOR LIVE TRADING WITH PAPER TRADING ACCOUNT
#Taking profit
profit_ratio = 100 + (take_profit_percent * 100)
def take_profit(close_price, sell_order_filled):
    if take_profit_automatically == True:
        try:
            position = api.get_position(asset)
            aep = float(api.get_position(asset).avg_entry_price)

            if sell_order_filled == False:
                if close_price >= aep * profit_ratio / 100:
                    n_shares = float(position.qty)
                    api.submit_order(symbol=asset,qty=n_shares,side='sell',type='market',time_in_force='gtc')
                    print("Take profit price is {}% from {:.2f}$ we paid for 1 {} = {:.2f}$. "
                      .format(profit_ratio, aep, asset, aep * profit_ratio / 100))
                    print('The current {:.2f}$ is good enough. We take profit with an order to sell {} shares/coins of {}.'
                          .format(close_price, n_shares, asset))
                else:
                    print('Take profit price is {}% from the price we used for buying: {:.2f}$ for 1 {} and that is {:.2f}$.'
                      .format(profit_ratio, aep, asset, aep * profit_ratio / 100))
                    print('Last close price {:.2f}$ is not enough.'.format(close_price))
        except:
            pass

        print()
    else:
        pass
    
#Stopping loss
stoploss_ratio = 100 - (stop_loss_percent * 100)
def stop_loss(close_price, sell_order_filled):
    if stop_loss_automatically == True:
        try:
            position = api.get_position(asset)
            aep = float(api.get_position(asset).avg_entry_price)

            if sell_order_filled == False:
                if close_price < aep * stoploss_ratio / 100:
                    n_shares = float(position.qty)
                    api.submit_order(symbol=asset,qty=n_shares,side='sell',type='market',time_in_force='gtc')
                    print("Stop loss price is {}% from {:.2f}$ we paid for 1 {} = {:.2f}$."
                      .format(stoploss_ratio, aep, asset, aep * stoploss_ratio / 100))
                    print('The current {:.2f}$ is less. We stop loss with an order to sell {} shares/coins of {}.'
                          .format(close_price, n_shares, asset))
                else:
                    print("Stop loss price is {}% from the price we used for buying: {:.2f}$ for 1 {} and that is {:.2f}$."
                      .format(stoploss_ratio, aep, asset, aep * stoploss_ratio / 100))
                    print("Last close price {:.2f}$ is not that low.".format(close_price))
        except:
            pass

        print()
    else:
        pass

# Caclulating Technical Indicators and Candlestick Patterns Signals
def cal_tech_ind(open_prices, high_prices, low_prices, close_prices, volume, buylesstime, selltime):
    #CALCULATING TECHNICAL INDICATORS SIGNALS    
    close_price = close_prices[-1]
    #Calculating MA Signals
    try:
        ma = ta.trend.ema_indicator(close_prices, window = ma_window_max)
        ma = np.round(ma, 2)
        ma_last = float(ma.iloc[-1])

        ma_fast = ta.trend.ema_indicator(close_prices, window = ma_fast_window_max)
        ma_fast = np.round(ma_fast, 2)
        ma_fast_last = float(ma_fast.iloc[-1])

        ma_slow = ta.trend.ema_indicator(close_prices, window = ma_slow_window_max)
        ma_slow = np.round(ma_slow, 2)
        ma_slow_last = float(ma_slow.iloc[-1])

        print("Last MA is: {:.3f}. Last Fast MA is: {:.3f}. Last Slow MA is: {:.3f}.\n".format(ma_last, 
                                                                                ma_fast_last, ma_slow_last))
    except:
        print ("MA signal doesn't work.\n")
        
    #Calculating MACD Signal
    try:
        macd = ta.trend.macd(close_prices, window_slow = macd_slow_window_max, window_fast = macd_fast_window_max)
        macd = np.round(macd, 2)
        macd_last = float(macd.iloc[-1])
        
        macd_diff = ta.trend.macd_diff(close_prices, window_slow = macd_slow_window_max, 
                                       window_fast = macd_fast_window_max, window_sign = macd_sign_window_max)
        macd_diff = np.round(macd_diff, 2)
        macd_diff_last = float(macd_diff.iloc[-1])
        
        macd_sign = ta.trend.macd_signal(close_prices, window_slow = macd_slow_window_max, 
                                       window_fast = macd_fast_window_max, window_sign = macd_sign_window_max)
        macd_sign = np.round(macd_sign, 2)
        macd_sign_last = float(macd_sign.iloc[-1])

        print("Last MACD is: {:.3f}. Last MACD_DIFF is: {:.3f}. Last MACD_SIGNAL is: {:.3f}.\n".format(macd_last, 
                                                                                   macd_diff_last, macd_sign_last))
    except:
        print ("MACD signal doesn't work.\n")
        
    #Calculating RSI Signal
    try:
        rsi = ta.momentum.rsi(close_prices, window = rsi_window_max)
        rsi = np.round(rsi, 2)
        rsi_last = rsi.iloc[-1]
        rsi_entry = rsi_entry_max
        rsi_exit = rsi_exit_max
        
        print("Last RSI is {:.3f}. RSI thresholds are: {:.2f} - {:.2f}.\n".format(rsi_last, rsi_entry, rsi_exit))
    except:
        print("RSI signal doesn't work.\n")

    #Calculating Stochastic Signal
    try:
        stoch = ta.momentum.stoch(high_prices, low_prices, close_prices, 
                                window = stoch_window_max, smooth_window = stoch_smooth_window_max)
        stoch = np.round(stoch, 2)
        stoch_last = stoch.iloc[-1]
        
        stoch_sign = ta.momentum.stoch_signal(high_prices, low_prices, close_prices, 
                                window = stoch_window_max, smooth_window = stoch_smooth_window_max)
        stoch = np.round(stoch_sign, 2)
        stoch_sign_last = stoch_sign.iloc[-1]
        
        stoch_entry = stoch_entry_max
        stoch_exit = stoch_exit_max
        
        print("Last Stochastic is {:.3f}. Stochastic thresholds are: {:.2f} - {:.2f}.\n".format(
            stoch_last, stoch_entry, stoch_exit))
        print("Last Stochastic Signal is {:.3f}. Stochastic thresholds are: {:.2f} - {:.2f}.\n".format(
            stoch_sign_last, stoch_entry, stoch_exit))
    except:
        print("Stochastic signal doesn't work.\n")
    
    #Calculating Bollinger Bands Signal
    try:
        bb_low = ta.volatility.bollinger_lband(close_prices, window=bb_window_max, window_dev = bb_dev_max)
        bb_low = np.round(bb_low, 2)
        bb_high = ta.volatility.bollinger_hband(close_prices, window=bb_window_max, window_dev = bb_dev_max)
        bb_high = np.round(bb_high, 2)
        
        bb_low_last = float(bb_low.iloc[-1])
        bb_high_last = float(bb_high.iloc[-1])

        print("Last price is: {}$. Bollinger Bands are: Lower: {:.3f}, Upper: {:.3f}.\n".format(close_price, 
                                                                                              bb_low_last, bb_high_last))
    except:
        print ("Bollinger Bands signal doesn't work.\n")
        
    #Calculating MFI Signal
    try:
        mfi = ta.volume.money_flow_index(high_prices, low_prices, close_prices, volume, window = mfi_window_max)
        mfi = np.round(mfi, 2)
        mfi_last = mfi.iloc[-1]
        mfi_entry = mfi_entry_max
        mfi_exit = mfi_exit_max

        print("Last MFI is {:.3f}. MFI thresholds are: {:.2f} - {:.2f}.\n".format(mfi_last, mfi_entry, mfi_exit))
    except:
        print("MFI signal doesn't work.\n")
        
    return (ma, ma_fast, ma_slow, 
                  macd, macd_diff, macd_sign,
                  rsi, rsi_entry, rsi_exit, 
                  stoch, stoch_sign, stoch_entry, stoch_exit, 
                  bb_low, bb_high, 
                  mfi, mfi_entry, mfi_exit)

#SuperAI Trading Bot
def cal_can_pat(open_prices, high_prices, low_prices, close_prices, volume, buylesstime, selltime):
    #CALCULATING CANDLESTICK PATTERNS AND SIGNALS    
    #Hammer
    candle_buy_signal_1 = ta_lib.CDLHAMMER(open_prices, high_prices, low_prices, close_prices)
    last_candle_buy_signal_1 = candle_buy_signal_1.iloc[-1]
    print("Last Candle Buy Signal 1: {}.".format(last_candle_buy_signal_1))

    #Morning Star
    candle_buy_signal_2 = ta_lib.CDLMORNINGSTAR(open_prices, high_prices, low_prices, close_prices)
    last_candle_buy_signal_2 = candle_buy_signal_2.iloc[-1]
    print("Last Candle Buy Signal 2: {}.".format(last_candle_buy_signal_2))
    
    #Three White Soldiers
    candle_buy_signal_3 = ta_lib.CDL3WHITESOLDIERS(open_prices, high_prices, low_prices, close_prices)
    last_candle_buy_signal_3 = candle_buy_signal_3.iloc[-1]
    print("Last Candle Buy Signal 3: {}.".format(last_candle_buy_signal_3))
    
    #Shooting Star
    candle_sell_signal_1 = ta_lib.CDLSHOOTINGSTAR(open_prices, high_prices, low_prices, close_prices)
    last_candle_sell_signal_1 = candle_sell_signal_1.iloc[-1]
    print("Last Candle Sell Signal 1: {}.".format(last_candle_sell_signal_1))
    
    #Evening Star
    candle_sell_signal_2 = ta_lib.CDLEVENINGSTAR(open_prices, high_prices, low_prices, close_prices)
    last_candle_sell_signal_2 = candle_sell_signal_2.iloc[-1]
    print("Last Candle Sell Signal 2: {}.".format(last_candle_sell_signal_2))
    
    #3 Black Crows
    candle_sell_signal_3 = ta_lib.CDL3BLACKCROWS(open_prices, high_prices, low_prices, close_prices)
    last_candle_sell_signal_3 = candle_sell_signal_3.iloc[-1]
    print("Last Candle Sell Signal 3: {}.".format(last_candle_sell_signal_3))
    
    #Engulfing (Bullish (buy) / Bearish (Sell))
    candle_buy_sell_signal_1 = ta_lib.CDLENGULFING(open_prices, high_prices, low_prices, close_prices)
    last_candle_buy_sell_signal_1 = candle_buy_sell_signal_1.iloc[-1]
    print("Last Candle Buy Sell Signal 1: {}.".format(last_candle_buy_sell_signal_1))
    
    #Three Outside: Up (buy) / Down (sell)
    candle_buy_sell_signal_2 = ta_lib.CDL3OUTSIDE(open_prices, high_prices, low_prices, close_prices)
    last_candle_buy_sell_signal_2 = candle_buy_sell_signal_2.iloc[-1]
    print("Last Candle Buy Sell Signal 2: {}.".format(last_candle_buy_sell_signal_2))
    
    return (candle_buy_signal_1, candle_buy_signal_2, candle_buy_signal_3,
                  candle_sell_signal_1, candle_sell_signal_2, candle_sell_signal_3,
                  candle_buy_sell_signal_1, candle_buy_sell_signal_2)

# Check if market is open
def check_if_market_open():
    if api.get_clock().is_open == False:
        print("The market is closed at the moment.")
        print("Time to open is around: {:.0f} minutes. So I'll stop working for now. Hope you don't mind."
              .format((api.get_clock().next_open.timestamp()- api.get_clock().timestamp.timestamp())/60))
        sys.exit("I'm out. Turn me back on when it's time. Yours, SuperAI trader.")
    else:
        pass

#Buyless time
def buyless_time(time_now):
    (trading_hour_start, trading_hour_stop, 
            buyless_time_start_1, buyless_time_end_1, buyless_time_start_2, buyless_time_end_2,
            selltime_start, selltime_end) = trading_buy_sell_time()

    buyless1_start = datetime.time(int(buyless_time_start_1[:2]), int(buyless_time_start_1[3:]))
    buyless1_end = datetime.time(int(buyless_time_end_1[:2]), int(buyless_time_end_1[3:]))
    buyless2_start = datetime.time(int(buyless_time_start_2[:2]), int(buyless_time_start_2[3:]))
    buyless2_end = datetime.time(int(buyless_time_end_2[:2]), int(buyless_time_end_2[3:]))

    buylesstime = (buyless1_start < time_now < buyless1_end) | (buyless2_start < time_now < buyless2_end) 
    print('is it buyless time? ', buylesstime)
    return buylesstime

#Sell time
def sell_time(time_now):
    (trading_hour_start, trading_hour_stop, 
                buyless_time_start_1, buyless_time_end_1, buyless_time_start_2, buyless_time_end_2,
                selltime_start, selltime_end) = trading_buy_sell_time()

    sell_time_start = datetime.time(int(selltime_start[:2]), int(selltime_start[3:]))
    sell_time_end = datetime.time(int(selltime_end[:2]), int(selltime_end[3:]))    

    selltime = (sell_time_start < time_now < sell_time_end) 
    print('is it sell time? ', selltime)
    return selltime

#Waiting for a bar to close
#check for more: https://alpaca.markets/learn/code-cryptocurrency-live-trading-bot-python-alpaca/
def wait_for_bar_to_close():
    time_now = datetime.datetime.now()
    next_min = time_now.replace(second=5, microsecond=0) + timedelta(minutes=1)
    pause = math.ceil((next_min - time_now).seconds)
    print("I'll wait {} seconds for the bar to close.".format(pause))
    print("\n* * * * * * * * * * * * * * * * * * * * * * * * *\n")
    
    return pause

# Function to run after starting the bot (on_open function) 
def on_open():
    print("\nI'm connected to Alpaca API and ready to work. I'm starting to watch the prices.\n")  
    cash = float(api.get_account().cash)
    print("We have {:.2f}$ cash.".format(cash))
    
    try:
        position = api.get_position(asset)
        n_shares = float(position.qty)
        print("We have {} shares/coins of {}.\n".format(n_shares, asset))
    except:
        print("We don't have any shares/coins of {} at the moment.\n".format(asset))

    funds = cash * funds_percentage / 100
    print("Funds we will use for trading: {:.2f}$.\n".format(funds))
    print("I will be trading {}.\n".format(asset))
    try:
        print("The last backtest I did was for {}.\n".format(backtested_asset))
    except:
        print("I didn't do any backtesting yet.\n")
        
    global trading_hour_start, trading_hour_stop
    global buyless_time_start_1, buyless_time_end_1, buyless_time_start_2, buyless_time_end_2
    global selltime_start, selltime_end
     
    (trading_hour_start, trading_hour_stop, 
            buyless_time_start_1, buyless_time_end_1, buyless_time_start_2, buyless_time_end_2,
            selltime_start, selltime_end) = trading_buy_sell_time()
    
    print("""    I will be trading between {} and {}. 
    I won't be buying between {} and {} and between {} and {}.
    I will sell the shares/coins of {} when it's between {} and {} no matter the trading signals.\n""".format(
            trading_hour_start, trading_hour_stop, 
            buyless_time_start_1, buyless_time_end_1, buyless_time_start_2, buyless_time_end_2, asset,
            selltime_start, selltime_end, "\n"))  
    
    if take_profit_automatically:
        print("Take profit is set to +{}% from the average entry price.".format(take_profit_percent*100))
        print("I will be trading when the technical indicators and candlestick patterns say so, but also")
        print("if entry price is e.g. 100$ I'll automatically sell when last close price is more than 100$+{}%*100$={:.2f}$"
                          .format(take_profit_percent * 100, 100*profit_ratio/100))
    else:
        print("Take profit automatically is turned off.")
        print("I will use technical indicators and candlestick patterns to get as much profit as I can.")
    
    if stop_loss_automatically:
        print("\nStop loss is set to -{}% from the average entry price.".format(stop_loss_percent*100))
        print("I will be trading when the technical indicators and candlestick patterns say so, but also")
        print("if entry price is e.g. 100$ I'll automatically sell when last close price is less than 100$-{}%*100$={:.2f}$"
                          .format(stop_loss_percent * 100, 100*stoploss_ratio/100))
    else:
        print("\nStop loss automatically is turned off.")
        print("I will use technical indicators and candlestick patterns so I don't lose money.")
        
    print("\nSo, here we go. Wish me luck.\n")
    print("* * * * * * * * * * * * * * * * * * * * * * * * *\n")
    
    if asset_type == "stock":
            check_if_market_open()
            
# Function to run after every message from Alpaca (on_message function)
def on_message():
    nyc_datetime = api.get_clock().timestamp.tz_convert('America/New_York')
    print("New York time:", str(nyc_datetime)[:16])
    
    open_prices, high_prices, low_prices, close_prices, volume, buylesstime, selltime = prepare_data(
                                                                str(datetime.date.today() - datetime.timedelta(days = 2)), 
                                                                str(datetime.date.today() +  datetime.timedelta(days = 2)))
    close_price = close_prices[-1]
    
    print("Close price of {}: {:.2f}$\n".format(asset, close_price))
    
    try:
        position = api.get_position(asset)
        n_shares = float(position.qty)
        print("We have {} shares/coins of {}.\n".format(n_shares, asset))
    except:
        print("We don't have any shares/coins of {} at the moment.\n".format(asset))

    cash = float(api.get_account().cash)
    print("We have {:.2f}$ cash.".format(cash))
    funds = cash * funds_percentage / 100
    print("Funds we will use for trading: {:.2f}$.\n".format(funds))

    #CALCULATING TECHNICAL INDICATORS SIGNALS
    (ma, ma_fast, ma_slow, macd, macd_diff, macd_sign, rsi, rsi_entry, rsi_exit, 
         stoch, stoch_sign, stoch_entry, stoch_exit, 
         bb_low, bb_high, mfi, mfi_entry, mfi_exit) = cal_tech_ind(
                                            open_prices, high_prices, low_prices, close_prices, volume, 
                                            buylesstime, selltime)

    #CALCULATING CANDLESTICK PATTERNS AND SIGNALS
    (candle_buy_signal_1, candle_buy_signal_2, candle_buy_signal_3,
    candle_sell_signal_1, candle_sell_signal_2, candle_sell_signal_3,
    candle_buy_sell_signal_1, candle_buy_sell_signal_2) = cal_can_pat(
                                            open_prices, high_prices, low_prices, close_prices, volume, 
                                            buylesstime, selltime)
    
    #Calculate final trade signal
    try:
        final_trade_signals = create_signal(open_prices, high_prices, low_prices, close_prices, volume,
                  buylesstime, selltime, 
                  ma, ma_fast, ma_slow, 
                  macd, macd_diff, macd_sign,
                  rsi, rsi_entry, rsi_exit, 
                  stoch, stoch_sign, stoch_entry, stoch_exit, 
                  bb_low, bb_high, 
                  mfi, mfi_entry, mfi_exit,
                  candle_buy_signal_1, candle_buy_signal_2, candle_buy_signal_3,
                  candle_sell_signal_1, candle_sell_signal_2, candle_sell_signal_3,
                  candle_buy_sell_signal_1, candle_buy_sell_signal_2)

        final_trade_signal = final_trade_signals[-1]
        if final_trade_signal == 0:
            print("\nFinal trade signal is: DO NOTHING\n")
        elif final_trade_signal == 1:
            print("\nFinal trade signal is: BUY\n")
        elif final_trade_signal == -1:
            print("\nFinal trade signal is: SELL\n")    
    except:
        print("\nFinal trade signal doesn't work.\n")
        final_trade_signal = False

    #Execute action after recieving the final trade signal: submitting an order
    sell_order_filled = False
    if final_trade_signal == 1: #"buy":
        try:
            api.get_position(asset)
            print("We hit the threshold to buy, but we already have some shares/coins, so we won't buy more.\n")
        except:
            n_shares = np.round(funds // close_price)
            if asset_type == 'crypto':
                #an overcomplicated way to round down the number of coins to buy to the declared number of places after comma
                n_shares = funds//close_price + float((str((funds / close_price)-(funds//close_price))[:(2+rounding)]))
            api.submit_order(symbol=asset,qty=n_shares,side="buy",type="market",time_in_force="gtc")
            print('We submitted the order to buy {} {} shares/coins.'.format(n_shares, asset))

            try:
                position = api.get_position(asset)
                aep = float(api.get_position(asset).avg_entry_price)
                print("The last close price was {}. We bought {} shares/coins of {} for the price of: {}$ for 1 {}.\n".format(
                    close_price, n_shares, asset, aep, asset))
            except:
                pass        
    
    elif final_trade_signal == -1: #"sell":
        try:
            position = api.get_position(asset)
            n_shares = float(position.qty)
            api.submit_order(symbol=asset,qty=n_shares,side='sell',type='market',time_in_force='gtc')
            sell_order_filled = True
            print('We submitted an order to sell {} {} shares/coins.'.format(n_shares, asset))
        except:
            print("We hit the threshold to sell, but we don't have anything to sell. Next time maybe.\n")

    else:
        print("The signal was inconclusive - probably indicators showed us we should wait, so we wait.\n")
    
    #Hand-made take profit
    take_profit(close_price, sell_order_filled)

    #Hand-made stop loss
    stop_loss(close_price, sell_order_filled)

    print("\n* * * * * * * * * * * * * * * * * * * * * * * * *\n")
      
    if asset_type == "stock":
            check_if_market_open()
            
# Function to run the bot until break or until the market is closed
def SuperAI_Trading_Bot():
    on_open()
    time.sleep(wait_for_bar_to_close())
        
    while True:
        on_message()
        time.sleep(wait_for_bar_to_close())
        
    print("You've interrupted me. That's it than. I hope I did good. Till the next time.")

"""
And now we have all we need. Now we can run the bot.

So, here we go!

HERE WE START RUNNING THE BOT!
DO SOME BACKTESTING
with 1 line of code if you want to backtest with the asset you've chosen previously
with 2 lines of code if you want to change the asset to backtest
"""
#SuperAI Trading Bot - optional cell

#the asset was declared at the beginning of the program, but you can change it here if you want

#(asset, asset_type, rounding) = ("BTCUSD", "crypto", 0)
#(asset, asset_type, data_source) = ("AAPL", "stock", "Yahoo")

SuperAI_Backtester()
"""
START LIVE TRADING WITH PAPER TRADING ACCOUNT
Run the bot to trade with crypto (e.g. BTCUSD) or stock (e.g. AAPL)
with 1 line of code if you want to trade the asset you've chosen previously
with 2 lines of code if you want to change the asset to trade
If you want to trade crypto, you have to choose one that Alpaca supports at the moment: https://alpaca.markets/support/what-cryptocurrencies-does-alpaca-currently-support/

You should also check the minimum quantity you can trade.

If you want to trade stocks you may choose ticker of the company you prefer: https://www.nyse.com/listings_directory/stock

Your data_source can be 'Alpaca' or 'Yahoo', just know that:

Alpaca gives you free access to more historical data, but in a free plan doesn't allow you to access data from last 15 minutes
Yahoo gives you access to data from last 15 minutes, but gives you only 7 days of historical data with 1-min interval at a time.
"""
#SuperAI Trading Bot - 7. mandatory cell to run

#the asset was declared at the beginning of the program, but you can change it here if you want
#(asset, asset_type, rounding) = ("BTCUSD", "crypto", 0)
#(asset, asset_type, data_source) = ("AAPL", "stock", "Yahoo")

"""
SuperAI_Trading_Bot()
And that's it. Now we can start testing different strategies.

TESTING THE 13 STRATEGIES
The 13 tested trading strategies are:
BUY when CLOSE PRICE crosses over MA ==> SELL when CLOSE PRICE crosses under MA
BUY when FAST MA crosses over SLOW MA ==> SELL when FAST MA crosses under SLOW MA
BUY when MACD crosses over MACD SIGNAL ==> SELL when MACD crosses under MACD SIGNAL
BUY when RSI crosses under RSI OVERSOLD THRESHOLD ==> SELL when RSI crosses over RSI OVERBOUGHT THRESHOLD
BUY when STOCHASTIC crosses under STOCHASTIC OVERSOLD THRESHOLD ==> SELL when STOCHASTIC crosses over STOCHASTIC OVERBOUGHT THRESHOLD
BUY when CLOSE PRICE crosses under LOWER BOLLINGER BAND ==> SELL when CLOSE PRICE crosses over HIGHER BOLLINGER BAND
BUY when MFI crosses under MFI ENTRY THRESHOLD ==> SELL when MFI crosses over MFI EXIT THRESHOLD
BUY when at least 1 of 5 CANDLESTICK PATTERNS shows to BUY ==> SELL when at least 1 of 5 CANDLESTICK PATTERNS shows to SELL
BUY when 1 of 7 TECHNICAL INDICATORS shows to BUY (CP > MA or FAST MA > SLOW MA or MACD > MACD SIGNAL or RSI > RSI ENTRY or STOCHASTIC > STOCHASTIC ENTRY or CP < BB LOW or MFI > MFI ENTRY) AND 1 of 5 CANDLESTICK PATTERNS shows to BUY ==> SELL when 1 of 7 TECHNICAL INDICATORS shows to SELL (CP < MA or FAST MA < SLOW MA or MACD < MACD SIGNAL or RSI > RSI EXIT or STOCHASTIC > STOCHASTIC EXIT or CP > BB HIGH or MFI > MFI EXIT) or 1 of 7 CANDLESTICK PATTERNS shows to SELL
BUY when ALL 7 TECHNICAL INDICATORS show to BUY (CP > MA and FAST MA > SLOW MA and MACD > MACD SIGNAL and RSI > RSI ENTRY and STOCHASTIC > STOCHASTIC ENTRY and CP < BB LOW, and MFI > MFI ENTRY) AND 1 of 5 CANDLESTICK PATTERNS shows to BUY ==> SELL when 1 of 7 TECHNICAL INDICATORS shows to SELL (CP < MA or FAST MA < SLOW MA or MACD < MACD SIGNAL or RSI > RSI EXIT or STOCHASTIC > STOCHASTIC EXIT or CP > BB HIGH or MFI > MFI EXIT) or 1 of 7 CANDLESTICK PATTERNS shows to SELL
BUY when MACD crosses over MACD SIGNAL only when MACD < 0 ==> SELL when MACD crosses under MACD SIGNAL
BUY when MACD crosses over MACD SIGNAL only when MACD < 0 and (MA is higher than MA from 2 minutes before and MA from 2 minutes before is higher than MA from 4 minutes before) ==> SELL when (STOCHASTIC is over STOCHASTIC EXIT and RSI over RSI EXIT) or (CP over BB HIGH and MFI over MFI EXIT)
BUY when last CLOSE PRICE is higher than the CLOSE PRICE from a minute before ==> SELL when last CLOSE PRICE is lower than the CLOSE PRICE from a minute before
and never buy when the time is set to 'buyless' (e.g., first 15 minutes of trading session)
and always sell when the time is set to 'selltime' (e.g., 5 minutes till the trading session ends)
and use take profit and stop loss to maximize gains and minimize risks
* '>' here represents the crossing over

* '<' here represents the crossing under
"""
#SuperAI Trading Bot - optional cell
# the same variables that were previously declared, copied here for convenience, 
# so we don't have to go to the beginning of the program every time we want to change them

(asset, asset_type, data_source) = ("AAPL", "stock", "Yahoo")
#(asset, asset_type, rounding) = ("BTCUSD", "crypto", 0)

optimization = False
validation = True

(data_start, data_end) = ('2022-07-01', '2022-07-31')
(valid_start, valid_end) = ('2022-08-01', '2022-08-15')
#SuperAI Trading Bot - optional cell
#DataFrame for testing various strategies
"""
strategies
1. TRADING STRATEGY
BUY when CLOSE PRICE crosses over MA
==> SELL when CLOSE PRICE crosses under MA
and never buy when the time is set to 'buyless' (e.g., first 15 minutes of trading session)
and always sell when the time is set to 'selltime' (e.g., 5 minutes till the trading session ends)
and use take profit and stop loss to maximize gains and minimize risks
crossing.png
"""

#SuperAI Trading Bot - optional cell
# 1. STRATEGY
# BUY when CLOSE PRICE crosses over MA ==> SELL when CLOSE PRICE crosses under MA

def create_signal(open_prices, high_prices, low_prices, close_prices, volume,
                  buylesstime, selltime, #Time to abstain from buying and forced selling
                  ma, ma_fast, ma_slow, #Moving Average
                  macd, macd_diff, macd_sign, #Moving Average Convergence Divergence
                  rsi, rsi_entry, rsi_exit, #Relative Strength Index
                  stoch, stoch_signal, stoch_entry, stoch_exit, #Stochastic
                  bb_low, bb_high,  #Bollinger Bands
                  mfi, mfi_entry, mfi_exit, #Money Flow Index
                  candle_buy_signal_1, candle_buy_signal_2, candle_buy_signal_3, #Candle signals to buy
                  candle_sell_signal_1, candle_sell_signal_2, candle_sell_signal_3, #Candle signals to sell
                  candle_buy_sell_signal_1, candle_buy_sell_signal_2): #Candle signals to buy or sell
    
    
    SuperAI_signal_buy = np.where( 
                                  (buylesstime != 1) 
                                  &
                                  ((close_prices > ma) & (shifted(close_prices, 1) <= shifted(ma, 1)))
                                  , 1, 0) #1 is buy, -1 is sell, 0 is do nothing
    SuperAI_signal = np.where( 
                                  (selltime == 1)
                                  |
                                  ((close_prices < ma) & (shifted(close_prices, 1) >= shifted(ma, 1)))
                                  , -1, SuperAI_signal_buy) #1 is buy, -1 is sell, 0 is do nothing
    
    global cs_name
    cs_name = 'Strategy 1 with Close Prices crossing Moving Average'
    
    return SuperAI_signal
#SuperAI Trading Bot - optional cell

SuperAI_Backtester()

#2. TRADING STRATEGY
"""
BUY when FAST MA crosses over SLOW MA
==> SELL when FAST MA crosses under SLOW MA
and never buy when the time is set to 'buyless' (e.g., first 15 minutes of trading session)
and always sell when the time is set to 'selltime' (e.g., 5 minutes till the trading session ends)
and use take profit and stop loss to maximize gains and minimize risks
#SuperAI Trading Bot - optional cell
# 2. STRATEGY
# BUY when FAST MA crosses over SLOW MA ==> SELL when FAST MA crosses under SLOW MA
"""

def create_signal(open_prices, high_prices, low_prices, close_prices, volume,
                  buylesstime, selltime, #Time to abstain from buying and forced selling
                  ma, ma_fast, ma_slow, #Moving Average
                  macd, macd_diff, macd_sign, #Moving Average Convergence Divergence
                  rsi, rsi_entry, rsi_exit, #Relative Strength Index
                  stoch, stoch_signal, stoch_entry, stoch_exit, #Stochastic
                  bb_low, bb_high,  #Bollinger Bands
                  mfi, mfi_entry, mfi_exit, #Money Flow Index
                  candle_buy_signal_1, candle_buy_signal_2, candle_buy_signal_3, #Candle signals to buy
                  candle_sell_signal_1, candle_sell_signal_2, candle_sell_signal_3, #Candle signals to sell
                  candle_buy_sell_signal_1, candle_buy_sell_signal_2): #Candle signals to buy or sell
    
    
    SuperAI_signal_buy = np.where( 
                                  (buylesstime != 1) 
                                  &
                                  ((ma_fast > ma_slow) & (shifted(ma_fast, 1) <= shifted(ma_slow, 1)))
                                  , 1, 0) #1 is buy, -1 is sell, 0 is do nothing
    SuperAI_signal = np.where( 
                                  (selltime == 1)
                                  |
                                  ((ma_fast < ma_slow) & (shifted(ma_fast, 1) >= shifted(ma_slow, 1)))
                                  , -1, SuperAI_signal_buy) #1 is buy, -1 is sell, 0 is do nothing
    
    global cs_name
    cs_name = 'Strategy 2 with Fast MA crossing Slow MA'
    
    return SuperAI_signal
#SuperAI Trading Bot - optional cell

SuperAI_Backtester()
"""
3. TRADING STRATEGY
BUY when MACD crosses over MACD SIGNAL
==> SELL when MACD crosses under MACD SIGNAL
and never buy when the time is set to 'buyless' (e.g., first 15 minutes of trading session)
and always sell when the time is set to 'selltime' (e.g., 5 minutes till the trading session ends)
and use take profit and stop loss to maximize gains and minimize risks
#SuperAI Trading Bot - optional cell
"""
# 3. STRATEGY
# BUY when MACD crosses over MACD SIGNAL ==> SELL when MACD crosses under MACD SIGNAL

def create_signal(open_prices, high_prices, low_prices, close_prices, volume,
                  buylesstime, selltime, #Time to abstain from buying and forced selling
                  ma, ma_fast, ma_slow, #Moving Average
                  macd, macd_diff, macd_sign, #Moving Average Convergence Divergence
                  rsi, rsi_entry, rsi_exit, #Relative Strength Index
                  stoch, stoch_signal, stoch_entry, stoch_exit, #Stochastic
                  bb_low, bb_high,  #Bollinger Bands
                  mfi, mfi_entry, mfi_exit, #Money Flow Index
                  candle_buy_signal_1, candle_buy_signal_2, candle_buy_signal_3, #Candle signals to buy
                  candle_sell_signal_1, candle_sell_signal_2, candle_sell_signal_3, #Candle signals to sell
                  candle_buy_sell_signal_1, candle_buy_sell_signal_2): #Candle signals to buy or sell
    
    
    SuperAI_signal_buy = np.where( 
                                  (buylesstime != 1) 
                                  &
                                  ((macd > macd_sign) & (shifted(macd, 1) <= shifted(macd_sign, 1)))
                                  , 1, 0) #1 is buy, -1 is sell, 0 is do nothing
    SuperAI_signal = np.where( 
                                  (selltime == 1)
                                  |
                                  ((macd < macd_sign) & (shifted(macd, 1) >= shifted(macd_sign, 1)))
                                  , -1, SuperAI_signal_buy) #1 is buy, -1 is sell, 0 is do nothing
    
    global cs_name
    cs_name = 'Strategy 3 with MACD crossing MACD Signal'
    
    return SuperAI_signal
#SuperAI Trading Bot - optional cell

SuperAI_Backtester()

"""
4. TRADING STRATEGY

BUY when RSI crosses under RSI OVERSOLD THRESHOLD
==> SELL when RSI crosses over RSI OVERBOUGHT THRESHOLD
and never buy when the time is set to 'buyless' (e.g., first 15 minutes of trading session)
and always sell when the time is set to 'selltime' (e.g., 5 minutes till the trading session ends)
and use take profit and stop loss to maximize gains and minimize risks
#SuperAI Trading Bot - optional cell
"""

# 4. STRATEGY
# BUY when RSI crosses under RSI OVERSOLD THRESHOLD ==> SELL when RSI crosses over RSI OVERBOUGHT THRESHOLD

def create_signal(open_prices, high_prices, low_prices, close_prices, volume,
                  buylesstime, selltime, #Time to abstain from buying and forced selling
                  ma, ma_fast, ma_slow, #Moving Average
                  macd, macd_diff, macd_sign, #Moving Average Convergence Divergence
                  rsi, rsi_entry, rsi_exit, #Relative Strength Index
                  stoch, stoch_signal, stoch_entry, stoch_exit, #Stochastic
                  bb_low, bb_high,  #Bollinger Bands
                  mfi, mfi_entry, mfi_exit, #Money Flow Index
                  candle_buy_signal_1, candle_buy_signal_2, candle_buy_signal_3, #Candle signals to buy
                  candle_sell_signal_1, candle_sell_signal_2, candle_sell_signal_3, #Candle signals to sell
                  candle_buy_sell_signal_1, candle_buy_sell_signal_2): #Candle signals to buy or sell
    
    
    SuperAI_signal_buy = np.where( 
                                  (buylesstime != 1) 
                                  &
                                  ((rsi < rsi_entry) & (shifted(rsi, 1) >= rsi_entry))
                                  , 1, 0) #1 is buy, -1 is sell, 0 is do nothing
    SuperAI_signal = np.where( 
                                  (selltime == 1)
                                  |
                                  ((rsi > rsi_exit) & (shifted(rsi, 1) <= rsi_exit))
                                  , -1, SuperAI_signal_buy) #1 is buy, -1 is sell, 0 is do nothing
    
    global cs_name
    cs_name = 'Strategy 4 with RSI crossing oversold / overbought thresholds'
    
    return SuperAI_signal
#SuperAI Trading Bot - optional cell

SuperAI_Backtester()

"""
5. TRADING STRATEGY
BUY when STOCHASTIC crosses under STOCHASTIC OVERSOLD THRESHOLD
==> SELL when STOCHASTIC crosses over STOCHASTIC OVERBOUGHT THRESHOLD
and never buy when the time is set to 'buyless' (e.g., first 15 minutes of trading session)
and always sell when the time is set to 'selltime' (e.g., 5 minutes till the trading session ends)
and use take profit and stop loss to maximize gains and minimize risks
#SuperAI Trading Bot - optional cell
"""

# 5. STRATEGY
# BUY when STOCHASTIC crosses under STOCHASTIC OVERSOLD THRESHOLD 
# ==> SELL when STOCHASTIC crosses over STOCHASTIC OVERBOUGHT THRESHOLD

def create_signal(open_prices, high_prices, low_prices, close_prices, volume,
                  buylesstime, selltime, #Time to abstain from buying and forced selling
                  ma, ma_fast, ma_slow, #Moving Average
                  macd, macd_diff, macd_sign, #Moving Average Convergence Divergence
                  rsi, rsi_entry, rsi_exit, #Relative Strength Index
                  stoch, stoch_signal, stoch_entry, stoch_exit, #Stochastic
                  bb_low, bb_high,  #Bollinger Bands
                  mfi, mfi_entry, mfi_exit, #Money Flow Index
                  candle_buy_signal_1, candle_buy_signal_2, candle_buy_signal_3, #Candle signals to buy
                  candle_sell_signal_1, candle_sell_signal_2, candle_sell_signal_3, #Candle signals to sell
                  candle_buy_sell_signal_1, candle_buy_sell_signal_2): #Candle signals to buy or sell
    
    
    SuperAI_signal_buy = np.where( 
                                  (buylesstime != 1) 
                                  &
                                  ((stoch < stoch_entry) & (shifted(stoch, 1) >= stoch_entry))
                                  , 1, 0) #1 is buy, -1 is sell, 0 is do nothing
    SuperAI_signal = np.where( 
                                  (selltime == 1)
                                  |
                                  ((stoch > stoch_exit) & (shifted(stoch, 1) <= stoch_exit))
                                  , -1, SuperAI_signal_buy) #1 is buy, -1 is sell, 0 is do nothing
    
    global cs_name
    cs_name = 'Strategy 5 with Stochastic crossing oversold / overbought thresholds'
    
    return SuperAI_signal
#SuperAI Trading Bot - optional cell

SuperAI_Backtester()

"""
6. TRADING STRATEGY
BUY when CLOSE PRICE crosses under LOWER BOLLINGER BAND
==> SELL when CLOSE PRICE crosses over HIGHER BOLLINGER BAND
and never buy when the time is set to 'buyless' (e.g., first 15 minutes of trading session)
and always sell when the time is set to 'selltime' (e.g., 5 minutes till the trading session ends)
and use take profit and stop loss to maximize gains and minimize risks
#SuperAI Trading Bot - optional cell
"""

# 6. STRATEGY
# BUY when CLOSE PRICE crosses under LOWER BOLLINGER BAND ==> SELL when CLOSE PRICE crosses over HIGHER BOLLINGER BAND

def create_signal(open_prices, high_prices, low_prices, close_prices, volume,
                  buylesstime, selltime, #Time to abstain from buying and forced selling
                  ma, ma_fast, ma_slow, #Moving Average
                  macd, macd_diff, macd_sign, #Moving Average Convergence Divergence
                  rsi, rsi_entry, rsi_exit, #Relative Strength Index
                  stoch, stoch_signal, stoch_entry, stoch_exit, #Stochastic
                  bb_low, bb_high,  #Bollinger Bands
                  mfi, mfi_entry, mfi_exit, #Money Flow Index
                  candle_buy_signal_1, candle_buy_signal_2, candle_buy_signal_3, #Candle signals to buy
                  candle_sell_signal_1, candle_sell_signal_2, candle_sell_signal_3, #Candle signals to sell
                  candle_buy_sell_signal_1, candle_buy_sell_signal_2): #Candle signals to buy or sell
    
    
    SuperAI_signal_buy = np.where( 
                                  (buylesstime != 1) 
                                  &
                                  ((close_prices < bb_low) & (shifted(close_prices, 1) >= shifted(bb_low, 1)))
                                  , 1, 0) #1 is buy, -1 is sell, 0 is do nothing
    SuperAI_signal = np.where( 
                                  (selltime == 1)
                                  |
                                  ((close_prices > bb_high) & (shifted(close_prices, 1) <= shifted(bb_high, 1)))
                                  , -1, SuperAI_signal_buy) #1 is buy, -1 is sell, 0 is do nothing
    
    global cs_name
    cs_name = 'Strategy 6 with Close Prices crossing Bollinger Bands'
    
    return SuperAI_signal
#SuperAI Trading Bot - optional cell

SuperAI_Backtester()

"""
7. TRADING STRATEGY
BUY when MFI crosses under MFI ENTRY THRESHOLD
==> SELL when MFI crosses over MFI EXIT THRESHOLD
and never buy when the time is set to 'buyless' (e.g., first 15 minutes of trading session)
and always sell when the time is set to 'selltime' (e.g., 5 minutes till the trading session ends)
and use take profit and stop loss to maximize gains and minimize risks
#SuperAI Trading Bot - optional cell
"""

# 7. STRATEGY
# BUY when MFI crosses under MFI ENTRY THRESHOLD ==> SELL when MFI crosses over MFI EXIT THRESHOLD

def create_signal(open_prices, high_prices, low_prices, close_prices, volume,
                  buylesstime, selltime, #Time to abstain from buying and forced selling
                  ma, ma_fast, ma_slow, #Moving Average
                  macd, macd_diff, macd_sign, #Moving Average Convergence Divergence
                  rsi, rsi_entry, rsi_exit, #Relative Strength Index
                  stoch, stoch_signal, stoch_entry, stoch_exit, #Stochastic
                  bb_low, bb_high,  #Bollinger Bands
                  mfi, mfi_entry, mfi_exit, #Money Flow Index
                  candle_buy_signal_1, candle_buy_signal_2, candle_buy_signal_3, #Candle signals to buy
                  candle_sell_signal_1, candle_sell_signal_2, candle_sell_signal_3, #Candle signals to sell
                  candle_buy_sell_signal_1, candle_buy_sell_signal_2): #Candle signals to buy or sell
    
    
    SuperAI_signal_buy = np.where( 
                                  (buylesstime != 1) 
                                  &
                                  ((mfi < mfi_entry) & (shifted(mfi, 1) >= mfi_entry))
                                  , 1, 0) #1 is buy, -1 is sell, 0 is do nothing
    SuperAI_signal = np.where( 
                                  (selltime == 1)
                                  |
                                  ((mfi > mfi_exit) & (shifted(mfi, 1) <= mfi_exit))
                                  , -1, SuperAI_signal_buy) #1 is buy, -1 is sell, 0 is do nothing
    
    global cs_name
    cs_name = 'Strategy 7 with MFI crossing entry / exit thresholds'
    
    return SuperAI_signal
#SuperAI Trading Bot - optional cell

SuperAI_Backtester()
"""
8. TRADING STRATEGY
BUY when at least 1 of 5 CANDLESTICK PATTERNS shows to BUY
==> SELL when at least 1 of 5 CANDLESTICK PATTERNS shows to SELL
and never buy when the time is set to 'buyless' (e.g., first 15 minutes of trading session)
and always sell when the time is set to 'selltime' (e.g., 5 minutes till the trading session ends)
and use take profit and stop loss to maximize gains and minimize risks
#SuperAI Trading Bot - optional cell
"""

# 8. STRATEGY
# BUY when at least 1 of 5 CANDLESTICK PATTERNS shows to BUY ==> SELL when at least 1 of 5 CANDLESTICK PATTERNS shows to SELL

def create_signal(open_prices, high_prices, low_prices, close_prices, volume,
                  buylesstime, selltime, #Time to abstain from buying and forced selling
                  ma, ma_fast, ma_slow, #Moving Average
                  macd, macd_diff, macd_sign, #Moving Average Convergence Divergence
                  rsi, rsi_entry, rsi_exit, #Relative Strength Index
                  stoch, stoch_signal, stoch_entry, stoch_exit, #Stochastic
                  bb_low, bb_high,  #Bollinger Bands
                  mfi, mfi_entry, mfi_exit, #Money Flow Index
                  candle_buy_signal_1, candle_buy_signal_2, candle_buy_signal_3, #Candle signals to buy
                  candle_sell_signal_1, candle_sell_signal_2, candle_sell_signal_3, #Candle signals to sell
                  candle_buy_sell_signal_1, candle_buy_sell_signal_2): #Candle signals to buy or sell
    
    
    SuperAI_signal_buy = np.where( 
                                  (buylesstime != 1) 
                                  &
                                  ((candle_buy_signal_1 > 0) | (candle_buy_signal_2 > 0) | (candle_buy_signal_3 > 0)
                                     | (candle_buy_sell_signal_1 > 0) | (candle_buy_sell_signal_2 > 0))
                                  , 1, 0) #1 is buy, -1 is sell, 0 is do nothing
    SuperAI_signal = np.where( 
                                  (selltime == 1)
                                  |
                                  ((candle_sell_signal_1 < 0) | (candle_sell_signal_2 < 0) | (candle_buy_signal_3 < 0)
                                     | (candle_buy_sell_signal_1 < 0) | (candle_buy_sell_signal_2 < 0))
                                  , -1, SuperAI_signal_buy) #1 is buy, -1 is sell, 0 is do nothing
    
    global cs_name
    cs_name = 'Strategy 8 with Candlestick Patterns'
    
    return SuperAI_signal
#SuperAI Trading Bot - optional cell

SuperAI_Backtester()

"""
9. TRADING STRATEGY
BUY when 1 of 7 TECHNICAL INDICATORS shows to BUY (CP > MA or FAST MA > SLOW MA or MACD > MACD SIGNAL or RSI > RSI ENTRY or STOCHASTIC > STOCHASTIC ENTRY or CP < BB LOW or MFI > MFI ENTRY)* AND 1 of 5 CANDLESTICK PATTERNS shows to BUY
==> SELL when 1 of 7 TECHNICAL INDICATORS shows to SELL (CP < MA or FAST MA < SLOW MA or MACD < MACD SIGNAL or RSI > RSI EXIT or STOCHASTIC > STOCHASTIC EXIT or CP > BB HIGH or MFI > MFI EXIT)* or 1 of 7 CANDLESTICK PATTERNS shows to SELL
and never buy when the time is set to 'buyless' (e.g., first 15 minutes of trading session)
and always sell when the time is set to 'selltime' (e.g., 5 minutes till the trading session ends)
and use take profit and stop loss to maximize gains and minimize risks
* '>' here represents the crossing over

* '<' here represents the crossing under

#SuperAI Trading Bot - optional cell
"""

# 9. STRATEGY
# BUY when 1 of 7 TECHNICAL INDICATORS shows to BUY (CP > MA or FAST MA > SLOW MA or MACD > MACD SIGNAL or RSI > RSI ENTRY 
# or STOCHASTIC > STOCHASTIC ENTRY or CP < BB LOW or MFI > MFI ENTRY)* AND 1 of 5 CANDLESTICK PATTERNS shows to BUY 
# ==> SELL when 1 of 7 TECHNICAL INDICATORS shows to SELL (CP < MA or FAST MA < SLOW MA or MACD < MACD SIGNAL or RSI > RSI EXIT 
# or STOCHASTIC > STOCHASTIC EXIT or CP > BB HIGH or MFI > MFI EXIT)* or 1 of 7 CANDLESTICK PATTERNS shows to SELL

def create_signal(open_prices, high_prices, low_prices, close_prices, volume,
                  buylesstime, selltime, #Time to abstain from buying and forced selling
                  ma, ma_fast, ma_slow, #Moving Average
                  macd, macd_diff, macd_sign, #Moving Average Convergence Divergence
                  rsi, rsi_entry, rsi_exit, #Relative Strength Index
                  stoch, stoch_signal, stoch_entry, stoch_exit, #Stochastic
                  bb_low, bb_high,  #Bollinger Bands
                  mfi, mfi_entry, mfi_exit, #Money Flow Index
                  candle_buy_signal_1, candle_buy_signal_2, candle_buy_signal_3, #Candle signals to buy
                  candle_sell_signal_1, candle_sell_signal_2, candle_sell_signal_3, #Candle signals to sell
                  candle_buy_sell_signal_1, candle_buy_sell_signal_2): #Candle signals to buy or sell
    
    
    SuperAI_signal_buy = np.where( 
                                  (buylesstime != 1) 
                                  &
                                  (
                                  (
                                  ((close_prices > ma) & (shifted(close_prices, 1) <= shifted(ma, 1)))
                                  |
                                  ((ma_fast > ma_slow) & (shifted(ma_fast, 1) <= shifted(ma_slow, 1)))
                                  |
                                  ((macd > macd_sign) & (shifted(macd, 1) <= shifted(macd_sign, 1)))
                                  |
                                  ((rsi < rsi_entry) & (shifted(rsi, 1) >= rsi_entry))
                                  |
                                  ((stoch < stoch_entry) & (shifted(stoch, 1) >= stoch_entry))
                                  |
                                  ((close_prices < bb_low) & (shifted(close_prices, 1) >= shifted(bb_low, 1)))
                                  |
                                  ((mfi < mfi_entry) & (shifted(mfi, 1) >= mfi_entry))
                                  )
                                  & 
                                  ((candle_buy_signal_1 > 0) | (candle_buy_signal_2 > 0) | (candle_buy_signal_3 > 0)
                                     | (candle_buy_sell_signal_1 > 0) | (candle_buy_sell_signal_2 > 0))
                                  )
                                  , 1, 0) #1 is buy, -1 is sell, 0 is do nothing
    SuperAI_signal = np.where( 
                                  (selltime == 1)
                                  |
                                  (
                                  ((close_prices < ma) & (shifted(close_prices, 1) >= shifted(ma, 1)))
                                  |
                                  ((ma_fast < ma_slow) & (shifted(ma_fast, 1) >= shifted(ma_slow, 1)))
                                  |
                                  ((macd < macd_sign) & (shifted(macd, 1) >= shifted(macd_sign, 1)))
                                  |
                                  ((rsi > rsi_exit) & (shifted(rsi, 1) <= rsi_exit))
                                  |
                                  ((stoch > stoch_exit) & (shifted(stoch, 1) <= stoch_exit))
                                  |
                                  ((close_prices > bb_high) & (shifted(close_prices, 1) <= shifted(bb_high, 1)))
                                  |
                                  ((mfi > mfi_exit) & (shifted(mfi, 1) <= mfi_exit))
                                  )
                                  |
                                  ((candle_sell_signal_1 < 0) | (candle_sell_signal_2 < 0) | (candle_buy_signal_3 < 0)
                                     | (candle_buy_sell_signal_1 < 0) | (candle_buy_sell_signal_2 < 0))
                                  , -1, SuperAI_signal_buy) #1 is buy, -1 is sell, 0 is do nothing
    
    global cs_name
    cs_name = 'Strategy 9 with at least 1 Technical Indicator and Candlestick Patterns'
    
    return SuperAI_signal
#SuperAI Trading Bot - optional cell

SuperAI_Backtester()
"""
10. TRADING STRATEGY
BUY when ALL 7 TECHNICAL INDICATORS show to BUY (CP > MA and FAST MA > SLOW MA and MACD > MACD SIGNAL and RSI > RSI ENTRY and STOCHASTIC > STOCHASTIC ENTRY and CP < BB LOW, and MFI > MFI ENTRY)* AND 1 of 5 CANDLESTICK PATTERNS shows to BUY
==> SELL when 1 of 7 TECHNICAL INDICATORS shows to SELL (CP < MA or FAST MA < SLOW MA or MACD < MACD SIGNAL or RSI > RSI EXIT or STOCHASTIC > STOCHASTIC EXIT or CP > BB HIGH or MFI > MFI EXIT)* or 1 of 7 CANDLESTICK PATTERNS shows to SELL
and never buy when the time is set to 'buyless' (e.g., first 15 minutes of trading session)
and always sell when the time is set to 'selltime' (e.g., 5 minutes till the trading session ends)
and use take profit and stop loss to maximize gains and minimize risks
* '>' here represents the crossing over

* '<' here represents the crossing under

#SuperAI Trading Bot - optional cell
"""
# 10. STRATEGY
# BUY when ALL 7 TECHNICAL INDICATORS show to BUY (CP > MA and FAST MA > SLOW MA and MACD > MACD SIGNAL and RSI > RSI ENTRY 
# and STOCHASTIC > STOCHASTIC ENTRY and CP < BB LOW, and MFI > MFI ENTRY)* AND 1 of 5 CANDLESTICK PATTERNS shows to BUY 
# ==> SELL when 1 of 7 TECHNICAL INDICATORS shows to SELL (CP < MA or FAST MA < SLOW MA or MACD < MACD SIGNAL or RSI > RSI EXIT 
# or STOCHASTIC > STOCHASTIC EXIT or CP > BB HIGH or MFI > MFI EXIT)* or 1 of 7 CANDLESTICK PATTERNS shows to SELL

def create_signal(open_prices, high_prices, low_prices, close_prices, volume,
                  buylesstime, selltime, #Time to abstain from buying and forced selling
                  ma, ma_fast, ma_slow, #Moving Average
                  macd, macd_diff, macd_sign, #Moving Average Convergence Divergence
                  rsi, rsi_entry, rsi_exit, #Relative Strength Index
                  stoch, stoch_signal, stoch_entry, stoch_exit, #Stochastic
                  bb_low, bb_high,  #Bollinger Bands
                  mfi, mfi_entry, mfi_exit, #Money Flow Index
                  candle_buy_signal_1, candle_buy_signal_2, candle_buy_signal_3, #Candle signals to buy
                  candle_sell_signal_1, candle_sell_signal_2, candle_sell_signal_3, #Candle signals to sell
                  candle_buy_sell_signal_1, candle_buy_sell_signal_2): #Candle signals to buy or sell
    
    
    SuperAI_signal_buy = np.where( 
                                  (buylesstime != 1) 
                                  &
                                  (
                                  (
                                  ((close_prices > ma) & (shifted(close_prices, 1) <= shifted(ma, 1)))
                                  &
                                  ((ma_fast > ma_slow) & (shifted(ma_fast, 1) <= shifted(ma_slow, 1)))
                                  &
                                  ((macd > macd_sign) & (shifted(macd, 1) <= shifted(macd_sign, 1)))
                                  &
                                  ((rsi < rsi_entry) & (shifted(rsi, 1) >= rsi_entry))
                                  &
                                  ((stoch < stoch_entry) & (shifted(stoch, 1) >= stoch_entry))
                                  &
                                  ((close_prices < bb_low) & (shifted(close_prices, 1) >= shifted(bb_low, 1)))
                                  &
                                  ((mfi < mfi_entry) & (shifted(mfi, 1) >= mfi_entry))
                                  )
                                  & 
                                  ((candle_buy_signal_1 > 0) | (candle_buy_signal_2 > 0) | (candle_buy_signal_3 > 0)
                                     | (candle_buy_sell_signal_1 > 0) | (candle_buy_sell_signal_2 > 0))
                                  )
                                  , 1, 0) #1 is buy, -1 is sell, 0 is do nothing
    SuperAI_signal = np.where( 
                                  (selltime == 1)
                                  |
                                  (
                                  ((close_prices < ma) & (shifted(close_prices, 1) >= shifted(ma, 1)))
                                  |
                                  ((ma_fast < ma_slow) & (shifted(ma_fast, 1) >= shifted(ma_slow, 1)))
                                  |
                                  ((macd < macd_sign) & (shifted(macd, 1) >= shifted(macd_sign, 1)))
                                  |
                                  ((rsi > rsi_exit) & (shifted(rsi, 1) <= rsi_exit))
                                  |
                                  ((stoch > stoch_exit) & (shifted(stoch, 1) <= stoch_exit))
                                  |
                                  ((close_prices > bb_high) & (shifted(close_prices, 1) <= shifted(bb_high, 1)))
                                  |
                                  ((mfi > mfi_exit) & (shifted(mfi, 1) <= mfi_exit))
                                  )
                                  |
                                  ((candle_sell_signal_1 < 0) | (candle_sell_signal_2 < 0) | (candle_buy_signal_3 < 0)
                                     | (candle_buy_sell_signal_1 < 0) | (candle_buy_sell_signal_2 < 0))
                                  , -1, SuperAI_signal_buy) #1 is buy, -1 is sell, 0 is do nothing
    
    global cs_name
    cs_name = 'Strategy 10 with all Technical Indicators and Candlestick Patterns'
    
    return SuperAI_signal
#SuperAI Trading Bot - optional cell

SuperAI_Backtester()
"""
11. TRADING STRATEGY
BUY when MACD crosses over MACD SIGNAL only when MACD < 0
==> SELL when MACD crosses under MACD SIGNAL
and never buy when the time is set to 'buyless' (e.g., first 15 minutes of trading session)
and always sell when the time is set to 'selltime' (e.g., 5 minutes till the trading session ends)
and use take profit and stop loss to maximize gains and minimize risks
#SuperAI Trading Bot - optional cell
"""

# 11. STRATEGY
# BUY when MACD crosses over MACD SIGNAL only when MACD < 0 ==> SELL when MACD crosses under MACD SIGNAL

def create_signal(open_prices, high_prices, low_prices, close_prices, volume,
                  buylesstime, selltime, #Time to abstain from buying and forced selling
                  ma, ma_fast, ma_slow, #Moving Average
                  macd, macd_diff, macd_sign, #Moving Average Convergence Divergence
                  rsi, rsi_entry, rsi_exit, #Relative Strength Index
                  stoch, stoch_signal, stoch_entry, stoch_exit, #Stochastic
                  bb_low, bb_high,  #Bollinger Bands
                  mfi, mfi_entry, mfi_exit, #Money Flow Index
                  candle_buy_signal_1, candle_buy_signal_2, candle_buy_signal_3, #Candle signals to buy
                  candle_sell_signal_1, candle_sell_signal_2, candle_sell_signal_3, #Candle signals to sell
                  candle_buy_sell_signal_1, candle_buy_sell_signal_2): #Candle signals to buy or sell
    
    
    SuperAI_signal_buy = np.where( 
                                  (buylesstime != 1) 
                                  &
                                  ((macd < 0) & (macd > macd_sign) & (shifted(macd, 1) <= shifted(macd_sign, 1)))
                                  , 1, 0) #1 is buy, -1 is sell, 0 is do nothing
    SuperAI_signal = np.where( 
                                  (selltime == 1)
                                  |
                                  ((macd < macd_sign) & (shifted(macd, 1) >= shifted(macd_sign, 1)))
                                  , -1, SuperAI_signal_buy) #1 is buy, -1 is sell, 0 is do nothing
    
    global cs_name
    cs_name = 'Strategy 11 with MACD crossing MACD Signal and MACD < 0'
    
    return SuperAI_signal
#SuperAI Trading Bot - optional cell

SuperAI_Backtester()
"""
12. TRADING STRATEGY
BUY when MACD crosses over MACD SIGNAL only when MACD < 0 and (MA is higher than MA from 2 minutes before and MA from 2 minutes before is higher than MA from 4 minutes before)
==> SELL when (STOCHASTIC is over STOCHASTIC EXIT and RSI over RSI EXIT) or (CP over BB HIGH and MFI over MFI EXIT)
and never buy when the time is set to 'buyless' (e.g., first 15 minutes of trading session)
and always sell when the time is set to 'selltime' (e.g., 5 minutes till the trading session ends)
and use take profit and stop loss to maximize gains and minimize risks
#SuperAI Trading Bot - optional cell

"""
# 12. STRATEGY (from previous tutorial)
# BUY when MACD crosses over MACD SIGNAL only when MACD < 0 and (MA is higher than MA from 2 minutes before 
# and MA from 2 minutes before is higher than MA from 4 minutes before) 
# ==> SELL when (STOCHASTIC is over STOCHASTIC EXIT and RSI over RSI EXIT) or (CP over BB HIGH and MFI over MFI EXIT)

def create_signal(open_prices, high_prices, low_prices, close_prices, volume,
                  buylesstime, selltime, #Time to abstain from buying and forced selling
                  ma, ma_fast, ma_slow, #Moving Average
                  macd, macd_diff, macd_sign, #Moving Average Convergence Divergence
                  rsi, rsi_entry, rsi_exit, #Relative Strength Index
                  stoch, stoch_signal, stoch_entry, stoch_exit, #Stochastic
                  bb_low, bb_high,  #Bollinger Bands
                  mfi, mfi_entry, mfi_exit, #Money Flow Index
                  candle_buy_signal_1, candle_buy_signal_2, candle_buy_signal_3, #Candle signals to buy
                  candle_sell_signal_1, candle_sell_signal_2, candle_sell_signal_3, #Candle signals to sell
                  candle_buy_sell_signal_1, candle_buy_sell_signal_2): #Candle signals to buy or sell
    
    
    SuperAI_signal_buy = np.where( 
                                  (buylesstime != 1) 
                                  &
                                  ((macd < 0) & (macd > macd_sign) & (shifted(macd, 1) <= shifted(macd_sign, 1)))
                                  &
                                  ((ma > (shifted(ma, 2))) & 
                                     (shifted(ma, 2) > (shifted(ma, 4))))
                                  , 1, 0) #1 is buy, -1 is sell, 0 is do nothing
    SuperAI_signal = np.where( 
                                  (selltime == 1)
                                  |
                                  ((stoch > stoch_exit) & (rsi > rsi_exit))
                                  |
                                  ((close_prices > bb_high) & (mfi > mfi_exit))
                                  , -1, SuperAI_signal_buy) #1 is buy, -1 is sell, 0 is do nothing
    
    global cs_name
    cs_name = 'Strategy 12 with various Technical Indicators'
    
    return SuperAI_signal
#SuperAI Trading Bot - optional cell

SuperAI_Backtester()

"""
13. TRADING STRATEGY
BUY when last close price is higher than the close price from a minute before
==> SELL when last close price is lower than the close price from a minute before
and never buy when the time is set to 'buyless' (e.g., first 15 minutes of trading session)
and always sell when the time is set to 'selltime' (e.g., 5 minutes till the trading session ends)
and use take profit and stop loss to maximize gains and minimize risks
#SuperAI Trading Bot - optional cell
"""

# 13. STRATEGY
# BUY when last close price is higher than the close price from a minute before 
# ==> SELL when last close price is lower than the close price from a minute before

def create_signal(open_prices, high_prices, low_prices, close_prices, volume,
                  buylesstime, selltime, #Time to abstain from buying and forced selling
                  ma, ma_fast, ma_slow, #Moving Average
                  macd, macd_diff, macd_sign, #Moving Average Convergence Divergence
                  rsi, rsi_entry, rsi_exit, #Relative Strength Index
                  stoch, stoch_signal, stoch_entry, stoch_exit, #Stochastic
                  bb_low, bb_high,  #Bollinger Bands
                  mfi, mfi_entry, mfi_exit, #Money Flow Index
                  candle_buy_signal_1, candle_buy_signal_2, candle_buy_signal_3, #Candle signals to buy
                  candle_sell_signal_1, candle_sell_signal_2, candle_sell_signal_3, #Candle signals to sell
                  candle_buy_sell_signal_1, candle_buy_sell_signal_2): #Candle signals to buy or sell
    
    SuperAI_signal_buy = np.where( 
                                  (buylesstime != 1) 
                                  &
                                  (close_prices > shifted(close_prices, 1))
                                  , 1, 0) #1 is buy, -1 is sell, 0 is do nothing
    SuperAI_signal = np.where( 
                                  (selltime == 1)
                                  |
                                  (close_prices <= shifted(close_prices, 1))
                                  , -1, SuperAI_signal_buy) #1 is buy, -1 is sell, 0 is do nothing
    
    global cs_name
    cs_name = 'Strategy 13 with buying when price go up and selling when price go down'
    
    return SuperAI_signal
#SuperAI Trading Bot - optional cell

SuperAI_Backtester()
#Summary of all 13 strategies tested
#SuperAI Trading Bot - optional cell
#Summary of all tested strategies

strategies.set_index('Strategy')
#Now we choose the best strategy from the 13 we tested.

#RUN THE BOT LIVE WITH ONE OF THE TESTED STRATEGIES - THE STRATEGY YOU BELIEVE IS THE BEST
#Paste the strategy you choose here
#SuperAI Trading Bot - optional cell

SuperAI_Trading_Bot()
#CREATE YOUR OWN STRATEGY AND RUN THE BOT WITH YOUR OWN STRATEGY
#SuperAI Trading Bot - optional cell
# YOUR STRATEGY
# BUY when  
# ==> SELL when 

def create_signal(open_prices, high_prices, low_prices, close_prices, volume,
                  buylesstime, selltime, #Time to abstain from buying and forced selling
                  ma, ma_fast, ma_slow, #Moving Average
                  macd, macd_diff, macd_sign, #Moving Average Convergence Divergence
                  rsi, rsi_entry, rsi_exit, #Relative Strength Index
                  stoch, stoch_signal, stoch_entry, stoch_exit, #Stochastic
                  bb_low, bb_high,  #Bollinger Bands
                  mfi, mfi_entry, mfi_exit, #Money Flow Index
                  candle_buy_signal_1, candle_buy_signal_2, candle_buy_signal_3, #Candle signals to buy
                  candle_sell_signal_1, candle_sell_signal_2, candle_sell_signal_3, #Candle signals to sell
                  candle_buy_sell_signal_1, candle_buy_sell_signal_2): #Candle signals to buy or sell
    
    SuperAI_signal_buy = np.where( 
                                  (buylesstime != 1) 
                                  &
                                  (True) #put here the condition that should be met to buy coins/shares
                                  , 1, 0) #1 is buy, -1 is sell, 0 is do nothing
    SuperAI_signal = np.where( 
                                  (selltime == 1)
                                  |
                                  (False) #put here the condition that should be met to sell the coins/shares
                                  , -1, SuperAI_signal_buy) #1 is buy, -1 is sell, 0 is do nothing
    
    global cs_name
    cs_name = 'Your Strategy'
    
    return SuperAI_signal
#SuperAI Trading Bot - optional cell

SuperAI_Backtester()
#SuperAI Trading Bot - optional cell
#Summary of all tested strategies

strategies.set_index('Strategy')
#SuperAI Trading Bot - optional cell

SuperAI_Trading_Bot()
