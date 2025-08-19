#so here I am going to finalise the mean reversion strategy and try to make it amazing
#make it backtestable

#this is the next stage of the code where I go over the code exactly and see how it works and finalize it
#will only concentrate on long signals, because we can only purchase them

import streamlit as st
import yfinance as yf
import pandas as pd
import pandas_ta as ta
import numpy as np
import datetime as dt
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from backtesting import Strategy
from backtesting import Backtest
from backtesting.lib import SignalStrategy, TrailingStrategy
from math import sqrt
from statsmodels.tsa.stattools import adfuller
from hurst import compute_Hc
import plotly.express as px
import statsmodels.api as sm
import seaborn as sns
import matplotlib.pyplot as plt

######################################################################################################

#title and other specifications
st.title("Long Only Bollinger Bands + RSI Strategy Dashboard")
col1, col2 = st.columns([0.14, 0.86], gap="small")
col1.write("code done by:")
linkedin = "https://www.linkedin.com/in/gergely-marton-2024-2026edi"
col2.markdown(
        f'<a href="{linkedin}" target="_blank" style="text-decoration: none; color: inherit;"><img src="https://cdn-icons-png.flaticon.com/512/174/174857.png" width="15" height="15" style="vertical-align: middle; margin-right: 10px;">`Gergely Marton`</a>',
        unsafe_allow_html=True,
    )

ticker = st.sidebar.text_input("Stock selected, symbol fetched from Yahoofinance website:", value="MSFT")
start_back = st.sidebar.number_input("Input for how many days we should fetch historicaly daily close prices, recommended: 3 years", min_value=365, max_value=3650, value=1095, step=1)
end_date = st.sidebar.date_input("Please input the end date for both the BB and Price", value=dt.date.today())
start_date = end_date - dt.timedelta(start_back)
st.sidebar.write("Start and end dates:", start_date,",", end_date)

#downloading data
price_df = yf.download(ticker, start_date, end_date)

price_series = price_df["Close"].squeeze()
price_series_clean = price_series.dropna()
fig = px.line(price_df, x = price_df.index, y = price_df['Close'].squeeze(), title = ticker, labels={'y': 'Close'})
st.plotly_chart(fig)

if st.checkbox("Show CleanClose Price Series"):
    st.write(price_series_clean)

######################################################################################################

#initializing the parameters so that after optimization it can be changed
if 'bb_ma' not in st.session_state:
    st.session_state.bb_ma = 20
if 'bb_std' not in st.session_state:
    st.session_state.bb_std = 2.0
if 'rsi_ma' not in st.session_state:
    st.session_state.rsi_ma = 14
if 'rsi_tresh_l' not in st.session_state:
    st.session_state.rsi_tresh_l = 30
if 'rsi_tresh_h' not in st.session_state:
    st.session_state.rsi_tresh_h = 70
if 'atr_mult' not in st.session_state:
    st.session_state.atr_mult = 2.0
if 'profit_mult' not in st.session_state:
    st.session_state.profit_mult = 2.0

######################################################################################################
#creating the sidebar parameters so that we can change it afterwards after optimization
bb_ma = st.sidebar.number_input(
    "Length for moving average for BBANDS (20 recommended): ", 
    min_value=5, max_value=100, 
    key='bb_ma'
)

bb_std = st.sidebar.number_input(
    "Standard deviation for BBANDS (2 recommended): ", 
    min_value=0.0, max_value=5.0,  
    step=0.01, format="%.4f",
    key='bb_std'
)

rsi_ma = st.sidebar.number_input(
    "Length for moving average for RSI (14 recommended): ", 
    min_value=1, max_value=30, 
    key='rsi_ma'
)

atr_mult = st.sidebar.number_input(
    "Stop-loss multiple of ATR:", 
    min_value=1.0, max_value=10.0,  
    step=0.1,
    key='atr_mult'
)

profit_mult = st.sidebar.number_input(
    "Profit target multiple of ATR:", 
    min_value=1.0, max_value=10.0,  
    step=0.1,
    key='profit_mult'
)

rsi_tresh_l = st.sidebar.number_input(
    "Input the value for RSI low threshold: ", 
    min_value=10, max_value=40, 
    value=st.session_state.rsi_tresh_l,
    key='rsi_tresh_l'
)

rsi_tresh_h = st.sidebar.number_input(
    "Input the value for RSI high threshold: ", 
    min_value=60, max_value=90, 
    value=st.session_state.rsi_tresh_h,
    key='rsi_tresh_h'
)

bb_width_t = st.sidebar.number_input("Input the value for BB width threshold: ", min_value=0.0005, max_value=0.003, value=0.001, step=0.0001, format="%.4f")

atr_length = st.sidebar.number_input("ATR period for stop-loss (e.g. 14):", min_value=5, max_value=100, value=14)

######################################################################################################
# now defining the LONG function based on the specifications: 

def generate_long_signals(df, rsi_low, bb_width_t):
    """
    Returns a Series with long signals (1 if signal, 0 otherwise) based on:
    1. Previous day's close < previous day's lower BB
    2. Previous day's RSI < rsi_low
    3. Today's close > today's lower BB
    4. Today's BB width > bb_width_t
    """
    long_signal = (
        (df['Close'].shift(1) < df['bbl'].shift(1)) &    # 1. Previous close < lower BB
        (df['rsi'].shift(1) < rsi_low) &                 # 2. Previous RSI < threshold
        (df['Close'] > df['bbl']) &                      # 3. Today close > lower BB
        (df['bb_width'] > bb_width_t)                    # 4. Today BB width > threshold
    )
    return long_signal.astype(int)  # 1 for signal, 0 otherwise

######################################################################################################
#now defining when and how to exit the trade

def simulate_long_trades_with_atr_stop_and_profit(df, atr_mult, profit_mult):
    """
    Simulate long trades with ATR-based stop-loss and ATR-based profit target.
    Only one position at a time. Entry on 'long_signal' == 1.
    Exit if price falls below (entry_price - entry_atr * atr_mult) [stop loss]
    or price rises above (entry_price + entry_atr * profit_mult) [profit target].
    ATR used is always the value at entry.
    Returns a signal column: 1 for entry, -1 for exit, 0 otherwise.
    """
    df = df.copy()
    signal = np.zeros(len(df))
    in_position = False
    entry_price = 0.0
    entry_atr = 0.0

    for i in range(1, len(df)):
        if not in_position:
            if df['long_signal'].iloc[i]:
                in_position = True
                entry_price = df['Close'].iloc[i]
                entry_atr = df['atr'].iloc[i]
                signal[i] = 1
        else:
            stop_price = entry_price - entry_atr * atr_mult
            profit_price = entry_price + entry_atr * profit_mult
            if df['Close'].iloc[i] <= stop_price or df['Close'].iloc[i] >= profit_price:
                in_position = False
                signal[i] = -1
                entry_price = 0.0
                entry_atr = 0.0
    return pd.Series(signal, index=df.index)
######################################################################################################

def prepare_new_df(ticker, start_date, end_date, bb_ma, bb_std, rsi_ma, atr_length):
    new_df = yf.download(ticker, start_date, end_date)

    if isinstance(new_df.columns, pd.MultiIndex):
        new_df.columns = new_df.columns.droplevel(1)

    new_df['atr'] = ta.atr(new_df['High'], new_df['Low'], new_df['Close'], length=atr_length)

    return new_df

######################################################################################################
def prepare_trading_dataframe(ticker, start_date, end_date, bb_ma, bb_std, rsi_ma, rsi_tresh_l, bb_width_t, atr_mult, profit_mult, atr_length):
    """
    Unified function to prepare a trading dataframe with all indicators and signals.
    This ensures consistency across all parts of the code.
    
    Returns:
        pd.DataFrame: Clean dataframe with all indicators and signals, or None if error
    """
    try:
        
        new_df = prepare_new_df(ticker, start_date, end_date, bb_ma, bb_std, rsi_ma, atr_length)
        
        bbands = ta.bbands(new_df["Close"], length=bb_ma, std=bb_std)
        rsi = ta.rsi(new_df["Close"], length=rsi_ma)

        # Combine all dataframes
        combined_df = pd.concat([new_df, bbands, rsi], axis=1)
        combined_df_clean = combined_df.dropna()
        
        if combined_df_clean.empty:
            return None
        
        # Rename columns consistently
        combined_df_clean.rename(columns={
            f'BBL_{bb_ma}_{float(bb_std):.1f}': 'bbl',
            f'BBM_{bb_ma}_{float(bb_std):.1f}': 'bbm',
            f'BBU_{bb_ma}_{float(bb_std):.1f}': 'bbh',
            f'RSI_{rsi_ma}': 'rsi',
            f'ATR_{atr_length}': 'atr'
        }, inplace=True)
        
        # Calculate BB width
        combined_df_clean['bb_width'] = (combined_df_clean['bbh'] - combined_df_clean['bbl']) / combined_df_clean['bbm']
        
        # Generate long signals
        combined_df_clean['long_signal'] = generate_long_signals(combined_df_clean, rsi_tresh_l, bb_width_t)
        
        # Generate trade signals (entry and exit)
        combined_df_clean['trade_signal'] = simulate_long_trades_with_atr_stop_and_profit(
            combined_df_clean, atr_mult, profit_mult
        )
        
        return combined_df_clean
        
    except Exception as e:
        st.write(f"Error preparing dataframe for {ticker}: {e}")
        return None

######################################################################################################
new_df = prepare_new_df(ticker, start_date, end_date, bb_ma, bb_std, rsi_ma, atr_length)

if st.checkbox("Show clean dataframe without BB&RSI:"):
    st.write(new_df)
######################################################################################################
# UPDATED: Main strategy implementation section
st.subheader("Implementing the BBANDS /RSI trading strategy")

# Use the unified function to prepare the dataframe
combined_df_clean = prepare_trading_dataframe(
    ticker, start_date, end_date, 
    bb_ma, bb_std, rsi_ma, rsi_tresh_l, bb_width_t, 
    atr_mult, profit_mult, atr_length
)

if combined_df_clean is not None:
    if st.checkbox("Show Price Series"):
        st.write(combined_df_clean[['Open', 'High', 'Low', 'Close', 'Volume']])    
    
    if st.checkbox("Show whole dataframe:"):
        st.write(combined_df_clean)
    
else:
    st.error(f"Failed to prepare dataframe for {ticker}")
######################################################################################################
#screening the tickers based on statistical tests
def calc_half_life(series) -> float | None:
    try:
        if isinstance(series, pd.DataFrame):
            if 'Close' in series.columns:
                series = series['Close']
            else:
                series = series.iloc[:, 0]
        
        if not isinstance(series, pd.Series):
            series = pd.Series(series)
        
        series = series.dropna()
        
        if len(series) < 30:
            return None

        lagged_values = series.shift(1)
        delta_values = series.diff()
        
        data_dict = {
            'delta': delta_values,
            'lagged': lagged_values
        }
        
        hl_df = pd.DataFrame(data_dict).dropna()
        
        if hl_df.empty or len(hl_df) < 5:
            return None

        X = sm.add_constant(hl_df["lagged"])
        y = hl_df["delta"]
        
        model_result = sm.OLS(y, X).fit()
        lambda_coef = model_result.params["lagged"]

        if lambda_coef >= 0:
            return None
        return -np.log(2) / lambda_coef
        
    except Exception:
        return None

######################################################################################################
def run_screening(tickers_list, max_half_life=100):
    """Run statistical tests on a list of tickers"""
    passed = []
    
    for ticker in tickers_list:
        try:
            df = yf.download(ticker, period="3y", interval="1d", progress=False)
            
            if df.empty:
                continue
                
            # Extract Close prices
            if isinstance(df.columns, pd.MultiIndex):
                close_col = [col for col in df.columns if 'Close' in str(col)]
                if close_col:
                    px = df[close_col[0]]
                else:
                    px = df.iloc[:, -1]
            else:
                if 'Close' in df.columns:
                    px = df['Close']
                else:
                    px = df.iloc[:, -1]
            
            px = px.dropna()
            if not isinstance(px, pd.Series):
                px = pd.Series(px.values.flatten() if hasattr(px.values, 'flatten') else px.values)

            if len(px) < 100:
                continue

            tests = []

            # 1) Augmented Dickey-Fuller
            try:
                p_val = adfuller(px, autolag="AIC")[1]
                if p_val <= 0.05:
                    tests.append("ADF")
            except:
                pass

            # 2) Hurst exponent
            try:
                H, _, _ = compute_Hc(px, kind="price", simplified=True)
                if H < 0.5:
                    tests.append("Hurst")
            except:
                pass

            # 3) Half-life
            try:
                hl = calc_half_life(px)
                if hl is not None and 1 < hl < max_half_life:     
                    tests.append("Half-Life")
            except:
                pass

            if tests:
                passed.append(ticker)

        except Exception:
            continue
    
    return passed

######################################################################################################
#def dfcreationbefore():
#    #creating the dataframe for the strategy"
#    st.subheader("Implementing the BBANDS /RSI trading strategy")
#
#    new_df = yf.download(ticker, start_date, end_date)
#
#    # If you downloaded data for a single ticker, remove the second level
#    if isinstance(new_df.columns, pd.MultiIndex):
#        new_df.columns = new_df.columns.droplevel(1)
#
#    if st.checkbox("Show Price Series"):
#        st.write(new_df)
#
#    bbands = ta.bbands(new_df["Close"], length=bb_ma, std=bb_std)
#    rsi = ta.rsi(new_df["Close"], length=rsi_ma)
#    if st.checkbox("Show BBANDS and RSI dataframe:"):
#        st.write(bbands, rsi)
#
#    new_df['atr'] = ta.atr(new_df['High'], new_df['Low'], new_df['Close'], length=atr_length)
#
#
#    combined_df = pd.concat([new_df, bbands, rsi], axis=1)
#    combined_df_clean = combined_df.dropna()
#    combined_df_clean.rename(columns={
#        f'BBL_{bb_ma}_{float(bb_std):.1f}': 'bbl',
#        f'BBM_{bb_ma}_{float(bb_std):.1f}': 'bbm',
#        f'BBU_{bb_ma}_{float(bb_std):.1f}': 'bbh',
#        f'RSI_{rsi_ma}': 'rsi'
#    }, inplace=True)
#    combined_df_clean['bb_width'] = (combined_df_clean['bbh'] - combined_df_clean['bbl']) / combined_df_clean['bbm']
#
#    if st.checkbox("Show nonclean dataframe"):
#        st.write(combined_df)
#
#    combined_df_clean['long_signal'] = generate_long_signals(combined_df_clean, rsi_tresh_l, bb_width_t)
#
#    if st.checkbox("Show whole dataframe:"):
#        st.write(combined_df_clean)
#
#    combined_df_clean['trade_signal'] = simulate_long_trades_with_atr_stop_and_profit(combined_df_clean, atr_mult, profit_mult)
#
#    if st.checkbox("See trading signals dataframe"):    
#        st.write(combined_df_clean)
######################################################################################################
def check_signals_for_tickers(tickers_list, bb_ma, bb_std, rsi_ma, rsi_tresh_l, bb_width_t, atr_mult, profit_mult, atr_length):
    """
    Check recent signals for a list of tickers using the unified prepare_trading_dataframe function.
    Returns detailed information about signals including entry/exit prices and levels.
    """
    buy_signals = []
    sell_signals = []
    signal_details = {}
    
    for ticker in tickers_list:
        try:
            # Use the unified function for consistency
            combined = prepare_trading_dataframe(
                ticker, start_date, end_date, 
                bb_ma, bb_std, rsi_ma, rsi_tresh_l, bb_width_t, 
                atr_mult, profit_mult, atr_length
            )
            
            if combined is None or combined.empty:
                continue
            
            # Check the last 5 trading days for any non-zero signal
            if len(combined) > 0:
                lookback_days = min(5, len(combined))
                recent_data = combined.iloc[-lookback_days:]
                
                # Find the most recent non-zero trade signal
                non_zero_signals = recent_data[recent_data['trade_signal'] != 0]
                
                if not non_zero_signals.empty:
                    # Get the most recent non-zero signal
                    last_signal_row = non_zero_signals.iloc[-1]
                    last_signal_value = last_signal_row['trade_signal']
                    signal_date = non_zero_signals.index[-1]
                    
                    # Calculate days since signal
                    days_since = len(combined) - 1 - combined.index.get_loc(signal_date)
                    
                    # Only consider recent signals (within last 2 trading days)
                    if days_since <= 1:  # 0 = today, 1 = yesterday
                        signal_info = {
                            'ticker': ticker,
                            'date': signal_date.strftime('%Y-%m-%d'),
                            'price': last_signal_row['Close'],
                            'rsi': last_signal_row['rsi'],
                            'bb_width': last_signal_row['bb_width'],
                            'atr': last_signal_row['atr'],
                            'days_ago': days_since
                        }
                        
                        if last_signal_value == 1:  # Buy signal
                            # Calculate stop-loss and take-profit levels
                            signal_info['stop_loss'] = signal_info['price'] - (signal_info['atr'] * atr_mult)
                            signal_info['take_profit'] = signal_info['price'] + (signal_info['atr'] * profit_mult)
                            buy_signals.append(ticker)
                            signal_details[ticker] = signal_info
                            
                        elif last_signal_value == -1:  # Sell/Exit signal
                            sell_signals.append(ticker)
                            signal_details[ticker] = signal_info
                    
        except Exception as e:
            # Optionally add debugging
            # st.error(f"Error processing {ticker}: {e}")
            continue
    
    return buy_signals, sell_signals, signal_details
######################################################################################################
dfpl = combined_df_clean.copy()  # or whatever your main DataFrame is called

# Entry points (long entry: trade_signal == 1)
dfpl['entry_marker'] = np.where(dfpl['trade_signal'] == 1, dfpl['Close'], np.nan)
# Exit points (long exit: trade_signal == -1)
dfpl['exit_marker'] = np.where(dfpl['trade_signal'] == -1, dfpl['Close'], np.nan)

fig = make_subplots(
    rows=2, cols=1,
    shared_xaxes=True,
    vertical_spacing=0.07,
    subplot_titles=("Candlestick & BBands", "RSI")
)

# --- Candlestick plot
fig.add_trace(
    go.Candlestick(
        x=dfpl.index,
        open=dfpl['Open'],
        high=dfpl['High'],
        low=dfpl['Low'],
        close=dfpl['Close'],
        name="Candlestick"
    ),
    row=1, col=1
)

# --- Bollinger Bands
fig.add_trace(
    go.Scatter(
        x=dfpl.index, y=dfpl['bbl'],
        line=dict(color='green', width=1),
        name="BB Lower", legendgroup="BB"
    ), row=1, col=1
)
fig.add_trace(
    go.Scatter(
        x=dfpl.index, y=dfpl['bbh'],
        line=dict(color='green', width=1),
        name="BB Upper", legendgroup="BB"
    ), row=1, col=1
)

# --- Entry Markers
fig.add_trace(
    go.Scatter(
        x=dfpl.index,
        y=dfpl['entry_marker'],
        mode="markers",
        marker=dict(size=9, color="MediumPurple", symbol='triangle-up'),
        name="Entry"
    ), row=1, col=1
)

# --- Exit Markers
fig.add_trace(
    go.Scatter(
        x=dfpl.index,
        y=dfpl['exit_marker'],
        mode="markers",
        marker=dict(size=9, color="red", symbol='x'),
        name="Exit"
    ), row=1, col=1
)

# --- RSI subplot
fig.add_trace(
    go.Scatter(
        x=dfpl.index, y=dfpl['rsi'],
        line=dict(color='orange', width=2),
        name="RSI"
    ), row=2, col=1
)

# Optional: Add RSI overbought/oversold lines
fig.add_shape(type='line', x0=dfpl.index[0], x1=dfpl.index[-1], y0=rsi_tresh_h, y1=rsi_tresh_h,
              line=dict(color='red', width=1, dash='dash'), row=2, col=1)
fig.add_shape(type='line', x0=dfpl.index[0], x1=dfpl.index[-1], y0=rsi_tresh_l, y1=rsi_tresh_l,
              line=dict(color='green', width=1, dash='dash'), row=2, col=1)

fig.update_layout(
    width=1200, height=800,
    title="BBands/RSI Strategy Signals",
    xaxis_rangeslider_visible=False
)
st.plotly_chart(fig, use_container_width=True)
######################################################################################################
#BACKTEST

# Precompute all indicators and merge into one DataFrame for Backtest
def prepare_bt_dataframe(df, bb_ma, bb_std, rsi_ma, atr_length):
    # Calculate BBANDS
    bb = ta.bbands(df['Close'], length=bb_ma, std=bb_std)
    rsi_2 = ta.rsi(df['Close'], length=rsi_ma)
    atr  = df['atr'] if 'atr' in df.columns else ta.atr(df['High'], df['Low'], df['Close'], length=atr_length).rename('atr')
    df_bt = pd.concat([df, bb, rsi_2, atr], axis=1)
    df_bt = df_bt.dropna().copy()
    # Rename columns for easy access
    df_bt.rename(columns={
        f'BBL_{bb_ma}_{float(bb_std):.1f}': 'bbl',
        f'BBM_{bb_ma}_{float(bb_std):.1f}': 'bbm',
        f'BBU_{bb_ma}_{float(bb_std):.1f}': 'bbh',
        f'RSI_{rsi_ma}': 'rsi',
        f'ATR_{atr_length}': 'atr'
    }, inplace=True)
    df_bt['bb_width'] = (df_bt['bbh'] - df_bt['bbl']) / df_bt['bbm']
    return df_bt

df_bt = prepare_bt_dataframe(
    new_df, bb_ma, bb_std, rsi_ma, atr_length
)

if st.checkbox("Dataframe for backtest and optimization:"):
    st.write(df_bt)

class BBRSIATRStrategy(Strategy):
    # These can be set as class variables or optimized later
    rsi_tresh_l = 30
    bb_std = 2.0
    atr_mult = 2.0
    profit_mult = 2.0

    bb_width_t = bb_width_t

    def init(self):
        super().init()
        # Store indicators as numpy arrays
        self.close = self.data.Close
        self.bbl = self.data.bbl
        self.rsi = self.data.rsi
        self.bb_width = self.data.bb_width
        self.atr = self.data.atr

    def next(self):
        i = len(self.data) - 1  # Current bar index

        # ENTRY CONDITION (only if not in position)
        if not self.position:
            prev = i - 1  # Previous day
            if (
                self.close[prev] < self.bbl[prev] and
                self.rsi[prev] < self.rsi_tresh_l and
                self.close[i] > self.bbl[i] and
                self.bb_width[i] > self.bb_width_t
            ):
                entry_price = self.close[i]
                entry_atr = self.atr[i]
                sl = entry_price - entry_atr * self.atr_mult
                tp = entry_price + entry_atr * self.profit_mult
                self.buy(sl=sl, tp=tp)
        # No short selling or re-entry until flat.

capital = st.sidebar.number_input("Input the starting capital:", min_value=1000, max_value=100000, value=10000, step=500)
commission = st.sidebar.number_input("Input the commission of the Backtest:", min_value=0.0, max_value=0.15, value=0.001,step=0.001, format='%4f')
spread = st.sidebar.number_input("Input the spread of the Backtest:", min_value=0.0, max_value=0.15, value=0.001,step=0.001, format='%4f')

# 1. Set strategy class parameters to sidebar input values
BBRSIATRStrategy.rsi_tresh_l = rsi_tresh_l  # from sidebar
BBRSIATRStrategy.bb_std = bb_std            # from sidebar
BBRSIATRStrategy.atr_mult = atr_mult        # from sidebar
BBRSIATRStrategy.profit_mult = profit_mult  # from sidebar
BBRSIATRStrategy.bb_width_t   = bb_width_t

# Run the backtest
bt = Backtest(
    df_bt,
    BBRSIATRStrategy,
    cash=capital,
    spread=spread,
    commission=commission,  # e.g. 0.1%
    exclusive_orders=True
)
stats = bt.run()
if st.button("See the plotted performance in an HTML:"):
    bt.plot()
st.write(stats)
######################################################################################################
#optimizing the parameter: 
# the stop loss level
# the take profit level
# the bb standard deviation
# the RSI treshold low

choice_max = st.selectbox(
    "Choose what you want to optimize:",
    options=[
        "Equity Final [$]",
        "Return [%]",
        "Sharpe Ratio",
        "Win Rate [%]",
        "Max. Drawdown [%]",
        "Sortino Ratio",
        "Expectancy [%]",
        "Profit Factor"
    ],
    index=0
)

def apply_optimal_parameters():
    """Callback to apply optimal parameters"""
    st.session_state.rsi_tresh_l = st.session_state.optimal_rsi_tresh_l
    st.session_state.bb_std = st.session_state.optimal_bb_std
    st.session_state.atr_mult = st.session_state.optimal_atr_mult
    st.session_state.profit_mult = st.session_state.optimal_profit_mult

if st.button("Run Optimization"):
    # Optimize only the 4 key parameters, keep others as sidebar/defaults
    optimization_results, heatmap = bt.optimize(
        rsi_tresh_l=range(20, 30, 1),
        bb_std=list(np.arange(1.5, 3.0, 0.5)),
        atr_mult=list(np.arange(1.0, 3.1, 0.5)),
        profit_mult=list(np.arange(1.0, 3.1, 0.5)),
        maximize=choice_max,
        return_heatmap=True
    )
    
    # Store optimal parameters in session state
    st.session_state.optimal_rsi_tresh_l = optimization_results._strategy.rsi_tresh_l
    st.session_state.optimal_bb_std = optimization_results._strategy.bb_std
    st.session_state.optimal_atr_mult = optimization_results._strategy.atr_mult
    st.session_state.optimal_profit_mult = optimization_results._strategy.profit_mult
    st.session_state.optimization_done = True
    
    st.write("Optimal Parameters:")
    st.write(f"RSI threshold (low): {optimization_results._strategy.rsi_tresh_l}")
    st.write(f"BB std: {optimization_results._strategy.bb_std}")
    st.write(f"ATR stop multiplier: {optimization_results._strategy.atr_mult}")
    st.write(f"Profit target multiplier: {optimization_results._strategy.profit_mult}")
    st.write(optimization_results)

# Add button to apply optimal parameters (only show if optimization has been done)
if 'optimization_done' in st.session_state and st.session_state.optimization_done:
    st.button(
        "Apply Optimal Parameters to Sidebar",
        on_click=apply_optimal_parameters
    )
    if st.session_state.get('parameters_applied', False):
        st.success("Optimal parameters applied! The sidebar values have been updated.")
######################################################################################################
#nextup: message if we should buy currently or not
#other nice things I can go on and do with this: 
#if in the past 5 days there would have been an entry, and then I can go on and enter the position
#if there would have been an entry, then it should tell me the atr level and the tp level as well with the specifications I told it
#I should also go on and change it so that I can test the strategy and the inputs on out-of-sample data, so that I will run it with 2024-June backwards until 2019 and then run the backtest on data from 2024 June until 2025 June

#next steps within the project: 
#the looped and filtered stocks need to be put into a loop within the backtest that tells if there was a signal in the past 5 days
#in and out of sample data  

# so now the things what I am going to do which is the workflow: 
#   1. optional now: have the backtest for optimization, to optimize the variables and then if it works for a certain period then I can progress onto the next period and then see if the backtest works
#   2. run the signal generation on a certain stock that is defined as the ticker and then if there is a signal then say it, if there was no signal then it should say that there was no signal
#   3. make sure that firstly the screening code is run and after certain stocks are selected they are being used to tell us whether there has been a sign for the past 5 days or not, so here the focus is on integration of the two codes
#   4. after it is done on the selected stocks, then it should tell us whether there has been a sell signal or a buy signal for the certain stocks that we want to trade
#   5. automate that it does this for every single day at certain periods of the day, and then send us an email for every single day so then it can run 24/7
######################################################################################################
# === Previous-day signal summary ===
# Uses your existing combined_df_clean['trade_signal'] with values in {-1, 0, 1}

def show_previous_day_signal(signal: pd.Series, tz="Europe/Budapest"):
    if signal is None or signal.empty:
        st.info("Previous day: 0")
        return

    s = signal.dropna().astype(int)

    # Take the last signal for each calendar day
    day_last = s.groupby(pd.Index(s.index).date).last()
    if day_last.empty:
        st.info("Previous day: 0")
        return

    today = pd.Timestamp.now(tz=tz).date()

    # Use the last *completed* day in your data
    if day_last.index[-1] == today and len(day_last) >= 2:
        target_date = day_last.index[-2]
    else:
        target_date = day_last.index[-1]

    val = int(day_last.get(target_date, 0))

    # Display with the requested styling
    msg = f"Previous day ({target_date}): {val}"
    if val == 1:
        st.success(msg)
    elif val == -1:
        st.error(msg)
    else:
        st.info(msg)

# Call it
show_previous_day_signal(combined_df_clean['trade_signal'])
######################################################################################################
st.subheader("Stock Screening and Signal Detection")

# Define the list of tickers to screen
SCREENING_TICKERS = ["INTC", "TSLA", "NVDA", "LCID", "WBD", "CNC", "AAL", "PSLV", "NIO", "PLTR",
    "QS", "AMD", "F", "RIVN", "VALE", "HBAN", "HIMS", "RIOT", "AAPL", "SOFI",
    "GOOGL", "KEY", "MARA", "ADT", "SOUN", "DOW", "JOBY", "PFE", "HOOD", "SMCI",
    "T", "CMG", "NU", "CMCSA", "ACHR", "CSX", "QBTS", "BTG", "GOOG", "NEM",
    "PCG", "SNAP", "AMZN", "BBD", "CLF", "KGC", "OSCR", "ITUB", "YMM", "PARA",
    "GRAB", "ASTS", "BAC", "RXRX", "NOK", "TRI", "STLA", "RIG", "RGTI", "HL",
    "IREN", "AEO", "FLG", "MSFT", "INFY", "PATH", "OKLO", "RF", "APLD", "NGD",
    "CLSK", "LUV", "CIFR", "RKT", "AGNC", "VZ", "UNH", "BE", "MU", "DOC",
    "MBLY", "BCS", "FCX", "ABEV", "HAL", "EW", "DECK", "KMI", "AG", "IAG",
    "AMCR", "RKLB", "CNH", "WFC", "LYFT", "UBER", "UEC", "C", "IONQ", "CCL", "GM", "ET", "APA", "KGC", "LAUR", "BKR", "MTCH", "KOS", "IAG", "SYF",
    "MAT", "DOLE", "AGNC", "NOV", "MET", "FLEX", "PGR", "HMY", "GS", "UNM",
    "EGO", "ACGL", "BBVA", "BSBR", "CNC", "HIG", "USA", "GDDY", "DBRG", "TRV",
    "BVN", "CIM", "BWA", "GBDC", "CAKE", "SFL", "FOX", "SB", "L", "PBA", "JAZZ",
    "FSLR", "GNK", "ALKS", "DNOW", "PPC", "THC", "VIST", "UHS", "HLIT", "HALO",
    "WFRD", "DVA", "NMR", "ELP", "GTN", "SPNT", "OSK", "EEFT", "TILE", "VAL",
    "MATX", "WES", "PTY", "ETV", "LRN", "IMMR", "TEI", "BRY", "CCS", "MSGE",
    "PBYI", "CNMD", "GBX", "KYN", "UTHR", "REVG", "TIMB", "NXST", "DIN", "VC", "ADX", "EZPW", "KELYA", "CGBD", "ETY", "PCN", "AB", "PFN", "IFN", "ETG",
    "GFF", "SMCI", "IDCC", "OPFI", "MITT", "EARN", "BLX", "SLRC", "HRTG", "EFT",
    "FCT", "EOS", "OPRA", "OPY", "CEPU", "HOV", "FPH", "MKL", "HY", "ASA",
    "RCKY", "JILL", "SGU", "VCTR", "HCI", "SCM", "TZOO", "PDS", "HBB", "NPK",
    "SITC", "VLT", "SPE", "MSB", "GHC", "IX", "WEA", "GAM", "INBK", "MLR",
    "VEL", "WTM", "ODC", "KF", "SOR", "AC", "CEE", "BH", "BRK-A", "AKO-A",
    "GTN-A", "CION", "CMBT", "EMBC", "HG", "NXT", "PR", "SKWD", "TCBX", "VBNK", "TLT", "0700.HK", 'SAP.BD', 'ABGOF', 'BMW.BD', 'NVDA.WA', 'BASF.BD', 'EON.BD', 'MSFT.WA', 'AAPL.WA', 'PBEV',
    'BAYER.BD', 'GOOGL.WA', 'AMZN.WA', 'OTP.BD', 'META.WA', 'RTL.VI', 'SHELL.PR', 'NVDA',
    'TSLA.WA', 'MSFT', 'NVD.DE', 'NVDA.VI', 'NVD.F', 'NVDG.F', 'DTE.PR', 'MSF0.F', 'AAPL',
    'MSFT.VI', 'MSF.DE', 'MSF.F', 'ULVR.PR', 'AEXA.F', 'TRE.VI', 'AAPL.VI', 'APC.DE',
    'APC8.F', 'APC.F', 'JPM.WA', 'AEXA.Y', 'GOOGL', 'GOOG', 'VIG.BD', 'MOL.BD', 'AMZN',
    'AZN.ST', 'ABE0.F', 'ABEC.F', 'ABEC.DE', 'GOOA.VI', 'GOOC.VI', 'ABEA.F', 'GE.VI',
    'ABEA.DE', 'AMZL.F', 'AMZN.VI', 'AMZF.DE', 'AMZ.F', 'META', 'FB2A.F', 'FB2A.DE',
    'META.VI', 'FB20.F', 'SAP.WA', 'AVGO', 'VOW.PR', 'TSM', '1YD.F', 'BROA.VI', '1YD.DE',
    '1YD0.F', 'ABB.ST', 'TSLA', 'TSFA.VI', 'TSFA.F', 'CEEI.F', 'ASML.WA', 'EOAN.PR',
    'BRK-A', 'BRK-B', 'JPM', 'TSLA.VI', 'TL0.DE', 'TL0.F', 'BRYN.F', 'BRH.F', 'BRYN.DE',
    'INVE-A.ST', 'INVE-B.ST', 'BRKA.VI', 'BRKB.VI', 'ERBAG.PR', 'BRHF.F', 'WMT', 'JPM.VI',
    'CMC.DE', 'CMC.F', 'SIE.WA', 'WMT.VI', 'WMTD.F', 'WMT.F', 'WMT.DE', 'ORCL', 'ATCO-B.ST',
    'TCTZ.F', 'ATCO-A.ST', 'CEZ.PR', 'CMC1.F', 'TCEHY', 'V', 'RWE.PR', 'UCG.WA', 'PKO.PR',
    'ORCL.VI', 'ORC.DE', 'ORC.F', 'NNN.F', 'VOLV-B.ST', 'VOLV-A.ST', 'LLY', '3V64.F',
    '3V64.DE', 'NNN1.VI', 'NNND.VI', 'PKN.PR', '3V6.F', 'VISA.VI', 'NNN1.F', 'BTL.SG',
    '4IG.BD', 'VER.PR', 'MA', 'SAN.WA', 'NFLX', 'LLY0.F', 'LLYC.VI', 'OTP.PR', 'LLY.DE',
    'LLY.F', 'NOKIA.PR', 'TSMW.F', 'M4I0.F', 'XOM', 'BAC', 'M4I.F', 'M4I.DE', 'MAST.VI',
    'NFC.F', 'NFLX.VI', 'NFC.DE', 'NFC1.F', 'CHAL.F', 'COST', 'NDA-SE.ST', 'JNJ', 'PLTR',
    'XONA.DE', 'OPUS.BD', 'XONA.F', 'XOM.VI', 'GEN.PR', 'EQT.ST', 'BOAC.VI', 'NCB.DE',
    'NCB.F', 'AGOA.F', 'HD', 'SEB-A.ST', 'SEB-C.ST', 'SHB-B.ST', 'COST.VI', 'CTO.DE',
    'JNJ.F', 'CTO.F', 'RNHEF', 'JNJ.DE', 'PLTR.VI', 'CTO0.F', 'PG', 'IDCBY', 'ASSA-B.ST',
    'JNJ.VI', 'OMV.PR', 'JNJ0.F', 'SSU.SG', 'PTX.DE', 'PTX.F', 'ABBV', 'IDCB.F', 'NCBO.F',
    'SAP.GF', 'SAP', 'JPM-PD', 'HDI0.F', 'HDI.F', 'HDI.DE', 'JPM-PC', 'HD.VI', 'BABA.F',
    'PRG.F',
    
    # Second batch
    'SAAB-B.ST', 'CCC3.SG', 'GCP.F', 'LVMUY', 'PM', 'LVMHF', 'KO.VI', 'EK7A.F', 'GDVTZ',
    'EK7.VI', 'EK7.F', 'WFC', 'CIS0.F', 'CICHY', '2RR.F', 'AHLA.F', 'BML-PJ', 'RHHBF',
    'BAC-PB', 'AHLA.DE', 'MS', 'PCCYF', 'CCC.F', 'ASML.VI', 'ASME.F', 'HESA.F', 'TM',
    'AMD.VI', 'AMD.DE', 'HESA.Y', 'TM5.F', 'ASME.DE', 'TM5.DE', 'AHLA.VI', 'TMUS.VI',
    'ASMF.F', 'RBL.PR', 'RHO.F', 'AMD0.F', 'ASMN.VI', 'CSCO.VI', 'CIS.F', 'RHO.DE',
    'AMD.F', 'CIS.DE', 'ERIC-A.ST', 'ERIC-B.ST', 'RHHVF', 'RHHBY', 'CHV.DE', 'LRLCF',
    'TOYOF', 'NESR.F', 'CHV.F', 'CVX.VI', 'SHB-A.ST', 'NVSEF', 'NESR.DE', 'LRLCY',
    'RHO5.F', 'NVS', 'CICHF', 'MBG.WA', 'UNH', 'C', 'AZNGF', 'MOHE.F', 'HSBC', 'MC.VI',
    'MOH.F', 'NONO.F', 'GS.VI', 'GOS.DE', 'MOH.DE', 'AZN', 'NOT.F', 'NSRGY', 'EPI-A.ST',
    '4I1.F', 'NWT.DE', 'NOT.DE', 'BACHY', 'PMOR.VI', 'NWT.F', 'NSRGF', '4I1.DE', 'NVO',
    'C6TB.F', 'LIN', 'CRM', 'BMW.WA', 'BACHF', 'EPI-B.ST', 'WFC.VI', 'PECN.VI', 'HMIA.F',
    'HML.F', 'CNCB.VI', 'HM-B.ST', 'HBCYF', 'TRVC.DE', 'RMS.VI', 'MCD', 'IBM', 'SHEL',
    'HMIA.SG', 'HML.DE', 'RYDA.F', 'PC6.F', 'TOM.VI', 'CBAU.F', 'UNH.F', 'NOKIA-SEK.ST',
    'MWD.VI', 'DWD.DE', 'TOMA.F', 'SSUN.SG', 'NOTA.F', 'C6T.F', 'TOM.F', 'VOW.WA',
    'DWD.F', 'RHO6.F', 'CMWAY', 'UNH.DE', 'AXP', 'BX', 'SMAW.F', 'SIEGY', 'RTX',
    'LORA.F', 'LOR.F', 'DIS', 'OR.VI',
    
    # Third batch
    'LOR.DE', 'MRK', 'T', 'UNH.VI', 'ABL.F', 'ABL.DE', 'CIL.F', 'INTU', 'PEP', 'HBC1.DE',
    'ZEG.F', 'IBMO.MU', 'AZNA.VI', 'W8VS.F', 'BOCN.VI', 'ZEG.DE', 'ROG.SW', 'CITI.VI',
    'NESM.F', 'CAT', 'CRM.VI', 'KOMB.PR', 'ABT.VI', 'HBC1.F', 'LIN.F', 'IBMO.F', 'TRVC.F',
    'LIN.DE', 'RO.SW', 'FOO.DE', 'ZEGA.F', 'FOO0.F', 'FOO.F', 'HBC2.F', 'WFC-PY', 'MBFJ.F',
    'SHR0.F', 'W8V.F', 'NOVA.F', 'LIN.VI', 'NOVN.SW', 'MDO0.F', 'UBER', 'SHOP', 'MCD.VI',
    'GOS0.F', 'NOVNEE.SW', 'R6C0.F', 'IBM.VI', 'IBM.DE', 'MDO.F', 'NOV.DE', 'NOV.F',
    'MDO.DE', 'R6C0.DE', 'WFC-PL', 'HDB', 'IBM.F', 'RY', 'UT8.SG', 'NESN.SW', 'L3H.F',
    'RLI.VI', 'UT8.F', '5UR.DE', '5UR.F', 'NNO2.VI', 'VZ', 'EVO.ST', 'AEC1.F', 'BBN1.F',
    'ITU.F', 'AEC1.DE', '5UR0.F', 'CWW.F', 'AXP.VI', 'RLI.F', 'GEV', 'SCHW', 'SIE.DE',
    'SIE.F', 'SIE.VI', 'BKNG', 'UNCRY', 'SIEB.F', 'TMO', 'WDP.F', 'MRK.VI', 'CAT1.F',
    '8TRA.ST', 'WDP.DE', 'DIS.VI', 'TSFA.SG', 'NOW', 'UNCFF', 'WDP0.F', 'UT8.DE', 'INTU.VI',
    '6MK.F', 'SVNN.F', 'PEP.F', 'ATT.VI', 'BLK', 'MUFG', '6MK.DE', 'DTEGF', 'ANET', 'SOBA.F',
    'SOBA.DE', 'CAT1.DE', 'PEP.DE', 'ITU.DE', 'DTEGY', 'CAT.F', 'SPGI', 'ALFA.ST', 'CIHKY',
    'PEPS.VI', 'ESSITY-A.ST', 'ESSITY-B.ST', 'CAT.VI', 'FMXUF', 'BA', 'TXN', 'SNEJ.F',
    'ISRG', 'CHL.F', 'UT80.F', 'UBER.VI', 'SONY', '307.F', 'QCOM', 'HUNTF', 'ALIZY',
    'ALIZF', 'BAC.F', 'BAC.DE', 'EADSF', 'EADSY', 'HDFA.F', 'XIACF', 'LATO-B.ST', 'XIACY',
    'CWW0.F', 'RYC.F', 'INDU-A.ST', 'INDU-C.ST', '4S00.F', 'CAT.MU', 'PDD', 'BACB.F',
    'VZ.VI', 'APP', 'BOOK.VI', 'TN80.F', 'SCHW.VI', 'LIFCO-B.ST',
    
    # Fourth batch
    'SNOW.VI', 'Y5C.F', 'IDEXF', '4S0.F', 'SWG.F', '4S0.DE', 'ANDR.PR', 'PCE1.F', 'PCE1.DE',
    'TMOF.VI', 'UCG.VI', 'AMGN', 'MOL.PR', 'BLQ.F', 'KGH.PR', 'AMAT', 'FMX', 'TN8.F', 'BSX',
    'TIL.DE', 'UU2.F', 'UL', 'TIL.F', 'MFZ.F', 'M4BB.F', 'UNLYF', '1170.F', 'UU2.DE', 'PIAI.F',
    'CRIN.F', 'SPGI.VI', 'DTEA.F', 'SAFRY', 'MHL.F', 'GILD', 'MHL.DE', 'DTE.DE', 'SAF.F',
    'DTE.VI', 'ARM', 'DTE.F', 'NEE', 'ACN', 'MFZA.SG', 'TJX', 'UT80.MU', 'VIG.PR', 'ADBE',
    'PNGAY', 'BA.VI', 'BCO.DE', 'BCO0.F', 'ISRG.VI', 'TXN.VI', 'BCO.F', 'IUJ.F', 'SBGSF',
    'SYK', 'PGR', 'MFZA.F', 'SBGSY', 'QCOM.VI', 'ALVE.F', 'QCL.DE', 'ETN', 'BCDRF', 'SAN',
    'SON1.VI', 'ALDB.F', 'SFTBF', 'PROSF', 'M4B.F', 'SON1.F', 'ALV.DE', 'ALV.F', 'PROSY',
    'HON', 'SPOT', 'QCL.F', 'AIRA.F', 'SONA.F', 'AIR.F', 'PFE', 'AIR.DE', 'AIR.VI', 'DE',
    '3CPA.F', 'EMBRAC-B.ST', '3CP.F', 'HTHIF', 'TTE', 'SFTBY', 'CIHHF', 'MU', 'BHPLF', 'AMG.DE',
    'TELIA.ST', 'AMG.F', '9PDA.F', 'TTFNF', 'LOW', 'BHP', 'TD', 'ESL0.F', 'AP2.F', 'ESLOY',
    'GRV.F', 'AAMA.F', 'CHZQ', 'AP2.DE', 'AMAT.VI', 'AMGN.VI', 'UNP', 'DAP.F', 'APH', 'IXD1.F',
    'ITX.VI', 'BSXC.VI', 'LRCX', 'IXD1.DE', 'O9T.F', 'BSX.DE', 'KKR', 'DAP.DE', 'UNVA.F',
    'FP3.F', 'BSX.F', 'UNVB.F', 'LUG.ST', 'UNVB.DE', 'ADBE.VI',
    
    # Fifth batch
    'CSA.F', 'SAF.VI', 'CGXYY', 'BTI', 'SEJ1.DE', 'ADB.DE', 'TJX.DE', 'ADB0.F', 'NEE.VI',
    'GILD.VI', 'TJXC.VI', 'FOMA.F', 'PZX.VI', 'TJX.F', 'ABI.DE', 'GRV.MU', 'SYK.F', 'GIS.F',
    'ADS.WA', 'GIS.DE', 'ADB.F', 'RTNTF', '639.DE', 'SNDB.F', 'PZXB.F', 'DHRC.VI', 'ALD.F',
    'UBS', 'RYCEY', 'ALD.DE', '639.F', 'SYK.VI', 'PFE.DE', 'PZX.F', 'SU.VI', 'ADP', 'PFE.F',
    'SPOT.VI', 'ABJ.F', 'SEJU.F', 'DCO.F', 'ABLZ.F', 'SND.DE', 'SND.F', 'RYCEF', '3EC.F',
    'KLAC', 'PFE.VI', 'DCO.DE', 'DCO0.F', 'UNP.F', 'COP', 'HON.VI', 'SAN.VI', 'BSD2.F',
    'BUD', 'BSD2.DE', '1TY.DE', 'ABBNY', 'BNPQY', 'BNPQ.F', 'PFEB.F', 'CMCSA', 'PRX.VI',
    'LUND-B.ST', '1TY.F', 'OUB.F', 'CSUAY', 'TOTB.DE', 'SNY', 'TOTA.F', 'MU.VI', 'MTE.F',
    'SFT.F', 'BHPL.F', 'TOTB.F', 'MTE.DE', 'SFT.VI', 'IBDSF', 'SFTU.F', '1YL.F', 'RLLV.F',
    'MELI', 'ESLC.F', 'LWE.F', 'BUDF.F', 'WFC-PC', 'LOWE.VI', 'SNYN.F', 'BTAF.F', 'BHP.F',
    'RWE.WA', 'AIQUY', 'IBN', 'EL.VI', 'CEZ.WA', 'AIQU.F', 'APH.VI', 'UNPC.VI', 'IBDRY',
    'ESL.DE', 'ESL.F', 'KR51.F', 'HY9H.F', 'NTDOF', 'ANY.BD', 'LAR0.F', 'XPH.F', 'LRC2.VI',
    'FOMC.F', 'TDB.F', 'DBSDF', 'NTDOY', 'BYNEF', 'FRCO.F', 'DBSDY', 'ADI', 'MO', 'SLL.SG',
    'NKE', 'ISNPY', 'IITSF', 'BMT.F', 'BMT.DE', 'ADP.F', 'CB', 'BAY.WA', 'TEL2-B.ST',
    'BMTA.F', 'TEL2-A.ST', 'SKF-A.ST', 'YCP.DE', 'ADP.VI', 'YCP.F', 'BBVA', 'RRU.F',
    'RRU1.F', 'ICE', 'AXAHY', 'DASH', 'KLAC.VI', 'KLA.DE', 'ITKA.F', 'KLA.F', 'SKF-B.ST',
    'MLB1.F', 'CRWD', 'CRA1.F', 'ALTEO.BD', 'WELL', 'COPH.VI', 'ABJA.F', 'CTP2.F',
    'CTP2.DE', 'AAIGF', 'SBUX', 'CMCS.VI', '2M6.F', 'CEG', '1NBA.DE', 'BIF.BD', '2M6.DE',
    'ZFSVF', '1NBA.F', 'UBSGE.SW', 'ENB', 'MELI.VI', 'PKO.WA', 'SO', 'MLB1.DE',
    
    # Sixth batch (final)
    'BBVXF', 'SMFG', 'SNW.F', 'SANO.VI', 'MDT.VI', 'BNP.F', 'UBSG.SW', 'IKFC.F', 'SNW.DE',
    'HOOD', 'ZFIN.F', 'BNP.DE', 'BNP.VI', 'AAGIY', 'SNW2.F', 'AXAH.F', 'AILA.F', 'BNPH.F',
    '5AP0.F', 'RIO', '12DA.MU', 'ZFIN.DE', 'IBE5.F', 'MMC', 'AI.VI', 'AIL.F', 'IBE1.F',
    'AIL.DE', 'IBE.VI', 'IBE1.DE', 'ICBA.F', 'DEVL.F', 'RLLGF', 'ABBNE.SW', 'ABBN.SW', 'NTO.VI',
    
    "00001.HK", "00002.HK", "00003.HK", "00005.HK", "00006.HK", "00011.HK", 
    "00012.HK", "00016.HK", "00027.HK", "00066.HK", "00083.HK", "00101.HK", 
    "00144.HK", "00175.HK", "00241.HK", "00267.HK", "00285.HK", "00288.HK", 
    "00291.HK", "00300.HK", "00316.HK", "00322.HK", "00338.HK", "00386.HK", 
    "00388.HK", "00669.HK", "00683.HK", "00688.HK", "00700.HK", "00728.HK", 
    "00762.HK", "00788.HK", "00836.HK", "00857.HK", "00868.HK", "00883.HK", 
    "00939.HK", "00941.HK", "00960.HK", "00968.HK", "00981.HK", "00992.HK", 
    "01024.HK", "01038.HK", "01044.HK", "01088.HK", "01093.HK", "01099.HK", 
    "01109.HK", "01113.HK", "01128.HK", "01177.HK", "01193.HK", "01211.HK", 
    "01288.HK", "01299.HK", "01310.HK", "01313.HK", "01336.HK", "01339.HK", 
    "01347.HK", "01359.HK", "01368.HK", "01378.HK", "01398.HK", "01618.HK", 
    "01658.HK", "01698.HK", "01766.HK", "01772.HK", "01788.HK", "01797.HK", 
    "01801.HK", "01810.HK", "01816.HK", "01818.HK", "01833.HK", "01876.HK", 
    "01881.HK", "01898.HK", "01910.HK", "01928.HK", "01929.HK", "01972.HK", 
    "01997.HK", "02013.HK", "02015.HK", "02018.HK", "02020.HK", "02057.HK", 
    "02202.HK", "02208.HK", "02238.HK", "02269.HK", "02313.HK", "02318.HK", 
    "02319.HK", "02328.HK", "02331.HK", "02338.HK", "02382.HK", "02388.HK", 
    "02601.HK", "02607.HK", "02618.HK", "02628.HK", "02688.HK", "02899.HK", 
    "03328.HK", "03618.HK", "03690.HK", "03908.HK", "03968.HK", "03988.HK", 
    "06030.HK", "06060.HK", "06098.HK", "06099.HK", "06160.HK", "06618.HK", 
    "06808.HK", "06818.HK", "06823.HK", "06837.HK", "06862.HK", "06881.HK", 
    "06969.HK", "09618.HK", "09626.HK", "09633.HK", "09863.HK", "09866.HK", 
    "09868.HK", "09880.HK", "09888.HK", "09896.HK", "09961.HK", "09969.HK", 
    "09988.HK", "09992.HK", "09999.HK",
    
    "OTP", "MBHBANK", "RICHTER", "MOL", "MTELEKOM", "4IG", "OPUS", "ANY", 
    "BIF", "ALTEO", "MBHJB", "AUTOWALLIS", "WABERERS", "SPLUS", "ZWACK", 
    "MASTERPLAST", "GSPARK", "BET", "DUNAHOUSE", "APPENINN", "NATURLAND", 
    "CIGPANNONIA", "PANNERGY", "DELTA", "RABA", "MULTIHOME", "AKKO", 
    "GLOSTER", "VERTIKAL", "NAP", "ESENSE", "OXOTECH", "EPDUFERR", "CIVITA", 
    "ASTRASUN", "STRT", "MEGAKRAN", "AMIXA", "DMKER", "VVT", "NUTEX", 
    "ENEFI", "ORMESTER", "NORDGENERAL", "FUTURAQUA",
    
    "600000.SS", "600004.SS", "600009.SS", "600010.SS", "600011.SS", "600015.SS", 
    "600016.SS", "600018.SS", "600019.SS", "600021.SS", "600026.SS", "600027.SS", 
    "600028.SS", "600029.SS", "600030.SS", "600031.SS", "600036.SS", "600048.SS", 
    "600050.SS", "600061.SS", "600066.SS", "600085.SS", "600089.SS", "600096.SS", 
    "600104.SS", "600109.SS", "600111.SS", "600118.SS", "600150.SS", "600160.SS", 
    "600176.SS", "600188.SS", "600196.SS", "600219.SS", "600233.SS", "600271.SS", 
    "600276.SS", "600298.SS", "600309.SS", "600332.SS", "600346.SS", "600362.SS", 
    "600372.SS", "600406.SS", "600418.SS", "600436.SS", "600438.SS", "600460.SS", 
    "600489.SS", "600511.SS", "600519.SS", "600521.SS", "600522.SS", "600547.SS", 
    "600570.SS", "600583.SS", "600585.SS", "600588.SS", "600600.SS", "600612.SS", 
    "600637.SS", "600660.SS", "600690.SS", "600703.SS", "600745.SS", "600760.SS", 
    "600763.SS", "600779.SS", "600795.SS", "600809.SS", "600837.SS", "600845.SS", 
    "600859.SS", "600872.SS", "600875.SS", "600886.SS", "600887.SS", "600893.SS", 
    "600900.SS", "600905.SS", "600919.SS", "600941.SS", "600958.SS", "600989.SS", 
    "600998.SS", "600999.SS", "601012.SS", "601066.SS", "601088.SS", "601098.SS", 
    "601111.SS", "601117.SS", "601138.SS", "601155.SS", "601162.SS", "601166.SS", 
    "601169.SS", "601186.SS", "601198.SS", "601211.SS", "601216.SS", "601225.SS", 
    "601231.SS", "601236.SS", "601238.SS", "601288.SS", "601298.SS", "601319.SS", 
    "601326.SS", "601328.SS", "601336.SS", "601360.SS", "601377.SS", "601388.SS", 
    "601390.SS", "601398.SS", "601555.SS", "601577.SS", "601588.SS", "601600.SS", 
    "601601.SS", "601607.SS", "601618.SS", "601628.SS", "601633.SS", "601658.SS", 
    "601668.SS", "601669.SS", "601688.SS", "601698.SS", "601727.SS", "601728.SS", 
    "601766.SS", "601788.SS", "601799.SS", "601800.SS", "601808.SS", "601811.SS", 
    "601818.SS", "601828.SS", "601838.SS", "601857.SS", "601872.SS", "601877.SS", 
    "601878.SS", "601881.SS", "601888.SS", "601898.SS", "601899.SS", "601901.SS", 
    "601919.SS", "601933.SS", "601939.SS", "601985.SS", "601988.SS", "601989.SS", 
    "601995.SS", "601997.SS", "601998.SS", "601999.SS", "603019.SS", "603160.SS", 
    "603195.SS", "603259.SS", "603260.SS", "603288.SS", "603290.SS", "603369.SS", 
    "603392.SS", "603486.SS", "603501.SS", "603658.SS", "603659.SS", "603799.SS", 
    "603806.SS", "603833.SS", "603939.SS", "603986.SS", "603993.SS", "605117.SS", 
    "605358.SS", "605499.SS", "688001.SS", "688008.SS", "688012.SS", "688036.SS", 
    "688041.SS", "688063.SS", "688098.SS", "688111.SS", "688122.SS", "688169.SS", 
    "688223.SS", "688256.SS", "688271.SS", "688363.SS", "688390.SS", "688561.SS", 
    "688599.SS", "688981.SS",
    
    "AZN.L", "SHEL.L", "HSBA.L", "ULVR.L", "RIO.L", "BP.L", "GSK.L", "REL.L", 
    "DGE.L", "BATS.L", "GLEN.L", "BARC.L", "LLOY.L", "VOD.L", "BA.L", "AAL.L", 
    "AHT.L", "ANTO.L", "AV.L", "HLN.L", "PRU.L", "STAN.L", "NG.L", "SVT.L", 
    "UU.L", "TSCO.L", "SBRY.L", "CPG.L", "IMB.L", "NXT.L", "AUTO.L", "FRES.L", 
    "LSEG.L", "SGE.L", "HLMA.L", "HIK.L", "WTB.L", "ABF.L", "ADM.L", "LGEN.L", 
    "PSON.L", "SSE.L", "CNA.L", "JD.L", "RR.L", "BDEV.L", "PSN.L", "TW.L", 
    "IAG.L", "KGF.L", "WPP.L", "EXPN.L", "RS1.L", "BKG.L", "ITV.L", "PHNX.L", 
    "SMIN.L", "SN.L", "CRDA.L", "WEIR.L", "CTEC.L", "BNZL.L", "CCH.L", "INF.L", 
    "RMV.L", "HSX.L", "RKT.L", "SPX.L", "BT-A.L", "NWG.L", "HBR.L", "III.L", 
    "ENT.L", "FERG.L", "ABDN.L", "ASHM.L", "ASC.L", "BOO.L", "YOU.L", "FEVR.L", 
    "SMWH.L", "JDW.L", "FXPO.L", "CMCX.L", "IGG.L", "HL.L", "FOUR.L", "ESNT.L", 
    "INCH.L", "BAB.L", "BWY.L", "CRST.L", "DLG.L", "RTO.L", "EZJ.L", "IHG.L", 
    "VCT.L", "SMDS.L", "GNS.L", "RWS.L", "WIZZ.L", "SYN.L", "EDV.L", "BRBY.L", 
    "IDS.L", "WOSG.L", "QQ.L", "SPT.L", "TRN.L", "INVP.L", "ITRK.L", "RCH.L", 
    "TIFS.L", "POLR.L", "IMI.L", "SCT.L", "CCC.L", "AVON.L", "BOY.L", "RAT.L", 
    "TPK.L", "PETS.L", "RSW.L", "DNLM.L", "WEIR.L", "BAH.L", "BTEM.L", "SMT.L", 
    "STJ.L", "DLN.L", "LAND.L", "BLND.L", "SGRO.L", "STEM.L", "PAGE.L", "PFG.L", 
    "MTRO.L", "MONY.L", "MNG.L", "CRH.L", "HLCL.L", "FRAS.L", "ICG.L", "JUP.L", 
    "JMAT.L", "ITM.L", "OTB.L", "SAIN.L", "TEP.L", "SVS.L", "FOXT.L", "RWI.L", 
    "CBG.L", "RS1.L", "TRST.L", "BAE.L", "SKG.L", "FOUR.L", "PZC.L", "UU.L", 
    "SVT.L",
    
    "MC.PA", "TTE.PA", "AIR.PA", "SAN.PA", "SU.PA", "AI.PA", "BN.PA", "RI.PA", 
    "DG.PA", "VIE.PA", "BNP.PA", "GLE.PA", "ACA.PA", "CS.PA", "CAP.PA", "HO.PA", 
    "SAF.PA", "EL.PA", "ORA.PA", "CA.PA", "RNO.PA", "PUB.PA", "VIV.PA", "LR.PA", 
    "SGO.PA", "STM.PA", "KER.PA", "WLN.PA", "ENGI.PA", "STLA.PA", "DSY.PA", 
    "EDEN.PA", "TEP.PA", "RMS.PA", "ALO.PA", "ML.PA", "FR.PA", "EN.PA", "IPN.PA", 
    "AF.PA", "ATE.PA", "BVI.PA", "FGR.PA", "RUI.PA", "ERF.PA", "ENX.PA", "SW.PA", 
    "GET.PA", "COFA.PA", "VLA.PA", "DBV.PA", "IPS.PA", "CGG.PA", "ALD.PA", 
    "DIM.PA", "AM.PA", "RCO.PA", "SOP.PA", "SPIE.PA", "NEOEN.PA", "SOI.PA", 
    "AMUN.PA", "MDM.PA", "UBI.PA", "BB.PA", "AKW.PA", "VRLA.PA", "COIL.PA", 
    "ASML.AS", "ADYEN.AS", "PRX.AS", "HEIA.AS", "HEIO.AS", "PHIA.AS", "INGA.AS", 
    "ABN.AS", "AKZA.AS", "ASRNL.AS", "KPN.AS", "NN.AS", "RAND.AS", "AALB.AS", 
    "WKL.AS", "MT.AS", "DSFIR.AS", "BESI.AS", "ASM.AS", "IMCD.AS", "FLOW.AS", 
    "TKWY.AS", "OCI.AS", "SBMO.AS", "TOM2.AS", "AMG.AS", "LIGHT.AS", "POSTNL.AS", 
    "TKH.AS", "BAMNB.AS", "AGN.AS", "AD.AS", "PHARM.AS", "GLPG.AS", "FUR.AS", 
    "BOKA.AS", "ALFEN.AS", "CMCOM.AS", "APAM.AS", "VPK.AS", "JDEP.AS", "ABI.BR", 
    "KBC.BR", "AGS.BR", "UCB.BR", "UMI.BR", "PROX.BR", "COLR.BR", "ACKB.BR", 
    "GBLB.BR", "SOLB.BR", "ELI.BR", "TNET.BR", "BAR.BR", "BPOST.BR", "LOTB.BR", 
    "DIE.BR", "EVS.BR", "IBA.BR", "BEKB.BR", "RECT.BR", "MELE.BR", "KIN.BR", 
    "SOF.BR", "EDP.LS", "EDPR.LS", "GALP.LS", "JMT.LS", "BCP.LS", "NOS.LS", 
    "NVG.LS", "REN.LS", "CTT.LS", "COR.LS", "ALTR.LS", "SEM.LS", "SON.LS", 
    "MOTA.LS", "IBS.LS", "PHR.LS", "NBA.LS", "RYA.IR", "BIRG.IR", "KRZ.IR", 
    "KSP.IR", "SKG.IR", "GL9.IR", "GNC.IR", "A5G.IR", "DALY.IR", "GLN.IR", 
    "GNR.IR", "EQNR.OL", "DNB.OL", "TEL.OL", "NHY.OL", "YAR.OL", "MOWI.OL", 
    "ORK.OL", "AKRBP.OL", "KOG.OL", "FRO.OL", "GOGL.OL", "BWLPG.OL", "SUBC.OL", 
    "ELK.OL", "PGS.OL", "SALM.OL", "NAS.OL", "AKSO.OL", "AKER.OL", "ODL.OL", 
    "HAFNI.OL", "VOW.OL", "HYARD.OL", "OKEA.OL", "TGS.OL", "NOD.OL", "TOM.OL", 
    "NEL.OL", "SCATC.OL", "BORR.OL", "KAHOT.OL", "HEX.OL", "AKAST.OL", "MPCC.OL", 
    "KID.OL", "POM.PA", "VK.PA", "OVH.PA", "FNAC.PA", "TE.PA",
    
    "ADANIENT.NS", "ADANIPORTS.NS", "APOLLOHOSP.NS", "ASIANPAINT.NS", "AXISBANK.NS", 
    "BAJAJ-AUTO.NS", "BAJFINANCE.NS", "BAJAJFINSV.NS", "BEL.NS", "BHARTIARTL.NS", 
    "CIPLA.NS", "COALINDIA.NS", "DRREDDY.NS", "EICHERMOT.NS", "GRASIM.NS", 
    "HCLTECH.NS", "HDFCBANK.NS", "HDFCLIFE.NS", "HEROMOTOCO.NS", "HINDALCO.NS", 
    "HINDUNILVR.NS", "ICICIBANK.NS", "INDUSINDBK.NS", "INFY.NS", "ITC.NS", 
    "JIOFIN.NS", "JSWSTEEL.NS", "KOTAKBANK.NS", "LT.NS", "M&M.NS", "MARUTI.NS", 
    "NESTLEIND.NS", "NTPC.NS", "ONGC.NS", "POWERGRID.NS", "RELIANCE.NS", 
    "SBILIFE.NS", "SHRIRAMFIN.NS", "SBIN.NS", "SUNPHARMA.NS", "TCS.NS", 
    "TATACONSUM.NS", "TATAMOTORS.NS", "TATASTEEL.NS", "TECHM.NS", "TITAN.NS", 
    "TRENT.NS", "ULTRACEMCO.NS", "WIPRO.NS", "ABB.NS", "ADANIENSOL.NS", 
    "ADANIGREEN.NS", "ADANIPOWER.NS", "AMBUJACEM.NS", "BAJAJHLDNG.NS", 
    "BAJAJHFL.NS", "BANKBARODA.NS", "BPCL.NS", "BRITANNIA.NS", "BOSCHLTD.NS", 
    "CANBK.NS", "CGPOWER.NS", "CHOLAFIN.NS", "DABUR.NS", "DIVISLAB.NS", "DLF.NS", 
    "DMART.NS", "GAIL.NS", "GODREJCP.NS", "HAVELLS.NS", "HAL.NS", "HYUNDAI.NS", 
    "ICICIGI.NS", "ICICIPRULI.NS", "INDHOTEL.NS", "IOC.NS", "INDIGO.NS", 
    "NAUKRI.NS", "IRFC.NS", "JINDALSTEL.NS", "JSWENERGY.NS", "LICI.NS", 
    "LODHA.NS", "LTIM.NS", "PIDILITIND.NS", "PFC.NS", "PNB.NS", "RECLTD.NS", 
    "MOTHERSON.NS", "SHREECEM.NS", "SIEMENS.NS", "SWIGGY.NS", "TATAPOWER.NS", 
    "TORNTPHARM.NS", "TVSMOTOR.NS", "UNITDSPR.NS", "VBL.NS", "VEDL.NS", 
    "ZYDUSLIFE.NS", "POLICYBZR.NS", "PETRONET.NS", "POLYCAB.NS", "SAIL.NS", 
    "TIINDIA.NS", "IDEA.NS", "YESBANK.NS", "COLPAL.NS", "MARICO.NS", 
    "MUTHOOTFIN.NS", "AUROPHARMA.NS", "MRF.NS", "OIL.NS", "PHOENIXLTD.NS", 
    "NMDC.NS", "PAGEIND.NS", "FEDERALBNK.NS", "COFORGE.NS", "PERSISTENT.NS", 
    "MAXHEALTH.NS", "HDFCAMC.NS", "LUPIN.NS", "DIXON.NS", "TORNTPOWER.NS", 
    "GODREJPROP.NS", "PRESTIGE.NS", "OBEROIRLTY.NS", "BHARATFORG.NS", "ASTRAL.NS", 
    "SRF.NS", "VOLTAS.NS", "GLAND.NS", "INDUSTOWER.NS", "APLAPOLLO.NS", 
    "CONCOR.NS", "BALKRISIND.NS", "TATACHEM.NS", "PIIND.NS", "LTFH.NS", 
    "GMRINFRA.NS", "ADANIWILMAR.NS", "DEEPAKNTR.NS", "UBL.NS", "BANDHANBNK.NS", 
    "CANFINHOME.NS", "IPCALABS.NS", "SUNTV.NS", "TATACOMM.NS", "GUJGASLTD.NS", 
    "AUBANK.NS",
    
    "ABB.ST","ADDT-B.ST","ALFA.ST","ASSA-B.ST","AZN.ST",
    "ATCO-A.ST","BOL.ST","EPI-A.ST","EQT.ST","ERIC-B.ST",
    "ESSITY-B.ST","EVO.ST","HM-B.ST","HEXA-B.ST","INDU-C.ST",
    "INVE-B.ST","LIFCO-B.ST","NIBE-B.ST","NDA-SE.ST","SAAB-B.ST",
    "SAND.ST","SEB-A.ST","SKA-B.ST","SKF-B.ST","SCA-B.ST",
    "SHB-A.ST","SWED-A.ST","TEL2-B.ST","TELIA.ST","VOLV-B.ST",
    
    "BHP.AX","CBA.AX","CSL.AX","NAB.AX","WBC.AX",
    "ANZ.AX","MQG.AX","WES.AX","WOW.AX","TLS.AX",
    "WDS.AX","RIO.AX","FMG.AX","GMG.AX","TCL.AX",
    "REA.AX","SEK.AX","XRO.AX","WTC.AX","CPU.AX",
    "CAR.AX","NXT.AX","QBE.AX","IAG.AX","SUN.AX",
    "ASX.AX","APA.AX","ORG.AX","AGL.AX","COH.AX",
    "RHC.AX","SHL.AX","RMD.AX","JHX.AX","ALL.AX",
    "COL.AX","EDV.AX","QAN.AX","S32.AX","NST.AX",
    "EVN.AX","IGO.AX","MIN.AX","ALD.AX","AMC.AX",
    "BXB.AX","SCG.AX","SGP.AX","MGR.AX","TPG.AX",
    
    "005930.KS","000660.KS","373220.KS","207940.KS","005380.KS",
    "000270.KS","051910.KS","006400.KS","035420.KS","035720.KS",
    "005490.KS","068270.KS","012330.KS","003670.KS","017670.KS",
    "032830.KS","105560.KS","055550.KS","086790.KS","000810.KS",
    "028260.KS","034730.KS","015760.KS","010950.KS","096770.KS",
    "010130.KS","003550.KS","066570.KS","034220.KS","009540.KS",
    "267250.KS","086280.KS","003490.KS","011200.KS","030200.KS",
    "032640.KS","097950.KS","011170.KS","004020.KS","051900.KS",
    "018260.KS","090430.KS","009150.KS","028050.KS","000720.KS",
    "012450.KS","009830.KS","011070.KS","033780.KS","002380.KS",
    
    "ADEN.SW","AMS.SW","AVOL.SW","BALN.SW","BARN.SW","BEAN.SW","BKWN.SW","CLN.SW","DOCS.SW","EMSN.SW",
    "FHZN.SW","GALN.SW","GFN.SW","HELN.SW","BAER.SW","LISP.SW","MBTN.SW","PSPN.SW","SDZ.SW","SCHP.SW",
    "SIGN.SW","STMN.SW","UHR.SW","SPSN.SW","TECN.SW","TEMN.SW","VACN.SW","BANB.SW","DKSH.SW","SRAIL.SW",
    "ABBN.SW","ALC.SW","CFR.SW","GEBN.SW","GIVN.SW","HOLN.SW","KNIN.SW","LOGN.SW","LONN.SW","NESN.SW",
    "NOVN.SW","PGHN.SW","ROG.SW","SCMN.SW","SIKA.SW","SLHN.SW","SREN.SW","SOON.SW","UBSG.SW","ZURN.SW",
    
    "2330.TW","2317.TW","2454.TW","2308.TW","2382.TW",
    "2412.TW","2881.TW","2882.TW","2891.TW","3711.TW",
    "2886.TW","2884.TW","2303.TW","2357.TW","1216.TW",
    "2603.TW","2885.TW","2892.TW","2887.TW","6505.TW",
    "2880.TW","5880.TW","2890.TW","1303.TW","3045.TW",
    "3008.TW","2207.TW","2002.TW","4904.TW","2327.TW",
    "2379.TW","2395.TW","2912.TW","2883.TW","3034.TW",
    "2615.TW","1301.TW","2801.TW","5876.TW","2609.TW",
    "3037.TW","5871.TW","4938.TW","1101.TW","1326.TW",
    "1590.TW","1402.TW","3231.TW","2345.TW","6669.TW",
    
    "2222.SR", "1120.SR", "1180.SR", "7010.SR", "2010.SR",
    "1211.SR", "5110.SR", "2082.SR", "1150.SR", "1140.SR",
    "1010.SR", "1060.SR", "7203.SR", "2223.SR", "2020.SR",
    "2280.SR", "2050.SR", "2310.SR", "2350.SR", "2380.SR",
    "4003.SR", "4190.SR", "4002.SR", "4004.SR", "7020.SR",
    "7030.SR", "4030.SR", "4300.SR", "4220.SR", "4250.SR",
    "4001.SR", "4161.SR", "4164.SR", "7202.SR", "7200.SR",
    "4005.SR", "4013.SR", "8210.SR", "8010.SR", "4200.SR",
    "4321.SR", "1111.SR", "3030.SR", "3020.SR", "3050.SR",
    "3010.SR", "2330.SR", "2060.SR", "2250.SR", "6004.SR",
    
    "RY.TO","SHOP.TO","TD.TO","BN.TO","BAM.TO","ENB.TO","CNQ.TO","SU.TO","TRP.TO","CNR.TO",
    "CP.TO","BNS.TO","BMO.TO","CM.TO","NA.TO","MFC.TO","SLF.TO","IFC.TO","POW.TO","NTR.TO",
    "TECK-B.TO","ABX.TO","AEM.TO","FNV.TO","WPM.TO","K.TO","FM.TO","CCO.TO","CVE.TO","IMO.TO",
    "PPL.TO","TOU.TO","H.TO","FTS.TO","EMA.TO","RCI-B.TO","T.TO","BCE.TO","TRI.TO","OTEX.TO",
    "CSU.TO","GIB-A.TO","DOL.TO","ATD.TO","L.TO","WN.TO","MRU.TO","SAP.TO","QSR.TO","MG.TO",
    "GIL.TO","CTC-A.TO","CCL-B.TO","WCN.TO","WSP.TO","STN.TO","CAE.TO","TFII.TO","FSV.TO","AQN.TO",
    
    "300896.SZ","300418.SZ","300059.SZ","300017.SZ","300735.SZ","300122.SZ","300408.SZ","300347.SZ","300750.SZ","300760.SZ",
    "300433.SZ","301308.SZ","300308.SZ","300604.SZ","300759.SZ","300496.SZ","300014.SZ","300274.SZ","302132.SZ","300073.SZ",
    "300346.SZ","300373.SZ","300394.SZ","300476.SZ","300339.SZ","300782.SZ","301301.SZ","300413.SZ","300502.SZ","300223.SZ",
    "300124.SZ","300015.SZ","300024.SZ","300724.SZ","300458.SZ","300450.SZ","300001.SZ","300251.SZ","300442.SZ","300763.SZ",
    "300058.SZ","300316.SZ","300033.SZ","301236.SZ","300474.SZ","300207.SZ","300002.SZ","300803.SZ","300115.SZ","300136.SZ",
    "002594.SZ","000157.SZ","300999.SZ","000063.SZ","300661.SZ","000002.SZ","002920.SZ","000596.SZ","002049.SZ","000166.SZ",
    "000776.SZ","000100.SZ","000876.SZ","002459.SZ","300832.SZ","000999.SZ","002304.SZ","000538.SZ","002252.SZ","000938.SZ",
    "000963.SZ","002493.SZ","000338.SZ","000858.SZ","002714.SZ","000768.SZ","002422.SZ","001979.SZ","000977.SZ","001289.SZ",
    "002601.SZ","000661.SZ","002028.SZ","002001.SZ","001872.SZ","002230.SZ","002736.SZ","002648.SZ",
    "000001.SZ","000333.SZ","000651.SZ","000725.SZ","002475.SZ","002241.SZ","002352.SZ","002142.SZ","000625.SZ","002460.SZ",
    "002466.SZ","002371.SZ","002916.SZ","002938.SZ","002841.SZ","002410.SZ","300498.SZ","300142.SZ","002709.SZ","002821.SZ",
    "300628.SZ","002129.SZ","002415.SZ","000568.SZ","002271.SZ","000792.SZ","002311.SZ","002180.SZ","300919.SZ","002938.SZ",
    "002007.SZ","002432.SZ","002271.SZ","002456.SZ","002415.SZ","002594.SZ","002138.SZ","002709.SZ","000063.SZ","002024.SZ",
    "002007.SZ","002602.SZ","002709.SZ","002129.SZ","002841.SZ","002410.SZ","002475.SZ","002241.SZ","002352.SZ","002142.SZ",
    "000333.SZ","000651.SZ","000725.SZ","000001.SZ","000625.SZ","000568.SZ","002460.SZ","002466.SZ","002371.SZ","002916.SZ"]

col1, col2 = st.columns(2)

with col1:
    if st.button("Run Statistical Screening"):
        with st.spinner("Running statistical tests on stocks..."):
            passed_tickers = run_screening(SCREENING_TICKERS, max_half_life=100)
            st.session_state.screened_tickers = passed_tickers
            
        if passed_tickers:
            st.success(f"✅ {len(passed_tickers)} stocks passed mean-reversion tests")
            st.write("**Passed tickers:**", ', '.join(passed_tickers))
        else:
            st.warning("No stocks passed the screening criteria.")

with col2:
    if st.button("Check Trading Signals"):
        # Use all tickers if no screening has been done
        tickers_to_check = st.session_state.get('screened_tickers', SCREENING_TICKERS[:10])
        
        with st.spinner(f"Checking signals for {len(tickers_to_check)} stocks..."):
            buy_signals, sell_signals, signal_details = check_signals_for_tickers(
                tickers_to_check,
                bb_ma, bb_std, rsi_ma, rsi_tresh_l, bb_width_t, 
                atr_mult, profit_mult, atr_length,
            )
        
        # Display results in organized format
        st.markdown("### 📊 Signal Summary")
        
        col_buy, col_sell = st.columns(2)
        
        with col_buy:
            if buy_signals:
                st.success(f"**🟢 BUY Signals ({len(buy_signals)})**")
                for ticker in buy_signals:
                    details = signal_details[ticker]
                    with st.expander(f"{ticker} - {details['days_ago']} day(s) ago"):
                        st.write(f"**Date:** {details['date']}")
                        st.write(f"**Entry Price:** ${details['price']:.2f}")
                        st.write(f"**Stop Loss:** ${details['stop_loss']:.2f} (-{atr_mult:.1f} ATR)")
                        st.write(f"**Take Profit:** ${details['take_profit']:.2f} (+{profit_mult:.1f} ATR)")
                        st.write(f"**RSI:** {details['rsi']:.2f}")
                        st.write(f"**BB Width:** {details['bb_width']:.4f}")
                        st.write(f"**ATR:** ${details['atr']:.2f}")
            else:
                st.info("No buy signals found")
        
        with col_sell:
            if sell_signals:
                st.error(f"**🔴 EXIT Signals ({len(sell_signals)})**")
                for ticker in sell_signals:
                    details = signal_details[ticker]
                    with st.expander(f"{ticker} - {details['days_ago']} day(s) ago"):
                        st.write(f"**Date:** {details['date']}")
                        st.write(f"**Exit Price:** ${details['price']:.2f}")
                        st.write(f"**RSI:** {details['rsi']:.2f}")
                        st.write(f"**BB Width:** {details['bb_width']:.4f}")
            else:
                st.info("No exit signals found")
        
        # Summary statistics
        st.markdown("---")
        st.markdown("### 📈 Summary Statistics")
        col_stat1, col_stat2, col_stat3 = st.columns(3)
        with col_stat1:
            st.metric("Total Stocks Checked", len(tickers_to_check))
        with col_stat2:
            st.metric("Buy Signals", len(buy_signals))
        with col_stat3:
            st.metric("Exit Signals", len(sell_signals))

if 'screened_tickers' not in st.session_state:
    st.info("👆 Press 'Run Statistical Screening' first to identify mean-reverting stocks, or press 'Check Trading Signals' to check a default set.")

st.markdown("---")
######################################################################################################
