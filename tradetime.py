import yfinance as yf
import pandas as pd
import numpy as np
import pytz
from datetime import datetime
import streamlit as st
import plotly.graph_objects as go

# ---------------- Constants ----------------
RISK_PERCENTAGE = 0.5  # % of account per trade
STOP_MULTIPLIER = 1.5  # ATR multiple for stop-loss
TARGET_PCT = 0.01      # 1% profit target
TIMEZONE = pytz.timezone("Asia/Kolkata")

# ---------------- Helper Functions ----------------
def safe_float(x):
    if isinstance(x, pd.Series):
        return float(x.iloc[0])
    return float(x)

def fetch_data(symbol, period="5d", interval="5m"):
    df = yf.download(symbol, period=period, interval=interval, progress=False)
    if df.empty:
        return df
    df.index = pd.to_datetime(df.index)
    if df.index.tz is None:
        df.index = df.index.tz_localize('UTC')
    df.index = df.index.tz_convert(TIMEZONE)
    return df.dropna()

def calculate_ema(df, span):
    return df['Close'].ewm(span=span, adjust=False).mean()

def calculate_atr(df, period=14):
    high, low, close = df['High'], df['Low'], df['Close']
    tr = pd.concat([high - low, (high - close.shift()).abs(), (low - close.shift()).abs()], axis=1).max(axis=1)
    return tr.rolling(window=period).mean()

def calculate_vwap(df):
    return (df['Close'] * df['Volume']).cumsum() / df['Volume'].cumsum()

def generate_signal(latest):
    try:
        ema8 = safe_float(latest['EMA8'])
        ema21 = safe_float(latest['EMA21'])
        vwap = safe_float(latest['VWAP'])
        close = safe_float(latest['Close'])
    except Exception:
        return 'HOLD'

    if any(np.isnan(x) for x in [ema8, ema21, vwap, close]):
        return 'HOLD'

    if ema8 > ema21 and close > vwap:
        return 'BUY'
    elif ema8 < ema21 and close < vwap:
        return 'SELL'
    else:
        return 'HOLD'

def calculate_position_size(account_balance, atr, price):
    stop_loss_distance = atr * STOP_MULTIPLIER
    risk_amount = account_balance * (RISK_PERCENTAGE / 100)
    qty = int(risk_amount / stop_loss_distance) if stop_loss_distance > 0 else 0
    return max(qty, 0)

def plot_chart(df, signal, symbol):
    latest_time_str = df.index[-1].strftime("%Y-%m-%d %H:%M:%S %Z")
    latest_price = safe_float(df['Close'].iloc[-1])

    fig = go.Figure()
    fig.add_trace(go.Candlestick(
        x=df.index,
        open=df['Open'], high=df['High'], low=df['Low'], close=df['Close'],
        name='Candlesticks'
    ))
    fig.add_trace(go.Scatter(x=df.index, y=df['EMA8'], mode='lines', name='EMA8', line=dict(color='orange')))
    fig.add_trace(go.Scatter(x=df.index, y=df['EMA21'], mode='lines', name='EMA21', line=dict(color='blue')))
    fig.add_trace(go.Scatter(x=df.index, y=df['VWAP'], mode='lines', name='VWAP', line=dict(color='green')))

    # Signal arrow
    if signal != 'HOLD':
        marker_color = 'lime' if signal=='BUY' else 'red'
        fig.add_trace(go.Scatter(
            x=[df.index[-1]],
            y=[latest_price],
            mode='markers+text',
            marker=dict(symbol='arrow-up' if signal=='BUY' else 'arrow-down', color=marker_color, size=15),
            text=[signal],
            textposition='top center' if signal=='BUY' else 'bottom center',
            name='Signal'
        ))

    # Latest time marker
    fig.add_trace(go.Scatter(
        x=[df.index[-1]],
        y=[latest_price],
        mode='markers+text',
        marker=dict(symbol='circle', color='yellow', size=10),
        text=[df.index[-1].strftime("%H:%M:%S")],
        textposition='bottom center',
        name='Latest Time'
    ))

    fig.update_layout(
        title=f"{symbol} — Last Data: {latest_time_str}",
        template='plotly_dark',
        height=500,
        yaxis=dict(autorange=True)
    )

    return fig

# ---------------- Streamlit App ----------------
st.set_page_config(page_title="AI Intraday Scanner", layout="wide")
st.title("📈 AI Intraday Nifty50 Scanner (Real-time)")
st.caption("EMA8 / EMA21 + VWAP Scanner — Educational Use Only")

account_balance = st.number_input("Account Size (₹)", value=100000.0, step=5000.0)

if st.button("Run Scanner"):
    symbols = ['RELIANCE.NS', 'INFY.NS', 'TCS.NS', 'HDFCBANK.NS', 'ICICIBANK.NS',
               'SBIN.NS', 'LT.NS', 'AXISBANK.NS', 'ITC.NS', 'ONGC.NS', 'HINDUNILVR.NS',
               'MARUTI.NS', 'BHARTIARTL.NS', 'BAJFINANCE.NS', 'ASIANPAINT.NS',
               'TECHM.NS', 'KOTAKBANK.NS', 'ULTRACEMCO.NS', 'HCLTECH.NS', 'ADANIPORTS.NS']

    watchlist = []
    for symbol in symbols:
        df = fetch_data(symbol)
        if df.empty:
            continue
        df['EMA8'] = calculate_ema(df, 8)
        df['EMA21'] = calculate_ema(df, 21)
        df['ATR5'] = calculate_atr(df, 5)
        df['VWAP'] = calculate_vwap(df)

        latest = df.iloc[-1]
        signal = generate_signal(latest)
        if signal != 'HOLD':
            price = safe_float(latest['Close'])
            atr = safe_float(latest['ATR5'])
            qty = calculate_position_size(account_balance, atr, price)
            stop_loss = price - STOP_MULTIPLIER*atr if signal=='BUY' else price + STOP_MULTIPLIER*atr
            target = price * (1 + TARGET_PCT) if signal=='BUY' else price * (1 - TARGET_PCT)
            strength = abs(safe_float(latest['EMA8']) - safe_float(latest['EMA21']))

            watchlist.append({
                'Symbol': symbol,
                'Signal': signal,
                'Price': round(price,2),
                'Stop-Loss': round(stop_loss,2),
                'Target (1%)': round(target,2),
                'Qty': qty,
                'Strength': round(strength,4)
            })

    df_watch = pd.DataFrame(watchlist)
    if not df_watch.empty:
        df_watch = df_watch.sort_values(by='Strength', ascending=False)
        st.subheader("📊 Ranked Intraday Watchlist")
        st.dataframe(df_watch.reset_index(drop=True))

        st.subheader("⭐ Top 5 BUY/SELL Calls")
        st.dataframe(df_watch.head(5).reset_index(drop=True))

        # Show chart for top 1 signal
        top1 = df_watch.iloc[0]
        df_top = fetch_data(top1['Symbol'])
        df_top['EMA8'] = calculate_ema(df_top, 8)
        df_top['EMA21'] = calculate_ema(df_top, 21)
        df_top['VWAP'] = calculate_vwap(df_top)
        st.plotly_chart(plot_chart(df_top.tail(200), top1['Signal'], top1['Symbol']), use_container_width=True)
    else:
        st.info("No BUY/SELL opportunities detected right now.")
