import yfinance as yf
import pandas as pd
import numpy as np
import pytz
from datetime import datetime
import streamlit as st
import plotly.graph_objects as go
import os
import schedule
import threading
import time
import requests

# ---------------- Configuration ----------------
EXCEL_FILE = "paper_orders.xlsx"
RISK_PERCENTAGE = 0.5      # % of account per trade
STOP_MULTIPLIER = 1.5      # ATR multiple for stop-loss
TARGET_PCT = 0.01          # 1% profit target
TIMEZONE = pytz.timezone("Asia/Kolkata")

# --- Telegram ---
TELEGRAM_BOT_TOKEN = "8320189677:AAEH-fBVQ5auPjyBZ9Bvg1mHKYV6F_wt3kE"  # replace with your token
TELEGRAM_CHAT_ID = "5546511884"  # replace with your chat ID

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

def append_order_to_excel(new_order):
    if os.path.exists(EXCEL_FILE):
        df_existing = pd.read_excel(EXCEL_FILE)
        df_full = pd.concat([df_existing, pd.DataFrame([new_order])], ignore_index=True)
    else:
        df_full = pd.DataFrame([new_order])
    df_full.to_excel(EXCEL_FILE, index=False)

def send_telegram_message(message: str):
    url = f"https://api.telegram.org/bot{TELEGRAM_BOT_TOKEN}/sendMessage"
    payload = {"chat_id": TELEGRAM_CHAT_ID, "text": message, "parse_mode": "Markdown"}
    try:
        requests.post(url, data=payload, timeout=10)
    except Exception as e:
        print("❌ Telegram send failed:", e)

# ---------------- Auto Scanner ----------------
def run_auto_scanner():
    symbols = [
        'RELIANCE.NS', 'INFY.NS', 'TCS.NS', 'HDFCBANK.NS', 'ICICIBANK.NS', 'SBIN.NS',
        'LT.NS', 'AXISBANK.NS', 'ITC.NS', 'ONGC.NS', 'HINDUNILVR.NS', 'MARUTI.NS',
        'BHARTIARTL.NS', 'BAJFINANCE.NS', 'ASIANPAINT.NS', 'TECHM.NS',
        'KOTAKBANK.NS', 'ULTRACEMCO.NS', 'HCLTECH.NS', 'ADANIPORTS.NS'
    ]
    account_balance = 100000
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
                'Price': round(price, 2),
                'Stop-Loss': round(stop_loss, 2),
                'Target (1%)': round(target, 2),
                'Qty': qty,
                'Strength': round(strength, 4)
            })

    if not watchlist:
        send_telegram_message("⚠️ No active BUY/SELL signals found.")
        return

    df_watch = pd.DataFrame(watchlist).sort_values(by='Strength', ascending=False)
    top5 = df_watch.head(5)

    msg = "📊 *Top 5 Intraday Signals*\n\n"
    for _, row in top5.iterrows():
        emoji = "🟢" if row['Signal'] == 'BUY' else "🔴"
        msg += f"{emoji} *{row['Symbol']}* — {row['Signal']}\n"
        msg += f"💰 ₹{row['Price']} | 🎯 ₹{row['Target (1%)']} | 🛑 ₹{row['Stop-Loss']}\n\n"
    msg += f"_Updated at {datetime.now().strftime('%d-%b %H:%M:%S')}_"

    send_telegram_message(msg)
    print("✅ Telegram update sent at", datetime.now())

# ---------------- Scheduler ----------------
def start_scheduler():
    schedule.every(3).minutes.do(run_auto_scanner)
    while True:
        schedule.run_pending()
        time.sleep(30)

# ---------------- Streamlit App ----------------
st.set_page_config(page_title="AI Intraday Scanner + Telegram", layout="wide")
st.title("📈 AI Intraday Nifty50 Scanner (Auto + Telegram)")
st.caption("Sends Top 5 BUY/SELL calls to Telegram every 15 minutes")

if 'scheduler_started' not in st.session_state:
    threading.Thread(target=start_scheduler, daemon=True).start()
    st.session_state.scheduler_started = True
    st.success("📡 Scheduler started — will send Telegram updates every 15 min.")

if st.button("Run Scanner Now"):
    run_auto_scanner()
    st.info("Manual scan executed and message sent to Telegram.")
