import streamlit as st
import yfinance as yf
import pandas as pd
import os

EXCEL_FILE = "paper_orders.xlsx"

def load_orders():
    if os.path.exists(EXCEL_FILE):
        return pd.read_excel(EXCEL_FILE)
    return pd.DataFrame()

st.set_page_config(page_title="📈 Paper Trading Dashboard", layout="wide")
st.title("📋 Paper Trading Order Book")

if 'paper_orders' not in st.session_state:
    st.session_state.paper_orders = load_orders()

def fetch_live_price(symbol):
    try:
        data = yf.download(tickers=symbol, period="1d", interval="1m", progress=False)
        if not data.empty:
            price = data["Close"].iloc[-1]
            return float(price)
    except:
        pass
    return None

def refresh_prices():
    df = st.session_state.paper_orders.copy().reset_index(drop=True)
    if df.empty:
        return df

    if "Live Price" not in df.columns:
        df["Live Price"] = None
    if "P&L" not in df.columns:
        df["P&L"] = None

    for i in range(len(df)):
        symbol = df.iloc[i]["Symbol"]
        price = fetch_live_price(symbol)
        if isinstance(price, (float, int)):
            df.iloc[i, df.columns.get_loc("Live Price")] = price
        else:
            df.iloc[i, df.columns.get_loc("Live Price")] = None

        if price is not None:
            qty = df.iloc[i]["Qty"]
            entry = df.iloc[i]["Entry Price"]
            signal = df.iloc[i]["Signal"]
            if signal == "BUY":
                pnl = round((price - entry) * qty, 2)
            else:
                pnl = round((entry - price) * qty, 2)
            df.iloc[i, df.columns.get_loc("P&L")] = pnl
        else:
            df.iloc[i, df.columns.get_loc("P&L")] = None
    return df

col1, col2 = st.columns([1, 1])
with col1:
    if st.button("🔄 Refresh Prices"):
        df_updated = refresh_prices()
        st.session_state.paper_orders = df_updated
        st.success("Live prices updated successfully!")
with col2:
    if st.button("↩️ Reload from Excel"):
        st.session_state.paper_orders = load_orders()
        st.info("Reloaded latest data from Excel")

df_display = st.session_state.paper_orders.copy()

if df_display.empty:
    st.info("No paper trading orders placed yet.")
else:
    if "Live Price" not in df_display.columns:
        df_display["Live Price"] = None
    if "P&L" not in df_display.columns:
        df_display["P&L"] = None

    def color_pnl(val):
        try:
            if pd.isna(val):
                return ""
            if val > 0:
                return "color: green; font-weight: bold"
            elif val < 0:
                return "color: red; font-weight: bold"
            else:
                return ""
        except:
            return ""

    total_open = len(df_display[df_display["Status"] == "Open"])
    total_closed = len(df_display[df_display["Status"] == "Closed"])

    total_pnl = 0.0
    if "P&L" in df_display.columns:
        total_pnl = pd.to_numeric(df_display["P&L"], errors='coerce').fillna(0).sum()

    kpicol = st.columns(3)
    kpicol[0].metric("Open Trades", total_open)
    kpicol[1].metric("Closed Trades", total_closed)
    kpicol[2].metric("Net P&L (₹)", f"{total_pnl:,.2f}")

    st.markdown("---")
    st.dataframe(
        df_display.style.applymap(color_pnl, subset=["P&L"]),
        use_container_width=True,
        height=450,
    )

    st.caption("⚠️ This page reads from Excel and updates live prices only when you click 'Refresh Prices'.")
