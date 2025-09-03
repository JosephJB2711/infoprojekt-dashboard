import streamlit as st
import pandas as pd
import numpy as np
import yfinance as yf

st.set_page_config(page_title="Wirtschafts-Dashboard", page_icon="📈", layout="wide")
st.title("📈 Mini-Wirtschafts-Dashboard")
st.caption("Täglich ~1h coden • Python • GitHub • Live-KPIs")

# --- Auswahl links
symbols = st.sidebar.multiselect(
    "Assets/Indizes wählen",
    ["^GSPC", "^NDX", "BTC-USD", "ETH-USD", "EURUSD=X", "GC=F"],
    default=["^GSPC", "BTC-USD"]
)
period_map = {"1M": "1mo", "3M": "3mo", "6M": "6mo", "1Y": "1y", "5Y": "5y"}
rng = st.sidebar.selectbox("Zeitraum", list(period_map.keys()), index=1)

if not symbols:
    st.info("Links etwas auswählen, z. B. **S&P 500 (^GSPC)** oder **Bitcoin (BTC-USD)**.")
    st.stop()

# --- Daten laden (gecacht)
@st.cache_data(ttl=3600, show_spinner=False)
def load(symbols, period):
    out = {}
    for s in symbols:
        df = yf.download(s, period=period, interval="1d", auto_adjust=True, progress=False)
        if not df.empty:
            df["Symbol"] = s
            out[s] = df
    return out

frames = load(symbols, period_map[rng])
if not frames:
    st.error("Keine Daten geladen.")
    st.stop()

# --- KPI-Berechnung
def kpis(df):
    if df.shape[0] < 2:
        return np.nan, np.nan, np.nan, np.nan
    last = df["Close"].iloc[-1]
    prev = df["Close"].iloc[-2]
    i7 = max(0, len(df) - 7)
    i30 = max(0, len(df) - 30)
    day = (last / prev - 1) * 100
    w7 = (last / df["Close"].iloc[i7] - 1) * 100
    m30 = (last / df["Close"].iloc[i30] - 1) * 100
    return last, day, w7, m30

# --- KPI-Kacheln
def fmt(x, unit=""):
    if pd.isna(x):
        return "—"
    return f"{x:,.2f}{unit}"

cols = st.columns(len(frames))
for i, (sym, df) in enumerate(frames.items()):
    price, d, w, m = kpis(df)
    with cols[i]:
        st.subheader(sym)
        st.metric("Preis", fmt(price))
        c1, c2 = st.columns(2)
        c1.metric("24h", fmt(d, "%"))
        c2.metric("7 Tage", fmt(w, "%"))
        st.caption(f"30 Tage: {fmt(m, '%')}")

st.markdown("---")

# --- Charts
tab1, tab2 = st.tabs(["📉 Verlauf", "📊 Korrelation"])
with tab1:
    for sym, df in frames.items():
        st.write(f"**{sym}**")
        st.line_chart(df["Close"])

with tab2:
    merged = None
    for sym, df in frames.items():
        s = df["Close"].rename(sym)
        merged = s if merged is None else merged.to_frame().join(s, how="outer")
    corr = merged.pct_change().corr().round(2)
    st.dataframe(corr, use_container_width=True)

st.caption("ℹ️ Symbole: ^GSPC=S&P 500, ^NDX=Nasdaq 100, BTC-USD=Bitcoin, EURUSD=X=Euro/US-Dollar, GC=F=Gold")
