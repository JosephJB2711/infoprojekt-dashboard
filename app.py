import streamlit as st
import pandas as pd
import numpy as np
import yfinance as yf

st.set_page_config(page_title="Wirtschafts-Dashboard", page_icon="📈", layout="wide")
st.title("📈 Mini-Wirtschafts-Dashboard")
st.caption("Täglich ~1h coden • Python • GitHub • Live-KPIs")

# --- Sidebar-Auswahl
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
        if not df.empty and "Close" in df.columns:
            df = df.dropna(subset=["Close"])
            if not df.empty:
                df["Symbol"] = s
                out[s] = df
    return out

frames = load(symbols, period_map[rng])
if not frames:
    st.error("Keine Daten geladen (evtl. Symbol/Zeitraum ändern).")
    st.stop()

# --- Helpers
def to_scalar(x):
    """Robust: extrahiere Einzelwert aus Series/ndarray/list, sonst unverändert zurück."""
    try:
        if isinstance(x, (pd.Series, list, tuple, np.ndarray)):
            if len(x) == 0:
                return np.nan
            return to_scalar(x[-1])  # nimm den letzten Wert
        if hasattr(x, "item"):
            return x.item()
    except Exception:
        pass
    return x

def fmt(x, unit=""):
    """Sichere Formatierung mit NaN/Inf-Handling."""
    x = to_scalar(x)
    try:
        # None, NaN, +/-Inf
        if x is None or (isinstance(x, (float, int, np.floating, np.integer)) and not np.isfinite(x)):
            return "—"
        # pandas NA
        if pd.isna(x):
            return "—"
        return f"{float(x):,.2f}{unit}"
    except Exception:
        return "—"

def pct(a, b):
    """Prozentänderung in %, NaN-sicher."""
    a, b = to_scalar(a), to_scalar(b)
    if a is None or b is None:
        return np.nan
    try:
        if pd.isna(a) or pd.isna(b) or b == 0:
            return np.nan
    except Exception:
        pass
    try:
        return (float(a) / float(b) - 1.0) * 100.0
    except Exception:
        return np.nan

# --- KPI-Berechnung
def kpis(df: pd.DataFrame):
    n = len(df)
    if n < 2:
        return np.nan, np.nan, np.nan, np.nan
    close = df["Close"]
    last = to_scalar(close.iloc[-1])
    prev = to_scalar(close.iloc[-2])
    i7 = max(0, n - 7)
    i30 = max(0, n - 30)
    d = pct(last, prev)
    w7 = pct(last, to_scalar(close.iloc[i7]))
    m30 = pct(last, to_scalar(close.iloc[i30]))
    return last, d, w7, m30

# --- KPI-Kacheln
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
