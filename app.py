import streamlit as st
import pandas as pd
import numpy as np
import yfinance as yf
import plotly.graph_objects as go
# -----------------------------------------------------------------------------
# Grundkonfiguration
# -----------------------------------------------------------------------------
st.set_page_config(page_title="Wirtschafts-Dashboard", page_icon="📈", layout="wide")
st.title("📈 Mini-Wirtschafts-Dashboard")
st.caption("Tipp: In den App-Settings → Theme den Dark Mode aktivieren")
st.caption("Täglich ~1h coden • Python • GitHub • Live-KPIs")

# -----------------------------------------------------------------------------
# Sidebar (Presets + Zeitraum) mit Session State
# -----------------------------------------------------------------------------
period_map = {"1M": "1mo", "3M": "3mo", "6M": "6mo", "1Y": "1y", "5Y": "5y"}

if "symbols" not in st.session_state:
    st.session_state.symbols = ["^GSPC", "BTC-USD"]
if "rng" not in st.session_state:
    st.session_state.rng = "3M"

with st.sidebar:
    st.header("⚡ Presets")
    row1_c1, row1_c2, row1_c3 = st.columns(3)
    if row1_c1.button("🪙 Krypto"):
      st.session_state.symbols = ["BTC-USD", "ETH-USD"]
    if row1_c2.button("📈 Indizes"):
      st.session_state.symbols = ["^GSPC", "^NDX"]
    if row1_c3.button("💱 Währungen"):
      st.session_state.symbols = ["EURUSD=X", "GBPUSD=X", "JPY=X", "CHFUSD=X"]


    row2_c1, row2_c2 = st.columns(2)
    if row2_c1.button("💻 Tech"):
      st.session_state.symbols = ["AAPL", "TSLA", "NVDA"]
    if row2_c2.button("⛽ Rohstoffe"):
      st.session_state.symbols = ["GC=F", "CL=F"]


    st.write("— oder manuell wählen —")
    st.session_state.symbols = st.multiselect(
    "Assets/Indizes wählen",
    [
        "^GSPC",     # S&P 500 Index
        "^NDX",      # Nasdaq 100 Index
        "BTC-USD",   # Bitcoin
        "ETH-USD",   # Ethereum
        "EURUSD=X",  # Euro / US-Dollar
        "GBPUSD=X", 
        "JPY=X", 
        "CHFUSD=X",
        "GC=F",      # Gold
        "CL=F",      # Öl (WTI Crude)
        "AAPL",      # Apple
        "TSLA",      # Tesla
        "NVDA",      # Nvidia
    ],
    default=st.session_state.symbols
)

    st.session_state.rng = st.selectbox(
        "Zeitraum",
        list(period_map.keys()),
        index=list(period_map.keys()).index(st.session_state.rng)
    )

symbols = st.session_state.symbols
rng = st.session_state.rng

if not symbols:
    st.info("Links etwas auswählen, z. B. **S&P 500 (^GSPC)** oder **Bitcoin (BTC-USD)**.")
    st.stop()

# -----------------------------------------------------------------------------
# Utils
# -----------------------------------------------------------------------------
def has_close_data(df: pd.DataFrame) -> bool:
    try:
        return ("Close" in df.columns) and (not df["Close"].dropna().empty)
    except Exception:
        return False

def get_news(symbol: str, limit: int = 5):
    """Hole die neuesten Nachrichten-Headlines für ein Symbol (falls verfügbar)."""
    try:
        t = yf.Ticker(symbol)
        news = getattr(t, "news", None)
        if not news:
            return []
        return news[:limit]
    except Exception:
        return []
def add_mas(df: pd.DataFrame):
    d = df.copy()
    d["MA20"] = d["Close"].rolling(20).mean()
    d["MA50"] = d["Close"].rolling(50).mean()
    return d

def fig_with_mas(df: pd.DataFrame, sym: str, show_ma20: bool, show_ma50: bool):
    d = add_mas(df)
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=d.index, y=d["Close"], mode="lines", name=f"{sym} Close"))
    if show_ma20 and d["MA20"].notna().any():
        fig.add_trace(go.Scatter(x=d.index, y=d["MA20"], mode="lines", name="MA20"))
    if show_ma50 and d["MA50"].notna().any():
        fig.add_trace(go.Scatter(x=d.index, y=d["MA50"], mode="lines", name="MA50"))
    fig.update_layout(margin=dict(l=0,r=0,t=30,b=0), legend=dict(orientation="h"))
    return fig


def to_scalar(x):
    if isinstance(x, (pd.Series, list, tuple, np.ndarray)):
        return np.nan if len(x) == 0 else to_scalar(x[-1])
    return x.item() if hasattr(x, "item") else x

def fmt(x, unit=""):
    """Zahlen hübsch formatieren, NaN/Inf sicher abfangen."""
    x = to_scalar(x)
    try:
        if x is None or (isinstance(x, (float, int, np.floating, np.integer)) and not np.isfinite(x)):
            return "—"
        if pd.isna(x):
            return "—"
        return f"{float(x):,.2f}{unit}"
    except Exception:
        return "—"

def color_pct_html(x, label):
    """Gibt eine HTML-Zeile mit farbiger % Zahl zurück (grün/rot)."""
    # Leere/ungültige Werte
    if x is None:
        return f"<div><strong>{label}:</strong> —</div>"
    try:
        v = float(to_scalar(x))
        if not np.isfinite(v) or pd.isna(v):
            return f"<div><strong>{label}:</strong> —</div>"
    except Exception:
        return f"<div><strong>{label}:</strong> —</div>"
    color = "green" if v >= 0 else "red"
    return f"<div><strong>{label}:</strong> <span style='color:{color}'>{v:+.2f}%</span></div>"

def pct(a, b):
    """Prozentänderung in % (NaN-sicher)."""
    a, b = to_scalar(a), to_scalar(b)
    try:
        if a is None or b is None or pd.isna(a) or pd.isna(b) or float(b) == 0.0:
            return np.nan
        return (float(a) / float(b) - 1.0) * 100.0
    except Exception:
        return np.nan

def extract_close(df: pd.DataFrame):
    """Gib eine einspaltige Serie mit Schlusskursen zurück – robust gegen MultiIndex/Adj Close."""
    if df is None or df.empty:
        return None
    # MultiIndex-Fälle
    if isinstance(df.columns, pd.MultiIndex):
        try:
            s = df.xs("Close", axis=1, level=0)
        except KeyError:
            try:
                s = df.xs("Adj Close", axis=1, level=0)
            except KeyError:
                s = None
        if s is not None:
            if isinstance(s, pd.DataFrame) and s.shape[1] >= 1:
                s = s.iloc[:, 0]
            return s.dropna() if s is not None else None
    # Normale Spalten
    for name in ["Close", "Adj Close", "close", "adjclose"]:
        if name in df.columns:
            return df[name].dropna()
    # Fallback: letzte Spalte
    try:
        return df.iloc[:, -1].dropna()
    except Exception:
        return None


# -----------------------------------------------------------------------------
# Daten laden (Cache)
# -----------------------------------------------------------------------------
@st.cache_data(ttl=3600, show_spinner=True)
def load(symbols, period):
    out = {}
    for s in symbols:
        df = yf.download(s, period=period, interval="1d", auto_adjust=True, progress=False, group_by="column")
        c = extract_close(df)
        if c is not None and not c.empty:
            out[s] = pd.DataFrame({"Close": c})
    return out

frames = load(symbols, period_map[rng])
if not frames:
    st.error("Keine Daten geladen (Symbol/Zeitraum wechseln und erneut versuchen).")
    st.stop()



# -----------------------------------------------------------------------------
# KPIs & Volatilität
# -----------------------------------------------------------------------------
def kpis(df: pd.DataFrame):
    n = len(df)
    if n < 2:
        return np.nan, np.nan, np.nan, np.nan
    close = df["Close"]
    last = to_scalar(close.iloc[-1])
    prev = to_scalar(close.iloc[-2])
    i7, i30 = max(0, n - 7), max(0, n - 30)
    d = pct(last, prev)
    w7 = pct(last, to_scalar(close.iloc[i7]))
    m30 = pct(last, to_scalar(close.iloc[i30]))
    return last, d, w7, m30

def volatility(df: pd.DataFrame, days: int = 30):
    close = df["Close"].dropna()
    if len(close) < 2:
        return np.nan
    returns = close.pct_change().dropna()
    if len(returns) < days:
        return np.nan
    return returns[-days:].std() * np.sqrt(252) * 100  # annualisiert in %

# -----------------------------------------------------------------------------
# TABS: KPIs / CHARTS / NEWS
# -----------------------------------------------------------------------------
tab_kpi, tab_charts, tab_news = st.tabs(["📊 KPIs", "📈 Charts", "📰 News"])

# ---------- KPI TAB ----------
for sym, df in frames.items():
    if not has_close_data(df):
        st.warning(f"{sym}: Keine Daten verfügbar.")
        continue
    # normale KPI-Berechnung

with tab_kpi:
    cols = st.columns(len(frames))
    rows = []

    for i, (sym, df) in enumerate(frames.items()):
        price, d, w, m = kpis(df)
        vol = volatility(df)

        with cols[i]:
            # BTC-Logo neben Titel
            if sym == "BTC-USD":
                head_l, head_r = st.columns([1, 5])
                with head_l:
                    try:
                        st.image("bitcoin_PNG7.png", width=28)
                    except Exception:
                        pass
                with head_r:
                    st.subheader(sym)
            else:
                st.subheader(sym)
            delta = fmt(d, "%")  # d ist dein 24h %
            st.metric("Preis", fmt(price), delta=delta)


            c1, c2 = st.columns(2)
            with c1:
              st.markdown(color_pct_html(d, "24h"), unsafe_allow_html=True)
            with c2:
              st.markdown(color_pct_html(w, "7 Tage"), unsafe_allow_html=True)

            st.markdown(color_pct_html(m, "30 Tage"), unsafe_allow_html=True)
            st.markdown(color_pct_html(vol, "Volatilität (30T)"), unsafe_allow_html=True)




        rows.append({
            "Symbol": sym,
            "Preis": to_scalar(price),
            "24h_%": to_scalar(d),
            "7d_%": to_scalar(w),
            "30d_%": to_scalar(m),
            "Vol_30T_%": to_scalar(vol),
        })

    kpi_df = pd.DataFrame(rows)
    # Button oben rechts ausrichten
    btn_l, btn_r = st.columns([3, 1])
    with btn_r:
        st.download_button(
            label="⬇️ KPIs als CSV",
            data=kpi_df.to_csv(index=False).encode("utf-8"),
            file_name="kpis.csv",
            mime="text/csv",
            key="kpi_csv_download_button"
        )

# ---------- CHARTS TAB ----------
with tab_charts:
    sub1, sub2 = st.tabs(["📉 Verlauf", "📊 Korrelation"])
    show_ma20 = st.checkbox("MA20 anzeigen", value=True)
    show_ma50 = st.checkbox("MA50 anzeigen", value=False)

    with sub1:
      for sym, df in frames.items():
        st.write(f"**{sym}**")
        fig = fig_with_mas(df, sym, show_ma20, show_ma50)
        st.plotly_chart(fig, use_container_width=True)

    with sub2:
    # Robuster Merge für unterschiedlich lange Indizes
      series_list = []
      for sym, df in frames.items():
        if "Close" in df.columns and not df["Close"].empty:
            series_list.append(df["Close"].rename(sym))

      if series_list:
        merged = pd.concat(series_list, axis=1)  # automatisch am Index ausrichten
        corr = merged.pct_change().corr().round(2)
        st.dataframe(corr, use_container_width=True)
      else:
        st.info("Keine Daten für Korrelation verfügbar.")


# ---------- NEWS TAB ----------
with tab_news:
    st.subheader("Aktuelle Nachrichten")
    for sym in symbols:
        items = get_news(sym, limit=5)
        if not items:
            st.write(f"Keine News gefunden für {sym}")
            continue
        st.markdown(f"### {sym}")
        for item in items:
            title = item.get("title", "Ohne Titel")
            link = item.get("link", "#")
            st.markdown(f"- [{title}]({link})")

# -----------------------------------------------------------------------------
# Hinweis
# -----------------------------------------------------------------------------
st.caption("ℹ️ Symbole: ^GSPC=S&P 500, ^NDX=Nasdaq 100, BTC-USD=Bitcoin, EURUSD=X=Euro/US-Dollar, GC=F=Gold")

