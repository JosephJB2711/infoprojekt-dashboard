import streamlit as st
import pandas as pd
import numpy as np
import yfinance as yf
st.set_page_config(page_title="Wirtschafts-Dashboard", page_icon="📈", layout="wide")

st.title("📈 Mini-Wirtschafts-Dashboard")
st.caption("Tipp: In den App-Settings → Theme den Dark Mode aktivieren")
st.caption("Täglich ~1h coden • Python • GitHub • Live-KPIs")

# --- Sidebar-Auswahl
# --- Sidebar mit Presets & State ----------------------------------
period_map = {"1M": "1mo", "3M": "3mo", "6M": "6mo", "1Y": "1y", "5Y": "5y"}

# Initiale Defaults nur einmal setzen
if "symbols" not in st.session_state:
    st.session_state.symbols = ["^GSPC", "BTC-USD"]
if "rng" not in st.session_state:
    st.session_state.rng = "3M"

with st.sidebar:
    st.header("⚡ Presets")
    cols = st.columns(3)
    if cols[0].button("Krypto"):
        st.session_state.symbols = ["BTC-USD", "ETH-USD"]
    if cols[1].button("Aktien"):
        st.session_state.symbols = ["^GSPC", "^NDX"]
    if cols[2].button("Währungen"):
        st.session_state.symbols = ["EURUSD=X", "GC=F"]

    # Manuelle Auswahl bleibt möglich
    st.write("— oder manuell wählen —")
    st.session_state.symbols = st.multiselect(
        "Assets/Indizes wählen",
        ["^GSPC", "^NDX", "BTC-USD", "ETH-USD", "EURUSD=X", "GC=F"],
        default=st.session_state.symbols
    )

    st.session_state.rng = st.selectbox("Zeitraum", list(period_map.keys()),
                                        index=list(period_map.keys()).index(st.session_state.rng))

symbols = st.session_state.symbols
rng = st.session_state.rng


if not symbols:
    st.info("Links etwas auswählen, z. B. **S&P 500 (^GSPC)** oder **Bitcoin (BTC-USD)**.")
    st.stop()

# --- Utils --------------------------------------------------------------------
def get_news(symbol: str, limit: int = 5):
    """Hole die neuesten Nachrichten-Headlines für ein Symbol (falls verfügbar)."""
    try:
        t = yf.Ticker(symbol)
        if hasattr(t, "news"):
            news = t.news[:limit]
            return news
    except Exception:
        return []
    return []

def to_scalar(x):
    if isinstance(x, (pd.Series, list, tuple, np.ndarray)):
        return np.nan if len(x) == 0 else to_scalar(x[-1])
    return x.item() if hasattr(x, "item") else x

def fmt(x, unit=""):
    x = to_scalar(x)
    try:
        if x is None or (isinstance(x, (float, int, np.floating, np.integer)) and not np.isfinite(x)):
            return "—"
        if pd.isna(x):
            return "—"
        return f"{float(x):,.2f}{unit}"
    except Exception:
        return "—"

def pct(a, b):
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
    # MultiIndex (kommt vor, je nach yfinance-Version/Symbol)
    if isinstance(df.columns, pd.MultiIndex):
        try:
            s = df.xs("Close", axis=1, level=0)
        except KeyError:
            # Fallback: Adj Close
            try:
                s = df.xs("Adj Close", axis=1, level=0)
            except KeyError:
                s = None
        if s is not None:
            # Wenn DataFrame mit einer Spalte -> Serie daraus machen
            if isinstance(s, pd.DataFrame):
                if s.shape[1] >= 1:
                    s = s.iloc[:, 0]
            return s.dropna() if s is not None else None

    # Normaler DataFrame
    for name in ["Close", "Adj Close", "close", "adjclose"]:
        if name in df.columns:
            return df[name].dropna()

    # letzter Fallback: nimm letzte Spalte
    try:
        return df.iloc[:, -1].dropna()
    except Exception:
        return None

# --- Daten laden (gecacht) ----------------------------------------------------
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

# --- KPIs ---------------------------------------------------------------------
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
    """Berechnet die annualisierte Volatilität aus täglichen Renditen (Standardabweichung)."""
    close = df["Close"].dropna()
    if len(close) < 2:
        return np.nan
    returns = close.pct_change().dropna()
    if len(returns) < days:
        return np.nan
    vol = returns[-days:].std() * np.sqrt(252) * 100  # annualisiert, in %
    return vol

# --- KPI-Kacheln (mit BTC-Logo neben dem Titel) -------------------

cols = st.columns(len(frames))
for i, (sym, df) in enumerate(frames.items()):
    price, d, w, m = kpis(df)

    with cols[i]:
        # Überschrift: für BTC-USD Logo + Text nebeneinander
        if sym == "BTC-USD":
            head_l, head_r = st.columns([1, 5])
            with head_l:
                try:
                    st.image("bitcoin_PNG7.png", width=28)  # Bild liegt im Repo-Root
                except Exception:
                    pass  # falls Bild fehlt, einfach ohne
            with head_r:
                    st.markdown(f"### {sym}")

        else:
            st.subheader(sym)

        # KPIs
        st.metric("Preis", fmt(price))
        c1, c2 = st.columns(2)
        c1.metric("24h",  fmt(d, "%"))
        c2.metric("7 Tage", fmt(w, "%"))
        st.caption(f"30 Tage: {fmt(m, '%')}")
        vol = volatility(df)
        st.caption(f"Volatilität (30T): {fmt(vol, '%')}")

# --- CSV-Export der KPIs -----------------------------------------
rows = []
for sym, df in frames.items():
    price, d, w, m = kpis(df)
    rows.append({
        "Symbol": sym,
        "Preis": to_scalar(price),
        "24h_%": to_scalar(d),
        "7d_%": to_scalar(w),
        "30d_%": to_scalar(m)
    })

kpi_df = pd.DataFrame(rows)

st.download_button(
    "⬇️ KPIs als CSV",
    kpi_df.to_csv(index=False).encode("utf-8"),
    "kpis.csv",
    "text/csv"
)

# --- Tabs ----------------------------------------------------------
tab_kpi, tab_charts, tab_news = st.tabs(["📊 KPIs", "📈 Charts", "📰 News"])


# ---------- KPI-TAB ----------
with tab_kpi:
    # KPI-Grid
    cols = st.columns(len(frames))
    rows = []  # für CSV

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

            st.metric("Preis", fmt(price))
            c1, c2 = st.columns(2)
            c1.metric("24h",  fmt(d, "%"))
            c2.metric("7 Tage", fmt(w, "%"))
            st.caption(f"30 Tage: {fmt(m, '%')}")
            st.caption(f"Volatilität (30T): {fmt(vol, '%')}")

        # für CSV sammeln
        rows.append({
            "Symbol": sym,
            "Preis": to_scalar(price),
            "24h_%": to_scalar(d),
            "7d_%": to_scalar(w),
            "30d_%": to_scalar(m),
            "Vol_30T_%": to_scalar(vol),
        })

    # CSV-Download (einmal, mit eindeutigem Key)
    kpi_df = pd.DataFrame(rows)
    st.download_button(
        label="⬇️ KPIs als CSV",
        data=kpi_df.to_csv(index=False).encode("utf-8"),
        file_name="kpis.csv",
        mime="text/csv",
        key="kpi_csv_download_button"
    )

# --- Tabs ----------------------------------------------------------
tab_kpi, tab_charts, tab_news = st.tabs(["📊 KPIs", "📈 Charts", "📰 News"])

# ---------- KPI-TAB ----------
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

            st.metric("Preis", fmt(price))
            c1, c2 = st.columns(2)
            c1.metric("24h",  fmt(d, "%"))
            c2.metric("7 Tage", fmt(w, "%"))
            st.caption(f"30 Tage: {fmt(m, '%')}")
            st.caption(f"Volatilität (30T): {fmt(vol, '%')}")

        rows.append({
            "Symbol": sym,
            "Preis": to_scalar(price),
            "24h_%": to_scalar(d),
            "7d_%": to_scalar(w),
            "30d_%": to_scalar(m),
            "Vol_30T_%": to_scalar(vol),
        })

    kpi_df = pd.DataFrame(rows)
    st.download_button(
        label="⬇️ KPIs als CSV",
        data=kpi_df.to_csv(index=False).encode("utf-8"),
        file_name="kpis.csv",
        mime="text/csv",
        key="kpi_csv_download_button"
    )

# ---------- CHARTS-TAB ----------
with tab_charts:
    sub1, sub2 = st.tabs(["📉 Verlauf", "📊 Korrelation"])
    with sub1:
        for sym, df in frames.items():
            st.write(f"**{sym}**")
            st.line_chart(df["Close"])
    with sub2:
        merged = None
        for sym, df in frames.items():
            s = df["Close"].rename(sym)
            merged = s if merged is None else merged.to_frame().join(s, how="outer")
        corr = merged.pct_change().corr().round(2)
        st.dataframe(corr, use_container_width=True)

# ---------- NEWS-TAB ----------
with tab_news:
    st.subheader("Aktuelle Nachrichten")
    for sym in symbols:
        news_items = get_news(sym, limit=5)
        if not news_items:
            st.write(f"Keine News gefunden für {sym}")
            continue

        st.markdown(f"### {sym}")
        for item in news_items:
            # Fallbacks, falls Keys fehlen
            title = item.get("title", "Ohne Titel")
            link = item.get("link", "#")
            st.markdown(f"- [{title}]({link})")

# Hinweis unten (optional)
st.caption("ℹ️ Symbole: ^GSPC=S&P 500, ^NDX=Nasdaq 100, BTC-USD=Bitcoin, EURUSD=X=Euro/US-Dollar, GC=F=Gold")
