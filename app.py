import datetime as dt
import json
import re
import time
from difflib import SequenceMatcher
from email.utils import parsedate_to_datetime
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Set
from urllib.parse import parse_qs, quote_plus, urlparse, urlsplit
from xml.etree import ElementTree as ET

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import requests
import streamlit as st
import yfinance as yf
# -----------------------------------------------------------------------------
# Grundkonfiguration
# -----------------------------------------------------------------------------
# --- Session Defaults (MÜSSEN ganz oben stehen) ---
if "symbols" not in st.session_state:
    st.session_state.symbols = ["^GSPC", "BTC-USD"]
if "rng" not in st.session_state:
    st.session_state.rng = "3M"

with st.container():
    st.markdown("### 📈 Mini-Wirtschafts-Dashboard")
    st.markdown("Täglich ~1h coden • Python • GitHub • Live-KPIs")
    st.caption("💡 Tipp: In den App-Settings → Theme den Dark Mode aktivieren")
st.markdown("<hr style='opacity:0.3'>", unsafe_allow_html=True)

# -----------------------------------------------------------------------------
# Sidebar (Presets + Zeitraum) mit Session State
# -----------------------------------------------------------------------------
period_map = {"1M": "1mo", "3M": "3mo", "6M": "6mo", "1Y": "1y", "5Y": "5y","MAX": "max" }
if "rng" not in st.session_state:
    st.session_state.rng = "3M"

# ... (Presets & Multiselect wie gehabt)

st.session_state.rng = st.selectbox(
    "Zeitraum",
    list(period_map.keys()),
    index=list(period_map.keys()).index(st.session_state.rng)
)

symbols = st.session_state.symbols
rng = st.session_state.rng

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
# --- NEWS Utils (robust & defensiv) ---

NEWS_PROXY_CHAIN = {
    "^GSPC": ["SPY", "^GSPC"],
    "^NDX":  ["QQQ", "^NDX"],
    "GC=F":  ["GLD", "GC=F"],
    "CL=F":  ["USO", "CL=F"],
    "BTC-USD": ["BTC-USD", "GBTC", "BITO", "COIN", "MSTR"],
    "ETH-USD": ["ETH-USD", "ETHE", "COIN"],
    "EURUSD=X": ["FXE", "EURUSD=X"],
    "GBPUSD=X": ["FXB", "GBPUSD=X"],
    "JPY=X":    ["FXY", "JPY=X"],
    "CHFUSD=X": ["FXF", "CHFUSD=X"],
    "AAPL": ["AAPL"], "TSLA": ["TSLA"], "NVDA": ["NVDA"],
}
def perf_pct(df: pd.DataFrame, days: int) -> float:
    if df is None or df.empty or "Close" not in df.columns:
        return np.nan
    s = df["Close"].dropna()
    if s.size < 2:
        return np.nan
    if days <= 0:
        a, b = s.iloc[-1], s.iloc[-2]
        return (a / b - 1.0) * 100.0 if b else np.nan
    if s.size <= days:
        return np.nan
    a, b = s.iloc[-1], s.iloc[-1 - days]
    return (a / b - 1.0) * 100.0 if b else np.nan


def time_ago(epoch_secs):
    try:
        ts = dt.datetime.fromtimestamp(float(epoch_secs), tz=dt.timezone.utc)
    except Exception:
        return ""
    now = dt.datetime.now(dt.timezone.utc)
    s = int((now - ts).total_seconds())
    if s < 60: return "gerade eben"
    m = s // 60
    if m < 60: return f"vor {m} min"
    h = m // 60
    if h < 24: return f"vor {h} h"
    d = h // 24
    return f"vor {d} Tagen"

def _fallback_title_from_link(link):
    try:
        p = urlparse(link or "")
        slug = p.path.strip("/").split("/")[-1].replace("-", " ").strip()
        host = p.netloc.replace("www.", "")
        return (slug.title() if slug else host) or "News"
    except Exception:
        return "News"

def normalize_news_item(item):
    if not isinstance(item, dict):
        return None
    link = (item.get("link") or "").strip()
    if not link:
        return None
    title = (item.get("title") or "").strip() or _fallback_title_from_link(link)
    publisher = (item.get("publisher") or "").strip()
    ts = item.get("providerPublishTime")
    ago = time_ago(ts) if ts else ""
    thumb = None
    try:
        res = (item.get("thumbnail") or {}).get("resolutions") or []
        if res:
            thumb = res[0].get("url")
    except Exception:
        pass
    return {
        "title": title,
        "link": link,
        "publisher": publisher,
        "ago": ago,
        "thumb": thumb,
        "raw": item,
    }


def _normalize_google_link(link: str) -> str:
    if not link:
        return ""
    try:
        parsed = urlsplit(link)
    except Exception:
        return link

    if "news.google.com" not in parsed.netloc:
        return link

    params = parse_qs(parsed.query)
    target = params.get("url")
    if target and target[0]:
        return target[0]
    return link


def google_news_items(symbol: str, limit: int = 6) -> list[dict]:
    """Fallback via Google News RSS feed (no API key required)."""
    url = (
        "https://news.google.com/rss/search?q="
        f"{quote_plus(symbol)}&hl=de&gl=DE&ceid=DE:de"
    )
    try:
        response = requests.get(url, timeout=10)
        response.raise_for_status()
    except requests.RequestException:
        return []

    try:
        root = ET.fromstring(response.content)
    except Exception:
        return []

    channel = root.find("channel") if root is not None else None
    if channel is None:
        return []

    items: list[dict] = []
    for node in channel.findall("item"):
        title = (node.findtext("title") or "").strip()
        link = _normalize_google_link((node.findtext("link") or "").strip())
        if not title or not link:
            continue

        source = node.find("source")
        publisher = source.text.strip() if source is not None and source.text else "Google News"

        pub_date = node.findtext("pubDate")
        ts = None
        ago = ""
        if pub_date:
            try:
                dt_obj = parsedate_to_datetime(pub_date)
                if dt_obj.tzinfo is None:
                    dt_obj = dt_obj.replace(tzinfo=dt.timezone.utc)
                ts = int(dt_obj.timestamp())
                ago = time_ago(ts)
            except Exception:
                ts = None
                ago = ""

        items.append(
            {
                "title": title,
                "link": link,
                "publisher": publisher,
                "ago": ago,
                "thumb": None,
                "raw": {"providerPublishTime": ts},
            }
        )
        if len(items) >= limit:
            break

    return items

def _yfinance_news(symbol: str) -> list[dict]:
    """Safely fetch the raw Yahoo Finance news list for a single symbol."""
    try:
        ticker = yf.Ticker(symbol)
        return getattr(ticker, "news", None) or []
    except Exception:
        return []


def get_news(symbol, limit=6):
    """Teste mehrere Proxies; filter/normalize/dedupe; vermeide KeyErrors."""
    proxies = NEWS_PROXY_CHAIN.get(symbol, [symbol])
    seen = set()
    out = []
    for proxy in proxies:
        raw_items = _yfinance_news(proxy)
        for it in raw_items[: limit * 2]:
            n = normalize_news_item(it)
            if not n:
                continue
            link = (n.get("link") or "").strip()
            title = (n.get("title") or "").strip()
            if not link or not title or link in seen:
                continue
            seen.add(link)
            out.append(n)
            if len(out) >= limit:
                return out
        time.sleep(0.2)  # sanfte Pause gg. Throttling
    if len(out) < limit:
        for alt in google_news_items(symbol, limit=limit):
            link = (alt.get("link") or "").strip()
            title = (alt.get("title") or "").strip()
            if not link or not title or link in seen:
                continue
            seen.add(link)
            out.append(alt)
            if len(out) >= limit:
                break
    return out

def get_news_multi(symbols, per_symbol=5):
    """Sammelt News über alle Symbole, dedupliziert & sortiert."""
    items, seen = [], set()
    for sym in symbols:
        for n in get_news(sym, limit=per_symbol):
            link = (n.get("link") or "").strip()
            if not link or link in seen:
                continue
            seen.add(link)
            n["sym"] = sym
            n["ts"] = (n.get("raw") or {}).get("providerPublishTime") or 0
            items.append(n)
    items.sort(key=lambda x: x.get("ts", 0), reverse=True)
    return items

def fallback_news_links(sym: str) -> list[tuple[str, str]]:
    """Gibt (Label, URL)-Paare zurück – kein Rendering hier drin!"""
    return [
        (f"🌐 Google News",  f"https://news.google.com/search?q={quote_plus(sym)}"),
        (f"📊 Yahoo Finance", f"https://finance.yahoo.com/quote/{quote_plus(sym)}/news"),
    ]






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
    try:
        v = float(to_scalar(x))
        if not np.isfinite(v) or pd.isna(v):
            raise ValueError
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

def has_close_data(df: pd.DataFrame) -> bool:
    try:
        return ("Close" in df.columns) and (not df["Close"].dropna().empty)
    except Exception:
        return False

# ——— Chart-Tools (MA, Plotly, Normalisierung) ———
def add_bbands(df: pd.DataFrame, window: int = 20, n_std: float = 2.0) -> pd.DataFrame:
    """Bollinger-Bänder auf Close (MID, UPPER, LOWER)."""
    d = df.copy()
    roll = d["Close"].rolling(window)
    mid = roll.mean()
    std = roll.std()
    d["BB_MID"] = mid
    d["BB_UP"]  = mid + n_std * std
    d["BB_LOW"] = mid - n_std * std
    return d

def calc_rsi(close: pd.Series, period: int = 14) -> pd.Series:
    """RSI(14) via EMA (robust, NaN-safe)."""
    if close is None or close.dropna().empty:
        return pd.Series(index=close.index if hasattr(close, "index") else None, dtype=float)
    delta = close.diff()
    up = pd.Series(np.where(delta > 0,  delta, 0.0), index=close.index)
    down = pd.Series(np.where(delta < 0, -delta, 0.0), index=close.index)
    roll_up = up.ewm(alpha=1/period, adjust=False).mean()
    roll_down = down.ewm(alpha=1/period, adjust=False).mean()
    rs = roll_up / (roll_down.replace(0, np.nan))
    rsi = 100 - (100 / (1 + rs))
    return rsi.clip(0, 100)

def add_mas(df: pd.DataFrame):
    d = df.copy()
    d["MA20"] = d["Close"].rolling(20).mean()
    d["MA50"] = d["Close"].rolling(50).mean()
    return d

def fig_with_mas(df: pd.DataFrame, sym: str, show_ma20: bool, show_ma50: bool, normalize: bool):
    d = add_mas(df)
    series = d["Close"].dropna()

    # optional: Index=100 Normalisierung
    if normalize and not series.empty:
        base = series.iloc[0]
        if base != 0:
            factor = 100.0 / base
            d = d.copy()
            d["Close"] = d["Close"] * factor
            if "MA20" in d: d["MA20"] = d["MA20"] * factor
            if "MA50" in d: d["MA50"] = d["MA50"] * factor

    fig = go.Figure()
    # Close
    fig.add_trace(go.Scatter(
        x=d.index, y=d["Close"], mode="lines", name=f"{sym} {'(Index=100)' if normalize else 'Close'}"
    ))
    # MAs
    if show_ma20 and "MA20" in d and d["MA20"].notna().any():
        fig.add_trace(go.Scatter(x=d.index, y=d["MA20"], mode="lines", name="MA20"))
    if show_ma50 and "MA50" in d and d["MA50"].notna().any():
        fig.add_trace(go.Scatter(x=d.index, y=d["MA50"], mode="lines", name="MA50"))

    # Profi-Layout
    fig.update_layout(
        margin=dict(l=0, r=0, t=30, b=0),
        legend=dict(orientation="h"),
        hovermode="x unified",
        xaxis=dict(
            rangeselector=dict(
                buttons=list([
                    dict(count=1, label="1M", step="month", stepmode="backward"),
                    dict(count=3, label="3M", step="month", stepmode="backward"),
                    dict(count=6, label="6M", step="month", stepmode="backward"),
                    dict(step="all", label="All")
                ])
            ),
            rangeslider=dict(visible=True),
            type="date"
        )
    )
    return fig

# -----------------------------------------------------------------------------
# Daten laden (Cache) – robust, mit Wochenintervallen bei MAX
# -----------------------------------------------------------------------------
@st.cache_data(ttl=3600, show_spinner=True)
def load(symbols, period):
    out = {}
    for s in symbols:
        df = yf.download(
            s,
            period=period,
            interval="1d",
            auto_adjust=True,
            progress=False,
            group_by="column",
        )
        c = extract_close(df)
        if c is not None and not c.empty:
            out[s] = pd.DataFrame({"Close": c})
    return out


@st.cache_data(ttl=3600, show_spinner=False)
def load_ohlc(symbol: str, period: str) -> pd.DataFrame:
    """Load OHLC data once per symbol/period and flatten MultiIndex frames."""
    try:
        df = yf.download(
            symbol,
            period=period,
            interval="1d",
            auto_adjust=False,
            progress=False,
            group_by="column",
        )
    except Exception:
        return pd.DataFrame()

    if df is None or df.empty:
        return pd.DataFrame()

    if isinstance(df.columns, pd.MultiIndex):
        df.columns = [level[0] if isinstance(level, tuple) else level for level in df.columns]

    required = [col for col in ["Open", "High", "Low", "Close"] if col in df.columns]
    if len(required) < 4:
        return pd.DataFrame()

    clean = df[required].dropna(how="any")
    return clean


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
# --- Tabs ----------------------------------------------------------
tab_kpi, tab_charts, tab_news, tab_project = st.tabs([
    "📊 KPIs",
    "📈 Charts",
    "📰 News",
    "🗂️ Projekt",
])


# ---------- KPI TAB ----------
with tab_kpi:
    # feiner Trenner oben
    st.markdown("<hr style='opacity:0.2'>", unsafe_allow_html=True)

    items = list(frames.items())
    rows_for_csv = []

    # Immer 3 Karten pro Reihe (stabiles Layout)
    chunk = 3
    for start in range(0, len(items), chunk):
        cols = st.columns(chunk)
        for col, (sym, df) in zip(cols, items[start:start+chunk]):
            with col:
                # Daten-Check
                if not has_close_data(df):
                    st.warning(f"{sym}: Keine Daten verfügbar.")
                    continue

                # KPIs berechnen
                price, d, w, m = kpis(df)
                vol = volatility(df)

                # Titel mit BTC-Logo nur für BTC-USD
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

                # Preis + Delta (24h) – Streamlit färbt den Delta-Wert automatisch grün/rot
                st.metric("Preis", fmt(price), delta=fmt(d, "%"))

                # Farbige Prozentwerte (eigene, kräftige Darstellung)
                c1, c2 = st.columns(2)
                with c1:
                    st.markdown(color_pct_html(d, "24h"),     unsafe_allow_html=True)
                with c2:
                    st.markdown(color_pct_html(w, "7 Tage"),  unsafe_allow_html=True)
                st.markdown(color_pct_html(m,  "30 Tage"),    unsafe_allow_html=True)
                st.markdown(color_pct_html(vol,"Volatilität (30T)"), unsafe_allow_html=True)

                # Für CSV sammeln
                rows_for_csv.append({
                    "Symbol": sym,
                    "Preis": to_scalar(price),
                    "24h_%": to_scalar(d),
                    "7d_%": to_scalar(w),
                    "30d_%": to_scalar(m),
                    "Vol_30T_%": to_scalar(vol),
                })

    # CSV-Download (einmal, rechts ausgerichtet)
    kpi_df = pd.DataFrame(rows_for_csv)
    btn_l, btn_r = st.columns([3, 1])
    with btn_r:
        st.download_button(
            label="⬇️ KPIs als CSV",
            data=kpi_df.to_csv(index=False).encode("utf-8"),
            file_name="kpis.csv",
            mime="text/csv",
            key="kpi_csv_download_button"
        )

    # feiner Trenner unten
    st.markdown("<hr style='opacity:0.2'>", unsafe_allow_html=True)

# ---------- CHARTS TAB ----------
with tab_charts:
    # Optionen oberhalb der Unter-Tabs
    show_ma20  = st.checkbox("MA20 anzeigen", value=True,  key="opt_ma20")
    show_ma50  = st.checkbox("MA50 anzeigen", value=False, key="opt_ma50")
    show_bb    = st.checkbox("Bollinger (20, 2σ)", value=False, key="opt_bb")
    show_rsi   = st.checkbox("RSI(14)", value=False, key="opt_rsi")
    normalize  = st.checkbox("Verlauf auf 100 normieren", value=False, key="opt_norm")

    # WICHTIG: hier wirklich 4 Tabs erzeugen
    sub1, sub2, sub3, sub4 = st.tabs(["📉 Verlauf", "📊 Korrelation", "🕯️ Candlesticks", "📛 Performance"])

    # --- Verlauf (Plotly, MAs, BB, Normalisierung, Slider) ---
    with sub1:
        for sym, df in frames.items():
            if not has_close_data(df):
                st.info(f"{sym}: Keine Daten für Verlauf.")
                continue

            # Basisdaten + MAs
            d = add_mas(df)

            # Optional Bollinger-Bänder
            if show_bb:
                d = add_bbands(d, window=20, n_std=2.0)

            # Optional auf 100 normieren (alle gezeigten Linien)
            if normalize and not d["Close"].dropna().empty:
                base = d["Close"].dropna().iloc[0]
                if base != 0:
                    scale = 100.0 / base
                    for col in ["Close", "MA20", "MA50", "BB_MID", "BB_UP", "BB_LOW"]:
                        if col in d.columns:
                            d[col] = d[col] * scale

            # Preis/Indikator-Chart
            fig = go.Figure()
            fig.add_trace(go.Scatter(x=d.index, y=d["Close"], mode="lines",
                                     name=f"{sym} {'(Index=100)' if normalize else 'Close'}"))

            if show_ma20 and "MA20" in d and d["MA20"].notna().any():
                fig.add_trace(go.Scatter(x=d.index, y=d["MA20"], mode="lines", name="MA20"))
            if show_ma50 and "MA50" in d and d["MA50"].notna().any():
                fig.add_trace(go.Scatter(x=d.index, y=d["MA50"], mode="lines", name="MA50"))

            # Bollinger-Bänder als Band (Upper zuerst, dann Lower mit Fill)
            if show_bb and "BB_UP" in d and "BB_LOW" in d:
                if d["BB_UP"].notna().any() and d["BB_LOW"].notna().any():
                    fig.add_trace(go.Scatter(x=d.index, y=d["BB_UP"], mode="lines",
                                             name="BB Upper", line=dict(width=0)))
                    fig.add_trace(go.Scatter(x=d.index, y=d["BB_LOW"], mode="lines",
                                             name="BB Lower", fill='tonexty',
                                             line=dict(width=0), opacity=0.2))

            fig.update_layout(
                margin=dict(l=0, r=0, t=30, b=0),
                legend=dict(orientation="h"),
                hovermode="x unified",
                xaxis=dict(
                    rangeselector=dict(
                        buttons=[
                            dict(count=1, label="1M", step="month", stepmode="backward"),
                            dict(count=3, label="3M", step="month", stepmode="backward"),
                            dict(count=6, label="6M", step="month", stepmode="backward"),
                            dict(step="all", label="All"),
                        ]
                    ),
                    rangeslider=dict(visible=True),
                    type="date",
                ),
            )

            st.markdown(f"**{sym}**")
            st.plotly_chart(fig, use_container_width=True)

            # RSI unterhalb (eigener kleiner Plot)
            if show_rsi:
                rsi = calc_rsi(df["Close"])
                if rsi is not None and not rsi.dropna().empty:
                    fig_rsi = go.Figure()
                    fig_rsi.add_trace(go.Scatter(x=rsi.index, y=rsi, mode="lines", name="RSI(14)"))
                    fig_rsi.add_hrect(y0=30, y1=70, fillcolor="lightgray", opacity=0.2, line_width=0)
                    fig_rsi.update_yaxes(range=[0, 100])
                    fig_rsi.update_layout(margin=dict(l=0, r=0, t=10, b=0), height=160, showlegend=False)
                    st.plotly_chart(fig_rsi, use_container_width=True)

    # --- Korrelation als Heatmap (statt Tabelle) ---
    with sub2:
        series_list = []
        for sym, df in frames.items():
            if has_close_data(df):
                series_list.append(df["Close"].rename(sym))
        if series_list:
            merged = pd.concat(series_list, axis=1)
            corr = merged.pct_change().corr()
            fig_corr = px.imshow(corr.round(2), text_auto=True, aspect="auto",
                                 title="Korrelationsmatrix (Daily Returns)")
            fig_corr.update_layout(margin=dict(l=0, r=0, t=40, b=0))
            st.plotly_chart(fig_corr, use_container_width=True)
        else:
            st.info("Keine Daten für Korrelation verfügbar.")

    # --- Candlesticks ---
    with sub3:
        for sym in symbols:
            df_ohlc = load_ohlc(sym, period_map[rng])
            if df_ohlc is None or df_ohlc.empty:
                st.info(
                    f"{sym}: Keine Candlestick-Daten verfügbar (ggf. Proxy/Netzwerk prüfen)."
                )
                continue

            st.markdown(f"**{sym}**")
            fig = go.Figure(
                data=[
                    go.Candlestick(
                        x=df_ohlc.index,
                        open=df_ohlc["Open"],
                        high=df_ohlc["High"],
                        low=df_ohlc["Low"],
                        close=df_ohlc["Close"],
                        name="Candlesticks",
                    )
                ]
            )
            fig.update_layout(
                xaxis_rangeslider_visible=False,
                margin=dict(l=0, r=0, t=30, b=0),
                hovermode="x unified",
            )
            st.plotly_chart(fig, use_container_width=True)

    # --- Performance (1T, 7T, 30T) als Gruppen-Barchart ---
    with sub4:
        rows = []
        for sym, df in frames.items():
            if not has_close_data(df):
                continue
            r_1d  = perf_pct(df, 0)   # ~ letzte 24h (letzter vs. vorheriger Close)
            r_7d  = perf_pct(df, 7)
            r_30d = perf_pct(df, 30)
            rows.append({"Symbol": sym, "1T": r_1d, "7T": r_7d, "30T": r_30d})

        if not rows:
            st.info("Keine Daten für Performance verfügbar.")
        else:
            perf_df = pd.DataFrame(rows)
            perf_long = perf_df.melt(id_vars="Symbol", var_name="Fenster", value_name="Rendite_%")
            fig_perf = px.bar(
                perf_long,
                x="Symbol", y="Rendite_%", color="Fenster", barmode="group",
                text="Rendite_%",
                title="Performancevergleich (1T / 7T / 30T)"
            )
            fig_perf.update_traces(
                texttemplate="%{y:.2f}%",
                hovertemplate="<b>%{x}</b><br>%{legendgroup}: %{y:.2f}%<extra></extra>"
            )
            fig_perf.update_layout(margin=dict(l=0, r=0, t=40, b=0), legend=dict(orientation="h"))
            st.plotly_chart(fig_perf, use_container_width=True)



# ---------- NEWS TAB ----------
with tab_news:
    st.subheader("📰 Markt-News & Analysen")
    st.caption("Bleib auf dem Laufenden mit den neuesten Finanz- und Wirtschaftsnachrichten.")
    st.markdown("<hr style='opacity:0.3'>", unsafe_allow_html=True)



    mode = st.radio("Ansicht", ["Kombiniert (alle Symbole)", "Pro Symbol"], horizontal=True, key="news_mode")
    per_symbol = st.slider("Anzahl pro Symbol", 1, 10, 5, key="news_per_symbol")
    st.button("🔄 Aktualisieren")  # triggert Rerun

    def render_item(n, show_sym_tag=False):
        if not isinstance(n, dict):
            return
        title = (n.get("title") or "News").strip()
        link  = (n.get("link")  or "").strip()
        if not link:
            return
        publisher = (n.get("publisher") or "").strip()
        ago       = (n.get("ago") or "").strip()
        sym_tag   = (n.get("sym") or "").strip() if show_sym_tag else ""

        c0, c1 = st.columns([1, 8])
        with c0:
            thumb = n.get("thumb")
            if thumb:
                try:
                    st.image(thumb, use_container_width=True)
                except Exception:
                    st.empty()
        with c1:
            st.markdown(f"**[{title}]({link})**")
            meta_parts = [publisher, ago]
            if show_sym_tag and sym_tag:
                meta_parts.append(sym_tag)
            meta = " · ".join([p for p in meta_parts if p])
            if meta:
                st.markdown(f"<span style='opacity:.6'>{meta}</span>", unsafe_allow_html=True)

    if mode.startswith("Kombiniert"):
        feed = get_news_multi(symbols, per_symbol=per_symbol)
        if not feed:
          st.warning("⚠️ Keine News via yfinance gefunden – hier ein paar Alternativen:")
          for sym in symbols[:3]:
            with st.container():
              st.markdown(f"#### 🔎 {sym}")
              c1, c2 = st.columns(2)
              links = fallback_news_links(sym)
              with c1:
                st.markdown(f"[{links[0][0]}]({links[0][1]})")
              with c2:
                st.markdown(f"[{links[1][0]}]({links[1][1]})")

        else:
            for n in feed:
                render_item(n, show_sym_tag=True)
        st.caption("🔎 Quelle: Yahoo Finance News (yfinance) – mit Proxy-Ketten & Fallback-Links")

    else:  # Pro Symbol
        for sym in symbols:
            st.markdown(f"### {sym}")
            items = get_news(sym, limit=per_symbol)
            if not items:
                st.write("Keine News via yfinance gefunden. Alternativen:")
                for label, url in fallback_news_links(sym):
                    st.markdown(f"- [{label}]({url})")
                continue
            for n in items:
                render_item(n, show_sym_tag=False)



# ---------- PROJECT TAB ----------
with tab_project:
    st.subheader("🗂️ Projekt-Steuerung & Kommunikation")
    st.caption(
        "Nutze Projektmethoden, um dein InfoDashboard in 12 Tagen fertigzustellen und beeindruckend zu präsentieren."
    )
    st.markdown("<hr style='opacity:0.3'>", unsafe_allow_html=True)

    st.markdown("### 🎯 SMART-Ziel")
    st.markdown(
        """
        **Spezifisch:** Ein einsatzbereites Finanz-Dashboard mit KPIs, Charts und News, ergänzt um Projektkommunikationstools.

        **Messbar:** Mindestens 4 Daten-Visualisierungen, 1 automatisierte News-Quelle, 1 Projektreport-Download und 1 neue Nutzerinteraktion.

        **Akzeptiert:** Abgestimmt mit Stakeholdern (Mentor, Community, Recruiter).

        **Realistisch:** Aufbauend auf Streamlit, yfinance und bestehenden Komponenten, zusätzliche Features sind leichtgewichtig.

        **Terminiert:** Fertigstellung innerhalb der nächsten 12 Tage inkl. Review & Demo.
        """
    )

    st.markdown("### 🛣️ Projektphasen")
    phases = [
        "1. **Vision & Scope** – Zielbild finalisieren, Stakeholder abholen",
        "2. **Planung** – Deliverables, Ressourcen, Risks, Kommunikationsplan",
        "3. **Umsetzung** – Features coden, Inhalte kuratieren, Qualität sichern",
        "4. **Feedback-Loop** – Soll-Ist-Abgleich, Iterationen basierend auf Tests & Feedback",
        "5. **Abschluss & Launch** – Demo, Dokumentation, LinkedIn-Post vorbereiten",
    ]
    st.markdown("\n".join(phases))

    st.markdown("### 🗓️ 12-Tage Gantt-Plan")
    today = dt.date.today()
    gantt_data = pd.DataFrame(
        [
            {
                "Phase": "Vision & Scope",
                "Start": today,
                "Ende": today + dt.timedelta(days=1),
                "Owner": "Product Owner",
            },
            {
                "Phase": "Planung",
                "Start": today + dt.timedelta(days=1),
                "Ende": today + dt.timedelta(days=3),
                "Owner": "Projektleitung",
            },
            {
                "Phase": "Umsetzung",
                "Start": today + dt.timedelta(days=3),
                "Ende": today + dt.timedelta(days=9),
                "Owner": "Dev-Team",
            },
            {
                "Phase": "Feedback-Loop",
                "Start": today + dt.timedelta(days=5),
                "Ende": today + dt.timedelta(days=10),
                "Owner": "QA & Stakeholder",
            },
            {
                "Phase": "Abschluss",
                "Start": today + dt.timedelta(days=9),
                "Ende": today + dt.timedelta(days=12),
                "Owner": "Projektleitung",
            },
        ]
    )
    fig_gantt = px.timeline(
        gantt_data,
        x_start="Start",
        x_end="Ende",
        y="Phase",
        color="Owner",
        title="Fokus-Sprint zum Launch",
    )
    fig_gantt.update_yaxes(autorange="reversed")
    fig_gantt.update_layout(margin=dict(l=0, r=0, t=40, b=0))
    st.plotly_chart(fig_gantt, use_container_width=True)

    st.markdown("### ⚠️ Risiko-Matrix")
    risk_df = pd.DataFrame(
        [
            {
                "Risiko": "API-Rate-Limits/Proxy-Blockaden",
                "Eintritt": "Mittel",
                "Auswirkung": "Hoch",
                "Mitigation": "Caching, Fallback-Links, lokale CSV-Snapshots",
            },
            {
                "Risiko": "Zeitüberschreitung",
                "Eintritt": "Mittel",
                "Auswirkung": "Mittel",
                "Mitigation": "Daily Stand-up, klarer Scope & Kanban",
            },
            {
                "Risiko": "Scope Creep",
                "Eintritt": "Niedrig",
                "Auswirkung": "Mittel",
                "Mitigation": "Change-Log pflegen, Priorisierung mit Stakeholdern",
            },
            {
                "Risiko": "Qualitätsprobleme",
                "Eintritt": "Niedrig",
                "Auswirkung": "Hoch",
                "Mitigation": "Tests, Peer-Review, Demo-Checkliste",
            },
        ]
    )
    st.dataframe(risk_df, hide_index=True, use_container_width=True)

    st.markdown("### 🗃️ Kanban Snapshot")
    kanban = {
        "To Do": [
            "Projektstory für LinkedIn entwerfen",
            "Newsletter-Kopie schreiben",
            "Screenshot-Galerie vorbereiten",
        ],
        "In Progress": [
            "Candlestick-Fix testen",
            "Newsfeed-Proxy validieren",
        ],
        "Done": [
            "KPI-Tab verfeinert",
            "Projekt-Tab angelegt",
        ],
    }
    col_todo, col_progress, col_done = st.columns(3)
    for col, key in zip([col_todo, col_progress, col_done], kanban.keys()):
        with col:
            st.markdown(f"#### {key}")
            for item in kanban[key]:
                st.markdown(f"- {item}")

    st.markdown("### 🤝 Stakeholderanalyse")
    stakeholder_df = pd.DataFrame(
        [
            {
                "Stakeholder": "Mentor / Dozent",
                "Interesse": "Lernfortschritt, Präsentation",
                "Einbindung": "Wöchentliche Demo & Feedback",
            },
            {
                "Stakeholder": "Tech-Community",
                "Interesse": "Best Practices teilen",
                "Einbindung": "Blog-Post, Discord-Updates",
            },
            {
                "Stakeholder": "Recruiter",
                "Interesse": "Skills & Story",
                "Einbindung": "LinkedIn-Showcase, Portfolio-Link",
            },
        ]
    )
    st.dataframe(stakeholder_df, hide_index=True, use_container_width=True)

    st.markdown("### 🧭 RACI Matrix")
    raci_df = pd.DataFrame(
        [
            {"Aufgabe": "Feature-Implementierung", "R": "Dev", "A": "Projektleitung", "C": "Mentor", "I": "Community"},
            {"Aufgabe": "Qualitätssicherung", "R": "QA", "A": "Projektleitung", "C": "Dev", "I": "Stakeholder"},
            {"Aufgabe": "Kommunikation & Updates", "R": "Projektleitung", "A": "Projektleitung", "C": "Mentor", "I": "Recruiter"},
            {"Aufgabe": "Launch-Story", "R": "Marketing", "A": "Projektleitung", "C": "Dev", "I": "Community"},
        ]
    )
    st.dataframe(raci_df, hide_index=True, use_container_width=True)

    st.markdown("### 📨 Newsletter & Updates")
    if "newsletter_signups" not in st.session_state:
        st.session_state.newsletter_signups = []
    with st.form("newsletter_form"):
        email = st.text_input("E-Mail für Projekt-Updates")
        submit = st.form_submit_button("Anmelden")
    if submit:
        if email:
            st.session_state.newsletter_signups.append({"email": email, "timestamp": dt.datetime.utcnow()})
            st.success("Danke! Du erhältst vor dem Launch ein Update.")
        else:
            st.warning("Bitte eine gültige E-Mail eintragen.")

    st.markdown("### ❓ Projekt Q&A Bot")

    def _tokenize(text: str) -> List[str]:
        return re.findall(r"[a-z0-9äöüß]+", (text or "").lower())

    def _normalize(text: str) -> str:
        return " ".join(_tokenize(text))

    def _score_match(question_tokens: Set[str], phrase_tokens: Set[str]) -> float:
        if not question_tokens or not phrase_tokens:
            return 0.0
        overlap = len(question_tokens & phrase_tokens)
        return overlap / len(phrase_tokens)

    def resolve_faq_answer(question: str, knowledge_base: List[Dict[str, Sequence[str]]]) -> Optional[str]:
        normalized_question = _normalize(question)
        question_tokens = set(_tokenize(question))
        best_answer: Optional[str] = None
        best_score = 0.0

        for entry in knowledge_base:
            for phrase in entry.get("keywords", []):
                phrase_tokens = set(_tokenize(phrase))
                score = _score_match(question_tokens, phrase_tokens)
                if score > best_score:
                    best_score = score
                    best_answer = entry.get("answer")

        if best_answer and (best_score >= 0.6 or (best_score >= 0.4 and len(question_tokens) <= 3)):
            return best_answer

        best_ratio = 0.0
        for entry in knowledge_base:
            for phrase in entry.get("keywords", []):
                ratio = SequenceMatcher(None, _normalize(phrase), normalized_question).ratio()
                if ratio > best_ratio:
                    best_ratio = ratio
                    best_answer = entry.get("answer")

        if best_answer and best_ratio >= 0.65:
            return best_answer

        return None

    CUSTOM_FAQ_PATH = Path(__file__).resolve().parent / "assets" / "faq_custom.json"

    def _load_custom_faq_entries() -> List[Dict[str, Sequence[str]]]:
        """Lädt optionale FAQ-Einträge aus assets/faq_custom.json."""
        if not CUSTOM_FAQ_PATH.exists():
            return []

        try:
            with CUSTOM_FAQ_PATH.open("r", encoding="utf-8") as handle:
                data = json.load(handle)
        except Exception:
            return []

        entries: List[Dict[str, Sequence[str]]] = []
        for item in data:
            if not isinstance(item, dict):
                continue

            keywords = item.get("keywords")
            answer = item.get("answer")
            if not answer or not keywords:
                continue

            if isinstance(keywords, str):
                keywords = [keywords]

            cleaned_keywords = []
            for keyword in keywords:
                if isinstance(keyword, str):
                    normalized_keyword = keyword.strip()
                    if normalized_keyword:
                        cleaned_keywords.append(normalized_keyword)

            if not cleaned_keywords:
                continue

            entries.append({
                "keywords": cleaned_keywords,
                "answer": str(answer).strip(),
            })

        return entries

    # Neue Antworten können direkt unten ergänzt oder in assets/faq_custom.json hinterlegt werden.
    faq_knowledge: List[Dict[str, Sequence[str]]] = [
        {
            "keywords": ["ziel", "projektziel", "smart ziel"],
            "answer": "Das Ziel ist ein datengetriebenes Finanzdashboard mit Projektstory für dein Portfolio.",
        },
        {
            "keywords": ["umfang", "scope", "features"],
            "answer": "Der Scope umfasst KPIs, Charts, News, Projektplan, Kommunikationsfeatures und Dokumentation.",
        },
        {
            "keywords": ["deadline", "zeitplan", "dauer"],
            "answer": "Von Anfang September bis Ende Oktober",
        },
        {
            "keywords": ["linkedin", "lebenslauf", "portfolio"],
            "answer": "Ich habe dieses Projekt in mein Lebenslauf hinzugefügt",
        },
        {
            "keywords": ["risiko", "risk", "risikomatrix"],
            "answer": "Haupt-Risiken: Zeit, API-Limits, fehlendes Feedback – mit Kanban, Backups und Fallback-News mitigieren.",
        },
        {
            "keywords": ["bollinger", "bollinger bänder"],
            "answer": "Bollinger-Bänder zeigen ein Mittelband (MA20) plus/minus zwei Standardabweichungen und markieren mögliche Überkauft-/Überverkauft-Zonen.",
        },
        {
            "keywords": ["rsi", "relative strength index", "rsi(14)"],
            "answer": "RSI(14) misst das Momentum aus 14 Perioden. Werte über 70 gelten als überkauft, unter 30 als überverkauft.",
        },
        {
            "keywords": ["normieren", "index 100", "normalisierung"],
            "answer": "Beim Normieren auf 100 wird der erste Kurswert auf 100 gesetzt, damit sich mehrere Assets prozentual vergleichen lassen.",
        },
        {
            "keywords": ["kanban", "board", "todo"],
            "answer": "Das Kanban-Board zeigt Aufgaben nach Status (To Do, In Progress, Done) und hält den Flow im Blick.",
        },
        {
            "keywords": ["gantt", "zeitplan", "meilenstein"],
            "answer": "Das Gantt-Chart visualisiert die Projektphasen: Ziele, Planung, Umsetzung, Review und Abschluss.",
        },
        {
            "keywords": ["stakeholder", "analyse", "interessen"],
            "answer": "Die Stakeholderanalyse mappt Mentor, Community und Recruiter auf Interesse und Einbindung.",
        },
        {
            "keywords": ["raci", "verantwortung", "rollen"],
            "answer": "Die RACI-Matrix klärt Verantwortungen: Responsible, Accountable, Consulted und Informed pro Aufgabe.",
        },
        {
            "keywords": ["newsletter", "update", "email"],
            "answer": "Über das Newsletter-Formular sammelst du E-Mails, um vor dem Launch ein Update zu verschicken.",
        },
         {
            "keywords": ["ma20", "MA20", "Ma20"],
            "answer": "MA20 ist der gleitende Durchschnitt der letzten 20 Handelstage, häufig genutzt, um kurzfristige Trends in Aktien zu erkennen.",
        },
        {
            "keywords": ["ma50", "MA50", "Ma50"],
            "answer": "MA steht für „Moving Averages“, zu Deutsch „gleitende Durchschnitte“. Die 50 Tagelinie ist ein Indikator, der den Durchschnitt der letzten 50 Handelstage berücksichtigt.",
        },
        {
            "keywords": ["candlesticks", "Candlesticks", "Kerzencharts"],
            "answer": "Candlesticks, auch Kerzencharts genannt, sind eine visuelle Darstellung der Kursbewegungen eines Assets innerhalb eines bestimmten Zeitraums. Sie bestehen aus einem Kerzenkörper, der den Eröffnungs- und Schlusskurs zeigt, sowie Dochten, die die Höchst- und Tiefstkurse repräsentieren. ",
        },
        {
            "keywords": ["KPI", "KPIs", "kpi"],
            "answer": "KPIs bei Aktien sind Kennzahlen, die die finanzielle Leistung, Bewertung und Erfolg eines Unternehmens messen, z.B. KGV, Umsatz, Gewinn, Dividendenrendite, Eigenkapitalquote und Verschuldungsgrad. ",
        },
        {
            "keywords": ["KPI", "KPIs", "kpi"],
            "answer": "Die Performance misst die Wertentwicklung eines Investments oder eines Portfolios. Meist wird zum Vergleich ein sogenannter Benchmark als Referenz genommen, um die Performance im Vergleich zum Gesamtmarkt oder zu den Branchen darzustellen. Die Performance wird in Prozent ermittelt.",
        },    
    ] + _load_custom_faq_entries()

    if "qa_history" not in st.session_state:
        st.session_state.qa_history = [
            {
                "role": "assistant",
                "content": "Frag mich alles rund um Zielsetzung, Scope oder Launch deines Projekts!",
            }
        ]

    for message in st.session_state.qa_history:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])

    user_question = st.chat_input("Welche Frage hast du zum Projekt?")
    if user_question:
        st.session_state.qa_history.append({"role": "user", "content": user_question})
        with st.chat_message("user"):
            st.markdown(user_question)

        resolved = resolve_faq_answer(user_question, faq_knowledge)
        answer = (
            resolved
            or "Dazu habe ich aktuell keine Details – ergänze gern neue Knowledge-Keywords oberhalb im Code."
        )

        st.session_state.qa_history.append({"role": "assistant", "content": answer})
        with st.chat_message("assistant"):
            st.markdown(answer)





# Hinweis unten (optional)
st.caption("ℹ️ Symbole: ^GSPC=S&P 500, ^NDX=Nasdaq 100, BTC-USD=Bitcoin, EURUSD=X=Euro/US-Dollar, GC=F=Gold")
st.markdown("<hr style='opacity:0.2'>", unsafe_allow_html=True)
st.caption("📊 Datenquelle: Yahoo Finance • Build mit Streamlit 🚀")
