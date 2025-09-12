import streamlit as st
import pandas as pd
import numpy as np
import yfinance as yf
import plotly.graph_objects as go
import datetime as dt
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

# -----------------------------------------------------------------------------
# Utils
# --- NEWS Utils (robust) ---
import datetime as dt
from urllib.parse import urlparse

# Für jedes Symbol mehrere Alternativen ausprobieren (bis echte News kommen)
NEWS_PROXY_CHAIN = {
    # Indizes
    "^GSPC":  ["SPY", "^GSPC"],
    "^NDX":   ["QQQ", "^NDX"],
    # Rohstoffe
    "GC=F":   ["GLD", "GC=F"],
    "CL=F":   ["USO", "CL=F"],
    # Krypto – mehrere Quellen (ETF, Trust, Börse, direkt)
    "BTC-USD": ["BTC-USD", "GBTC", "BITO", "COIN"],
    "ETH-USD": ["ETH-USD", "ETHE", "COIN"],
    # FX – Währungs-ETFs zuerst (haben eher News)
    "EURUSD=X": ["FXE", "EURUSD=X"],
    "GBPUSD=X": ["FXB", "GBPUSD=X"],
    "JPY=X":    ["FXY", "JPY=X"],
    "CHFUSD=X": ["FXF", "CHFUSD=X"],
    # Aktien
    "AAPL": ["AAPL"],
    "TSLA": ["TSLA"],
    "NVDA": ["NVDA"],
}

def time_ago(epoch_secs: float) -> str:
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

def _fallback_title_from_link(link: str) -> str:
    try:
        p = urlparse(link or "")
        slug = p.path.strip("/").split("/")[-1].replace("-", " ").strip()
        host = p.netloc.replace("www.", "")
        return (slug.title() if slug else host) or "News"
    except Exception:
        return "News"

def normalize_news_item(item: dict) -> dict:
    link = item.get("link") or ""
    title = (item.get("title") or "").strip()
    if not link:
        return {}  # ohne Link unbrauchbar
    if not title:
        title = _fallback_title_from_link(link)
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
    return {"title": title, "link": link, "publisher": publisher, "ago": ago, "thumb": thumb, "raw": item}

def get_news(symbol: str, limit: int = 6) -> list[dict]:
    """Probiert mehrere Proxies; gibt bereits normalisierte Items zurück (gefiltert)."""
    proxies = NEWS_PROXY_CHAIN.get(symbol, [symbol])
    for p in proxies:
        try:
            t = yf.Ticker(p)
            raw = getattr(t, "news", None) or []
            items = []
            for it in raw[:limit*2]:  # mehr ziehen, später filtern/deduplizieren
                n = normalize_news_item(it)
                if n and n["title"] and n["link"]:
                    items.append(n)
            if items:
                # dedupe per link
                seen = set()
                deduped = []
                for n in items:
                    if n["link"] in seen: 
                        continue
                    seen.add(n["link"])
                    deduped.append(n)
                return deduped[:limit]
        except Exception:
            continue
    return []

def get_news_multi(symbols: list[str], per_symbol: int = 5) -> list[dict]:
    """Sammelt News über alle Symbole, dedupliziert & sortiert nach Zeit."""
    items, seen = [], set()
    for sym in symbols:
        for n in get_news(sym, limit=per_symbol):
            if n["link"] in seen:
                continue
            seen.add(n["link"])
            n["sym"] = sym
            n["ts"] = n["raw"].get("providerPublishTime") or 0
            items.append(n)
    items.sort(key=lambda x: x["ts"], reverse=True)
    return items



def time_ago(epoch_secs: float) -> str:
    """Wandelt Unix-Sekunden in 'vor X min/h/t' um."""
    try:
        ts = dt.datetime.fromtimestamp(float(epoch_secs), tz=dt.timezone.utc)
    except Exception:
        return ""
    now = dt.datetime.now(dt.timezone.utc)
    delta = now - ts
    s = int(delta.total_seconds())
    if s < 60:
        return "gerade eben"
    m = s // 60
    if m < 60:
        return f"vor {m} min"
    h = m // 60
    if h < 24:
        return f"vor {h} h"
    d = h // 24
    return f"vor {d} Tagen"

def normalize_news_item(item: dict) -> dict:
    """Extrahiert robuste Felder aus yfinance-News."""
    title = item.get("title", "Ohne Titel")
    link = item.get("link", "#")
    publisher = item.get("publisher", "")
    # providerPublishTime ist Unix (Sekunden)
    ts = item.get("providerPublishTime")
    ago = time_ago(ts) if ts else ""
    # Thumbnail (optional; Struktur variiert)
    thumb = None
    try:
        th = item.get("thumbnail") or {}
        res = th.get("resolutions") or []
        if res:
            thumb = res[0].get("url")
    except Exception:
        pass
    return {"title": title, "link": link, "publisher": publisher, "ago": ago, "thumb": thumb, "raw": item}

def get_news_multi(symbols: list[str], per_symbol: int = 5) -> list[dict]:
    """Holt News für mehrere Symbole (mit Proxy-Mapping) und sortiert sie nach Zeit absteigend."""
    all_items = []
    seen_links = set()
    for sym in symbols:
        for raw in get_news(sym, limit=per_symbol):
            n = normalize_news_item(raw)
            # Dedupe per Link
            if n["link"] in seen_links:
                continue
            seen_links.add(n["link"])
            # Zeit für Sortierung (Fallback: 0)
            ts = raw.get("providerPublishTime") or 0
            n["ts"] = ts
            n["sym"] = sym
            all_items.append(n)
    # Neueste zuerst
    all_items.sort(key=lambda x: x.get("ts", 0), reverse=True)
    return all_items

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

# ——— News mit Symbol-Mapping (bessere Trefferquote) ———
_NEWS_PROXY = {
    "^GSPC": "SPY",    # S&P 500 ETF
    "^NDX":  "QQQ",    # Nasdaq 100 ETF
    "EURUSD=X": "EURUSD=X",
    "GBPUSD=X": "GBPUSD=X",
    "JPY=X": "JPY=X",
    "CHFUSD=X": "CHFUSD=X",
    "GC=F": "GC=F",
    "CL=F": "CL=F",
    "BTC-USD": "BTC-USD",
    "ETH-USD": "ETH-USD",
    "AAPL": "AAPL",
    "TSLA": "TSLA",
    "NVDA": "NVDA",
}

def get_news(symbol: str, limit: int = 5):
    """Hole Headlines; für Indizes auf liquide ETFs mappen, damit Yahoo-News etwas liefert."""
    try:
        proxy = _NEWS_PROXY.get(symbol, symbol)
        t = yf.Ticker(proxy)
        news = getattr(t, "news", None)
        if not news:
            return []
        return news[:limit]
    except Exception:
        return []


# -----------------------------------------------------------------------------
# Daten laden (Cache) – robust, mit Wochenintervallen bei MAX
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
# --- Tabs ----------------------------------------------------------
tab_kpi, tab_charts, tab_news = st.tabs(["📊 KPIs", "📈 Charts", "📰 News"])


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
    normalize  = st.checkbox("Verlauf auf 100 normieren", value=False, key="opt_norm")

    sub1, sub2 = st.tabs(["📉 Verlauf", "📊 Korrelation"])

    # --- Verlauf (Plotly, MAs, Normalisierung, Slider) ---
    with sub1:
        for sym, df in frames.items():
            if not has_close_data(df):
                st.info(f"{sym}: Keine Daten für Verlauf.")
                continue
            st.markdown(f"**{sym}**")
            fig = fig_with_mas(df, sym, show_ma20, show_ma50, normalize)
            st.plotly_chart(fig, use_container_width=True)

    # --- Korrelation (robust via concat) ---
    with sub2:
        series_list = []
        for sym, df in frames.items():
            if has_close_data(df):
                series_list.append(df["Close"].rename(sym))
        if series_list:
            merged = pd.concat(series_list, axis=1)
            corr = merged.pct_change().corr().round(2)
            st.dataframe(corr, use_container_width=True)
        else:
            st.info("Keine Daten für Korrelation verfügbar.")

# ---------- NEWS TAB ----------
# ---------- NEWS TAB ----------
with tab_news:
    st.subheader("Aktuelle Nachrichten")

    mode = st.radio(
        "Ansicht",
        ["Kombiniert (alle Symbole)", "Pro Symbol"],
        horizontal=True,
        key="news_mode"
    )
    per_symbol = st.slider("Anzahl pro Symbol", 1, 10, 5, key="news_per_symbol")
    st.button("🔄 Aktualisieren")  # triggert Rerun

    if mode.startswith("Kombiniert"):
        feed = get_news_multi(symbols, per_symbol=per_symbol)
        if not feed:
            st.info("Keine News gefunden. Tipp: Preset 'Tech' wählen oder andere Symbole.")
        else:
            for n in feed:
                c0, c1 = st.columns([1, 8])
                with c0:
                    thumb = n.get("thumb")  # <-- sicher abfragen
                    if thumb:
                        try:
                            st.image(thumb, use_container_width=True)
                        except Exception:
                            st.empty()
                with c1:
                    st.markdown(f"**[{n['title']}]({n['link']})**")
                    meta = " · ".join([p for p in [n.get('publisher',''), n.get('ago',''), n.get('sym','')] if p])
                    if meta:
                        st.markdown(f"<span style='opacity:.6'>{meta}</span>", unsafe_allow_html=True)
        st.caption("🔎 Quelle: Yahoo Finance News (yfinance)")

    else:  # Pro Symbol
        for sym in symbols:
            st.markdown(f"### {sym}")
            items = get_news(sym, limit=per_symbol)
            if not items:
                st.write("Keine News gefunden.")
                continue
            for n in items:
                c0, c1 = st.columns([1, 8])
                with c0:
                    thumb = n.get("thumb")  # <-- sicher abfragen
                    if thumb:
                        try:
                            st.image(thumb, use_container_width=True)
                        except Exception:
                            st.empty()
                with c1:
                    st.markdown(f"**[{n['title']}]({n['link']})**")
                    meta = " · ".join([p for p in [n.get('publisher',''), n.get('ago','')] if p])
                    if meta:
                        st.markdown(f"<span style='opacity:.6'>{meta}</span>", unsafe_allow_html=True)




# Hinweis unten (optional)
st.caption("ℹ️ Symbole: ^GSPC=S&P 500, ^NDX=Nasdaq 100, BTC-USD=Bitcoin, EURUSD=X=Euro/US-Dollar, GC=F=Gold")
st.markdown("<hr style='opacity:0.2'>", unsafe_allow_html=True)
st.caption("📊 Datenquelle: Yahoo Finance • Build mit Streamlit 🚀")
