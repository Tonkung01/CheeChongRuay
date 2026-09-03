"""
ชี้ช่องรวย by โค้ชต้น💰 — Signal Bot
----------------
Fetches market data (free, no API key, via Yahoo Finance through yfinance)
on the M15 timeframe and evaluates a 3-step trend-following system:

  Step 1: EMA(50) sets the only allowed direction (price above -> BUY only,
          below -> SELL only), and EMA(9/21) must confirm that same momentum.
  Step 2: ADX(14) must be > 25 (a real trend, not sideways/forming).
  Step 3: RSI(14) or a fresh EMA(9/21) crossover triggers the entry --
          either a pullback into the 40-50 (BUY) / 50-60 (SELL) zone that
          then bounces back through 50, a classic oversold(<30)/
          overbought(>70) cross, OR EMA(9) crossing EMA(21) on this bar.

TP/SL are set from ATR(14): SL = 1.5x ATR, TP = 3x ATR (2:1 reward:risk).
An alert is sent to Telegram only when all steps agree.

Designed to be run on a schedule (e.g. every 15 minutes) by GitHub Actions,
so it needs no server of its own. State (to avoid duplicate alerts) is kept
in state.json, which the workflow commits back to the repo after each run.

This is a rule-based technical tool only. It is NOT financial advice.
Past performance of any strategy does not guarantee future results. Trade
at your own risk and manage your own risk.
"""

import json
import os
import sys
from datetime import datetime, timezone, timedelta

import pandas as pd
import requests
import yfinance as yf

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

# Map our display name -> Yahoo Finance ticker.
# XAUUSD/USOIL use futures tickers since Yahoo doesn't offer clean free spot
# feeds for them; prices track spot closely but are not identical to Exness.
INSTRUMENTS = {
    "XAUUSD": {"ticker": "GC=F", "label": "Gold Futures", "dec": 2},
    "BTCUSD": {"ticker": "BTC-USD", "label": "Bitcoin", "dec": 0},
    "EURUSD": {"ticker": "EURUSD=X", "label": "Euro / USD", "dec": 5},
    "GBPUSD": {"ticker": "GBPUSD=X", "label": "Pound / USD", "dec": 5},
    "USDJPY": {"ticker": "JPY=X", "label": "USD / Yen", "dec": 3},
    "USOIL":  {"ticker": "CL=F", "label": "WTI Crude Futures", "dec": 2},
    "SPX500": {"ticker": "^GSPC", "label": "S&P 500", "dec": 1},
    "NAS100": {"ticker": "^NDX", "label": "Nasdaq 100", "dec": 1},
    # Stocks
    "NVDA":   {"ticker": "NVDA", "label": "Nvidia", "dec": 2},
    "AAPL":   {"ticker": "AAPL", "label": "Apple", "dec": 2},
    "MSFT":   {"ticker": "MSFT", "label": "Microsoft", "dec": 2},
    "AMZN":   {"ticker": "AMZN", "label": "Amazon", "dec": 2},
    "GOOGL":  {"ticker": "GOOGL", "label": "Alphabet", "dec": 2},
    "INTC":   {"ticker": "INTC", "label": "Intel", "dec": 2},
    "AMD":    {"ticker": "AMD", "label": "AMD", "dec": 2},
    "TSM":    {"ticker": "TSM", "label": "Taiwan Semiconductor (ADR)", "dec": 2},
    "ASML":   {"ticker": "ASML", "label": "ASML Holding (ADR)", "dec": 2},
    "SNDK":   {"ticker": "SNDK", "label": "SanDisk", "dec": 2},
    "MU":     {"ticker": "MU", "label": "Micron", "dec": 2},
    "SKHY":   {"ticker": "SKHY", "label": "SK Hynix (ADR)", "dec": 2},
    "SPCX":   {"ticker": "SPCX", "label": "SpaceX", "dec": 2},
    # More crypto / forex
    "ETHUSD": {"ticker": "ETH-USD", "label": "Ethereum", "dec": 2},
    "AUDUSD": {"ticker": "AUDUSD=X", "label": "Aussie / USD", "dec": 5},
    "USDTHB": {"ticker": "THB=X", "label": "USD / Baht", "dec": 3},
    "XAGUSD": {"ticker": "SI=F", "label": "Silver Futures", "dec": 3},
    # BTCTHB has no direct Yahoo Finance ticker, so it's built as a cross
    # rate from BTC-USD x USD/THB instead of a single symbol.
    "BTCTHB": {"cross": ("BTC-USD", "THB=X"), "label": "Bitcoin / Baht", "dec": 0},
}

EMA_FAST = 9
EMA_SLOW = 21
EMA_TREND = 50          # Step 1: sets allowed trade direction
RSI_LEN = 14
ATR_LEN = 14
ADX_LEN = 14
ADX_THRESHOLD = 25      # Step 2: only trade when ADX > 25 (real trend, not sideways/forming)
RSI_ZONE_LOOKBACK = 5   # bars to look back for "RSI was in the pullback zone"
RSI_BUY_ZONE = (40, 50)
RSI_SELL_ZONE = (50, 60)
RSI_OVERSOLD = 30
RSI_OVERBOUGHT = 70
SL_ATR_MULT = 1.5      # stop-loss distance = 1.5x ATR
RR = 2                  # reward:risk multiple -> TP = SL distance x RR (=> TP = 3x ATR)
COOLDOWN_BARS = 3       # don't repeat the same-side signal within N bars

INTERVAL = "15m"         # candle timeframe (M15)
PERIOD = "20d"            # how much history to pull each run (enough bars for EMA50 + lookback)

STATE_FILE = os.path.join(os.path.dirname(__file__), "state.json")

TELEGRAM_TOKEN = os.environ.get("TELEGRAM_BOT_TOKEN", "")
TELEGRAM_CHAT_ID = os.environ.get("TELEGRAM_CHAT_ID", "")


# ---------------------------------------------------------------------------
# Indicators
# ---------------------------------------------------------------------------

def ema(series: pd.Series, period: int) -> pd.Series:
    return series.ewm(span=period, adjust=False).mean()


def rsi(series: pd.Series, period: int = 14) -> pd.Series:
    delta = series.diff()
    gain = delta.clip(lower=0)
    loss = -delta.clip(upper=0)
    avg_gain = gain.ewm(alpha=1 / period, adjust=False).mean()
    avg_loss = loss.ewm(alpha=1 / period, adjust=False).mean()
    rs = avg_gain / avg_loss.replace(0, 1e-10)
    return 100 - (100 / (1 + rs))


def atr(df: pd.DataFrame, period: int = 14) -> pd.Series:
    high, low, close = df["High"], df["Low"], df["Close"]
    tr1 = high - low
    tr2 = (high - close.shift()).abs()
    tr3 = (low - close.shift()).abs()
    tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
    return tr.ewm(alpha=1 / period, adjust=False).mean()


def adx(df: pd.DataFrame, period: int = 14) -> pd.Series:
    """Average Directional Index: measures trend strength (0-100).
    Low values mean a choppy/sideways market where crossover systems tend
    to produce false signals."""
    high, low, close = df["High"], df["Low"], df["Close"]
    up_move = high.diff()
    down_move = -low.diff()
    plus_dm = ((up_move > down_move) & (up_move > 0)) * up_move
    minus_dm = ((down_move > up_move) & (down_move > 0)) * down_move
    tr1 = high - low
    tr2 = (high - close.shift()).abs()
    tr3 = (low - close.shift()).abs()
    tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
    atr_ = tr.ewm(alpha=1 / period, adjust=False).mean()
    plus_di = 100 * (plus_dm.ewm(alpha=1 / period, adjust=False).mean() / atr_.replace(0, 1e-10))
    minus_di = 100 * (minus_dm.ewm(alpha=1 / period, adjust=False).mean() / atr_.replace(0, 1e-10))
    dx = 100 * (plus_di - minus_di).abs() / (plus_di + minus_di).replace(0, 1e-10)
    return dx.ewm(alpha=1 / period, adjust=False).mean()


# ---------------------------------------------------------------------------
# State (used to avoid duplicate / repeated alerts across runs)
# ---------------------------------------------------------------------------

def load_state() -> dict:
    if os.path.exists(STATE_FILE):
        with open(STATE_FILE, "r") as f:
            return json.load(f)
    return {}


def save_state(state: dict) -> None:
    with open(STATE_FILE, "w") as f:
        json.dump(state, f, indent=2, default=str)


# ---------------------------------------------------------------------------
# Telegram
# ---------------------------------------------------------------------------

def send_telegram(text: str) -> None:
    if not TELEGRAM_TOKEN or not TELEGRAM_CHAT_ID:
        print("[warn] TELEGRAM_BOT_TOKEN / TELEGRAM_CHAT_ID not set - skipping alert send")
        return
    url = f"https://api.telegram.org/bot{TELEGRAM_TOKEN}/sendMessage"
    try:
        r = requests.post(
            url,
            json={"chat_id": TELEGRAM_CHAT_ID, "text": text, "parse_mode": "HTML"},
            timeout=15,
        )
        r.raise_for_status()
        if not r.json().get("ok"):
            print("[error] Telegram API returned not-ok:", r.text)
    except Exception as e:
        print("[error] Failed to send Telegram message:", e)


# ---------------------------------------------------------------------------
# Core evaluation
# ---------------------------------------------------------------------------

def _download(ticker: str) -> pd.DataFrame:
    df = yf.download(ticker, period=PERIOD, interval=INTERVAL, progress=False, auto_adjust=False)
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.get_level_values(0)
    return df


def fetch_bars(cfg: dict) -> pd.DataFrame:
    """Returns an OHLC dataframe for either a direct ticker or a synthetic
    cross rate (e.g. BTCTHB = BTC-USD x USD/THB)."""
    if "cross" in cfg:
        base_ticker, quote_ticker = cfg["cross"]
        base = _download(base_ticker)
        quote = _download(quote_ticker)
        if base.empty or quote.empty:
            return pd.DataFrame()
        quote_close = quote["Close"].reindex(base.index, method="ffill")
        df = pd.DataFrame(index=base.index)
        for col in ("Open", "High", "Low", "Close"):
            df[col] = base[col] * quote_close
        return df.dropna()
    return _download(cfg["ticker"])


def evaluate_instrument(name: str, cfg: dict, state: dict) -> None:
    try:
        df = fetch_bars(cfg)
    except Exception as e:
        print(f"[error] {name}: download failed: {e}")
        return

    min_bars = EMA_TREND + RSI_ZONE_LOOKBACK + 2
    if df is None or df.empty or len(df) < min_bars:
        print(f"[warn] {name}: not enough data ({0 if df is None else len(df)} bars, need {min_bars})")
        return

    df["ema_fast"] = ema(df["Close"], EMA_FAST)
    df["ema_slow"] = ema(df["Close"], EMA_SLOW)
    df["ema_trend"] = ema(df["Close"], EMA_TREND)
    df["rsi"] = rsi(df["Close"], RSI_LEN)
    df["atr"] = atr(df, ATR_LEN)
    df["adx"] = adx(df, ADX_LEN)

    last = df.iloc[-1]
    prev = df.iloc[-2]

    if (pd.isna(last["ema_trend"]) or pd.isna(last["ema_fast"]) or pd.isna(last["ema_slow"])
            or pd.isna(prev["ema_fast"]) or pd.isna(prev["ema_slow"])
            or pd.isna(last["rsi"]) or pd.isna(prev["rsi"])
            or pd.isna(last["atr"]) or pd.isna(last["adx"])):
        print(f"[info] {name}: indicators not warmed up yet")
        return

    # Step 1: EMA50 sets the only allowed direction
    if last["Close"] > last["ema_trend"]:
        direction = "UP"
    elif last["Close"] < last["ema_trend"]:
        direction = "DOWN"
    else:
        direction = None

    # Step 2: ADX must show a real trend
    trending = last["adx"] > ADX_THRESHOLD

    # EMA(9/21) momentum must also align with the trade direction
    momentum_up = last["ema_fast"] > last["ema_slow"]
    momentum_down = last["ema_fast"] < last["ema_slow"]

    # fresh EMA(9/21) crossover this bar -- an entry trigger on its own
    ema_cross_up = prev["ema_fast"] <= prev["ema_slow"] and last["ema_fast"] > last["ema_slow"]
    ema_cross_down = prev["ema_fast"] >= prev["ema_slow"] and last["ema_fast"] < last["ema_slow"]

    side = None
    if direction and trending:
        rsi_window = df["rsi"].iloc[-(RSI_ZONE_LOOKBACK + 1):-1]  # bars before the current one
        if direction == "UP" and momentum_up:
            was_in_buy_zone = rsi_window.between(*RSI_BUY_ZONE).any()
            bounced = was_in_buy_zone and prev["rsi"] <= 50 and last["rsi"] > 50
            oversold_cross = prev["rsi"] <= RSI_OVERSOLD and last["rsi"] > RSI_OVERSOLD
            if bounced or oversold_cross or ema_cross_up:
                side = "BUY"
        elif direction == "DOWN" and momentum_down:
            was_in_sell_zone = rsi_window.between(*RSI_SELL_ZONE).any()
            bounced = was_in_sell_zone and prev["rsi"] >= 50 and last["rsi"] < 50
            overbought_cross = prev["rsi"] >= RSI_OVERBOUGHT and last["rsi"] < RSI_OVERBOUGHT
            if bounced or overbought_cross or ema_cross_down:
                side = "SELL"

    bar_time = str(df.index[-1])
    inst_state = state.get(name, {"last_bar_time": None, "last_side": None, "bars_since": 999})

    # advance cooldown counter only when we've moved to a genuinely new bar
    if inst_state.get("last_bar_time") != bar_time:
        inst_state["bars_since"] = inst_state.get("bars_since", 999) + 1
    inst_state["last_bar_time"] = bar_time

    if side and not (inst_state.get("last_side") == side and inst_state.get("bars_since", 999) < COOLDOWN_BARS):
        price = float(last["Close"])
        atr_val = float(last["atr"])
        dec = cfg["dec"]
        if side == "BUY":
            sl = price - SL_ATR_MULT * atr_val
            tp = price + SL_ATR_MULT * RR * atr_val
            emoji = "🟢"
        else:
            sl = price + SL_ATR_MULT * atr_val
            tp = price - SL_ATR_MULT * RR * atr_val
            emoji = "🔴"

        thailand_tz = timezone(timedelta(hours=7))
        msg = (
            f"{emoji} <b>{side} {name}</b> ({cfg['label']})\n"
            f"Entry: {price:.{dec}f}\n"
            f"TP: {tp:.{dec}f}\n"
            f"SL: {sl:.{dec}f}\n"
            f"RSI: {last['rsi']:.1f}  |  ADX: {last['adx']:.1f}  |  Timeframe: {INTERVAL}\n"
            f"เวลา (ไทย): {datetime.now(thailand_tz).strftime('%Y-%m-%d %H:%M')}"
        )
        print(f"[signal] {name}: {side} @ {price}")
        send_telegram(msg)

        inst_state["last_side"] = side
        inst_state["bars_since"] = 0
    else:
        dir_txt = direction or "none"
        mom_txt = "up" if momentum_up else ("down" if momentum_down else "flat")
        print(f"[info] {name}: no new signal (rsi={last['rsi']:.1f}, adx={last['adx']:.1f}, "
              f"direction={dir_txt}, trending={trending}, momentum={mom_txt})")

    state[name] = inst_state


def main() -> int:
    state = load_state()
    for name, cfg in INSTRUMENTS.items():
        evaluate_instrument(name, cfg, state)
    save_state(state)
    return 0


if __name__ == "__main__":
    sys.exit(main())
