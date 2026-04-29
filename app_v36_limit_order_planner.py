from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
import json
import os
from collections import Counter
from pathlib import Path
from threading import Lock
import requests
import pandas as pd
import numpy as np
from datetime import datetime, timezone

ENGINE_VERSION = "v36-limit-order-planner"
DEFAULT_STRATEGY_MODE = "balanced"

app = FastAPI(title=f"XRP Engine {ENGINE_VERSION}")

DEFAULT_CORS_ORIGINS = [
    "http://localhost:3000",
    "http://127.0.0.1:3000",
]


def get_cors_origins() -> list[str]:
    configured = os.getenv("FRONTEND_ORIGINS", "")
    origins = [origin.strip() for origin in configured.split(",") if origin.strip()]
    return DEFAULT_CORS_ORIGINS + origins


app.add_middleware(
    CORSMiddleware,
    allow_origins=get_cors_origins(),
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

COINBASE_CANDLES_URL = "https://api.exchange.coinbase.com/products/{product_id}/candles"
PRODUCT_ID = "XRP-USD"

# User-facing timeframes
TIMEFRAME_CONFIG = {
    "1m": {"granularity": 60, "limit": 200, "resample": None},
    "5m": {"granularity": 300, "limit": 200, "resample": None},
    "15m": {"granularity": 900, "limit": 200, "resample": None},
    "1h": {"granularity": 3600, "limit": 200, "resample": None},
    "4h": {"granularity": 3600, "limit": 240, "resample": "4h"},
    "1d": {"granularity": 86400, "limit": 200, "resample": None},
}


REVERSAL_SETUPS = {
    "Oversold Reversal",
    "Overbought Reversal Risk",
    "Early Bullish Reversal",
    "Early Bearish Reversal",
    "Bullish Exhaustion Reversal",
    "Bearish Exhaustion",
}


TRAP_SETUP_NAMES = {"Bull Trap Risk", "Bear Trap Risk"}

# Mirror-image setup to capitulation bounce:
# pump / failed upside continuation / buyer exhaustion / rejection fade.
EXHAUSTION_REJECTION_SETUPS = {
    "Bull Trap Risk",
    "Overbought Reversal Risk",
    "Bearish Exhaustion",
    "Early Bearish Reversal",
}


BULLISH_BREAKOUT_SETUPS = {"Bullish Breakout", "Volatility Expansion Bullish"}
BEARISH_BREAKOUT_SETUPS = {"Bearish Breakdown", "Volatility Expansion Bearish"}


BULLISH_SIGNALS = {"bullish_breakout", "bullish_trend"}
BEARISH_SIGNALS = {"bearish_breakdown", "bearish_trend"}
NEUTRAL_SIGNALS = {"range_neutral", "neutral"}


FAILURE_RISKS = {"breakout_failure_risk", "breakdown_failure_risk"}


ENTRY_TIMEFRAMES = ["5m", "15m"]
ACTIVE_TIMEFRAMES = ["5m", "15m", "1h"]
HIGHER_TIMEFRAMES = ["1h", "4h", "1d"]



STRATEGY_MODE_CONFIG = {
    "balanced": {
        "name": "balanced",
        "label": "Balanced",
        "soft_fail_limit": 1,
        "transition_limit": 2,
        "higher_timeframe_opposition_is_hard": True,
        "trap_entry_severity": "soft",
        "strategy_trap_block_weight": 2.5,
        "failure_risk_hard_count": 2,
        "timing_wait_profile": "moderate",
        "enter_confidence_bonus": 2,
        "watch_confidence_bonus": 2,
        "early_bias_enabled": True,
        "compression_bias_threshold": 4.5,
        "reversal_bias_threshold": 4.5,
        "early_bias_score_edge": 1.8,
        "trap_reversal_bias_bonus": 1.5,
        "countertrend_reversal_enabled": False,
        "compression_activation_min_cluster": 1,
        "reversal_activation_min_transition": 2,
        "reversal_activation_min_setups": 2,
        "execution_stop_padding_atr": 0.38,
        "execution_stop_padding_buffer": 0.18,
        "execution_chase_buffer_atr": 0.22,
        "execution_scale_out": [60, 25, 15],
        "execution_break_even_after": "tp1_touch",
        "execution_trail_after": "tp2_touch",
        "execution_trail_reference": "5m_ema9",
    },
}

STRATEGY_BIAS_MEMORY: dict[str, dict] = {}

BIAS_MEMORY_DEFAULT_TTL_SECONDS = 1800
BIAS_MEMORY_MAX_AGE_SECONDS = 3600

LOG_DIR = Path(__file__).resolve().parent / "engine_logs"
SNAPSHOT_LOG_PATH = LOG_DIR / "engine_snapshots.jsonl"
SNAPSHOT_LOG_LOCK = Lock()
MAX_LOG_LIMIT = 1000
DEFAULT_OUTCOME_HORIZONS = [1, 3, 6, 12]





def normalize_strategy_mode(mode: str | None) -> str:
    return DEFAULT_STRATEGY_MODE


def get_strategy_mode_config(mode: str | None) -> dict:
    return dict(STRATEGY_MODE_CONFIG[DEFAULT_STRATEGY_MODE])


def build_strategy_mode_metadata(mode: str | None) -> dict:
    active = STRATEGY_MODE_CONFIG[DEFAULT_STRATEGY_MODE]
    return {
        "active": DEFAULT_STRATEGY_MODE,
        "label": active["label"],
        "available": [DEFAULT_STRATEGY_MODE],
    }


def build_bias_memory_key(symbol: str, mode_name: str) -> str:
    return f"{symbol}:{DEFAULT_STRATEGY_MODE}"


def get_bias_memory_ttl_seconds(mode_name: str) -> int:
    return 1800


def get_bias_memory_min_confidence(mode_name: str) -> int:
    return 54


def current_utc_timestamp() -> float:
    return datetime.now(timezone.utc).timestamp()


def compute_bias_confidence(selected_score: float, opposite_score: float, threshold: float, edge_required: float, mode_name: str) -> int:
    edge = max(selected_score - opposite_score, 0.0)
    overshoot = max(selected_score - threshold, 0.0)
    confidence = 46.0 + (overshoot * 8.0) + (edge * 5.0)
    return max(35, min(92, int(round(confidence))))



def infer_early_directional_bias(
    *,
    results: dict,
    mode_config: dict,
    compression_cluster: int,
    transition_cluster: int,
    reversal_setup_count: int,
    higher_bullish_pressure: bool,
    higher_bearish_pressure: bool,
    bull_trap_count: int,
    bear_trap_count: int,
    short_breakout_count: int,
    short_breakdown_count: int,
) -> dict | None:
    if not mode_config.get("early_bias_enabled", False):
        return None

    compression_min_cluster = int(mode_config.get("compression_activation_min_cluster", 2))
    reversal_min_transition = int(mode_config.get("reversal_activation_min_transition", 2))
    reversal_min_setups = int(mode_config.get("reversal_activation_min_setups", 2))

    if compression_cluster >= compression_min_cluster:
        activation_state = "compression"
        threshold = float(mode_config.get("compression_bias_threshold", 99.0))
    elif transition_cluster >= reversal_min_transition or reversal_setup_count >= reversal_min_setups:
        activation_state = "reversal"
        threshold = float(mode_config.get("reversal_bias_threshold", 99.0))
    else:
        return None

    edge_required = float(mode_config.get("early_bias_score_edge", 1.0))
    trap_bonus = float(mode_config.get("trap_reversal_bias_bonus", 0.0))
    countertrend_enabled = bool(mode_config.get("countertrend_reversal_enabled", False))

    bullish_score = 0.0
    bearish_score = 0.0
    bull_reasons: list[str] = []
    bear_reasons: list[str] = []

    def add(direction: str, points: float, reason: str) -> None:
        nonlocal bullish_score, bearish_score
        if points <= 0:
            return
        if direction == "long":
            bullish_score += points
            if reason not in bull_reasons:
                bull_reasons.append(reason)
        else:
            bearish_score += points
            if reason not in bear_reasons:
                bear_reasons.append(reason)

    tf_1m = results.get("1m", {})
    tf_5m = results.get("5m", {})
    tf_15m = results.get("15m", {})
    tf_1h = results.get("1h", {})
    tf_4h = results.get("4h", {})
    tf_1d = results.get("1d", {})

    details_1m = tf_1m.get("signal_details", {})
    details_5m = tf_5m.get("signal_details", {})
    details_15m = tf_15m.get("signal_details", {})
    details_1h = tf_1h.get("signal_details", {})
    details_4h = tf_4h.get("signal_details", {})
    details_1d = tf_1d.get("signal_details", {})

    signal_1m = tf_1m.get("signal")
    signal_5m = tf_5m.get("signal")
    signal_15m = tf_15m.get("signal")
    signal_1h = tf_1h.get("signal")
    signal_4h = tf_4h.get("signal")
    signal_1d = tf_1d.get("signal")

    setup_1m = tf_1m.get("setup")
    setup_5m = tf_5m.get("setup")
    setup_15m = tf_15m.get("setup")
    setup_1h = tf_1h.get("setup")
    setup_4h = tf_4h.get("setup")
    setup_1d = tf_1d.get("setup")

    def score_trend(details: dict, tf_label: str, major: float = 1.0, minor: float = 0.5) -> None:
        trend_bias = details.get("trend_bias")
        momentum_state = details.get("momentum_state")
        rsi_state = details.get("rsi_state")
        pos_ema = details.get("position_vs_ema9")
        pos_sma = details.get("position_vs_sma20")

        if trend_bias == "bullish":
            add("long", major, f"{tf_label} trend bias is bullish.")
        elif trend_bias == "bearish":
            add("short", major, f"{tf_label} trend bias is bearish.")

        if momentum_state == "rising":
            add("long", major, f"{tf_label} momentum is rising.")
        elif momentum_state == "falling":
            add("short", major, f"{tf_label} momentum is falling.")

        if rsi_state in {"bullish", "oversold_recovering"}:
            add("long", minor, f"{tf_label} RSI is improving for bulls.")
        elif rsi_state in {"bearish", "overbought_falling"}:
            add("short", minor, f"{tf_label} RSI is leaning bearish.")

        if pos_ema == "above" and pos_sma == "above":
            add("long", minor, f"{tf_label} price is holding above its fast trend averages.")
        elif pos_ema == "below" and pos_sma == "below":
            add("short", minor, f"{tf_label} price is holding below its fast trend averages.")

    score_trend(details_5m, "5m", major=0.9, minor=0.4)
    score_trend(details_15m, "15m", major=0.8, minor=0.4)

    if signal_5m in BULLISH_SIGNALS:
        add("long", 2.0, "5m trigger already shows bullish pressure.")
    elif signal_5m in BEARISH_SIGNALS:
        add("short", 2.0, "5m trigger already shows bearish pressure.")

    if signal_15m in BULLISH_SIGNALS:
        add("long", 2.0, "15m setup already leans bullish.")
    elif signal_15m in BEARISH_SIGNALS:
        add("short", 2.0, "15m setup already leans bearish.")

    bullish_setup_names = BULLISH_BREAKOUT_SETUPS | {"Trend Continuation", "Bullish Pressure Build", "Early Bullish Reversal"}
    bearish_setup_names = BEARISH_BREAKOUT_SETUPS | {"Early Bearish Reversal"}

    if setup_5m in bullish_setup_names:
        add("long", 1.0, "5m setup supports upside continuation.")
    if setup_5m in bearish_setup_names:
        add("short", 1.0, "5m setup supports downside continuation.")
    if setup_15m in bullish_setup_names:
        add("long", 1.0, "15m setup supports upside continuation.")
    if setup_15m in bearish_setup_names:
        add("short", 1.0, "15m setup supports downside continuation.")

    bullish_reversal_setups = {"Bear Trap Risk", "Oversold Reversal", "Early Bullish Reversal", "Bullish Exhaustion Reversal"}
    bearish_reversal_setups = {"Bull Trap Risk", "Overbought Reversal Risk", "Early Bearish Reversal", "Bearish Exhaustion"}

    if setup_1h in bullish_reversal_setups:
        add("long", 2.5, "1h reversal evidence favors an upside turn.")
    if setup_1h in bearish_reversal_setups:
        add("short", 2.5, "1h reversal evidence favors a downside turn.")

    if signal_1h in {"oversold_reversal_watch", "bearish_but_bouncing"}:
        add("long", 1.5, "1h behavior is consistent with a bullish reversal attempt.")
    if signal_1h in {"overbought_pullback_watch", "bullish_but_pulling_back"}:
        add("short", 1.5, "1h behavior is consistent with a bearish reversal attempt.")

    if signal_4h in BULLISH_SIGNALS:
        add("long", 1.0, "4h structure supports upside continuation.")
    elif signal_4h in BEARISH_SIGNALS:
        add("short", 1.0, "4h structure supports downside continuation.")

    if signal_1d in BULLISH_SIGNALS:
        add("long", 1.0, "Daily structure is supportive on the upside.")
    elif signal_1d in BEARISH_SIGNALS:
        add("short", 1.0, "Daily structure is still leaning lower.")

    if higher_bullish_pressure:
        add("long", 2.0, "Higher timeframe pressure still leans bullish.")
    if higher_bearish_pressure:
        add("short", 2.0, "Higher timeframe pressure still leans bearish.")

    if short_breakout_count >= 1:
        add("long", 1.5, "Short-term breakout pressure is already forming.")
    if short_breakdown_count >= 1:
        add("short", 1.5, "Short-term breakdown pressure is already forming.")

    if bear_trap_count >= 1:
        add("long", trap_bonus * bear_trap_count, "Active bear trap risk favors an upside reversal bias.")
    if bull_trap_count >= 1:
        add("short", trap_bonus * bull_trap_count, "Active bull trap risk favors a downside reversal bias.")

    if activation_state == "compression":
        if details_5m.get("momentum_state") == "rising" and details_15m.get("rsi_state") in {"bullish", "oversold_recovering"}:
            add("long", 0.8, "Compression is leaning upward on the trigger stack.")
        if details_5m.get("momentum_state") == "falling" and details_15m.get("rsi_state") in {"bearish", "overbought_falling"}:
            add("short", 0.8, "Compression is leaning downward on the trigger stack.")

        if mode_config.get("name") == "balanced":
            one_min_bullish_impulse = signal_1m == "bullish_breakout" or setup_1m in BULLISH_BREAKOUT_SETUPS
            one_min_bearish_impulse = signal_1m == "bearish_breakdown" or setup_1m in BEARISH_BREAKOUT_SETUPS
            higher_bullish_backdrop = (
                signal_4h in BULLISH_SIGNALS
                or details_4h.get("trend_bias") == "bullish"
                or setup_4h in {"Trend Continuation", "Bullish Pullback"}
                or signal_1d in BULLISH_SIGNALS
                or details_1d.get("trend_bias") == "bullish"
                or setup_1d in {"Early Bullish Reversal", "Bullish Exhaustion Reversal", "Range Compression"}
            )
            higher_bearish_backdrop = (
                signal_4h in BEARISH_SIGNALS
                or details_4h.get("trend_bias") == "bearish"
                or setup_4h in {"Trend Continuation", "Bearish Bounce Attempt"}
                or signal_1d in BEARISH_SIGNALS
                or details_1d.get("trend_bias") == "bearish"
                or setup_1d in {"Early Bearish Reversal", "Bearish Exhaustion", "Range Compression"}
            )

            if one_min_bullish_impulse and higher_bullish_backdrop:
                bearish_score -= 4.0
                bullish_score += 1.0
            if one_min_bearish_impulse and higher_bearish_backdrop:
                bullish_score -= 4.0
                bearish_score += 1.0

            if details_5m.get("momentum_state") == "rising" and details_15m.get("momentum_state") == "rising":
                bearish_score -= 1.5
            if details_5m.get("momentum_state") == "falling" and details_15m.get("momentum_state") == "falling":
                bullish_score -= 1.5
    else:
        reversal_bullish_count = sum(1 for setup in [setup_5m, setup_15m, setup_1h, setup_4h, setup_1d] if setup in bullish_reversal_setups)
        reversal_bearish_count = sum(1 for setup in [setup_5m, setup_15m, setup_1h, setup_4h, setup_1d] if setup in bearish_reversal_setups)
        if reversal_bullish_count >= 2:
            add("long", 1.0, "Multiple reversal markers are favoring a bullish turn.")
        if reversal_bearish_count >= 2:
            add("short", 1.0, "Multiple reversal markers are favoring a bearish turn.")

    if not countertrend_enabled:
        if higher_bearish_pressure and activation_state == "reversal":
            bullish_score -= 0.5
        if higher_bullish_pressure and activation_state == "reversal":
            bearish_score -= 0.5

    setup_15m_bullish_structure = (
        signal_15m in BULLISH_SIGNALS
        or setup_15m in {"Bullish Pullback", "Trend Continuation", "Bullish Pressure Build", "Early Bullish Reversal", "Oversold Reversal", "Bullish Exhaustion Reversal"}
        or details_15m.get("trend_bias") == "bullish"
    )
    setup_15m_bearish_structure = (
        signal_15m in BEARISH_SIGNALS
        or setup_15m in {"Bearish Bounce Attempt", "Trend Continuation", "Early Bearish Reversal", "Overbought Reversal Risk", "Bearish Exhaustion"}
        or details_15m.get("trend_bias") == "bearish"
    )
    real_bearish_trigger = signal_5m in BEARISH_SIGNALS or short_breakdown_count >= 1
    real_bullish_trigger = signal_5m in BULLISH_SIGNALS or short_breakout_count >= 1

    if activation_state == "compression":
        if setup_15m_bullish_structure and not real_bearish_trigger and not higher_bearish_pressure:
            bearish_score -= 5.5
        if setup_15m_bearish_structure and not real_bullish_trigger and not higher_bullish_pressure:
            bullish_score -= 5.5
        if setup_15m == "Bullish Pullback" and details_15m.get("trend_bias") == "bullish" and not real_bearish_trigger:
            bearish_score -= 4.0
        if setup_15m == "Bearish Bounce Attempt" and details_15m.get("trend_bias") == "bearish" and not real_bullish_trigger:
            bullish_score -= 4.0

    if activation_state == "reversal" and mode_config.get("name") == "balanced":
        higher_bullish_context = (
            signal_4h in BULLISH_SIGNALS
            or (tf_4h.get("structure") == "trend" and details_4h.get("trend_bias") == "bullish")
            or setup_4h == "Trend Continuation"
        )
        higher_bearish_context = (
            signal_4h in BEARISH_SIGNALS
            or (tf_4h.get("structure") == "trend" and details_4h.get("trend_bias") == "bearish")
            or (setup_4h == "Trend Continuation" and details_4h.get("trend_bias") == "bearish")
        )
        daily_not_bearish = signal_1d not in BEARISH_SIGNALS and setup_1d not in {"Overbought Reversal Risk", "Early Bearish Reversal", "Bearish Exhaustion"}
        daily_not_bullish = signal_1d not in BULLISH_SIGNALS and setup_1d not in {"Oversold Reversal", "Early Bullish Reversal", "Bullish Exhaustion Reversal"}
        bearish_trigger_ready = signal_5m in BEARISH_SIGNALS or short_breakdown_count >= 1
        bullish_trigger_ready = signal_5m in BULLISH_SIGNALS or short_breakout_count >= 1

        if higher_bullish_context and daily_not_bearish and not bearish_trigger_ready:
            bearish_score -= 3.0
            bullish_score += 1.0
        if higher_bearish_context and daily_not_bullish and not bullish_trigger_ready:
            bullish_score -= 3.0
            bearish_score += 1.0

        if setup_15m in {"Early Bullish Reversal", "Oversold Reversal", "Bullish Exhaustion Reversal"} and not bearish_trigger_ready:
            bearish_score -= 2.0
        if setup_15m in {"Early Bearish Reversal", "Overbought Reversal Risk", "Bearish Exhaustion"} and not bullish_trigger_ready:
            bullish_score -= 2.0

    selected_direction = None
    selected_score = 0.0
    opposite_score = 0.0
    selected_reasons: list[str] = []

    if bullish_score >= threshold and bullish_score >= (bearish_score + edge_required):
        selected_direction = "long"
        selected_score = bullish_score
        opposite_score = bearish_score
        selected_reasons = bull_reasons[:3]
    elif bearish_score >= threshold and bearish_score >= (bullish_score + edge_required):
        selected_direction = "short"
        selected_score = bearish_score
        opposite_score = bullish_score
        selected_reasons = bear_reasons[:3]

    if selected_direction is None:
        return None

    if activation_state == "compression":
        selected_action = "watch_breakout" if selected_direction == "long" else "watch_breakdown"
    else:
        selected_action = "watch_reversal"

    bias_confidence = compute_bias_confidence(
        selected_score=selected_score,
        opposite_score=opposite_score,
        threshold=threshold,
        edge_required=edge_required,
        mode_name=mode_config.get("name", DEFAULT_STRATEGY_MODE),
    )

    return {
        "strategy_bias": selected_direction,
        "state": activation_state,
        "action": selected_action,
        "score": round(selected_score, 2),
        "opposite_score": round(opposite_score, 2),
        "edge": round(max(selected_score - opposite_score, 0.0), 2),
        "confidence": bias_confidence,
        "reasons": selected_reasons,
        "origin": f"{activation_state}_early_bias",
    }


def round_or_none(value, digits=6):
    if value is None:
        return None
    if pd.isna(value):
        return None
    return round(float(value), digits)



def get_unix_range(granularity: int, limit: int) -> tuple[int, int]:
    end = int(datetime.now(timezone.utc).timestamp())
    start = end - (granularity * limit)
    return start, end



def fetch_candles(product_id: str, granularity: int, limit: int = 200) -> pd.DataFrame:
    start, end = get_unix_range(granularity, limit)

    url = COINBASE_CANDLES_URL.format(product_id=product_id)
    params = {
        "start": start,
        "end": end,
        "granularity": granularity,
    }

    headers = {
        "Accept": "application/json",
        "User-Agent": "xrp-engine/1.0",
    }

    response = requests.get(url, params=params, headers=headers, timeout=10)
    response.raise_for_status()

    raw = response.json()

    if not raw:
        raise HTTPException(status_code=502, detail="No candle data returned from Coinbase.")

    # Coinbase Exchange format:
    # [time, low, high, open, close, volume]
    df = pd.DataFrame(raw, columns=["time", "low", "high", "open", "close", "volume"])

    # API often returns newest first; sort ascending
    df["time"] = pd.to_datetime(df["time"], unit="s", utc=True)

    numeric_cols = ["low", "high", "open", "close", "volume"]
    for col in numeric_cols:
        df[col] = pd.to_numeric(df[col], errors="coerce")

    df = df.sort_values("time").drop_duplicates(subset=["time"]).reset_index(drop=True)
    return df



def resample_candles(df: pd.DataFrame, rule: str) -> pd.DataFrame:
    if df.empty:
        return df

    resampled = (
        df.set_index("time")
        .resample(rule)
        .agg(
            {
                "open": "first",
                "high": "max",
                "low": "min",
                "close": "last",
                "volume": "sum",
            }
        )
        .dropna()
        .reset_index()
    )

    return resampled



def add_indicators(df: pd.DataFrame) -> pd.DataFrame:
    if df.empty:
        return df

    close = df["close"]
    volume = df["volume"]

    # Core moving averages
    df["sma_5"] = close.rolling(5).mean()
    df["sma_20"] = close.rolling(20).mean()
    df["ema_9"] = close.ewm(span=9, adjust=False).mean()

    # RSI(14) with stable edge-case handling for one-way candles
    delta = close.diff()
    gain = delta.clip(lower=0)
    loss = -delta.clip(upper=0)
    avg_gain = gain.rolling(14).mean()
    avg_loss = loss.rolling(14).mean()
    rs = avg_gain / avg_loss.replace(0, np.nan)
    rsi = 100 - (100 / (1 + rs))
    rsi = rsi.where(avg_loss != 0, 100.0)
    rsi = rsi.where(avg_gain != 0, 0.0)
    rsi = rsi.where(~((avg_gain == 0) & (avg_loss == 0)), 50.0)
    df["rsi_14"] = rsi

    # Momentum
    df["momentum"] = close.diff()

    # Bollinger Bands (20, 2)
    bb_mid = close.rolling(20).mean()
    bb_std = close.rolling(20).std()
    df["bb_mid"] = bb_mid
    df["bb_upper"] = bb_mid + (2 * bb_std)
    df["bb_lower"] = bb_mid - (2 * bb_std)

    # ATR(14)
    prev_close = close.shift(1)
    tr_components = pd.concat(
        [
            (df["high"] - df["low"]).abs(),
            (df["high"] - prev_close).abs(),
            (df["low"] - prev_close).abs(),
        ],
        axis=1,
    )
    true_range = tr_components.max(axis=1)
    df["atr_14"] = true_range.rolling(14).mean()

    # OBV
    obv = [0]
    for i in range(1, len(df)):
        if close.iloc[i] > close.iloc[i - 1]:
            obv.append(obv[-1] + volume.iloc[i])
        elif close.iloc[i] < close.iloc[i - 1]:
            obv.append(obv[-1] - volume.iloc[i])
        else:
            obv.append(obv[-1])
    df["obv"] = obv

    return df





def add_fibonacci(df: pd.DataFrame, lookback: int = 50) -> dict:
    if len(df) < 2:
        return {}

    recent = df.tail(lookback)
    swing_high = recent["high"].max()
    swing_low = recent["low"].min()
    price_range = swing_high - swing_low

    if price_range <= 0:
        return {}

    return {
        "high": round_or_none(swing_high),
        "low": round_or_none(swing_low),
        "levels": {
            "0.236": round_or_none(swing_high - price_range * 0.236),
            "0.382": round_or_none(swing_high - price_range * 0.382),
            "0.5": round_or_none(swing_high - price_range * 0.5),
            "0.618": round_or_none(swing_high - price_range * 0.618),
            "0.786": round_or_none(swing_high - price_range * 0.786),
        },
    }



def generate_signal(df: pd.DataFrame) -> str:
    if len(df) < 20:
        return "neutral"

    last = df.iloc[-1]
    prev = df.iloc[-2] if len(df) > 1 else last

    price = last["close"]
    ema_9 = last["ema_9"]
    sma_20 = last["sma_20"]
    rsi_14 = last["rsi_14"]
    bb_upper = last["bb_upper"]
    bb_lower = last["bb_lower"]
    momentum = last["momentum"]

    prev_rsi = prev["rsi_14"] if pd.notna(prev["rsi_14"]) else None

    required = [price, ema_9, sma_20, rsi_14, bb_upper, bb_lower, momentum]
    if any(pd.isna(x) for x in required):
        return "neutral"

    above_ema = price > ema_9
    below_ema = price < ema_9
    above_sma20 = price > sma_20
    below_sma20 = price < sma_20

    momentum_up = momentum > 0
    momentum_down = momentum < 0

    near_upper_bb = price >= (bb_upper * 0.995)
    near_lower_bb = price <= (bb_lower * 1.005)

    rsi_bullish = rsi_14 >= 55
    rsi_bearish = rsi_14 <= 45
    rsi_oversold = rsi_14 <= 35
    rsi_overbought = rsi_14 >= 70

    rsi_recovering = prev_rsi is not None and rsi_14 > prev_rsi
    rsi_falling = prev_rsi is not None and rsi_14 < prev_rsi

    if above_ema and above_sma20 and rsi_bullish and momentum_up and near_upper_bb:
        return "bullish_breakout"

    if below_ema and below_sma20 and rsi_bearish and momentum_down and near_lower_bb:
        return "bearish_breakdown"

    if above_ema and above_sma20 and rsi_bullish and momentum_up:
        return "bullish_trend"

    if below_ema and below_sma20 and rsi_bearish and momentum_down:
        return "bearish_trend"

    if below_ema and rsi_oversold and rsi_recovering and momentum_up:
        return "oversold_reversal_watch"

    if above_ema and rsi_overbought and rsi_falling and momentum_down:
        return "overbought_pullback_watch"

    if below_sma20 and momentum_up and rsi_recovering:
        return "bearish_but_bouncing"

    if above_sma20 and momentum_down and rsi_falling:
        return "bullish_but_pulling_back"

    return "range_neutral"



def get_signal_score(signal: str) -> int:
    score_map = {
        "bullish_breakout": 2,
        "bullish_trend": 1,
        "oversold_reversal_watch": 1,
        "range_neutral": 0,
        "neutral": 0,
        "bullish_but_pulling_back": 0,
        "bearish_but_bouncing": 0,
        "overbought_pullback_watch": -1,
        "bearish_trend": -1,
        "bearish_breakdown": -2,
    }
    return score_map.get(signal, 0)



def get_signal_details(df: pd.DataFrame) -> dict:
    if len(df) < 20:
        return {
            "trend_bias": "unknown",
            "momentum_state": "unknown",
            "rsi_state": "unknown",
            "bollinger_state": "unknown",
            "position_vs_ema9": "unknown",
            "position_vs_sma20": "unknown",
        }

    last = df.iloc[-1]
    prev = df.iloc[-2] if len(df) > 1 else last

    price = last["close"]
    ema_9 = last["ema_9"]
    sma_20 = last["sma_20"]
    rsi_14 = last["rsi_14"]
    bb_upper = last["bb_upper"]
    bb_lower = last["bb_lower"]
    momentum = last["momentum"]

    prev_rsi = prev["rsi_14"] if pd.notna(prev["rsi_14"]) else None

    if any(pd.isna(x) for x in [price, ema_9, sma_20, rsi_14, bb_upper, bb_lower, momentum]):
        return {
            "trend_bias": "unknown",
            "momentum_state": "unknown",
            "rsi_state": "unknown",
            "bollinger_state": "unknown",
            "position_vs_ema9": "unknown",
            "position_vs_sma20": "unknown",
        }

    if price > ema_9 and price > sma_20:
        trend_bias = "bullish"
    elif price < ema_9 and price < sma_20:
        trend_bias = "bearish"
    else:
        trend_bias = "mixed"

    if momentum > 0:
        momentum_state = "rising"
    elif momentum < 0:
        momentum_state = "falling"
    else:
        momentum_state = "flat"

    if rsi_14 >= 70:
        rsi_state = "overbought"
    elif rsi_14 <= 30:
        rsi_state = "oversold"
    elif rsi_14 >= 55:
        rsi_state = "bullish"
    elif rsi_14 <= 45:
        rsi_state = "bearish"
    else:
        rsi_state = "neutral"

    if prev_rsi is not None:
        if rsi_14 > prev_rsi and rsi_state == "oversold":
            rsi_state = "oversold_recovering"
        elif rsi_14 < prev_rsi and rsi_state == "overbought":
            rsi_state = "overbought_falling"

    if price >= bb_upper:
        bollinger_state = "above_upper_band"
    elif price <= bb_lower:
        bollinger_state = "below_lower_band"
    else:
        band_range = bb_upper - bb_lower
        if band_range > 0:
            band_position = (price - bb_lower) / band_range
            if band_position >= 0.8:
                bollinger_state = "near_upper_band"
            elif band_position <= 0.2:
                bollinger_state = "near_lower_band"
            else:
                bollinger_state = "inside_bands"
        else:
            bollinger_state = "inside_bands"

    position_vs_ema9 = "above" if price > ema_9 else "below" if price < ema_9 else "at"
    position_vs_sma20 = "above" if price > sma_20 else "below" if price < sma_20 else "at"

    return {
        "trend_bias": trend_bias,
        "momentum_state": momentum_state,
        "rsi_state": rsi_state,
        "bollinger_state": bollinger_state,
        "position_vs_ema9": position_vs_ema9,
        "position_vs_sma20": position_vs_sma20,
    }



def build_summary(
    signal: str,
    signal_details: dict,
    timeframe: str,
    symbol: str,
    setup: str | None = None,
    setup_confidence: int = 0,
    structure: str | None = None,
    trap_risk: str | None = None,
) -> str:
    trend_bias = signal_details.get("trend_bias", "unknown")
    momentum_state = signal_details.get("momentum_state", "unknown")
    rsi_state = signal_details.get("rsi_state", "unknown")
    bollinger_state = signal_details.get("bollinger_state", "unknown")
    position_vs_ema9 = signal_details.get("position_vs_ema9", "unknown")
    position_vs_sma20 = signal_details.get("position_vs_sma20", "unknown")

    signal_map = {
        "bullish_breakout": "strong bullish breakout conditions",
        "bearish_breakdown": "strong bearish breakdown conditions",
        "bullish_trend": "a bullish trend",
        "bearish_trend": "a bearish trend",
        "oversold_reversal_watch": "a possible oversold reversal setup",
        "overbought_pullback_watch": "a possible overbought pullback setup",
        "bearish_but_bouncing": "a bearish-leaning structure with a short-term bounce attempt",
        "bullish_but_pulling_back": "a bullish-leaning structure with a short-term pullback",
        "range_neutral": "a neutral ranging market",
        "neutral": "a neutral market state",
    }

    signal_text = signal_map.get(signal, signal.replace("_", " "))
    base = f"On the {timeframe} timeframe, {symbol} is showing {signal_text}. "

    if setup:
        if setup in TRAP_SETUP_NAMES:
            setup_sentence = (
                f"The current setup is {setup}, warning that the latest move may be vulnerable to a failed continuation or reversal trap. "
            )
        elif setup in REVERSAL_SETUPS:
            setup_sentence = (
                f"The current setup is {setup}, suggesting a potential shift in direction may be developing. "
            )
        elif setup == "Range Compression":
            setup_sentence = (
                f"The current setup is {setup}, indicating momentum is tightening and a larger move may be building. "
            )
        elif setup_confidence >= 75:
            setup_sentence = (
                f"The current setup is {setup} with high confidence, "
                f"suggesting a strong directional phase may be underway. "
            )
        elif setup_confidence >= 65:
            setup_sentence = (
                f"The current setup is {setup} with moderate-to-strong confidence, "
                f"reinforcing the present market bias. "
            )
        elif setup_confidence >= 55:
            setup_sentence = (
                f"The current setup is {setup}, offering additional context for the current price structure. "
            )
        else:
            setup_sentence = f"The current setup is {setup}, though conviction remains limited. "
    else:
        setup_sentence = ""

    detail_sentence = (
        f"Trend bias is {trend_bias}, momentum is {momentum_state}, and RSI is {rsi_state}. "
        f"Price is {position_vs_ema9} EMA9 and {position_vs_sma20} SMA20, "
        f"with Bollinger position at {bollinger_state}."
    )

    extra = ""
    if structure:
        extra += f" Market structure is currently {structure.replace('_', ' ')}."
    if trap_risk:
        extra += f" There is a potential {trap_risk.replace('_', ' ')} in this timeframe."

    return base + setup_sentence + detail_sentence + extra



def classify_bias_from_average(avg_score: float) -> str:
    if avg_score >= 1.5:
        return "strong_bullish"
    if avg_score >= 0.5:
        return "bullish"
    if avg_score <= -1.5:
        return "strong_bearish"
    if avg_score <= -0.5:
        return "bearish"
    return "neutral"



def build_dashboard_summary(
    short_term_bias: str,
    higher_timeframe_bias: str,
    overall_bias: str,
    symbol: str,
    results: dict,
    strategy: dict | None = None,
) -> str:
    short_term_keys = ["1m", "5m", "15m"]
    higher_timeframe_keys = ["1h", "4h", "1d"]
    active_timeframe_keys = [tf for tf in ACTIVE_TIMEFRAMES if tf in results]

    trap_weights = {
        "1m": 0.75,
        "5m": 1.5,
        "15m": 2.0,
        "1h": 2.0,
        "4h": 1.25,
        "1d": 1.0,
    }

    def trap_count(keys: list[str], trap_name: str) -> int:
        return sum(1 for tf in keys if results.get(tf, {}).get("trap_risk") == trap_name)

    def weighted_trap_score(trap_name: str) -> float:
        return sum(
            trap_weights.get(tf, 1.0)
            for tf in results
            if results.get(tf, {}).get("trap_risk") == trap_name
        )

    setups = [results[tf].get("setup") for tf in results if results[tf].get("setup")]
    structures = [results[tf].get("structure") for tf in results if results[tf].get("structure")]
    trap_risks = [results[tf].get("trap_risk") for tf in results if results[tf].get("trap_risk")]

    higher_structures = [results[tf].get("structure") for tf in higher_timeframe_keys if tf in results]
    short_signals = [results[tf].get("signal") for tf in short_term_keys if tf in results]
    higher_signals = [results[tf].get("signal") for tf in higher_timeframe_keys if tf in results]

    breakout_count = sum(1 for s in setups if s in BULLISH_BREAKOUT_SETUPS)
    bearish_breakout_count = sum(1 for s in setups if s in BEARISH_BREAKOUT_SETUPS)
    continuation_count = sum(1 for s in setups if s == "Trend Continuation")
    pullback_count = sum(1 for s in setups if s == "Bullish Pullback")
    reversal_count = sum(1 for s in setups if s in {"Oversold Reversal", "Early Bullish Reversal"})

    compression_count = sum(1 for s in structures if s == "compression")
    transition_count = sum(1 for s in structures if s == "transition")
    trend_count = sum(1 for s in structures if s == "trend")
    expansion_count = sum(1 for s in structures if s == "expansion")

    higher_compression_count = sum(1 for s in higher_structures if s == "compression")

    bull_trap_count = sum(1 for t in trap_risks if t == "bull_trap_risk")
    bear_trap_count = sum(1 for t in trap_risks if t == "bear_trap_risk")
    failure_risk_count = sum(1 for t in trap_risks if t in FAILURE_RISKS)

    active_bull_trap_count = trap_count(active_timeframe_keys, "bull_trap_risk")
    active_bear_trap_count = trap_count(active_timeframe_keys, "bear_trap_risk")
    higher_bull_trap_count = trap_count(higher_timeframe_keys, "bull_trap_risk")
    higher_bear_trap_count = trap_count(higher_timeframe_keys, "bear_trap_risk")

    bull_trap_weight = weighted_trap_score("bull_trap_risk")
    bear_trap_weight = weighted_trap_score("bear_trap_risk")

    mixed_trap_environment = bull_trap_count >= 1 and bear_trap_count >= 1
    active_mixed_trap_environment = active_bull_trap_count >= 1 and active_bear_trap_count >= 1
    dominant_bull_trap = bull_trap_weight >= (bear_trap_weight + 1.0) and active_bull_trap_count >= 1
    dominant_bear_trap = bear_trap_weight >= (bull_trap_weight + 1.0) and active_bear_trap_count >= 1

    short_neutral_count = sum(1 for s in short_signals if s in NEUTRAL_SIGNALS)
    higher_bullish_count = sum(1 for s in higher_signals if s in BULLISH_SIGNALS)

    strategy_action = strategy.get("action") if strategy else None
    strategy_state = strategy.get("state") if strategy else None
    strategy_bias = strategy.get("strategy_bias") if strategy else None
    trigger_context = strategy.get("trigger_context") if strategy else None
    trigger_state = (trigger_context or {}).get("state")

    tf_5m = results.get("5m", {})
    tf_15m = results.get("15m", {})
    signal_5m = tf_5m.get("signal")
    signal_15m = tf_15m.get("signal")
    setup_15m = tf_15m.get("setup")
    trend_15m = tf_15m.get("signal_details", {}).get("trend_bias")
    structure_15m = tf_15m.get("structure")

    fifteen_bullish_structure = (
        signal_15m in BULLISH_SIGNALS
        or setup_15m in {"Bullish Pullback", "Trend Continuation", "Bullish Pressure Build", "Early Bullish Reversal", "Oversold Reversal", "Bullish Exhaustion Reversal"}
        or trend_15m == "bullish"
    )
    fifteen_bearish_structure = (
        signal_15m in BEARISH_SIGNALS
        or setup_15m in {"Bearish Bounce Attempt", "Trend Continuation", "Early Bearish Reversal", "Overbought Reversal Risk", "Bearish Exhaustion"}
        or trend_15m == "bearish"
    )

    if strategy_state == "countertrend_short_watch" or trigger_state == "countertrend_short_watch":
        return (
            f"{symbol} is still holding a broadly bullish 15m structure, but a short-term pullback is trying to develop inside compression. "
            f"That makes this a countertrend short watch, not a qualified bearish continuation setup yet."
        )

    if strategy_state == "countertrend_long_watch" or trigger_state == "countertrend_long_watch":
        return (
            f"{symbol} is still holding a broadly bearish 15m structure, but a short-term bounce is trying to develop inside compression. "
            f"That makes this a countertrend long watch, not a qualified bullish continuation setup yet."
        )

    if strategy_state == "compression" and strategy_bias == "neutral" and fifteen_bullish_structure and signal_5m in {"bullish_but_pulling_back", "range_neutral", None}:
        return (
            f"{symbol} is still holding a bullish 15m structure while price pulls back inside compression. "
            f"That keeps bearish continuation unqualified for now, so the better read is a bullish pause rather than confirmed short control."
        )

    if strategy_state == "compression" and strategy_bias == "neutral" and fifteen_bearish_structure and signal_5m in {"bearish_but_bouncing", "range_neutral", None}:
        return (
            f"{symbol} is still holding a bearish 15m structure while price bounces inside compression. "
            f"That keeps bullish continuation unqualified for now, so the better read is a bearish pause rather than confirmed long control."
        )

    if trigger_state == "direction_supported_but_setup_unqualified":
        if strategy_bias == "short" and fifteen_bullish_structure:
            return (
                f"{symbol} is still holding a bullish 15m structure, but price is pulling back inside compression. "
                f"That keeps bearish continuation unqualified for now, so the better read is a bullish pause rather than confirmed short control."
            )
        if strategy_bias == "long" and fifteen_bearish_structure:
            return (
                f"{symbol} is still holding a bearish 15m structure, but price is trying to bounce inside compression. "
                f"That keeps bullish continuation unqualified for now, so the better read is a bearish pause rather than confirmed long control."
            )

    if trigger_state == "indicator_break_without_band_break":
        if strategy_bias == "long":
            return (
                f"{symbol} has a bullish trigger on the 5m chart, but price is still inside the monitored compression band. "
                f"That means buyers have an indicator breakout, not a fully confirmed band break yet."
            )
        if strategy_bias == "short":
            return (
                f"{symbol} has a bearish trigger on the 5m chart, but price is still inside the monitored compression band. "
                f"That means sellers have an indicator breakdown, not a fully confirmed band break yet."
            )

    if strategy_state == "pre_breakdown":
        trigger_signal = (trigger_context or {}).get("trigger_signal") or results.get("5m", {}).get("signal")
        return (
            f"{symbol} already has bearish pressure on the 15m setup timeframe, but the 5m trigger has not fully confirmed yet. "
            f"That suggests sellers have the lead, though the short still depends on a failed bounce or clean 5m breakdown instead of a blind chase from {str(trigger_signal).replace('_', ' ')} conditions."
        )

    if strategy_state == "pre_breakout":
        trigger_signal = (trigger_context or {}).get("trigger_signal") or results.get("5m", {}).get("signal")
        return (
            f"{symbol} already has bullish pressure on the 15m setup timeframe, but the 5m trigger has not fully confirmed yet. "
            f"That suggests buyers have the lead, though the long still depends on a pullback hold or clean 5m breakout instead of a blind chase from {str(trigger_signal).replace('_', ' ')} conditions."
        )

    if strategy_state == "exhaustion_rejection":
        return (
            f"{symbol} is showing a failed upside continuation attempt, with bull-trap or buyer-exhaustion evidence active on the trading stack. "
            f"That makes this a tactical exhaustion-rejection short watch, not a generic trend short."
        )

    # 0. Strategy-level trap states take top priority so the dashboard story matches the execution posture.
    if strategy_state == "trap":
        if strategy_action == "avoid_short_trap" and (active_bear_trap_count >= 1 or bear_trap_weight >= bull_trap_weight):
            return (
                f"{symbol} is in a trap-heavy transition environment, with reversal pressure building against recent downside pressure on the active trading timeframes. "
                f"That suggests short continuation is vulnerable to a squeeze or reversal until sellers regain clean control."
            )
        if strategy_action == "avoid_long_trap" and (active_bull_trap_count >= 1 or bull_trap_weight >= bear_trap_weight):
            return (
                f"{symbol} is in a trap-heavy transition environment, with follow-through weakening against recent upside pressure on the active trading timeframes. "
                f"That suggests long continuation is vulnerable to a fade or reversal until buyers reclaim clean control."
            )
        return (
            f"{symbol} is in a trap-heavy and unstable environment, with conflicting trap and reversal signals across timeframes. "
            f"That suggests directional follow-through is unreliable until one side regains clear control."
        )

    if strategy_state in {"continuation", "continuation_watch"} and compression_count >= 2 and strategy and strategy.get("strategy_bias") == "long":
        if strategy_action == "enter_long":
            return (
                f"{symbol} is resolving higher out of a prior compression regime, with the active trading timeframes starting to expand upward. "
                f"That suggests buyers are taking control as the market exits its recent range."
            )
        if strategy_action == "watch_breakout":
            return (
                f"{symbol} is pressing higher against the edge of a prior compression regime, but full upside follow-through is not confirmed yet. "
                f"That suggests buyers are probing for a breakout, though confirmation is still needed."
            )

    if strategy_state in {"continuation", "continuation_watch"} and compression_count >= 2 and strategy and strategy.get("strategy_bias") == "short":
        if strategy_action == "enter_short":
            return (
                f"{symbol} is resolving lower out of a prior compression regime, with the active trading timeframes starting to expand downward. "
                f"That suggests sellers are taking control as the market exits its recent range."
            )
        if strategy_action == "watch_breakdown":
            return (
                f"{symbol} is pressing lower against the edge of a prior compression regime, but full downside follow-through is not confirmed yet. "
                f"That suggests sellers are probing for a breakdown, though confirmation is still needed."
            )

    if strategy_state == "compression" and strategy and strategy.get("strategy_bias") == "long":
        return (
            f"{symbol} is compressing across active trading timeframes, but early upside pressure is starting to build inside the range. "
            f"That suggests buyers are trying to base a bullish breakout, though confirmation is still required."
        )

    if strategy_state == "compression" and strategy and strategy.get("strategy_bias") == "short":
        return (
            f"{symbol} is compressing across active trading timeframes, but early downside pressure is starting to build inside the range. "
            f"That suggests sellers are leaning toward a bearish resolution, though confirmation is still required."
        )

    if strategy_state == "reversal" and strategy and strategy.get("strategy_bias") == "long":
        return (
            f"{symbol} is moving through a reversal environment, with early bullish pressure beginning to build against the recent downside move. "
            f"That suggests an upside turn may be developing, though trigger confirmation is still required."
        )

    if strategy_state == "reversal" and strategy and strategy.get("strategy_bias") == "short":
        return (
            f"{symbol} is moving through a reversal environment, with early bearish pressure beginning to build against the recent upside move. "
            f"That suggests a downside turn may be developing, though trigger confirmation is still required."
        )

    # 1. Mixed timeframe alignment first
    if short_term_bias in ["bearish", "strong_bearish"] and higher_timeframe_bias in ["bullish", "strong_bullish"]:
        return (
            f"{symbol} is showing short-term weakness inside a stronger higher-timeframe bullish structure. "
            f"This looks more like a pullback phase than a confirmed bearish shift."
        )

    if short_term_bias in ["bullish", "strong_bullish"] and higher_timeframe_bias in ["bearish", "strong_bearish"]:
        return (
            f"{symbol} is showing a short-term bounce against a weaker higher-timeframe backdrop. "
            f"That suggests relief strength, but not yet a confirmed broader reversal."
        )

    # 1b. Fragile bullish structure
    if (
        short_term_bias in ["bearish", "strong_bearish"]
        and higher_timeframe_bias == "neutral"
        and results.get("1h", {}).get("signal") in BULLISH_SIGNALS
        and (
            results.get("4h", {}).get("structure") == "transition"
            or results.get("4h", {}).get("setup") == "Overbought Reversal Risk"
            or results.get("1d", {}).get("signal") == "bearish_trend"
        )
    ):
        return (
            f"{symbol} is showing short-term breakdown pressure while higher timeframes remain mixed. "
            f"The 1h still holds bullish structure, but higher timeframes are weakening, with reversal risk or bearish trend conditions present. "
            f"That suggests the prior bullish move is losing strength and the market is entering a more fragile transition phase."
        )

    # 2. Trap / failure risk takes priority, with active trading timeframes weighted most heavily.
    if mixed_trap_environment or active_mixed_trap_environment:
        return (
            f"{symbol} is in a trap-heavy transition environment, with conflicting bull-trap and bear-trap signals across timeframes. "
            f"Reversal pressure is building, so follow-through looks unreliable until one side regains clean control."
        )

    if dominant_bull_trap or (
        active_bull_trap_count > active_bear_trap_count
        and higher_bull_trap_count >= higher_bear_trap_count
    ):
        return (
            f"{symbol} is showing upside pressure, but bull trap risk is dominant on the active trading timeframes. "
            f"That suggests upside continuation may be vulnerable if follow-through weakens."
        )

    if dominant_bear_trap or (
        active_bear_trap_count > active_bull_trap_count
        and higher_bear_trap_count >= higher_bull_trap_count
    ):
        return (
            f"{symbol} remains under downside pressure, but bear trap risk is dominant on the active trading timeframes. "
            f"That suggests sellers may be losing control and downside continuation is vulnerable to reversal."
        )

    if failure_risk_count >= 1:
        return (
            f"{symbol} is showing unstable recent directional pressure, with failure risk appearing on breakout or breakdown attempts. "
            f"That suggests follow-through is not fully reliable right now."
        )

    # 3. Bullish expansion inside broader compression
    if overall_bias == "bullish" and expansion_count >= 2 and higher_compression_count >= 1:
        return (
            f"{symbol} is entering a bullish expansion phase on the active trading timeframes, "
            f"while at least one higher timeframe remains in compression. "
            f"That suggests this move may be an early breakout attempt from a larger range."
        )

    # 4. Short-term pause inside bullish structure
    if overall_bias == "bullish" and higher_bullish_count >= 2 and short_neutral_count >= 2:
        return (
            f"{symbol} remains bullish overall, with strength holding on the higher timeframes while short-term charts pause near recent highs. "
            f"That suggests consolidation rather than confirmed reversal."
        )

    # 5. Compression-dominant conditions
    if compression_count >= 3 and expansion_count == 0:
        return (
            f"{symbol} is in a broad compression phase across multiple timeframes, "
            f"suggesting the market may be coiling before a larger directional move."
        )

    if compression_count >= 2 and trend_count >= 2:
        return (
            f"{symbol} is showing compression on some timeframes while the broader trend structure remains intact. "
            f"That suggests the market is pausing before its next clearer move."
        )

    # 6. Transition / reversal environment
    if transition_count >= 2 and reversal_count >= 1:
        return (
            f"{symbol} is moving through a transition phase, with reversal pressure appearing across multiple timeframes. "
            f"That suggests prior trend conditions may be weakening, though confirmation is still needed."
        )

    if pullback_count >= 1 and reversal_count >= 1:
        return (
            f"{symbol} is showing conflicting pullback and reversal signals across timeframes, "
            f"suggesting an unstable market environment with reduced directional clarity."
        )

    # 7. Strong bullish
    if overall_bias == "strong_bullish":
        if breakout_count >= 2 or expansion_count >= 2:
            return (
                f"{symbol} is strongly bullish, with breakout and expansion pressure visible across multiple timeframes, "
                f"indicating aggressive upside momentum."
            )
        if continuation_count >= 1 and pullback_count >= 1:
            return (
                f"{symbol} remains in a strong bullish structure, with continuation conditions supported by healthy pullbacks."
            )
        return f"{symbol} is aligned bullish across most timeframes, with strong continuation conditions."

    # 8. Bullish
    if overall_bias == "bullish":
        if breakout_count >= 1:
            return (
                f"{symbol} is maintaining a bullish structure, with breakout pressure visible on key timeframes "
                f"and continued upside momentum still present."
            )
        if continuation_count >= 1:
            return (
                f"{symbol} is trending bullish overall, with continuation structure supporting the current upward bias."
            )
        if compression_count >= 2:
            return (
                f"{symbol} is leaning bullish overall, though several timeframes are compressing. "
                f"That suggests bullish structure remains in place, but momentum is pausing before the next move."
            )
        if reversal_count >= 1:
            return f"{symbol} is leaning bullish, with reversal pressure beginning to support the upside."
        return f"{symbol} is leaning bullish overall, though some mixed signals are still present."

    # 9. Strong bearish
    if overall_bias == "strong_bearish":
        if bearish_breakout_count >= 2:
            return (
                f"{symbol} is strongly bearish, with breakdown and expansion pressure visible across multiple timeframes, "
                f"indicating aggressive downside momentum."
            )
        return f"{symbol} is aligned bearish across most timeframes, showing broad downside pressure."

    # 10. Bearish
    if overall_bias == "bearish":
        if bearish_breakout_count >= 1:
            return (
                f"{symbol} is leaning bearish overall, with breakdown pressure active on key timeframes and downside momentum still present."
            )
        if compression_count >= 2:
            return (
                f"{symbol} is leaning bearish overall, though several timeframes are compressing. "
                f"That suggests bearish structure remains in place, but momentum is pausing before the next move."
            )
        if reversal_count >= 1 or transition_count >= 1:
            return (
                f"{symbol} is leaning bearish overall, but reversal or transition pressure is beginning to appear, "
                f"which may signal weakening downside momentum."
            )
        return f"{symbol} is leaning bearish overall, though some bounce attempts or mixed signals may still be present."

    # 11. Neutral / mixed
    if transition_count >= 2:
        return (
            f"{symbol} is showing a mixed and unstable structure, with transition conditions appearing across multiple timeframes. "
            f"That suggests the market is losing directional clarity and may be shifting between phases rather than sustaining a clean trend."
        )

    if compression_count >= 2:
        return (
            f"{symbol} is currently in a compression phase across multiple timeframes, "
            f"suggesting the market is coiling before its next larger move. "
            f"However, higher timeframe signals remain mixed, with bullish and bearish forces still competing for control."
        )

    if compression_count >= 1:
        return (
            f"{symbol} is showing a mixed structure with some compression present, "
            f"but broader conditions remain unresolved across timeframes."
        )

    return f"{symbol} is showing a mixed market structure with no strong multi-timeframe consensus right now."

def clamp_confidence(value: float, min_confidence: int = 45, max_confidence: int = 92) -> int:
    return max(min_confidence, min(max_confidence, int(round(value))))



def clamp_score(value: float, min_score: int = 0, max_score: int = 100) -> int:
    return max(min_score, min(max_score, int(round(value))))



def get_directional_bias_score(bias: str, target: str) -> int:
    bias_map = {
        "strong_bullish": 2,
        "bullish": 1,
        "neutral": 0,
        "bearish": -1,
        "strong_bearish": -2,
    }

    raw_score = bias_map.get(bias, 0)
    return raw_score if target == "long" else -raw_score



def append_note(notes: list[str], note: str) -> None:
    if note and note not in notes:
        notes.append(note)



def bias_direction(bias: str) -> str:
    if bias in {"bullish", "strong_bullish"}:
        return "bullish"
    if bias in {"bearish", "strong_bearish"}:
        return "bearish"
    return "neutral"




def infer_market_phase(
    *,
    state: str,
    action: str,
    strategy_bias: str,
    overall_bias: str,
    compression_cluster: int,
    transition_cluster: int,
    reversal_setup_count: int,
    bull_trap_count: int,
    bear_trap_count: int,
) -> str:
    if state == "trap" or action in {"avoid_long_trap", "avoid_short_trap"} or bull_trap_count >= 1 or bear_trap_count >= 1:
        return "trap"

    if state in {"pre_breakout", "pre_breakdown"}:
        return "trigger_watch"

    if state in {"countertrend_long_watch", "countertrend_short_watch"}:
        return "countertrend"

    if state == "exhaustion_rejection":
        return "exhaustion_rejection"

    if state in {"continuation", "continuation_watch", "pullback"} and strategy_bias in {"long", "short"}:
        return "trend"

    if state == "compression":
        return "compression"

    if state == "reversal":
        return "reversal"

    if state == "failed_move" or transition_cluster >= 2 or reversal_setup_count >= 2:
        return "transition"

    if compression_cluster >= 2 and overall_bias == "neutral" and strategy_bias == "neutral":
        return "compression"

    if strategy_bias in {"long", "short"} or overall_bias in {"bullish", "strong_bullish", "bearish", "strong_bearish"}:
        return "trend"

    return "transition"



def infer_trade_quality(
    *,
    action: str,
    state: str,
    strategy_bias: str,
    confidence: int,
    entry_permission: bool,
    reward_to_risk_estimate: float | None = None,
    extension_risk: bool = False,
) -> str:
    if state in {"trap", "failed_move"} or action in {"avoid_long_trap", "avoid_short_trap"}:
        return "D"

    if action in {"enter_long", "enter_short"} and entry_permission:
        rr = reward_to_risk_estimate
        if rr is None:
            return "B" if confidence >= 80 else "C"
        if confidence >= 85 and rr >= 1.8 and not extension_risk:
            return "A"
        if confidence >= 75 and rr >= 1.2:
            return "B"
        if confidence >= 65 and rr >= 0.9:
            return "C"
        return "D"

    if action == "wait_pullback" and strategy_bias in {"long", "short"}:
        if confidence >= 75:
            return "B"
        return "C"

    if state in {"countertrend_long_watch", "countertrend_short_watch"}:
        return "D" if confidence < 55 else "C"

    if state == "exhaustion_rejection":
        return "B" if confidence >= 64 else "C"

    if action in {"watch_breakout", "watch_breakdown", "watch_reversal"}:
        return "C"

    if action == "wait":
        return "C" if confidence >= 55 else "D"

    return "C"



def assess_entry_timing(
    *,
    decision: dict,
    context: dict,
    entry_validation: dict,
    execution_plan: dict,
) -> dict:
    action = decision.get("action")
    strategy_bias = decision.get("strategy_bias")
    mode_config = context.get("mode_config", get_strategy_mode_config(DEFAULT_STRATEGY_MODE))
    timing_profile = mode_config.get("timing_wait_profile", "strict")

    if action == "wait_pullback" and strategy_bias in {"long", "short"}:
        return {
            "timing_state": "wait_retest",
            "should_wait": True,
            "reasons": [
                "Immediate entry is intentionally paused until price retests the continuation area more efficiently."
            ],
            "timing_profile": timing_profile,
        }

    if action in {"watch_breakout", "watch_breakdown", "watch_reversal"}:
        return {
            "timing_state": "monitor",
            "should_wait": False,
            "reasons": [],
            "timing_profile": timing_profile,
        }

    if action in {"avoid_long_trap", "avoid_short_trap", "wait"}:
        return {
            "timing_state": "blocked",
            "should_wait": False,
            "reasons": [],
            "timing_profile": timing_profile,
        }

    if action not in {"enter_long", "enter_short"} or strategy_bias not in {"long", "short"}:
        return {
            "timing_state": "blocked" if not entry_validation.get("allowed") else "monitor",
            "should_wait": False,
            "reasons": [],
            "timing_profile": timing_profile,
        }

    if not entry_validation.get("allowed") or not execution_plan.get("execution_ready"):
        return {
            "timing_state": "blocked",
            "should_wait": False,
            "reasons": [],
            "timing_profile": timing_profile,
        }

    results = context["results"]
    trigger = results.get("5m", {})
    setup = results.get("15m", {})
    micro = results.get("1m", {})

    price = first_valid_number(trigger.get("price"), setup.get("price"))
    entry_zone = execution_plan.get("entry_zone") or {}
    invalidation_zone = execution_plan.get("invalidation_zone") or {}
    take_profit_zone = execution_plan.get("take_profit_zone") or {}

    entry_low = first_valid_number(entry_zone.get("low"))
    entry_high = first_valid_number(entry_zone.get("high"))
    stop_mid = first_valid_number(invalidation_zone.get("mid"))
    tp1 = first_valid_number(take_profit_zone.get("tp1"))

    atr_5m = first_valid_number(trigger.get("indicators", {}).get("atr_14"))
    atr_15m = first_valid_number(setup.get("indicators", {}).get("atr_14"))
    atr_reference = first_valid_number(atr_5m, atr_15m)

    trigger_prev_high = first_valid_number(trigger.get("previous_candle", {}).get("high"))
    trigger_prev_low = first_valid_number(trigger.get("previous_candle", {}).get("low"))
    setup_prev_high = first_valid_number(setup.get("previous_candle", {}).get("high"))
    setup_prev_low = first_valid_number(setup.get("previous_candle", {}).get("low"))

    if price is None:
        return {
            "timing_state": "immediate" if entry_validation.get("allowed") else "blocked",
            "should_wait": False,
            "reasons": [],
            "timing_profile": timing_profile,
        }

    zone_width = max((entry_high - entry_low) if entry_high is not None and entry_low is not None else 0.0, 0.0)
    tolerance_candidates = [price * 0.00018]
    if atr_reference is not None:
        tolerance_candidates.append(atr_reference * 0.22)
    if zone_width > 0:
        tolerance_candidates.append(zone_width * 0.75)
    tolerance = max(tolerance_candidates)

    activation_source = decision.get("activation_source")
    momentum_1m = micro.get("signal_details", {}).get("momentum_state")
    reasons: list[str] = []
    should_wait = False

    if strategy_bias == "short":
        broken_support_candidates = [value for value in [trigger_prev_low, setup_prev_low] if value is not None]
        broken_support = min(broken_support_candidates) if broken_support_candidates else entry_low
        extension_distance = max(0.0, (broken_support - price) if broken_support is not None else 0.0)
        stop_distance = (stop_mid - price) if stop_mid is not None and stop_mid > price else None
        target_distance = (price - tp1) if tp1 is not None and price > tp1 else None
        outside_entry_band = extension_distance > tolerance
        severe_extension = extension_distance > max(tolerance * 1.35, (atr_reference or 0) * 0.55)
        poor_rr_now = (
            stop_distance is not None and target_distance is not None and stop_distance > 0 and (target_distance / stop_distance) < 1.15
        )
        close_to_target = (
            target_distance is not None and (
                (atr_reference is not None and target_distance <= atr_reference * 0.45)
                or (stop_distance is not None and target_distance <= stop_distance * 0.45)
            )
        )
        bounce_risk = momentum_1m == "rising"

        if outside_entry_band:
            reasons.append("Price is trading well below the recent breakdown level, so selling here would chase the move.")
        if poor_rr_now and outside_entry_band:
            reasons.append("Current downside distance to the first target is too thin relative to the stop distance.")
        if close_to_target and outside_entry_band:
            reasons.append("The nearest downside target is close enough that a retest entry is more efficient than a chase entry.")
        if bounce_risk and outside_entry_band:
            reasons.append("The 1m trigger is already bouncing against the move, which increases the odds of a retest before continuation.")

        strict_wait = activation_source == "compression_breakout" or severe_extension or poor_rr_now or close_to_target or bounce_risk
        moderate_wait = severe_extension or poor_rr_now or (activation_source == "compression_breakout" and close_to_target)
        loose_wait = severe_extension or (poor_rr_now and close_to_target)
    else:
        broken_resistance_candidates = [value for value in [trigger_prev_high, setup_prev_high] if value is not None]
        broken_resistance = max(broken_resistance_candidates) if broken_resistance_candidates else entry_high
        extension_distance = max(0.0, (price - broken_resistance) if broken_resistance is not None else 0.0)
        stop_distance = (price - stop_mid) if stop_mid is not None and price > stop_mid else None
        target_distance = (tp1 - price) if tp1 is not None and tp1 > price else None
        outside_entry_band = extension_distance > tolerance
        severe_extension = extension_distance > max(tolerance * 1.35, (atr_reference or 0) * 0.55)
        poor_rr_now = (
            stop_distance is not None and target_distance is not None and stop_distance > 0 and (target_distance / stop_distance) < 1.15
        )
        close_to_target = (
            target_distance is not None and (
                (atr_reference is not None and target_distance <= atr_reference * 0.45)
                or (stop_distance is not None and target_distance <= stop_distance * 0.45)
            )
        )
        bounce_risk = momentum_1m == "falling"

        if outside_entry_band:
            reasons.append("Price is trading well above the recent breakout level, so buying here would chase the move.")
        if poor_rr_now and outside_entry_band:
            reasons.append("Current upside distance to the first target is too thin relative to the stop distance.")
        if close_to_target and outside_entry_band:
            reasons.append("The nearest upside target is close enough that a retest entry is more efficient than a chase entry.")
        if bounce_risk and outside_entry_band:
            reasons.append("The 1m trigger is already pulling back against the move, which increases the odds of a retest before continuation.")

        strict_wait = activation_source == "compression_breakout" or severe_extension or poor_rr_now or close_to_target or bounce_risk
        moderate_wait = severe_extension or poor_rr_now or (activation_source == "compression_breakout" and close_to_target)
        loose_wait = severe_extension or (poor_rr_now and close_to_target)

    if timing_profile == "strict":
        should_wait = outside_entry_band and strict_wait
    elif timing_profile == "moderate":
        should_wait = outside_entry_band and moderate_wait
    else:
        should_wait = outside_entry_band and loose_wait

    timing_state = "wait_retest" if should_wait else "immediate"
    if should_wait and not reasons:
        reasons.append("Immediate entry is stretched away from the preferred continuation area, so wait for a retest first.")

    return {
        "timing_state": timing_state,
        "should_wait": should_wait,
        "reasons": reasons[:3] if should_wait else [],
        "timing_profile": timing_profile,
    }





def apply_timing_wait_to_entry_validation(entry_validation: dict, timing: dict) -> dict:
    updated = {
        "allowed": False,
        "status": "timing_wait",
        "checklist": list(entry_validation.get("checklist", [])),
        "blocking_reasons": list(entry_validation.get("blocking_reasons", [])),
    }

    updated["checklist"].append(
        {
            "condition": "Immediate entry is not overextended beyond the preferred continuation band",
            "passed": False,
        }
    )

    reasons = list(timing.get("reasons", []))
    for reason in updated["blocking_reasons"]:
        if reason not in reasons:
            reasons.append(reason)

    updated["blocking_reasons"] = reasons[:4]
    return updated



def build_actionable_strategy_summary(decision: dict, context: dict) -> str:
    symbol = context["symbol"]
    action = decision["action"]
    state = decision["state"]
    strategy_bias = decision["strategy_bias"]
    higher_timeframe_bias = context["higher_timeframe_bias"]
    failure_risk_count = context["failure_risk_count"]
    compression_activation = decision.get("activation_source") == "compression_breakout"
    compression_direction = decision.get("compression_break_direction")
    timing_state = decision.get("entry_timing")

    if state == "exhaustion_rejection":
        return (
            f"{symbol} is showing an exhaustion rejection short setup. "
            f"Price attempted upside continuation but failed to hold the move, while bull-trap or buyer-exhaustion evidence is active. "
            f"Treat this as a tactical fade: use quick scale-outs and avoid holding blindly if price reclaims the rejection high."
        )

    if action == "avoid_long_trap":
        return (
            f"{symbol} is showing upside pressure, but bull trap risk is active. "
            f"Avoid fresh longs and wait for the move to either reclaim cleanly or fail."
        )

    if action == "avoid_short_trap":
        return (
            f"{symbol} is showing downside pressure, but bear trap risk is active. "
            f"Avoid fresh shorts and wait for the move to either break cleanly again or reverse."
        )

    if state == "failed_move" or (action == "wait" and failure_risk_count >= 1):
        return (
            f"{symbol} is showing unstable follow-through. Stay flat until the latest directional move "
            f"either confirms on a retest or fully fails."
        )

    if state == "countertrend_short_watch":
        return (
            f"{symbol} is still holding a mostly bullish setup structure, but short-term pullback pressure is trying to build. "
            f"Treat this only as a countertrend short watch: require a failed bounce and a clear 5m bearish trigger before acting."
        )

    if state == "countertrend_long_watch":
        return (
            f"{symbol} is still holding a mostly bearish setup structure, but short-term bounce pressure is trying to build. "
            f"Treat this only as a countertrend long watch: require a reclaim and a clear 5m bullish trigger before acting."
        )

    if compression_activation and compression_direction == "down":
        if action == "enter_short":
            return (
                f"{symbol} has broken below its compression trigger band, activating the short continuation model. "
                f"Use 15m as the setup and 5m as the trigger, and favor retests of the broken lower band instead of chasing a flush lower."
            )
        if action == "watch_breakdown":
            return (
                f"{symbol} has started to resolve lower out of compression, but the full short confirmation stack is still incomplete. "
                f"Keep 15m and 5m on watch and activate the short model after a failed bounce or clean retest confirms the break."
            )

    if compression_activation and compression_direction == "up":
        if action == "enter_long":
            return (
                f"{symbol} has broken above its compression trigger band, activating the long continuation model. "
                f"Use 15m as the setup and 5m as the trigger, and favor breakout retests or controlled pullbacks instead of chasing the first impulse."
            )
        if action == "watch_breakout" and strategy_bias == "long":
            return (
                f"{symbol} has started to resolve higher out of compression, but the full long confirmation stack is still incomplete. "
                f"Keep the breakout on watch and activate the long model after a clean retest or pullback hold confirms the break."
            )

    if action == "watch_breakdown" and state == "pre_breakdown":
        return (
            f"{symbol} already has bearish pressure on the 15m setup timeframe, but the 5m trigger has not fully confirmed the breakdown yet. "
            f"Treat this as a lead-short setup: wait for a weak bounce into the entry band or a fresh 5m close lower before activating the short."
        )

    if action == "watch_breakout" and state == "pre_breakout":
        return (
            f"{symbol} already has bullish pressure on the 15m setup timeframe, but the 5m trigger has not fully confirmed the breakout yet. "
            f"Treat this as a lead-long setup: wait for a controlled pullback hold or a fresh 5m push higher before activating the long."
        )

    if action == "wait_pullback" and timing_state == "wait_retest" and strategy_bias == "short":
        if compression_activation and compression_direction == "down":
            return (
                f"{symbol} has resolved lower out of compression, but price is already stretched below the preferred short entry band. "
                f"Wait for a weak bounce back toward the broken band or nearby resistance before reactivating the short."
            )
        return (
            f"{symbol} still leans bearish, but the current move is too extended for a clean chase entry. "
            f"Wait for a weak bounce or breakdown retest before reactivating the short model."
        )

    if action == "wait_pullback" and timing_state == "wait_retest" and strategy_bias == "long":
        if compression_activation and compression_direction == "up":
            return (
                f"{symbol} has resolved higher out of compression, but price is already stretched above the preferred long entry band. "
                f"Wait for a controlled pullback back toward the broken band or nearby support before reactivating the long."
            )
        return (
            f"{symbol} still leans bullish, but the current move is too extended for a clean chase entry. "
            f"Wait for a controlled pullback or breakout retest before reactivating the long model."
        )

    if action == "enter_short":
        if higher_timeframe_bias == "neutral":
            return (
                f"{symbol} is in active short continuation, but the higher timeframe backdrop is only partly aligned. "
                f"Use 15m as the setup and 5m as the trigger, and favor entries on weak bounces or clean breakdown retests."
            )
        return (
            f"{symbol} is in active short continuation. Use 15m as the setup and 5m as the trigger. "
            f"Favor entries on weak bounces or clean breakdown retests, and avoid chasing late extension."
        )

    if action == "watch_breakdown":
        return (
            f"{symbol} is building downside continuation, but confirmation is not complete. "
            f"Keep 15m and 5m on watch, and only activate the short idea after a fresh breakdown retest or failed bounce confirms the move."
        )

    if action == "enter_long":
        if higher_timeframe_bias == "neutral":
            return (
                f"{symbol} is in active long continuation, but the higher timeframe backdrop is only partly aligned. "
                f"Use 15m as the setup and 5m as the trigger, and favor entries on pullback holds or breakout retests."
            )
        return (
            f"{symbol} is in active long continuation. Use 15m as the setup and 5m as the trigger. "
            f"Favor entries on controlled pullbacks or breakout retests, and avoid chasing the first impulse."
        )

    if action == "watch_breakout" and strategy_bias == "long":
        if state == "compression":
            return (
                f"{symbol} is compressing with an early bullish lean. Keep the breakout on watch and only activate the long idea after 5m follow-through confirms an upside break from the band."
            )
        return (
            f"{symbol} is building bullish continuation, but confirmation is incomplete. "
            f"Keep the breakout on watch and prefer retests or controlled pullbacks rather than chasing early strength."
        )

    if action == "watch_breakdown" and state == "compression":
        return (
            f"{symbol} is compressing with an early bearish lean. Keep the downside break on watch and only activate the short idea after 5m follow-through confirms a clean breakdown from the band."
        )

    if action == "watch_breakout" and state == "compression":
        return (
            f"{symbol} is coiling inside compression. Stay patient until price breaks the band with follow-through, "
            f"then activate the directional trade model."
        )

    if action == "watch_reversal" and strategy_bias == "long":
        return (
            f"{symbol} is moving through a developing bullish reversal phase. Wait for the 5m trigger to confirm a reclaim before treating the turn as tradable."
        )

    if action == "watch_reversal" and strategy_bias == "short":
        return (
            f"{symbol} is moving through a developing bearish reversal phase. Wait for the 5m trigger to confirm a rejection before treating the turn as tradable."
        )

    if action == "watch_reversal":
        return (
            f"{symbol} is moving through a developing reversal phase. Wait for the 5m trigger to confirm a reclaim or rejection "
            f"before treating the turn as tradable."
        )

    if action == "wait_pullback" and strategy_bias == "short":
        return (
            f"{symbol} still has bearish structure, but the move is stretched. Favor short entries on weak bounces or failed recoveries "
            f"rather than selling into extension."
        )

    if action == "wait_pullback" and strategy_bias == "long":
        return (
            f"{symbol} still has bullish structure, but the move is extended. Favor long entries on controlled pullbacks or support holds "
            f"rather than chasing strength into resistance."
        )

    return (
        f"{symbol} does not have a clean directional edge right now. Stay flat and wait for the setup and trigger stack to tighten."
    )



def build_strategy_confidence_components(
    *,
    strategy_bias: str,
    action: str,
    state: str | None = None,
    short_term_bias: str,
    higher_timeframe_bias: str,
    overall_bias: str,
    short_breakdown_count: int,
    short_breakout_count: int,
    compression_cluster: int,
    transition_cluster: int,
    reversal_setup_count: int,
    bull_trap_count: int,
    bear_trap_count: int,
    failure_risk_count: int,
    higher_bullish_pressure: bool,
    higher_bearish_pressure: bool,
    results: dict,
) -> dict:
    trend_notes = []
    entry_notes = []
    risk_notes = []

    short_direction = bias_direction(short_term_bias)
    higher_direction = bias_direction(higher_timeframe_bias)
    if short_direction == higher_direction and short_direction != "neutral":
        append_note(trend_notes, f"Short-term and higher timeframes are aligned {short_direction}.")
    elif short_direction == "neutral" and higher_direction == "neutral":
        append_note(trend_notes, "Both short-term and higher timeframe structure are neutral, so conviction stays capped.")
    elif higher_direction == "neutral":
        append_note(trend_notes, "Higher timeframe structure is neutral, so directional conviction stays capped.")
    else:
        append_note(trend_notes, "Short-term and higher timeframe bias are split, which caps directional conviction.")

    signal_5m = results.get("5m", {}).get("signal")
    setup_15m = results.get("15m", {}).get("setup")
    setup_1h = results.get("1h", {}).get("setup")

    momentum_5m = results.get("5m", {}).get("signal_details", {}).get("momentum_state")
    momentum_15m = results.get("15m", {}).get("signal_details", {}).get("momentum_state")
    rsi_state_5m = results.get("5m", {}).get("signal_details", {}).get("rsi_state")
    rsi_state_15m = results.get("15m", {}).get("signal_details", {}).get("rsi_state")
    bb_state_5m = results.get("5m", {}).get("signal_details", {}).get("bollinger_state")
    bb_state_15m = results.get("15m", {}).get("signal_details", {}).get("bollinger_state")

    bullish_entry_setups = BULLISH_BREAKOUT_SETUPS | {"Trend Continuation", "Bullish Pressure Build"}
    bearish_entry_setups = BEARISH_BREAKOUT_SETUPS | {"Trend Continuation"}
    reversal_signals = {
        "oversold_reversal_watch",
        "overbought_pullback_watch",
        "bearish_but_bouncing",
        "bullish_but_pulling_back",
    }

    if state == "exhaustion_rejection":
        trend_quality = 58
        entry_quality = 54
        risk_quality = 60

        if bull_trap_count >= 1:
            trend_quality += min(10, bull_trap_count * 5)
            entry_quality += min(8, bull_trap_count * 4)
            risk_notes.append("Bull-trap context is the primary reason this short fade is active.")
        if setup_15m in EXHAUSTION_REJECTION_SETUPS or setup_1h in EXHAUSTION_REJECTION_SETUPS:
            trend_quality += 6
            entry_quality += 6
            entry_notes.append("A setup layer is showing buyer exhaustion or bull-trap behavior.")
        if signal_5m in BEARISH_SIGNALS or momentum_5m == "falling":
            entry_quality += 8
            entry_notes.append("5m trigger behavior is rolling over in the short direction.")
        if momentum_15m == "falling":
            entry_quality += 4
        if higher_bullish_pressure:
            trend_quality -= 10
            risk_quality -= 12
            risk_notes.append("Higher timeframe bullish pressure keeps this a tactical fade rather than a trend short.")
        if failure_risk_count >= 1:
            risk_quality -= failure_risk_count * 8
            risk_notes.append("Failed-move risk caps short-side execution safety.")

        if not trend_notes:
            trend_notes.append("Exhaustion rejection is active after an upside push failed to hold.")
        if not entry_notes:
            entry_notes.append("Wait for the 5m rejection trigger to confirm before treating the fade as executable.")
        if not risk_notes:
            risk_notes.append("Use quick scale-outs and invalidate the setup if price reclaims the rejection high.")

        return {
            "trend_quality": clamp_score(trend_quality),
            "entry_quality": clamp_score(entry_quality),
            "risk_quality": clamp_score(risk_quality),
            "weights": {
                "trend_quality": 0.40,
                "entry_quality": 0.35,
                "risk_quality": 0.25,
            },
            "notes": {
                "trend": trend_notes[:3],
                "entry": entry_notes[:3],
                "risk": risk_notes[:3],
            },
        }

    if action == "watch_reversal":
        trend_quality = 42 + (transition_cluster * 12) + (reversal_setup_count * 10)
        entry_quality = 40 + (transition_cluster * 10) + (reversal_setup_count * 12)
        risk_quality = 58

        if short_term_bias != higher_timeframe_bias:
            trend_quality += 8
            trend_notes.append("Bias disagreement supports a reversal-watch context.")
        if overall_bias == "neutral":
            trend_quality += 6
        if signal_5m in reversal_signals:
            entry_quality += 6
            entry_notes.append("5m behavior is consistent with reversal monitoring.")
        if setup_1h in REVERSAL_SETUPS:
            entry_quality += 6
            entry_notes.append("Higher timeframe reversal context is active.")
        if compression_cluster >= 2:
            entry_quality -= compression_cluster * 3
        if failure_risk_count >= 1:
            entry_quality -= failure_risk_count * 10
        risk_quality -= compression_cluster * 2
        risk_quality -= bull_trap_count * 6
        risk_quality -= bear_trap_count * 6
        risk_quality -= failure_risk_count * 12
        if transition_cluster >= 2:
            risk_notes.append("Reversal state is defined, but still unstable by nature.")

    elif action in {"avoid_long_trap", "avoid_short_trap"}:
        trend_quality = 55
        entry_quality = 38
        risk_quality = 78

        if action == "avoid_long_trap":
            trend_quality += bull_trap_count * 4
            entry_quality += bull_trap_count * 3
            risk_quality += bull_trap_count * 8
            risk_notes.append("Bull trap detection strongly supports avoiding long entries.")
        else:
            trend_quality += bear_trap_count * 4
            entry_quality += bear_trap_count * 3
            risk_quality += bear_trap_count * 8
            risk_notes.append("Bear trap detection strongly supports avoiding short entries.")
        risk_quality -= failure_risk_count * 8

    elif action == "wait_pullback":
        directional_target = strategy_bias if strategy_bias in {"long", "short"} else "long"
        trend_quality = 50
        trend_quality += get_directional_bias_score(short_term_bias, directional_target) * 6
        trend_quality += get_directional_bias_score(higher_timeframe_bias, directional_target) * 8
        trend_quality += get_directional_bias_score(overall_bias, directional_target) * 8
        if directional_target == "long" and higher_bullish_pressure:
            trend_quality += 6
            trend_notes.append("Broader structure still supports the bullish pullback idea.")
        if directional_target == "short" and higher_bearish_pressure:
            trend_quality += 6
            trend_notes.append("Broader structure still supports the bearish pullback idea.")

        entry_quality = 34
        if compression_cluster >= 1:
            entry_quality += 4
            entry_notes.append("Compression supports waiting for a cleaner pullback re-entry.")
        if transition_cluster >= 1:
            entry_quality -= transition_cluster * 3
        if reversal_setup_count >= 1:
            entry_quality -= reversal_setup_count * 3

        risk_quality = 64
        risk_quality -= bull_trap_count * 8
        risk_quality -= bear_trap_count * 8
        risk_quality -= failure_risk_count * 10
        risk_quality -= transition_cluster * 3
        risk_notes.append("Waiting for pullback improves timing versus chasing extension.")

    elif strategy_bias == "long":
        trend_quality = 52
        trend_quality += get_directional_bias_score(short_term_bias, "long") * 8
        trend_quality += get_directional_bias_score(higher_timeframe_bias, "long") * 10
        trend_quality += get_directional_bias_score(overall_bias, "long") * 8

        if higher_bullish_pressure:
            trend_quality += 6
            trend_notes.append("Higher timeframe context supports long bias.")
        if higher_bearish_pressure:
            trend_quality -= 10
            trend_notes.append("Higher timeframe bearish pressure is capping trend quality.")
        if transition_cluster >= 1:
            trend_quality -= transition_cluster * 4
            trend_notes.append("Transition structure is reducing directional clarity.")
        if reversal_setup_count >= 1:
            trend_quality -= reversal_setup_count * 3
            trend_notes.append("Reversal setups are weakening clean continuation conditions.")
        if compression_cluster >= 1:
            trend_quality -= compression_cluster * 2

        entry_quality = 46
        if short_breakout_count == 2:
            entry_quality += 24
            entry_notes.append("Both short-term trigger layers show breakout confirmation.")
        elif short_breakout_count == 1:
            entry_quality += 12
            entry_notes.append("One short-term trigger layer is already in breakout mode.")
        if setup_15m in bullish_entry_setups:
            entry_quality += 6
            entry_notes.append("15m structure supports bullish continuation.")
        if signal_5m in BULLISH_SIGNALS:
            entry_quality += 6
            entry_notes.append("5m trigger remains aligned bullish.")
        if momentum_5m == "rising":
            entry_quality += 5
        elif momentum_5m == "falling":
            entry_quality -= 8
            entry_notes.append("5m momentum is fading, reducing immediate entry quality.")
        if momentum_15m == "rising":
            entry_quality += 4
        elif momentum_15m == "falling":
            entry_quality -= 6
        entry_quality -= compression_cluster * 5
        entry_quality -= transition_cluster * 6
        entry_quality -= reversal_setup_count * 4
        entry_quality -= bull_trap_count * 10
        entry_quality -= failure_risk_count * 12
        if higher_bearish_pressure:
            entry_quality -= 6
        if action == "watch_breakout":
            trend_quality -= 6
            entry_quality -= 12
            risk_notes.append("Continuation context exists, but immediate entry permission is not open.")

        risk_quality = 72
        if higher_bullish_pressure:
            risk_quality += 6
            risk_notes.append("Broader context improves continuation safety.")
        if higher_bearish_pressure:
            risk_quality -= 10
            risk_notes.append("Higher timeframe headwinds increase long-side risk.")
        if compression_cluster >= 1:
            risk_quality -= compression_cluster * 3
        if transition_cluster >= 1:
            risk_quality -= transition_cluster * 4
            risk_notes.append("Transition conditions keep long setups less stable.")
        if reversal_setup_count >= 1:
            risk_quality -= reversal_setup_count * 4
        if bull_trap_count >= 1:
            risk_quality -= bull_trap_count * 14
            risk_notes.append("Bull trap risk is directly reducing long-side safety.")
        if failure_risk_count >= 1:
            risk_quality -= failure_risk_count * 16
            risk_notes.append("Failed-move risk is materially lowering execution safety.")
        if bear_trap_count >= 1:
            risk_quality += min(6, bear_trap_count * 4)
        if rsi_state_5m in {"overbought", "overbought_falling"} or rsi_state_15m in {"overbought", "overbought_falling"} or bb_state_5m == "above_upper_band" or bb_state_15m == "above_upper_band":
            append_note(risk_notes, "Trend extension increases the chance of a short-term pullback even though structure stays bullish.")

        if action == "watch_breakout":
            risk_quality -= 4
            entry_notes.append("Entry permission is not open yet, so continuation remains watch-only.")

    elif strategy_bias == "short":
        trend_quality = 52
        trend_quality += get_directional_bias_score(short_term_bias, "short") * 8
        trend_quality += get_directional_bias_score(higher_timeframe_bias, "short") * 10
        trend_quality += get_directional_bias_score(overall_bias, "short") * 8

        if higher_bearish_pressure:
            trend_quality += 6
            trend_notes.append("Higher timeframe context supports short bias.")
        if higher_bullish_pressure:
            trend_quality -= 10
            trend_notes.append("Higher timeframe bullish pressure is capping downside continuation quality.")
        if transition_cluster >= 1:
            trend_quality -= transition_cluster * 4
            trend_notes.append("Transition structure is reducing directional clarity.")
        if reversal_setup_count >= 1:
            trend_quality -= reversal_setup_count * 3
            trend_notes.append("Reversal setups are weakening clean downside continuation conditions.")
        if compression_cluster >= 1:
            trend_quality -= compression_cluster * 2

        entry_quality = 46
        if short_breakdown_count == 2:
            entry_quality += 24
            entry_notes.append("Both short-term trigger layers show breakdown confirmation.")
        elif short_breakdown_count == 1:
            entry_quality += 12
            entry_notes.append("One short-term trigger layer is already in breakdown mode.")
        if setup_15m in bearish_entry_setups:
            entry_quality += 6
            entry_notes.append("15m structure supports bearish continuation.")
        if signal_5m in BEARISH_SIGNALS:
            entry_quality += 6
            entry_notes.append("5m trigger remains aligned bearish.")
        if momentum_5m == "falling":
            entry_quality += 5
        elif momentum_5m == "rising":
            entry_quality -= 8
            entry_notes.append("5m momentum is rising against the short idea.")
        if momentum_15m == "falling":
            entry_quality += 4
        elif momentum_15m == "rising":
            entry_quality -= 6
        entry_quality -= compression_cluster * 5
        entry_quality -= transition_cluster * 6
        entry_quality -= reversal_setup_count * 4
        entry_quality -= bear_trap_count * 10
        entry_quality -= failure_risk_count * 12
        if higher_bullish_pressure:
            entry_quality -= 6
        if action == "watch_breakdown":
            trend_quality -= 6
            entry_quality -= 12
            risk_notes.append("Continuation context exists, but immediate short permission is not open.")

        risk_quality = 72
        if higher_bearish_pressure:
            risk_quality += 6
            risk_notes.append("Broader context improves downside continuation safety.")
        if higher_bullish_pressure:
            risk_quality -= 10
            risk_notes.append("Higher timeframe strength increases short-side risk.")
        if compression_cluster >= 1:
            risk_quality -= compression_cluster * 3
        if transition_cluster >= 1:
            risk_quality -= transition_cluster * 4
            risk_notes.append("Transition conditions keep short setups less stable.")
        if reversal_setup_count >= 1:
            risk_quality -= reversal_setup_count * 4
        if bear_trap_count >= 1:
            risk_quality -= bear_trap_count * 14
            risk_notes.append("Bear trap risk is directly reducing short-side safety.")
        if failure_risk_count >= 1:
            risk_quality -= failure_risk_count * 16
            risk_notes.append("Failed-move risk is materially lowering execution safety.")
        if bull_trap_count >= 1:
            risk_quality += min(6, bull_trap_count * 4)
        if rsi_state_5m in {"oversold", "oversold_recovering"} or rsi_state_15m in {"oversold", "oversold_recovering"} or bb_state_5m == "below_lower_band" or bb_state_15m == "below_lower_band":
            append_note(risk_notes, "Trend extension increases the chance of a short-term bounce even though structure stays bearish.")

        if action == "watch_breakdown":
            risk_quality -= 4
            entry_notes.append("Entry permission is not open yet, so downside continuation remains watch-only.")

    elif action == "watch_breakout":
        trend_quality = 46 + (compression_cluster * 10)
        entry_quality = 42 + (compression_cluster * 12)
        risk_quality = 60

        if overall_bias == "neutral":
            trend_quality += 4
        if short_breakout_count >= 1 or short_breakdown_count >= 1:
            entry_quality += 6
            entry_notes.append("Early trigger activity is forming inside the compression zone.")
        if transition_cluster >= 1:
            trend_quality -= transition_cluster * 4
            entry_quality -= transition_cluster * 3
        risk_quality -= bull_trap_count * 8
        risk_quality -= bear_trap_count * 8
        risk_quality -= failure_risk_count * 12
        if compression_cluster >= 2:
            trend_notes.append("Multi-timeframe compression supports a breakout watch state.")

    else:
        trend_quality = 42
        entry_quality = 26
        risk_quality = 62
        if overall_bias == "neutral":
            trend_quality += 4
        if failure_risk_count >= 1:
            risk_quality += 6
            risk_notes.append("Waiting is supported by failed-move risk in the market.")
        risk_quality -= bull_trap_count * 4
        risk_quality -= bear_trap_count * 4

    if not trend_notes:
        if short_term_bias in {"bullish", "strong_bullish"} and higher_timeframe_bias in {"bullish", "strong_bullish"}:
            append_note(trend_notes, "Short-term and higher timeframes remain aligned bullish.")
        elif short_term_bias in {"bearish", "strong_bearish"} and higher_timeframe_bias in {"bearish", "strong_bearish"}:
            append_note(trend_notes, "Short-term and higher timeframes remain aligned bearish.")
        elif short_term_bias != higher_timeframe_bias:
            append_note(trend_notes, "Short-term and higher timeframe bias remain misaligned.")
        elif overall_bias == "neutral":
            append_note(trend_notes, "No durable directional edge is present across the multi-timeframe stack.")
        elif strategy_bias in {"long", "short"}:
            append_note(
                trend_notes,
                f"{strategy_bias.capitalize()} bias is present, but it is not yet strong enough across all layers.",
            )
        else:
            append_note(trend_notes, "Directional structure remains mixed rather than decisively aligned.")

    if not entry_notes:
        if strategy_bias == "long" and short_breakout_count == 2:
            append_note(entry_notes, "5m and 15m are both confirming bullish continuation.")
        elif strategy_bias == "short" and short_breakdown_count == 2:
            append_note(entry_notes, "5m and 15m are both confirming bearish continuation.")
        elif strategy_bias == "long":
            append_note(entry_notes, "The long trigger stack still needs cleaner 5m and 15m confirmation.")
        elif strategy_bias == "short":
            append_note(entry_notes, "The short trigger stack still needs cleaner 5m and 15m confirmation.")
        elif action == "wait_pullback":
            append_note(entry_notes, "The directional idea is intact, but the engine prefers a cleaner pullback entry.")
        elif action == "watch_reversal":
            append_note(entry_notes, "Reversal structure is forming, but it still requires trigger confirmation.")
        else:
            append_note(entry_notes, "The active trigger stack does not yet justify immediate execution.")

    if not risk_notes:
        if bull_trap_count >= 1 or bear_trap_count >= 1:
            append_note(risk_notes, "Trap risk is elevated enough to keep execution quality capped.")
        elif failure_risk_count >= 1:
            append_note(risk_notes, "Follow-through remains unstable, which lowers execution safety.")
        elif strategy_bias == "long" and (
            rsi_state_5m in {"overbought", "overbought_falling"}
            or rsi_state_15m in {"overbought", "overbought_falling"}
            or bb_state_5m == "above_upper_band"
            or bb_state_15m == "above_upper_band"
        ):
            append_note(risk_notes, "Trend extension increases the chance of a short-term pullback against the long idea.")
        elif strategy_bias == "short" and (
            rsi_state_5m in {"oversold", "oversold_recovering"}
            or rsi_state_15m in {"oversold", "oversold_recovering"}
            or bb_state_5m == "below_lower_band"
            or bb_state_15m == "below_lower_band"
        ):
            append_note(risk_notes, "Trend extension increases the chance of a short-term bounce against the short idea.")
        elif transition_cluster >= 1 or reversal_setup_count >= 1:
            append_note(risk_notes, "Transition and reversal pressure are keeping risk conditions mixed.")
        elif higher_timeframe_bias == "neutral":
            append_note(risk_notes, "Mixed higher timeframe structure argues for patience until confirmation improves.")
        else:
            append_note(risk_notes, "Risk conditions are acceptable, but not yet ideal for aggressive execution.")

    return {
        "trend_quality": clamp_score(trend_quality),
        "entry_quality": clamp_score(entry_quality),
        "risk_quality": clamp_score(risk_quality),
        "weights": {
            "trend_quality": 0.40,
            "entry_quality": 0.35,
            "risk_quality": 0.25,
        },
        "notes": {
            "trend": trend_notes[:3],
            "entry": entry_notes[:3],
            "risk": risk_notes[:3],
        },
    }





def evaluate_entry_permission(
    *,
    action: str,
    strategy_bias: str,
    results: dict,
    short_breakdown_count: int,
    short_breakout_count: int,
    higher_bullish_pressure: bool,
    higher_bearish_pressure: bool,
    bull_trap_count: int,
    bear_trap_count: int,
    failure_risk_count: int,
    transition_cluster: int,
    activation_source: str | None = None,
    compression_breakout_confirmed: bool = False,
    mode_config: dict | None = None,
) -> dict:
    mode_config = mode_config or get_strategy_mode_config(DEFAULT_STRATEGY_MODE)
    mode_name = mode_config["name"]
    soft_fail_limit = int(mode_config.get("soft_fail_limit", 0))
    transition_limit = int(mode_config.get("transition_limit", 1))
    trap_entry_severity = mode_config.get("trap_entry_severity", "hard")
    higher_timeframe_opposition_is_hard = bool(mode_config.get("higher_timeframe_opposition_is_hard", True))
    failure_risk_hard_count = int(mode_config.get("failure_risk_hard_count", 1))

    if strategy_bias not in {"long", "short"}:
        return {
            "allowed": False,
            "status": "standby",
            "checklist": [],
            "blocking_reasons": ["Immediate entry is disabled while the strategy bias remains neutral."],
            "hard_blocking_reasons": ["Immediate entry is disabled while the strategy bias remains neutral."],
            "soft_blocking_reasons": [],
            "hard_fail_count": 1,
            "soft_fail_count": 0,
            "soft_fail_limit": soft_fail_limit,
            "mode": mode_name,
            "advisories": [],
            "qualification": {
                "setup_support_directional": False,
                "setup_support_continuation": False,
                "trigger_indicator_confirmed": False,
            },
        }

    tf_5m = results.get("5m", {})
    tf_15m = results.get("15m", {})
    signal_5m = tf_5m.get("signal")
    signal_15m = tf_15m.get("signal")
    setup_15m = tf_15m.get("setup")
    structure_15m = tf_15m.get("structure")
    details_15m = tf_15m.get("signal_details", {})
    momentum_5m = tf_5m.get("signal_details", {}).get("momentum_state")
    momentum_15m = details_15m.get("momentum_state")

    setup_directional_support = directional_setup_supports(
        direction=strategy_bias,
        setup_signal=signal_15m,
        setup_name=setup_15m,
        structure=structure_15m,
        trend_bias=details_15m.get("trend_bias"),
        momentum_state=momentum_15m,
        rsi_state=details_15m.get("rsi_state"),
    )
    setup_continuation_support = continuation_setup_qualified(
        direction=strategy_bias,
        setup_signal=signal_15m,
        setup_name=setup_15m,
        structure=structure_15m,
        trend_bias=details_15m.get("trend_bias"),
        momentum_state=momentum_15m,
    )
    trigger_indicator_confirmed = trigger_indicator_confirms_direction(
        direction=strategy_bias,
        trigger_signal=signal_5m,
        activation_source=activation_source,
        compression_breakout_confirmed=compression_breakout_confirmed,
        compression_break_direction="up" if strategy_bias == "long" else "down",
    )

    compression_activation = activation_source == "compression_breakout" and compression_breakout_confirmed

    def item(condition: str, passed: bool, severity: str, failure_message: str | None = None) -> dict:
        return {
            "condition": condition,
            "passed": passed,
            "severity": severity,
            "failure_message": failure_message or condition,
        }

    if strategy_bias == "long":
        setup_condition = "15m setup is bullish continuation-qualified"
        trigger_condition = "5m trigger is bullish"
        confirmation_condition = "At least one short-term breakout trigger is confirmed"
        if compression_activation:
            setup_condition += " or the compression band has broken higher"
            trigger_condition += " or the compression band has broken higher"
            confirmation_condition += " or the compression break is confirmed"

        failure_severity = "hard" if failure_risk_count >= failure_risk_hard_count else "soft"
        checklist = [
            item(
                setup_condition,
                setup_continuation_support or compression_activation,
                "hard",
                "15m supports the bullish direction, but it is not yet continuation-qualified.",
            ),
            item(
                trigger_condition,
                trigger_indicator_confirmed or compression_activation,
                "hard",
                "5m has not confirmed a bullish indicator trigger yet.",
            ),
            item("5m momentum is not falling", momentum_5m in {"rising", "flat"}, "soft", "5m momentum is still fading against the long idea."),
            item("15m momentum is not falling", momentum_15m in {"rising", "flat"}, "soft", "15m momentum is still fading against the long idea."),
            item(
                "Higher timeframe bearish pressure is not dominant",
                not higher_bearish_pressure,
                "hard" if higher_timeframe_opposition_is_hard else "soft",
                "Higher timeframe bearish pressure is still too strong against the long.",
            ),
            item("No bull trap risk is active", bull_trap_count == 0, trap_entry_severity, "Bull trap risk is still active against the long."),
            item("No failed-move risk is active", failure_risk_count == 0, failure_severity, "Failed-move risk is still active, so the long should not be promoted yet."),
            item("Transition noise is limited", transition_cluster <= transition_limit, "soft", "Transition noise is still too high for a clean long entry."),
            item(
                confirmation_condition,
                short_breakout_count >= 1 or compression_activation,
                "hard",
                "The short-term breakout trigger stack is not confirmed yet.",
            ),
        ]
    else:
        setup_condition = "15m setup is bearish continuation-qualified"
        trigger_condition = "5m trigger is bearish"
        confirmation_condition = "At least one short-term breakdown trigger is confirmed"
        if compression_activation:
            setup_condition += " or the compression band has broken lower"
            trigger_condition += " or the compression band has broken lower"
            confirmation_condition += " or the compression break is confirmed"

        failure_severity = "hard" if failure_risk_count >= failure_risk_hard_count else "soft"
        checklist = [
            item(
                setup_condition,
                setup_continuation_support or compression_activation,
                "hard",
                "15m supports the bearish direction, but it is not yet continuation-qualified.",
            ),
            item(
                trigger_condition,
                trigger_indicator_confirmed or compression_activation,
                "hard",
                "5m has not confirmed a bearish indicator trigger yet.",
            ),
            item("5m momentum is not rising", momentum_5m in {"falling", "flat"}, "soft", "5m momentum is still rising against the short idea."),
            item("15m momentum is not rising", momentum_15m in {"falling", "flat"}, "soft", "15m momentum is still rising against the short idea."),
            item(
                "Higher timeframe bullish pressure is not dominant",
                not higher_bullish_pressure,
                "hard" if higher_timeframe_opposition_is_hard else "soft",
                "Higher timeframe bullish pressure is still too strong against the short.",
            ),
            item("No bear trap risk is active", bear_trap_count == 0, trap_entry_severity, "Bear trap risk is still active against the short."),
            item("No failed-move risk is active", failure_risk_count == 0, failure_severity, "Failed-move risk is still active, so the short should not be promoted yet."),
            item("Transition noise is limited", transition_cluster <= transition_limit, "soft", "Transition noise is still too high for a clean short entry."),
            item(
                confirmation_condition,
                short_breakdown_count >= 1 or compression_activation,
                "hard",
                "The short-term breakdown trigger stack is not confirmed yet.",
            ),
        ]

    hard_blocking_reasons = [item["failure_message"] for item in checklist if not item["passed"] and item["severity"] == "hard"]
    soft_blocking_reasons = [item["failure_message"] for item in checklist if not item["passed"] and item["severity"] == "soft"]

    allowed = (
        action in {"enter_long", "enter_short"}
        and not hard_blocking_reasons
        and len(soft_blocking_reasons) <= soft_fail_limit
    )

    watch_only = action in {"watch_breakout", "watch_breakdown", "watch_reversal"}
    if watch_only:
        if not setup_directional_support:
            status = "watch_blocked_no_directional_support"
        elif setup_directional_support and not setup_continuation_support:
            status = "watch_blocked_setup_unqualified"
        elif not trigger_indicator_confirmed and not compression_activation:
            status = "watch_blocked_waiting_trigger"
        else:
            status = "watch_ready" if not hard_blocking_reasons and len(soft_blocking_reasons) <= soft_fail_limit else "watch_blocked"
    else:
        status = "allowed" if allowed else "blocked"

    blocking_reasons = hard_blocking_reasons + soft_blocking_reasons
    advisories = soft_blocking_reasons[:soft_fail_limit] if allowed else []

    return {
        "allowed": allowed,
        "status": status,
        "checklist": checklist,
        "blocking_reasons": blocking_reasons,
        "hard_blocking_reasons": hard_blocking_reasons,
        "soft_blocking_reasons": soft_blocking_reasons,
        "hard_fail_count": len(hard_blocking_reasons),
        "soft_fail_count": len(soft_blocking_reasons),
        "soft_fail_limit": soft_fail_limit,
        "mode": mode_name,
        "advisories": advisories,
        "qualification": {
            "setup_support_directional": setup_directional_support,
            "setup_support_continuation": setup_continuation_support,
            "trigger_indicator_confirmed": trigger_indicator_confirmed,
        },
    }
def apply_blocked_component_caps(
    components: dict,
    entry_validation: dict,
    decision: dict,
    mode_config: dict | None = None,
) -> dict:
    mode_config = mode_config or get_strategy_mode_config(DEFAULT_STRATEGY_MODE)
    notes = components.get("notes", {}) or {}
    capped = {
        **components,
        "notes": {
            "trend": list(notes.get("trend", []) or []),
            "entry": list(notes.get("entry", []) or []),
            "risk": list(notes.get("risk", []) or []),
        },
    }

    if entry_validation.get("allowed"):
        return capped

    hard_fail_count = int(entry_validation.get("hard_fail_count", 0) or 0)
    soft_fail_count = int(entry_validation.get("soft_fail_count", 0) or 0)
    action = decision.get("action")
    mode_name = mode_config.get("name", DEFAULT_STRATEGY_MODE)

    if action == "watch_reversal" and hard_fail_count >= 2:
        capped["trend_quality"] = min(int(capped.get("trend_quality", 0) or 0), 72 if mode_name == "balanced" else 80)
        capped["entry_quality"] = min(int(capped.get("entry_quality", 0) or 0), 42 if mode_name == "balanced" else 50)
    elif action in {"watch_breakout", "watch_breakdown"} and hard_fail_count >= 2:
        capped["trend_quality"] = min(int(capped.get("trend_quality", 0) or 0), 74 if mode_name == "balanced" else 82)
        capped["entry_quality"] = min(int(capped.get("entry_quality", 0) or 0), 44 if mode_name == "balanced" else 52)
    elif action == "wait":
        capped["trend_quality"] = min(int(capped.get("trend_quality", 0) or 0), 48)
        capped["entry_quality"] = min(int(capped.get("entry_quality", 0) or 0), 24)
    elif hard_fail_count >= 1:
        capped["entry_quality"] = min(int(capped.get("entry_quality", 0) or 0), 50)

    if soft_fail_count >= 2:
        capped["risk_quality"] = min(int(capped.get("risk_quality", 0) or 0), 60)

    cap_note = "Blocked entry is capping execution confidence until the trigger stack improves."
    if cap_note not in capped["notes"]["entry"]:
        capped["notes"]["entry"].append(cap_note)

    capped["notes"]["trend"] = capped["notes"]["trend"][:3]
    capped["notes"]["entry"] = capped["notes"]["entry"][:3]
    capped["notes"]["risk"] = capped["notes"]["risk"][:3]
    return capped


def apply_blocked_confidence_cap(
    confidence: int,
    entry_validation: dict,
    decision: dict,
    mode_config: dict | None = None,
) -> int:
    mode_config = mode_config or get_strategy_mode_config(DEFAULT_STRATEGY_MODE)
    if entry_validation.get("allowed"):
        return confidence

    mode_name = mode_config.get("name", DEFAULT_STRATEGY_MODE)
    hard_fail_count = int(entry_validation.get("hard_fail_count", 0) or 0)
    soft_fail_count = int(entry_validation.get("soft_fail_count", 0) or 0)
    action = decision.get("action")
    strategy_bias = decision.get("strategy_bias")

    if action == "watch_reversal" and hard_fail_count >= 2:
        cap = 62 if mode_name == "balanced" else 70
    elif action in {"watch_breakout", "watch_breakdown"} and hard_fail_count >= 2:
        cap = 64 if mode_name == "balanced" else 72
    elif action == "wait_pullback":
        cap = 66 if mode_name == "balanced" else 72
    elif action == "wait":
        cap = 55 if strategy_bias == "neutral" else 60
    else:
        cap = 68 if mode_name == "balanced" else 74

    if hard_fail_count > 2:
        cap -= (hard_fail_count - 2) * 2
    if soft_fail_count >= 3:
        cap -= 2

    return min(confidence, clamp_confidence(cap))


def combine_strategy_confidence(
    *,
    components: dict,
    action: str,
    strategy_bias: str,
    entry_permission: bool,
    short_breakdown_count: int,
    short_breakout_count: int,
    compression_cluster: int,
    transition_cluster: int,
    reversal_setup_count: int,
    bull_trap_count: int,
    bear_trap_count: int,
    failure_risk_count: int,
    mode_config: dict | None = None,
) -> int:
    mode_config = mode_config or get_strategy_mode_config(DEFAULT_STRATEGY_MODE)
    weighted_score = (
        (components["trend_quality"] * components["weights"]["trend_quality"])
        + (components["entry_quality"] * components["weights"]["entry_quality"])
        + (components["risk_quality"] * components["weights"]["risk_quality"])
    )

    modifier = 0.0

    if action in {"enter_long", "enter_short"}:
        modifier += 6 if entry_permission else -10
        modifier += float(mode_config.get("enter_confidence_bonus", 0))
    elif action in {"watch_breakout", "watch_breakdown"}:
        if compression_cluster >= 2 or short_breakout_count >= 1 or short_breakdown_count >= 1:
            modifier += 4
        else:
            modifier += 1
        modifier += float(mode_config.get("watch_confidence_bonus", 0))
    elif action == "watch_reversal":
        modifier += 5 if transition_cluster >= 2 or reversal_setup_count >= 2 else 2
        modifier += float(mode_config.get("watch_confidence_bonus", 0)) * 0.5
    elif action == "wait_pullback":
        modifier += 4 if strategy_bias in {"long", "short"} else 1
        modifier += float(mode_config.get("watch_confidence_bonus", 0)) * 0.5
    elif action in {"avoid_long_trap", "avoid_short_trap"}:
        modifier += 6
    elif action == "wait":
        if failure_risk_count >= 1 or bull_trap_count >= 1 or bear_trap_count >= 1:
            modifier += 4

    final_score = weighted_score + modifier

    if action in {"watch_breakout", "watch_breakdown"}:
        final_score = min(final_score, 84 if mode_config["name"] != "conservative" else 82)
    elif action == "watch_reversal":
        final_score = min(final_score, 80 if mode_config["name"] != "conservative" else 79)
    elif action == "wait_pullback":
        final_score = min(final_score, 78 if mode_config["name"] != "conservative" else 76)
    elif action == "wait":
        final_score = min(final_score, 72)
    elif action in {"avoid_long_trap", "avoid_short_trap"}:
        final_score = min(final_score, 86)

    return clamp_confidence(final_score)





def first_valid_number(*values):
    for value in values:
        if value is None:
            continue
        try:
            if pd.isna(value):
                continue
        except TypeError:
            pass
        try:
            return float(value)
        except (TypeError, ValueError):
            continue
    return None




def detect_exhaustion_rejection_candidate(context: dict) -> dict | None:
    """Detect the tactical mirror-image of a capitulation bounce.

    This is intentionally not a generic short. It looks for an upside push that
    is failing: bull-trap or buyer-exhaustion evidence, 1m/5m rollover, and a
    failed hold above the recent high or fast trend level.

    Version 1 is watch-only so it can be outcome-tested cleanly before allowing
    automatic enter_short promotion.
    """
    results = context.get("results", {})

    tf_1m = results.get("1m", {}) or {}
    tf_5m = results.get("5m", {}) or {}
    tf_15m = results.get("15m", {}) or {}
    tf_1h = results.get("1h", {}) or {}

    signal_1m = tf_1m.get("signal")
    signal_5m = tf_5m.get("signal")
    signal_15m = tf_15m.get("signal")

    setup_5m = tf_5m.get("setup")
    setup_15m = tf_15m.get("setup")
    setup_1h = tf_1h.get("setup")

    details_1m = tf_1m.get("signal_details", {}) or {}
    details_5m = tf_5m.get("signal_details", {}) or {}
    details_15m = tf_15m.get("signal_details", {}) or {}

    momentum_1m = details_1m.get("momentum_state")
    momentum_5m = details_5m.get("momentum_state")
    momentum_15m = details_15m.get("momentum_state")
    trend_5m = details_5m.get("trend_bias")
    rsi_5m = details_5m.get("rsi_state")
    bb_5m = details_5m.get("bollinger_state")

    price = first_valid_number(tf_5m.get("price"), tf_15m.get("price"))
    ema9_5m = first_valid_number(tf_5m.get("indicators", {}).get("ema_9"))

    current_open = first_valid_number(tf_5m.get("candle", {}).get("open"))
    current_close = first_valid_number(tf_5m.get("candle", {}).get("close"), price)
    current_high = first_valid_number(tf_5m.get("candle", {}).get("high"))
    prev_high = first_valid_number(tf_5m.get("previous_candle", {}).get("high"))

    active_bull_trap = (
        context.get("bull_trap_count", 0) >= 1
        or setup_5m == "Bull Trap Risk"
        or setup_15m == "Bull Trap Risk"
        or setup_1h == "Bull Trap Risk"
        or tf_5m.get("trap_risk") == "bull_trap_risk"
        or tf_15m.get("trap_risk") == "bull_trap_risk"
        or tf_1h.get("trap_risk") == "bull_trap_risk"
    )

    exhaustion_setup = any(
        setup in EXHAUSTION_REJECTION_SETUPS
        for setup in [setup_5m, setup_15m, setup_1h]
    )

    upper_band_exhaustion = (
        trend_5m in {"bullish", "mixed"}
        and momentum_5m == "falling"
        and rsi_5m in {"overbought", "overbought_falling", "bullish"}
        and bb_5m in {"near_upper_band", "above_upper_band"}
    )

    failed_high = (
        current_high is not None
        and prev_high is not None
        and current_close is not None
        and current_high >= prev_high
        and current_close < prev_high
    )

    bearish_candle = (
        current_open is not None
        and current_close is not None
        and current_close < current_open
    )

    lost_fast_trend = (
        price is not None
        and ema9_5m is not None
        and price < ema9_5m
    )

    bearish_trigger = (
        signal_5m in BEARISH_SIGNALS
        or signal_1m in BEARISH_SIGNALS
        or momentum_1m == "falling"
    )

    score = 0.0
    reasons: list[str] = []

    def add(points: float, reason: str) -> None:
        nonlocal score
        score += points
        if reason not in reasons:
            reasons.append(reason)

    def subtract(points: float, reason: str) -> None:
        nonlocal score
        score -= points
        if reason not in reasons:
            reasons.append(reason)

    if active_bull_trap:
        add(3.0, "Bull-trap risk is active on the trading timeframes.")
    if exhaustion_setup:
        add(2.0, "A buyer-exhaustion setup is active on 5m, 15m, or 1h.")
    elif upper_band_exhaustion:
        add(2.0, "Buyer exhaustion is showing near the upper Bollinger band.")
    if momentum_5m == "falling":
        add(2.0, "5m momentum is rolling over.")
    if momentum_15m == "falling":
        add(1.0, "15m momentum is also fading.")
    if failed_high:
        add(2.0, "Price pushed above a recent high but failed to hold it.")
    if bearish_candle:
        add(1.0, "The active 5m candle is rejecting lower.")
    if lost_fast_trend:
        add(1.0, "Price has lost the 5m EMA9.")
    if bearish_trigger:
        add(2.0, "The short-term trigger layer is confirming bearish pressure.")

    if context.get("higher_bullish_pressure"):
        subtract(2.0, "Higher-timeframe bullish pressure is still a guardrail against oversized shorts.")
    if signal_15m in BULLISH_SIGNALS and not failed_high:
        subtract(2.0, "15m remains bullish without a confirmed failed-high rejection.")

    has_rejection_context = active_bull_trap or exhaustion_setup or failed_high or upper_band_exhaustion
    has_rollover = momentum_5m == "falling" or bearish_trigger or lost_fast_trend or bearish_candle
    if not has_rejection_context or not has_rollover or score < 6.0:
        return None

    setup_timeframes = [
        tf
        for tf in ["5m", "15m", "1h"]
        if (results.get(tf, {}) or {}).get("setup") in EXHAUSTION_REJECTION_SETUPS
    ] or ["5m", "15m"]

    risk_state = "mixed" if context.get("higher_bullish_pressure") else "favorable"
    confidence = max(55, min(78, int(round(50 + score * 3))))

    return {
        "strategy_bias": "short",
        "state": "exhaustion_rejection",
        "action": "watch_reversal",
        "setup_timeframes": setup_timeframes,
        "trigger_timeframes": ["5m"],
        "risk_timeframes": ["5m", "15m"],
        "risk_state": risk_state,
        "summary": (
            f'{context["symbol"]} is showing a potential exhaustion rejection: price pushed into upside pressure, '
            f'but bull-trap or buyer-exhaustion evidence is active and short-term momentum is rolling over. '
            f'Treat this as a tactical fade setup, not a long-duration trend short.'
        ),
        "bias_origin": "exhaustion_rejection",
        "bias_score": round(score, 2),
        "bias_reasons": reasons[:4],
        "bias_confidence": confidence,
        "bias_edge": round(score / 10.0, 2),
        "exhaustion_rejection": {
            "score": round(score, 2),
            "threshold": 6.0,
            "active_bull_trap": bool(active_bull_trap),
            "exhaustion_setup": bool(exhaustion_setup),
            "upper_band_exhaustion": bool(upper_band_exhaustion),
            "failed_high": bool(failed_high),
            "bearish_trigger": bool(bearish_trigger),
            "lost_fast_trend": bool(lost_fast_trend),
            "watch_only": True,
            "preferred_holding_window": "15m_to_60m",
        },
    }


def unique_price_levels(values: list[float], reverse: bool = False) -> list[float]:
    cleaned: list[float] = []
    for value in values:
        if value is None:
            continue
        try:
            if pd.isna(value):
                continue
        except TypeError:
            pass
        try:
            cleaned.append(round(float(value), 6))
        except (TypeError, ValueError):
            continue

    cleaned = sorted(set(cleaned), reverse=reverse)

    filtered: list[float] = []
    for value in cleaned:
        if not filtered or abs(value - filtered[-1]) > 1e-6:
            filtered.append(value)
    return filtered



def make_price_zone(
    low: float | None,
    high: float | None,
    *,
    label: str,
    description: str,
    reference_timeframes: list[str],
) -> dict | None:
    if low is None or high is None:
        return None

    zone_low = float(min(low, high))
    zone_high = float(max(low, high))

    return {
        "low": round_or_none(zone_low),
        "high": round_or_none(zone_high),
        "mid": round_or_none((zone_low + zone_high) / 2),
        "label": label,
        "description": description,
        "reference_timeframes": reference_timeframes,
    }



def build_take_profit_zone(
    *,
    direction: str,
    entry_mid: float,
    invalidation_zone: dict | None,
    favorable_levels: list[float],
    buffer: float,
    reference_timeframes: list[str],
) -> tuple[dict | None, float | None]:
    if invalidation_zone is None:
        return None, None

    if direction == "long":
        risk_distance = entry_mid - float(invalidation_zone["high"])
        if risk_distance <= 0:
            risk_distance = max(buffer, entry_mid * 0.002)

        projected_levels = [
            entry_mid + risk_distance * 1.0,
            entry_mid + risk_distance * 1.5,
            entry_mid + risk_distance * 2.0,
            entry_mid + risk_distance * 2.5,
        ]
        candidates = unique_price_levels([level for level in favorable_levels if level > entry_mid] + projected_levels)
        min_gap = max(buffer * 0.35, risk_distance * 0.50)

        targets: list[float] = []
        for target in candidates:
            if target <= entry_mid:
                continue
            if not targets or abs(target - targets[-1]) >= min_gap:
                targets.append(target)
            if len(targets) == 3:
                break

        while len(targets) < 3:
            next_target = entry_mid + risk_distance * (1.0 + (0.5 * len(targets)))
            if targets and abs(next_target - targets[-1]) < min_gap:
                next_target = targets[-1] + min_gap
            targets.append(round(next_target, 6))

        reward_to_risk = round((targets[1] - entry_mid) / risk_distance, 2) if risk_distance > 0 else None
        description = "Profit ladder based on nearby resistance and projected risk multiples above the entry zone."
    else:
        risk_distance = float(invalidation_zone["low"]) - entry_mid
        if risk_distance <= 0:
            risk_distance = max(buffer, entry_mid * 0.002)

        projected_levels = [
            entry_mid - risk_distance * 1.0,
            entry_mid - risk_distance * 1.5,
            entry_mid - risk_distance * 2.0,
            entry_mid - risk_distance * 2.5,
        ]
        candidates = unique_price_levels([level for level in favorable_levels if level < entry_mid] + projected_levels, reverse=True)
        min_gap = max(buffer * 0.35, risk_distance * 0.50)

        targets: list[float] = []
        for target in candidates:
            if target >= entry_mid:
                continue
            if not targets or abs(target - targets[-1]) >= min_gap:
                targets.append(target)
            if len(targets) == 3:
                break

        while len(targets) < 3:
            next_target = entry_mid - risk_distance * (1.0 + (0.5 * len(targets)))
            if targets and abs(next_target - targets[-1]) < min_gap:
                next_target = targets[-1] - min_gap
            targets.append(round(next_target, 6))

        reward_to_risk = round((entry_mid - targets[1]) / risk_distance, 2) if risk_distance > 0 else None
        description = "Profit ladder based on nearby support and projected risk multiples below the entry zone."

    return (
        {
            "tp1": round_or_none(targets[0]),
            "tp2": round_or_none(targets[1]),
            "tp3": round_or_none(targets[2]),
            "description": description,
            "reference_timeframes": reference_timeframes,
        },
        reward_to_risk,
    )





def build_execution_management(
    *,
    decision: dict,
    execution_plan: dict,
    context: dict,
) -> dict:
    mode_config = context.get("mode_config", get_strategy_mode_config(DEFAULT_STRATEGY_MODE))
    direction = execution_plan.get("direction")
    entry_zone = execution_plan.get("entry_zone") or {}
    invalidation_zone = execution_plan.get("invalidation_zone") or {}
    take_profit_zone = execution_plan.get("take_profit_zone") or {}

    scale_out_raw = list(mode_config.get("execution_scale_out", [60, 25, 15]))
    while len(scale_out_raw) < 3:
        scale_out_raw.append(0)

    if direction == "long":
        stop_anchor = first_valid_number(invalidation_zone.get("high"), invalidation_zone.get("mid"), invalidation_zone.get("low"))
    elif direction == "short":
        stop_anchor = first_valid_number(invalidation_zone.get("low"), invalidation_zone.get("mid"), invalidation_zone.get("high"))
    else:
        stop_anchor = None

    entry_mid = first_valid_number(entry_zone.get("mid"))
    stop_distance_pct = None
    if direction == "long" and entry_mid not in (None, 0.0) and stop_anchor is not None:
        stop_distance_pct = round(((entry_mid - stop_anchor) / entry_mid) * 100, 4)
    elif direction == "short" and entry_mid not in (None, 0.0) and stop_anchor is not None:
        stop_distance_pct = round(((stop_anchor - entry_mid) / entry_mid) * 100, 4)

    trailing_reference = mode_config.get("execution_trail_reference", "5m_ema9")
    break_even_after = mode_config.get("execution_break_even_after", "tp1_touch")
    trail_after = mode_config.get("execution_trail_after", "tp2_touch")

    scale_out_plan = [
        {
            "target": "tp1",
            "size_pct": int(scale_out_raw[0]),
            "price": first_valid_number(take_profit_zone.get("tp1")),
            "purpose": "Pay yourself quickly and de-risk the trade.",
        },
        {
            "target": "tp2",
            "size_pct": int(scale_out_raw[1]),
            "price": first_valid_number(take_profit_zone.get("tp2")),
            "purpose": "Keep size on for trend continuation after the first scale-out.",
        },
        {
            "target": "tp3",
            "size_pct": int(scale_out_raw[2]),
            "price": first_valid_number(take_profit_zone.get("tp3")),
            "purpose": "Leave a small runner for expansion when the move extends.",
        },
    ]

    management_notes: list[str] = []
    if direction in {"long", "short"}:
        management_notes.append("Balanced mode now uses tighter structure-based risk instead of a wide passive invalidation box.")
        management_notes.append("Scale out into strength: pay yourself at TP1, reduce more at TP2, and let a small runner test TP3.")
        management_notes.append("After TP1, move the stop to breakeven. After TP2, trail using the 5m structure instead of giving the whole move back.")
    else:
        management_notes.append("No active execution management plan is open while the engine is in standby.")

    return {
        "stop_policy": "tight_structure_stop" if direction in {"long", "short"} else "standby",
        "stop_distance_pct": stop_distance_pct,
        "break_even_after": break_even_after,
        "trailing_stop_after": trail_after,
        "trailing_stop_reference": trailing_reference,
        "runner_policy": "leave_runner_after_tp2" if direction in {"long", "short"} else "inactive",
        "scale_out_plan": scale_out_plan,
        "management_notes": management_notes[:3],
    }



def build_execution_review(
    *,
    decision: dict,
    context: dict,
    execution_plan: dict,
) -> dict:
    direction = execution_plan.get("direction")
    entry_zone = execution_plan.get("entry_zone") or {}
    take_profit_zone = execution_plan.get("take_profit_zone") or {}
    current_price = first_valid_number(
        context.get("results", {}).get("5m", {}).get("price"),
        context.get("results", {}).get("15m", {}).get("price"),
        context.get("results", {}).get("1h", {}).get("price"),
    )

    if direction not in {"long", "short"} or current_price is None:
        return {
            "market_entry_price": current_price,
            "planned_entry_price": None,
            "ideal_entry_price": None,
            "entry_zone_low": None,
            "entry_zone_high": None,
            "entry_location": None,
            "zone_width_pct": None,
            "market_vs_planned_pct": None,
            "market_vs_ideal_pct": None,
            "reward_to_risk_estimate": first_valid_number(execution_plan.get("reward_to_risk_estimate")),
            "execution_efficiency_score": None,
            "execution_efficiency_label": None,
            "quality": None,
            "planned_entry_touched_after_signal": None,
            "ideal_entry_touched_after_signal": None,
            "entry_zone_retested_after_signal": None,
            "market_entry_latest_pnl_pct": None,
            "planned_entry_latest_pnl_pct": None,
            "ideal_entry_latest_pnl_pct": None,
            "planned_edge_vs_market_latest_pct": None,
            "ideal_edge_vs_market_latest_pct": None,
            "notes": ["Execution review is inactive until a directional plan is available."],
        }

    zone_low = first_valid_number(entry_zone.get("low"))
    zone_high = first_valid_number(entry_zone.get("high"))
    planned_entry = first_valid_number(entry_zone.get("mid"), current_price)
    ideal_entry = first_valid_number(zone_low if direction == "long" else zone_high, planned_entry)

    zone_width_pct = None
    if planned_entry not in (None, 0.0) and zone_low is not None and zone_high is not None:
        zone_width_pct = round(((zone_high - zone_low) / planned_entry) * 100, 4)

    def edge_vs_market(reference_price: float | None) -> float | None:
        if reference_price in (None, 0.0):
            return None
        if direction == "long":
            return round(((current_price - reference_price) / reference_price) * 100, 4)
        return round(((reference_price - current_price) / reference_price) * 100, 4)

    planned_edge = edge_vs_market(planned_entry)
    ideal_edge = edge_vs_market(ideal_entry)

    tolerance_candidates = [current_price * 0.00015]
    zone_width_abs = None
    if zone_low is not None and zone_high is not None:
        zone_width_abs = abs(zone_high - zone_low)
        tolerance_candidates.append(zone_width_abs * 0.08)
    atr_reference = first_valid_number(
        context.get("results", {}).get("5m", {}).get("indicators", {}).get("atr_14"),
        context.get("results", {}).get("15m", {}).get("indicators", {}).get("atr_14"),
    )
    if atr_reference is not None:
        tolerance_candidates.append(atr_reference * 0.12)
    tolerance = max(tolerance_candidates)

    if zone_low is not None and zone_high is not None and (zone_low - tolerance) <= current_price <= (zone_high + tolerance):
        entry_location = "inside_zone"
    elif direction == "long" and zone_high is not None and current_price > zone_high + tolerance:
        entry_location = "above_zone"
    elif direction == "long" and zone_low is not None and current_price < zone_low - tolerance:
        entry_location = "below_zone"
    elif direction == "short" and zone_low is not None and current_price < zone_low - tolerance:
        entry_location = "below_zone"
    elif direction == "short" and zone_high is not None and current_price > zone_high + tolerance:
        entry_location = "above_zone"
    else:
        entry_location = "inside_zone"

    if entry_location == "inside_zone" and planned_edge is not None and abs(planned_edge) <= max(0.02, (zone_width_pct or 0) * 0.20):
        quality = "ideal"
        efficiency_score = 92
        efficiency_label = "excellent"
    elif entry_location == "inside_zone":
        quality = "in_band"
        quality_penalty = min(10.0, abs(planned_edge or 0.0) * 180)
        efficiency_score = max(80, int(round(90 - quality_penalty)))
        efficiency_label = "good" if efficiency_score >= 85 else "acceptable"
    elif (direction == "long" and entry_location == "above_zone") or (direction == "short" and entry_location == "below_zone"):
        quality = "chasing"
        stretch_penalty = min(22.0, abs(planned_edge or 0.0) * 220)
        efficiency_score = max(58, int(round(82 - stretch_penalty)))
        efficiency_label = "poor" if efficiency_score < 66 else "stretched"
    else:
        quality = "early"
        early_penalty = min(14.0, abs(planned_edge or 0.0) * 120)
        efficiency_score = max(72, int(round(86 - early_penalty)))
        efficiency_label = "good" if efficiency_score >= 82 else "acceptable"

    notes: list[str] = []
    if quality == "chasing":
        if direction == "long":
            notes.append("Market is above the preferred long entry band, so a fresh fill would be chasing instead of buying the pullback.")
        else:
            notes.append("Market is below the preferred short entry band, so a fresh fill would be chasing instead of selling the retest.")
    elif quality == "ideal":
        notes.append("Market is trading almost exactly where the plan wanted it, so the execution quality is excellent.")
    elif quality == "in_band":
        notes.append("Market is trading inside the planned entry band, which keeps execution aligned with the model.")
    else:
        notes.append("Market is still outside the ideal band in a favorable direction, so patience may improve the average fill.")

    planned_entry_latest_pnl_pct = edge_vs_market(planned_entry)
    ideal_entry_latest_pnl_pct = edge_vs_market(ideal_entry)

    return {
        "market_entry_price": current_price,
        "planned_entry_price": planned_entry,
        "ideal_entry_price": ideal_entry,
        "entry_zone_low": zone_low,
        "entry_zone_high": zone_high,
        "entry_location": entry_location,
        "zone_width_pct": zone_width_pct,
        "market_vs_planned_pct": planned_edge,
        "market_vs_ideal_pct": ideal_edge,
        "reward_to_risk_estimate": first_valid_number(execution_plan.get("reward_to_risk_estimate")),
        "execution_efficiency_score": efficiency_score,
        "execution_efficiency_label": efficiency_label,
        "quality": quality,
        "planned_entry_touched_after_signal": None,
        "ideal_entry_touched_after_signal": None,
        "entry_zone_retested_after_signal": None,
        "market_entry_latest_pnl_pct": 0.0,
        "planned_entry_latest_pnl_pct": planned_entry_latest_pnl_pct,
        "ideal_entry_latest_pnl_pct": ideal_entry_latest_pnl_pct,
        "planned_edge_vs_market_latest_pct": planned_edge,
        "ideal_edge_vs_market_latest_pct": ideal_edge,
        "notes": notes[:3],
    }




def price_buffer_from_context(context: dict, fallback_price: float | None = None) -> float:
    """Small helper for planner spacing and stale/chase thresholds."""
    results = context.get("results", {}) or {}
    price = first_valid_number(
        fallback_price,
        results.get("5m", {}).get("price"),
        results.get("15m", {}).get("price"),
        results.get("1h", {}).get("price"),
    )
    candidates: list[float] = []
    if price is not None:
        candidates.append(float(price) * 0.0015)
    for tf in ["5m", "15m"]:
        atr = first_valid_number(results.get(tf, {}).get("indicators", {}).get("atr_14"))
        if atr is not None:
            candidates.append(float(atr) * 0.35)
    return max(candidates) if candidates else 0.0



def build_limit_order_plan(
    *,
    decision: dict,
    context: dict,
    execution_plan: dict,
    execution_review: dict,
    entry_validation: dict,
    entry_score: dict | None = None,
) -> dict:
    """Convert a directional idea into a passive-entry plan when market entry is inefficient.

    This does not place orders. It emits a structured plan that can be logged,
    displayed, and outcome-tested before any automation is trusted.
    """
    direction = execution_plan.get("direction")
    action = decision.get("action")
    state = decision.get("state")
    entry_model = execution_plan.get("entry_model")
    entry_zone = execution_plan.get("entry_zone") or {}
    invalidation_zone = execution_plan.get("invalidation_zone") or {}
    take_profit_zone = execution_plan.get("take_profit_zone") or {}

    current_price = first_valid_number(execution_review.get("market_entry_price"))
    planned_entry = first_valid_number(execution_review.get("planned_entry_price"), entry_zone.get("mid"))
    ideal_entry = first_valid_number(execution_review.get("ideal_entry_price"))
    zone_low = first_valid_number(entry_zone.get("low"))
    zone_high = first_valid_number(entry_zone.get("high"))
    planned_edge = first_valid_number(
        execution_review.get("planned_edge_vs_market_latest_pct"),
        execution_review.get("market_vs_planned_pct"),
    )
    ideal_edge = first_valid_number(
        execution_review.get("ideal_edge_vs_market_latest_pct"),
        execution_review.get("market_vs_ideal_pct"),
    )
    efficiency_label = str(execution_review.get("execution_efficiency_label") or "").lower()
    quality = str(execution_review.get("quality") or "").lower()
    entry_score_value = int(first_valid_number((entry_score or {}).get("score"), 0) or 0)

    base_cancel_rules = [
        "Cancel if strategy bias flips neutral or opposite.",
        "Cancel if invalidation is touched before the limit fills.",
        "Cancel if trigger context loses directional setup support.",
    ]
    base_promotion_rules = [
        "Promote only after the limit fills inside the planned entry band.",
        "After TP1, move remaining risk to breakeven.",
    ]

    inactive = {
        "enabled": False,
        "planner_state": "inactive",
        "preference": "inactive",
        "side": None,
        "order_type": "limit",
        "limit_price": None,
        "backup_limit_price": None,
        "entry_zone": compact_price_zone(entry_zone),
        "market_entry_allowed": bool(execution_plan.get("execution_ready") and entry_validation.get("allowed")),
        "expiry_snapshots": 0,
        "expiry_minutes_estimate": 0,
        "cancel_if": [],
        "promotion_rules": [],
        "reason": "No directional limit plan is active while the strategy direction is neutral or unmapped.",
        "notes": [],
        "historical_hint": None,
        "source": "v36_limit_order_planner",
    }

    if direction not in {"long", "short"} or current_price is None:
        if entry_model == "compression_breakout_monitor" and entry_zone:
            return {
                **inactive,
                "planner_state": "monitor_band",
                "preference": "monitor_only",
                "entry_zone": compact_price_zone(entry_zone),
                "reason": "Compression is active, but the engine has not selected a tradable side yet. Watch the band before staging a limit.",
                "promotion_rules": ["Choose the side only after price breaks and holds outside the compression trigger band."],
            }
        return inactive

    if planned_entry is None and zone_low is not None and zone_high is not None:
        planned_entry = (zone_low + zone_high) / 2
    if ideal_entry is None:
        ideal_entry = zone_low if direction == "long" else zone_high
    if planned_entry is None:
        return inactive

    side = "buy" if direction == "long" else "sell"
    invalidation_price = first_valid_number(
        invalidation_zone.get("high") if direction == "long" else invalidation_zone.get("low"),
        invalidation_zone.get("mid"),
        invalidation_zone.get("low") if direction == "long" else invalidation_zone.get("high"),
    )
    take_profit_reference = first_valid_number(take_profit_zone.get("tp1"))

    # Positive planned_edge means the market has already moved away from the plan in the trade direction.
    stretched_from_plan = planned_edge is not None and planned_edge > 0.02
    strongly_stretched_from_plan = planned_edge is not None and planned_edge > 0.05
    chase_quality = quality == "chasing" or efficiency_label in {"stretched", "poor"}
    wait_pullback = action == "wait_pullback"
    exhaustion_fade = state == "exhaustion_rejection" or entry_model == "exhaustion_rejection_fade"
    breakout_watch = action in {"watch_breakout", "watch_breakdown"}
    market_ready = bool(execution_plan.get("execution_ready") and entry_validation.get("allowed"))

    preference = "market_or_limit"
    planner_state = "active"
    enabled = True
    market_entry_allowed = market_ready
    limit_price = ideal_entry
    backup_limit_price = planned_entry
    expiry_snapshots = 24
    expiry_minutes_estimate = 120
    notes: list[str] = []
    reason = "Directional idea is active; use the limit plan to improve average entry if price retests the band."

    if exhaustion_fade and direction == "short":
        preference = "limit_preferred"
        planner_state = "tactical_fade_retest"
        market_entry_allowed = market_ready and not strongly_stretched_from_plan
        reason = "Exhaustion rejection is a high-probability tactical fade, but the preferred fill is a retest into the rejection band rather than a late chase."
        notes.append("Keep this as a quick scale-out setup; do not treat it as a long-duration trend short.")
        expiry_snapshots = 12
        expiry_minutes_estimate = 60
    elif wait_pullback:
        preference = "limit_only"
        planner_state = "wait_for_retest"
        market_entry_allowed = False
        reason = "The engine already rejected immediate execution for timing. Stage the passive retest instead of chasing."
        notes.append("This is the cleanest use case for a passive order: planned/ideal entry first, market entry disabled.")
    elif direction == "long" and (stretched_from_plan or chase_quality or breakout_watch):
        preference = "limit_only" if strongly_stretched_from_plan or chase_quality else "limit_preferred"
        planner_state = "buy_retest"
        market_entry_allowed = market_ready and preference != "limit_only"
        reason = "Long plan edge is stretched versus the preferred band. Historical logs favored waiting for the retest over buying the extended print."
    elif direction == "short" and (stretched_from_plan or chase_quality):
        preference = "limit_preferred" if not strongly_stretched_from_plan else "limit_only"
        planner_state = "sell_retest"
        market_entry_allowed = market_ready and preference != "limit_only"
        reason = "Short plan edge is stretched versus the preferred band. Use a retest sell limit instead of selling late extension."
    elif market_ready:
        preference = "market_or_limit"
        planner_state = "inside_band_optional_limit"
        reason = "Market is close enough to the planned band that either staged market execution or a passive limit is acceptable."
    elif action in {"watch_reversal", "watch_breakout", "watch_breakdown"}:
        preference = "monitor_then_limit"
        planner_state = "armed_waiting_confirmation"
        market_entry_allowed = False
        reason = "The idea is directional but still watch-only. Arm the limit zone, then require trigger confirmation before any fill is trusted."
    else:
        preference = "limit_preferred"
        market_entry_allowed = False

    # For long buys the ideal price is the lower edge; for shorts it is the upper edge.
    limit_price = first_valid_number(ideal_entry, planned_entry)
    backup_limit_price = first_valid_number(planned_entry, limit_price)

    # Guard against stale or nonsensical prices. If the ideal side is already crossed, fall back to midpoint.
    buffer = price_buffer_from_context(context, current_price)
    if direction == "long" and limit_price is not None and limit_price > current_price + max(buffer * 0.20, current_price * 0.0001):
        limit_price = backup_limit_price
        notes.append("Ideal buy price was above market, so the planner fell back to the planned midpoint.")
    if direction == "short" and limit_price is not None and limit_price < current_price - max(buffer * 0.20, current_price * 0.0001):
        limit_price = backup_limit_price
        notes.append("Ideal sell price was below market, so the planner fell back to the planned midpoint.")

    if preference == "limit_only":
        notes.append("Market entry is disabled for this snapshot; only the planned retest should be considered.")
    if entry_score_value < 45:
        notes.append("Entry score is low, so treat the plan as observation/learning unless confirmation improves.")

    historical_hint = None
    if direction == "long":
        historical_hint = "Prior v35 logs showed long planned-entry retests were common and often cleaner than market buys."
    elif exhaustion_fade:
        historical_hint = "Prior v35 logs showed exhaustion-rejection shorts were the strongest tactical pattern, but mostly TP1-first."

    return {
        "enabled": enabled,
        "planner_state": planner_state,
        "preference": preference,
        "side": side,
        "order_type": "limit",
        "limit_price": round_or_none(limit_price),
        "backup_limit_price": round_or_none(backup_limit_price),
        "entry_zone": compact_price_zone(entry_zone),
        "invalidation_price": round_or_none(invalidation_price),
        "take_profit_reference": round_or_none(take_profit_reference),
        "market_entry_allowed": market_entry_allowed,
        "expiry_snapshots": expiry_snapshots,
        "expiry_minutes_estimate": expiry_minutes_estimate,
        "cancel_if": base_cancel_rules,
        "promotion_rules": base_promotion_rules,
        "reason": reason,
        "notes": notes[:4],
        "historical_hint": historical_hint,
        "source": "v36_limit_order_planner",
    }


def build_execution_plan(decision: dict, context: dict, entry_validation: dict) -> dict:
    results = context["results"]
    mode_config = context.get("mode_config", get_strategy_mode_config(DEFAULT_STRATEGY_MODE))

    trigger_tf = decision["trigger_timeframes"][0] if decision["trigger_timeframes"] else "5m"
    setup_tf = decision["setup_timeframes"][0] if decision["setup_timeframes"] else ("15m" if "15m" in results else trigger_tf)
    risk_tf = decision["risk_timeframes"][0] if decision["risk_timeframes"] else ("1h" if "1h" in results else setup_tf)

    trigger = results.get(trigger_tf, {})
    setup = results.get(setup_tf, {})
    risk = results.get(risk_tf, {})
    hourly = results.get("1h", {})

    price = first_valid_number(
        trigger.get("price"),
        setup.get("price"),
        results.get("5m", {}).get("price"),
        results.get("15m", {}).get("price"),
    )

    primary_timeframes = {
        "setup": setup_tf,
        "trigger": trigger_tf,
        "risk": risk_tf,
    }
    reference_timeframes = list(dict.fromkeys([tf for tf in [trigger_tf, setup_tf, risk_tf, "1h"] if tf in results]))

    if price is None:
        return {
            "execution_ready": False,
            "direction": "neutral",
            "entry_model": "standby",
            "timing_state": decision.get("entry_timing", "blocked"),
            "primary_timeframes": primary_timeframes,
            "entry_zone": None,
            "invalidation_zone": None,
            "take_profit_zone": None,
            "reward_to_risk_estimate": None,
            "notes": ["Execution plan is unavailable because no valid reference price was produced."],
        }

    atr_candidates: list[float] = []
    range_candidates: list[float] = []
    support_candidates: list[float] = []
    resistance_candidates: list[float] = []

    for tf_name in reference_timeframes:
        data = results.get(tf_name, {})
        indicators = data.get("indicators", {})
        candle = data.get("candle", {})
        fib = data.get("fibonacci", {})
        fib_levels = list((fib.get("levels") or {}).values())

        atr_value = first_valid_number(indicators.get("atr_14"))
        if atr_value is not None:
            atr_candidates.append(atr_value)

        candle_high = first_valid_number(candle.get("high"))
        candle_low = first_valid_number(candle.get("low"))
        if candle_high is not None and candle_low is not None:
            range_candidates.append(abs(candle_high - candle_low))

        support_candidates.extend(
            [
                indicators.get("ema_9"),
                indicators.get("sma_20"),
                indicators.get("bb_mid"),
                indicators.get("bb_lower"),
                candle.get("low"),
                fib.get("low"),
                *fib_levels,
            ]
        )
        resistance_candidates.extend(
            [
                indicators.get("ema_9"),
                indicators.get("sma_20"),
                indicators.get("bb_mid"),
                indicators.get("bb_upper"),
                candle.get("high"),
                fib.get("high"),
                *fib_levels,
            ]
        )

    supports_below = unique_price_levels([level for level in support_candidates if first_valid_number(level) is not None and float(level) < price], reverse=True)
    resistances_above = unique_price_levels([level for level in resistance_candidates if first_valid_number(level) is not None and float(level) > price])

    trigger_high = first_valid_number(trigger.get("candle", {}).get("high"))
    trigger_low = first_valid_number(trigger.get("candle", {}).get("low"))
    setup_high = first_valid_number(setup.get("candle", {}).get("high"))
    setup_low = first_valid_number(setup.get("candle", {}).get("low"))

    buffer_candidates = [price * 0.0015]
    if atr_candidates:
        buffer_candidates.append(max(atr_candidates) * 0.35)
    if range_candidates:
        buffer_candidates.append(max(range_candidates) * 0.50)
    buffer = max(buffer_candidates)

    atr_reference = first_valid_number(max(atr_candidates) if atr_candidates else None)
    stop_padding_candidates = [buffer * float(mode_config.get("execution_stop_padding_buffer", 0.18))]
    if atr_reference is not None:
        stop_padding_candidates.append(atr_reference * float(mode_config.get("execution_stop_padding_atr", 0.38)))
    stop_padding = max(stop_padding_candidates)

    entry_model = "standby"
    direction = decision["strategy_bias"] if decision["strategy_bias"] in {"long", "short"} else "neutral"
    activation_source = decision.get("activation_source")

    if activation_source == "compression_breakout" and direction == "long":
        entry_model = "compression_breakout_activation"
    elif activation_source == "compression_breakout" and direction == "short":
        entry_model = "compression_breakdown_activation"
    elif decision.get("state") == "compression" and decision["action"] in {"watch_breakout", "watch_breakdown"}:
        entry_model = "compression_breakout_monitor"
    elif decision["action"] in {"enter_long", "watch_breakout"} and direction == "long":
        entry_model = "breakout_confirmation"
    elif decision["action"] in {"enter_short", "watch_breakdown"} and direction == "short":
        entry_model = "breakdown_confirmation"
    elif decision["action"] == "wait_pullback" and direction == "long":
        entry_model = "bullish_pullback_reentry"
    elif decision["action"] == "wait_pullback" and direction == "short":
        entry_model = "bearish_pullback_reentry"
    elif decision.get("state") == "exhaustion_rejection":
        entry_model = "exhaustion_rejection_fade"
    elif decision["action"] == "watch_reversal":
        entry_model = "reversal_confirmation"
    elif decision["action"] in {"avoid_long_trap", "avoid_short_trap"}:
        entry_model = "trap_avoidance"

    execution_ready = decision["action"] in {"enter_long", "enter_short"} and entry_validation["allowed"]
    timing_state = decision.get("entry_timing", "immediate" if execution_ready else "monitor")
    notes: list[str] = []

    if decision.get("state") == "compression" and decision["action"] in {"watch_breakout", "watch_breakdown"}:
        prev_trigger_high = first_valid_number(trigger.get("previous_candle", {}).get("high"))
        prev_trigger_low = first_valid_number(trigger.get("previous_candle", {}).get("low"))
        prev_setup_high = first_valid_number(setup.get("previous_candle", {}).get("high"))
        prev_setup_low = first_valid_number(setup.get("previous_candle", {}).get("low"))

        anchor_highs = [level for level in [prev_trigger_high, prev_setup_high] if level is not None]
        anchor_lows = [level for level in [prev_trigger_low, prev_setup_low] if level is not None]

        if anchor_highs and anchor_lows:
            upper_trigger = max(anchor_highs)
            lower_trigger = min(anchor_lows)
        else:
            upper_trigger = max([level for level in [trigger_high, setup_high, price + (buffer * 0.40)] if level is not None])
            lower_trigger = min([level for level in [trigger_low, setup_low, price - (buffer * 0.40)] if level is not None])

        if upper_trigger <= lower_trigger:
            upper_trigger = max([level for level in [trigger_high, setup_high, price + (buffer * 0.40)] if level is not None])
            lower_trigger = min([level for level in [trigger_low, setup_low, price - (buffer * 0.40)] if level is not None])

        entry_zone = make_price_zone(
            lower_trigger,
            upper_trigger,
            label="compression trigger band",
            description="Wait for price to break outside this band before activating a directional breakout model.",
            reference_timeframes=reference_timeframes,
        )
        invalidation_zone = None
        take_profit_zone = None
        reward_to_risk = None
        if decision["action"] == "watch_breakdown":
            notes.append("Compression has a slight downside lean, but the band still needs to break before activating the short model.")
        elif direction == "long":
            notes.append("Compression has a slight upside lean, but the band still needs to break before activating the long model.")
        else:
            notes.append("This is a monitoring state: break the band first, then activate a directional entry model.")

    elif direction == "long":
        if activation_source == "compression_breakout":
            entry_zone = make_price_zone(
                max((supports_below[0] if supports_below else price - (buffer * 0.80)), price - buffer),
                price,
                label="post-compression long entry band",
                description="Long continuation zone built after price resolved above the compression trigger band.",
                reference_timeframes=reference_timeframes,
            )
            notes.append("Compression has resolved higher, so the long continuation model is now active." if execution_ready else "Compression has resolved higher, but the long confirmation stack still needs one more pass.")
        elif entry_model == "bullish_pullback_reentry":
            upper_support = supports_below[0] if supports_below else price - (buffer * 0.30)
            lower_support = supports_below[1] if len(supports_below) > 1 else upper_support - (buffer * 0.55)
            entry_zone = make_price_zone(
                lower_support,
                upper_support,
                label="bullish pullback entry band",
                description="Preferred long re-entry zone on a controlled pullback into nearby support.",
                reference_timeframes=reference_timeframes,
            )
            if decision.get("entry_timing") == "wait_retest":
                notes.append("Immediate long execution was rejected because price is extended away from the preferred continuation band.")
            notes.append("Wait for price to revisit the pullback band before activating a long entry.")
        else:
            floor_support = supports_below[0] if supports_below else price - (buffer * 0.70)
            entry_zone = make_price_zone(
                max(floor_support, price - (buffer * 0.80)),
                price,
                label="bullish continuation entry band",
                description="Long continuation zone built from current price and the nearest structural support below.",
                reference_timeframes=reference_timeframes,
            )
            notes.append("Long continuation conditions are execution-ready inside the current entry band." if execution_ready else "The long idea exists, but immediate entry permission is still blocked.")

        entry_mid = first_valid_number(entry_zone["mid"]) if entry_zone else price
        stop_touch = min(
            [
                level
                for level in [
                    trigger_low,
                    setup_low,
                    (entry_zone["low"] - (stop_padding * 0.15)) if entry_zone else None,
                ]
                if level is not None
            ]
            or [price - stop_padding]
        )
        invalidation_zone = make_price_zone(
            stop_touch - (stop_padding * 0.55),
            min(stop_touch + (stop_padding * 0.22), (entry_zone["low"] - (stop_padding * 0.05)) if entry_zone else stop_touch + (stop_padding * 0.22)),
            label="long invalidation zone",
            description="If price closes decisively below this tighter structure-based area, the current long idea is invalid.",
            reference_timeframes=reference_timeframes,
        )
        favorable_levels = unique_price_levels(
            [level for level in resistances_above if level > entry_mid]
            + [
                first_valid_number(trigger.get("indicators", {}).get("bb_upper")),
                first_valid_number(setup.get("indicators", {}).get("bb_upper")),
                first_valid_number(hourly.get("fibonacci", {}).get("high")),
            ]
        )
        take_profit_zone, reward_to_risk = build_take_profit_zone(
            direction="long",
            entry_mid=entry_mid,
            invalidation_zone=invalidation_zone,
            favorable_levels=favorable_levels,
            buffer=buffer,
            reference_timeframes=reference_timeframes,
        )

    elif direction == "short":
        if decision.get("state") == "exhaustion_rejection":
            rejection_high_candidates = [
                level
                for level in [
                    trigger_high,
                    setup_high,
                    first_valid_number(trigger.get("previous_candle", {}).get("high")),
                    first_valid_number(setup.get("previous_candle", {}).get("high")),
                ]
                if level is not None
            ]
            rejection_high = max(rejection_high_candidates) if rejection_high_candidates else price + buffer
            entry_zone = make_price_zone(
                price,
                min(rejection_high, price + (buffer * 0.85)),
                label="exhaustion rejection short entry band",
                description="Tactical short fade zone after a failed upside push and buyer-exhaustion rejection.",
                reference_timeframes=reference_timeframes,
            )
            notes.append("This is a tactical rejection short. Favor quick scale-outs and do not hold if price reclaims the rejection high.")
        elif activation_source == "compression_breakout":
            entry_zone = make_price_zone(
                price,
                min((resistances_above[0] if resistances_above else price + (buffer * 0.80)), price + buffer),
                label="post-compression short entry band",
                description="Short continuation zone built after price resolved below the compression trigger band.",
                reference_timeframes=reference_timeframes,
            )
            notes.append("Compression has resolved lower, so the short continuation model is now active." if execution_ready else "Compression has resolved lower, but the short confirmation stack still needs one more pass.")
        elif entry_model == "bearish_pullback_reentry":
            lower_resistance = resistances_above[0] if resistances_above else price + (buffer * 0.30)
            upper_resistance = resistances_above[1] if len(resistances_above) > 1 else lower_resistance + (buffer * 0.55)
            entry_zone = make_price_zone(
                lower_resistance,
                upper_resistance,
                label="bearish pullback entry band",
                description="Preferred short re-entry zone on a weak bounce into nearby resistance.",
                reference_timeframes=reference_timeframes,
            )
            if decision.get("entry_timing") == "wait_retest":
                notes.append("Immediate short execution was rejected because price is extended away from the preferred continuation band.")
            notes.append("Wait for price to retest the resistance band before activating a short entry.")
        else:
            ceiling_resistance = resistances_above[0] if resistances_above else price + (buffer * 0.70)
            entry_zone = make_price_zone(
                price,
                min(ceiling_resistance, price + (buffer * 0.80)),
                label="bearish continuation entry band",
                description="Short continuation zone built from current price and the nearest structural resistance above.",
                reference_timeframes=reference_timeframes,
            )
            notes.append("Short continuation conditions are execution-ready inside the current entry band." if execution_ready else "The short idea exists, but immediate entry permission is still blocked.")

        entry_mid = first_valid_number(entry_zone["mid"]) if entry_zone else price
        stop_touch = max(
            [
                level
                for level in [
                    trigger_high,
                    setup_high,
                    (entry_zone["high"] + (stop_padding * 0.15)) if entry_zone else None,
                ]
                if level is not None
            ]
            or [price + stop_padding]
        )
        invalidation_zone = make_price_zone(
            max(stop_touch - (stop_padding * 0.22), (entry_zone["high"] + (stop_padding * 0.05)) if entry_zone else stop_touch - (stop_padding * 0.22)),
            stop_touch + (stop_padding * 0.55),
            label="short invalidation zone",
            description="If price closes decisively above this tighter structure-based area, the current short idea is invalid.",
            reference_timeframes=reference_timeframes,
        )
        favorable_levels = unique_price_levels(
            [level for level in supports_below if level < entry_mid],
            reverse=True,
        )
        favorable_levels += [
            level
            for level in [
                first_valid_number(trigger.get("indicators", {}).get("bb_lower")),
                first_valid_number(setup.get("indicators", {}).get("bb_lower")),
                first_valid_number(hourly.get("fibonacci", {}).get("low")),
            ]
            if level is not None and level < entry_mid
        ]
        favorable_levels = unique_price_levels(favorable_levels, reverse=True)
        take_profit_zone, reward_to_risk = build_take_profit_zone(
            direction="short",
            entry_mid=entry_mid,
            invalidation_zone=invalidation_zone,
            favorable_levels=favorable_levels,
            buffer=buffer,
            reference_timeframes=reference_timeframes,
        )

    elif decision["action"] == "watch_reversal":
        reversal_low = supports_below[0] if supports_below else price - (buffer * 0.60)
        reversal_high = resistances_above[0] if resistances_above else price + (buffer * 0.60)
        entry_zone = make_price_zone(
            reversal_low,
            reversal_high,
            label="reversal decision band",
            description="Watch for a confirmed reclaim or rejection of this band before activating a reversal trade idea.",
            reference_timeframes=reference_timeframes,
        )
        invalidation_zone = None
        take_profit_zone = None
        reward_to_risk = None
        notes.append("Reversal setups require confirmation from the trigger layer before becoming executable.")

    else:
        entry_zone = None
        invalidation_zone = None
        take_profit_zone = None
        reward_to_risk = None
        if decision["action"] in {"avoid_long_trap", "avoid_short_trap"}:
            notes.append("Trap risk takes priority, so the engine is intentionally withholding execution levels.")
        else:
            notes.append("No active execution plan is open while the engine remains in a wait or standby state.")

    if not execution_ready and entry_validation.get("blocking_reasons"):
        notes.append(entry_validation["blocking_reasons"][0])

    return {
        "execution_ready": execution_ready,
        "direction": direction,
        "entry_model": entry_model,
        "timing_state": timing_state,
        "primary_timeframes": primary_timeframes,
        "entry_zone": entry_zone,
        "invalidation_zone": invalidation_zone,
        "take_profit_zone": take_profit_zone,
        "reward_to_risk_estimate": reward_to_risk,
        "notes": notes[:3],
    }





def infer_execution_playbook(
    *,
    decision: dict,
    execution_plan: dict,
    confidence: int,
    trade_quality: str,
    market_phase: str,
) -> dict:
    action = decision.get("action")
    direction = execution_plan.get("direction", "neutral")
    entry_model = execution_plan.get("entry_model", "standby")
    execution_ready = bool(execution_plan.get("execution_ready"))
    reward_to_risk = first_valid_number(execution_plan.get("reward_to_risk_estimate"))

    primary_timeframes = execution_plan.get("primary_timeframes", {}) or {}
    setup_tf = primary_timeframes.get("setup", "15m")
    trigger_tf = primary_timeframes.get("trigger", "5m")
    risk_tf = primary_timeframes.get("risk", "5m")

    management = execution_plan.get("execution_management") or {}
    scale_out_plan = management.get("scale_out_plan") or []
    scale_out_text = ", ".join(
        f"{step.get('target', '').upper()} {step.get('size_pct', 0)}%"
        for step in scale_out_plan
        if step.get("size_pct")
    ) or "TP1 60%, TP2 25%, TP3 15%"

    entry_style = "monitor_only"
    position_sizing = "probe" if market_phase == "reversal" else "reduced"
    aggressiveness = "low"
    execution_instruction = "No immediate execution is open. Keep monitoring the active decision band."
    playbook: list[str] = [
        execution_instruction,
        f"Keep {trigger_tf} as the live trigger while {setup_tf} defines the broader setup context.",
    ]

    if action in {"avoid_long_trap", "avoid_short_trap"}:
        execution_instruction = "Trap risk is active. Stand aside until price either reclaims the trigger band cleanly or the trap fully fails."
        playbook = [
            execution_instruction,
            "Do not deploy fresh directional size while trap risk remains active.",
            f"Reassess only after {trigger_tf} confirms the move is no longer vulnerable to failure.",
        ]
        position_sizing = "probe"

    elif decision.get("state") == "exhaustion_rejection":
        entry_style = "limit_retest"
        position_sizing = "probe"
        aggressiveness = "low"
        execution_instruction = "Premium short fade is forming, but treat it tactically: wait for 5m rejection confirmation and scale out quickly."
        playbook = [
            execution_instruction,
            f"Use {trigger_tf} for the rejection trigger and {setup_tf} to verify the failed upside push is not reclaiming.",
            f"If price reclaims the {risk_tf} rejection high or invalidation zone, cancel the fade instead of holding for a trend short.",
        ]

    elif action == "watch_reversal":
        execution_instruction = "Treat this as a developing reversal only. Wait for the trigger layer to confirm the turn before taking even probe size."
        playbook = [
            execution_instruction,
            f"Use {trigger_tf} for reclaim or rejection confirmation and let {setup_tf} validate the reversal structure.",
            f"Keep risk tight until {risk_tf} stops opposing the reversal attempt.",
        ]
        position_sizing = "probe"

    elif action in {"watch_breakout", "watch_breakdown"}:
        execution_instruction = "Compression is still resolving. Wait for a clean break and follow-through before deploying size."
        playbook = [
            execution_instruction,
            f"Promote to directional execution only after {trigger_tf} closes decisively outside the trigger band.",
            f"Until then, use {setup_tf} to map the band and {risk_tf} to judge whether the move is sticking.",
        ]
        position_sizing = "reduced"

    elif action == "wait_pullback":
        entry_style = "limit_retest"
        position_sizing = "reduced"
        aggressiveness = "moderate" if confidence >= 80 else "low"
        execution_instruction = "Immediate entry is too stretched. Stage passive orders in the retest zone instead of chasing."
        playbook = [
            execution_instruction,
            f"Work the pullback using {scale_out_text} after the position is filled instead of taking the whole trade off at TP1.",
            f"If price loses the {risk_tf} invalidation zone before the retest behaves correctly, cancel the idea rather than forcing a fill.",
        ]

    elif execution_ready and action in {"enter_long", "enter_short"}:
        if entry_model in {"compression_breakout_activation", "compression_breakdown_activation", "breakout_confirmation", "breakdown_confirmation"}:
            entry_style = "ladder_entry"
            position_sizing = "full" if confidence >= 84 and trade_quality in {"A", "B"} and (reward_to_risk or 0) >= 1.4 else "reduced"
            aggressiveness = "moderate"
            execution_instruction = "Enter inside the active band, pay yourself quickly at TP1, then convert the trade into a protected runner."
        elif entry_model in {"bullish_pullback_reentry", "bearish_pullback_reentry"}:
            entry_style = "limit_retest"
            position_sizing = "full" if confidence >= 85 and trade_quality in {"A", "B"} else "reduced"
            aggressiveness = "moderate"
            execution_instruction = "Work passive orders inside the pullback band rather than using market execution."
        else:
            entry_style = "ladder_entry"
            position_sizing = "reduced"
            aggressiveness = "moderate"
            execution_instruction = "Use staged execution inside the active band rather than a single aggressive fill."

        playbook = [
            execution_instruction,
            f"Scale out with {scale_out_text}. After TP1, move the stop to breakeven. After TP2, trail using the {management.get('trailing_stop_reference', '5m structure')}.",
            f"If price violates the {risk_tf} invalidation zone before TP1 pays you, cut the trade quickly instead of averaging into a loser.",
        ]

    return {
        "entry_style": entry_style,
        "position_sizing": position_sizing,
        "aggressiveness": aggressiveness,
        "execution_instruction": execution_instruction,
        "playbook": playbook[:3],
    }



def detect_compression_breakout_activation(
decision: dict, context: dict, execution_plan: dict) -> dict | None:
    if decision.get("state") != "compression" or decision.get("action") not in {"watch_breakout", "watch_breakdown"}:
        return None

    if not execution_plan or execution_plan.get("entry_model") != "compression_breakout_monitor":
        return None

    entry_zone = execution_plan.get("entry_zone") or {}
    lower_band = first_valid_number(entry_zone.get("low"))
    upper_band = first_valid_number(entry_zone.get("high"))
    if lower_band is None or upper_band is None:
        return None

    results = context["results"]
    tf_5m = results.get("5m", {})
    tf_15m = results.get("15m", {})

    price_5m = first_valid_number(tf_5m.get("price"))
    price_15m = first_valid_number(tf_15m.get("price"))
    high_5m = first_valid_number(tf_5m.get("candle", {}).get("high"))
    low_5m = first_valid_number(tf_5m.get("candle", {}).get("low"))
    high_15m = first_valid_number(tf_15m.get("candle", {}).get("high"))
    low_15m = first_valid_number(tf_15m.get("candle", {}).get("low"))

    momentum_5m = tf_5m.get("signal_details", {}).get("momentum_state")
    momentum_15m = tf_15m.get("signal_details", {}).get("momentum_state")
    signal_5m = tf_5m.get("signal")
    signal_15m = tf_15m.get("signal")
    setup_15m = tf_15m.get("setup")

    atr_5m = first_valid_number(tf_5m.get("indicators", {}).get("atr_14"))
    atr_15m = first_valid_number(tf_15m.get("indicators", {}).get("atr_14"))
    reference_price = first_valid_number(price_5m, price_15m, upper_band, lower_band)
    if reference_price is None:
        return None

    margin_candidates = [reference_price * 0.0002]
    if atr_5m is not None:
        margin_candidates.append(atr_5m * 0.08)
    if atr_15m is not None:
        margin_candidates.append(atr_15m * 0.05)
    margin = max(margin_candidates)

    closes_above = sum(1 for value in [price_5m, price_15m] if value is not None and value >= upper_band + margin)
    closes_below = sum(1 for value in [price_5m, price_15m] if value is not None and value <= lower_band - margin)

    wick_break_up = any(
        value is not None and value >= upper_band + margin
        for value in [high_5m, high_15m]
    )
    wick_break_down = any(
        value is not None and value <= lower_band - margin
        for value in [low_5m, low_15m]
    )

    broke_up = closes_above >= 1 or wick_break_up
    broke_down = closes_below >= 1 or wick_break_down

    strong_up = closes_above >= 2 or signal_5m in BULLISH_SIGNALS or signal_15m in BULLISH_SIGNALS or setup_15m in BULLISH_BREAKOUT_SETUPS
    strong_down = closes_below >= 2 or signal_5m in BEARISH_SIGNALS or signal_15m in BEARISH_SIGNALS or setup_15m in BEARISH_BREAKOUT_SETUPS

    bullish_follow_through = (
        broke_up
        and momentum_5m in {"rising", "flat"}
        and momentum_15m in {"rising", "flat"}
        and context["bull_trap_count"] == 0
        and context["failure_risk_count"] == 0
    )

    bearish_follow_through = (
        broke_down
        and momentum_5m in {"falling", "flat"}
        and momentum_15m in {"falling", "flat"}
        and context["bear_trap_count"] == 0
        and context["failure_risk_count"] == 0
    )

    if bullish_follow_through and not bearish_follow_through:
        action = "enter_long" if strong_up and not context["higher_bearish_pressure"] else "watch_breakout"
        state = "continuation" if action == "enter_long" else "continuation_watch"
        return {
            "strategy_bias": "long",
            "state": state,
            "action": action,
            "setup_timeframes": ["15m"],
            "trigger_timeframes": ["5m"],
            "risk_timeframes": ["5m"],
            "risk_state": "favorable" if context["higher_timeframe_bias"] in {"bullish", "strong_bullish"} else "mixed",
            "summary": (
                f'{context["symbol"]} has started to resolve higher out of compression, with the 5m trigger pushing above the monitored band. '
                f'That activates the bullish continuation model as long as follow-through holds.'
            ),
            "activation_source": "compression_breakout",
            "compression_breakout_confirmed": True,
            "compression_break_direction": "up",
        }

    if bearish_follow_through and not bullish_follow_through:
        action = "enter_short" if strong_down and not context["higher_bullish_pressure"] else "watch_breakdown"
        state = "continuation" if action == "enter_short" else "continuation_watch"
        return {
            "strategy_bias": "short",
            "state": state,
            "action": action,
            "setup_timeframes": ["15m"],
            "trigger_timeframes": ["5m"],
            "risk_timeframes": ["5m"],
            "risk_state": "favorable" if context["higher_timeframe_bias"] in {"bearish", "strong_bearish"} else "mixed",
            "summary": (
                f'{context["symbol"]} has started to resolve lower out of compression, with the 5m trigger pushing below the monitored band. '
                f'That activates the bearish continuation model as long as follow-through holds.'
            ),
            "activation_source": "compression_breakout",
            "compression_breakout_confirmed": True,
            "compression_break_direction": "down",
        }

    return None





def ladder_stage_from_score(score: float) -> str:
    if score >= 9.0:
        return "confirmed"
    if score >= 7.0:
        return "ready"
    if score >= 5.0:
        return "armed"
    if score >= 3.0:
        return "early"
    if score >= 1.0:
        return "building"
    return "inactive"



def directional_setup_supports(
    *,
    direction: str,
    setup_signal: str | None,
    setup_name: str | None,
    structure: str | None,
    trend_bias: str | None,
    momentum_state: str | None,
    rsi_state: str | None,
) -> bool:
    if direction == "long":
        if setup_signal in BULLISH_SIGNALS or setup_name in (BULLISH_BREAKOUT_SETUPS | {"Trend Continuation", "Bullish Pressure Build"}):
            return True
        if setup_name in {"Early Bullish Reversal", "Oversold Reversal", "Bullish Exhaustion Reversal", "Bullish Pullback"}:
            return True
        if setup_signal in {"bearish_but_bouncing", "oversold_reversal_watch", "bullish_but_pulling_back"}:
            return True
        if trend_bias == "bullish" and momentum_state in {"rising", "flat"}:
            return True
        if structure in {"transition", "compression"} and momentum_state == "rising" and rsi_state in {"neutral", "bullish", "oversold_recovering"}:
            return True
        return False

    if setup_signal in BEARISH_SIGNALS or setup_name in (BEARISH_BREAKOUT_SETUPS | {"Trend Continuation"}):
        return True
    if setup_name in {"Overbought Reversal Risk", "Bearish Exhaustion"}:
        return True
    if setup_name == "Early Bearish Reversal":
        return trend_bias in {"bearish", "mixed"}
    if setup_name == "Bearish Bounce Attempt":
        return trend_bias == "bearish" and momentum_state in {"falling", "flat"}
    if setup_signal == "overbought_pullback_watch":
        return True
    if setup_signal == "bearish_but_bouncing":
        return trend_bias == "bearish" and rsi_state in {"neutral", "bearish", "overbought_falling"}
    if trend_bias == "bearish" and momentum_state in {"falling", "flat"}:
        return True
    if structure in {"transition", "compression"} and trend_bias == "bearish" and momentum_state == "falling" and rsi_state in {"neutral", "bearish", "overbought_falling"}:
        return True
    return False



def continuation_setup_qualified(
    *,
    direction: str,
    setup_signal: str | None,
    setup_name: str | None,
    structure: str | None,
    trend_bias: str | None,
    momentum_state: str | None,
) -> bool:
    if direction == "long":
        return (
            setup_signal in BULLISH_SIGNALS
            or setup_name in (BULLISH_BREAKOUT_SETUPS | {"Trend Continuation", "Bullish Pressure Build"})
            or (
                structure == "expansion"
                and trend_bias == "bullish"
                and momentum_state in {"rising", "flat"}
            )
            or (
                setup_name == "Bullish Pullback"
                and trend_bias == "bullish"
                and momentum_state in {"rising", "flat"}
            )
        )

    return (
        setup_signal in BEARISH_SIGNALS
        or setup_name in (BEARISH_BREAKOUT_SETUPS | {"Trend Continuation"})
        or (
            structure == "expansion"
            and trend_bias == "bearish"
            and momentum_state in {"falling", "flat"}
        )
        or (
            setup_name == "Bearish Bounce Attempt"
            and trend_bias == "bearish"
            and momentum_state in {"falling", "flat"}
        )
    )



def trigger_indicator_confirms_direction(
    *,
    direction: str,
    trigger_signal: str | None,
    activation_source: str | None = None,
    compression_breakout_confirmed: bool = False,
    compression_break_direction: str | None = None,
) -> bool:
    if direction == "long":
        return trigger_signal in BULLISH_SIGNALS or (
            activation_source == "compression_breakout"
            and compression_breakout_confirmed
            and compression_break_direction == "up"
        )
    return trigger_signal in BEARISH_SIGNALS or (
        activation_source == "compression_breakout"
        and compression_breakout_confirmed
        and compression_break_direction == "down"
    )



def detect_trigger_band_break(
    *,
    direction: str,
    trigger: dict,
    setup: dict,
    entry_zone: dict | None,
    entry_model: str | None,
) -> dict:
    label = str((entry_zone or {}).get("label") or "").lower()
    required = bool(entry_model == "compression_breakout_monitor" or "trigger band" in label or "compression" in label)
    if not required:
        return {
            "required": False,
            "confirmed": None,
            "method": "not_required",
            "margin": 0.0,
            "upper_band": first_valid_number((entry_zone or {}).get("high")),
            "lower_band": first_valid_number((entry_zone or {}).get("low")),
        }

    upper_band = first_valid_number((entry_zone or {}).get("high"))
    lower_band = first_valid_number((entry_zone or {}).get("low"))
    if upper_band is None or lower_band is None:
        return {
            "required": True,
            "confirmed": False,
            "method": "missing_band",
            "margin": 0.0,
            "upper_band": upper_band,
            "lower_band": lower_band,
        }

    price_trigger = first_valid_number(trigger.get("price"))
    price_setup = first_valid_number(setup.get("price"))
    high_trigger = first_valid_number(trigger.get("candle", {}).get("high"))
    low_trigger = first_valid_number(trigger.get("candle", {}).get("low"))
    high_setup = first_valid_number(setup.get("candle", {}).get("high"))
    low_setup = first_valid_number(setup.get("candle", {}).get("low"))
    atr_trigger = first_valid_number(trigger.get("indicators", {}).get("atr_14"))
    atr_setup = first_valid_number(setup.get("indicators", {}).get("atr_14"))
    reference_price = first_valid_number(price_trigger, price_setup, upper_band, lower_band)

    if reference_price is None:
        return {
            "required": True,
            "confirmed": False,
            "method": "no_price",
            "margin": 0.0,
            "upper_band": upper_band,
            "lower_band": lower_band,
        }

    margin_candidates = [reference_price * 0.0002]
    if atr_trigger is not None:
        margin_candidates.append(atr_trigger * 0.08)
    if atr_setup is not None:
        margin_candidates.append(atr_setup * 0.05)
    margin = max(margin_candidates)

    if direction == "long":
        close_break = any(value is not None and value >= upper_band + margin for value in [price_trigger, price_setup])
        wick_break = any(value is not None and value >= upper_band + margin for value in [high_trigger, high_setup])
        confirmed = close_break or wick_break
        method = "close_break" if close_break else "wick_break" if wick_break else "inside_band"
    else:
        close_break = any(value is not None and value <= lower_band - margin for value in [price_trigger, price_setup])
        wick_break = any(value is not None and value <= lower_band - margin for value in [low_trigger, low_setup])
        confirmed = close_break or wick_break
        method = "close_break" if close_break else "wick_break" if wick_break else "inside_band"

    return {
        "required": True,
        "confirmed": confirmed,
        "method": method,
        "margin": round(float(margin), 6),
        "upper_band": upper_band,
        "lower_band": lower_band,
    }



def compute_trigger_confirmation_profile(
    *,
    directional_support: bool,
    continuation_qualified: bool,
    indicator_confirmed: bool,
    band_break_required: bool,
    band_break_confirmed: bool | None,
    momentum_ready: bool,
    micro_ready: bool,
    higher_guardrail_clear: bool,
    trigger_structure: str | None,
    in_entry_zone: bool,
) -> dict:
    structure_score = 0.0
    if trigger_structure in {"trend", "expansion"}:
        structure_score = 1.0
    elif trigger_structure in {"transition", "compression"} and momentum_ready:
        structure_score = 0.5

    band_break_score = 1.0 if (not band_break_required or band_break_confirmed) else 0.0
    zone_score = 0.5 if in_entry_zone else 0.0

    components = {
        "setup_directional_support": 1.0 if directional_support else 0.0,
        "setup_continuation_qualification": 1.0 if continuation_qualified else 0.0,
        "trigger_indicator_confirmation": 2.0 if indicator_confirmed else 0.0,
        "trigger_band_break_confirmation": band_break_score,
        "momentum_alignment": 1.5 if momentum_ready else 0.0,
        "micro_alignment": 1.0 if micro_ready else 0.0,
        "higher_timeframe_guardrail": 1.0 if higher_guardrail_clear else 0.0,
        "structure_quality": structure_score,
        "entry_zone_quality": zone_score,
    }

    confirmation_score = round(min(10.0, sum(components.values())), 2)
    confirmation_level = max(0, min(10, int(round(confirmation_score))))

    return {
        "score": confirmation_score,
        "level": confirmation_level,
        "max_score": 10,
        "stage": ladder_stage_from_score(confirmation_score),
        "components": {key: round(value, 2) for key, value in components.items()},
    }


def build_entry_score(
    *,
    decision: dict,
    context: dict,
    entry_validation: dict,
    execution_plan: dict,
    trigger_context: dict,
) -> dict:
    action = decision.get("action")
    direction = decision.get("strategy_bias")
    results = context["results"]

    confirmation_score = first_valid_number(trigger_context.get("confirmation_score"), trigger_context.get("confirmation_level"), 0.0) or 0.0
    trigger_alignment = min(35, int(round(confirmation_score * 3.5)))

    setup_directional_support = bool(trigger_context.get("setup_support_directional"))
    setup_continuation_support = bool(trigger_context.get("setup_support_continuation"))
    trigger_indicator_confirmed = bool(trigger_context.get("trigger_indicator_confirmed"))
    band_break_required = bool(trigger_context.get("trigger_band_break_required"))
    band_break_confirmed = trigger_context.get("trigger_band_break_confirmed")
    trigger_state = str(trigger_context.get("state") or "")

    if setup_directional_support and not setup_continuation_support:
        trigger_alignment = max(0, trigger_alignment - 6)
    if trigger_indicator_confirmed and band_break_required and band_break_confirmed is False:
        trigger_alignment = max(0, trigger_alignment - 4)
    elif not trigger_indicator_confirmed:
        trigger_alignment = max(0, trigger_alignment - 5)

    if entry_validation.get("allowed") and action in {"enter_long", "enter_short"}:
        trigger_alignment = min(35, trigger_alignment + 3)
    elif action in {"watch_breakout", "watch_breakdown", "watch_reversal"}:
        trigger_alignment = max(0, trigger_alignment - 2)

    state = decision.get("state")
    structure_quality = 10
    if state == "continuation":
        structure_quality = 24
    elif state in {"continuation_watch", "pullback"}:
        structure_quality = 20
    elif state in {"pre_breakout", "pre_breakdown"}:
        structure_quality = 18
    elif state in {"countertrend_long_watch", "countertrend_short_watch"}:
        structure_quality = 11
    elif state == "compression":
        structure_quality = 14
    elif state == "reversal":
        structure_quality = 12
    elif state == "exhaustion_rejection":
        structure_quality = 19

    if trigger_state == "direction_supported_but_setup_unqualified":
        structure_quality = min(structure_quality, 15)
    elif trigger_state == "indicator_break_without_band_break":
        structure_quality = min(structure_quality, 17)

    signal_15m = results.get("15m", {}).get("signal")
    if direction == "long" and signal_15m in BULLISH_SIGNALS:
        structure_quality = min(25, structure_quality + 3)
    elif direction == "short" and signal_15m in BEARISH_SIGNALS:
        structure_quality = min(25, structure_quality + 3)

    reward_to_risk = first_valid_number(execution_plan.get("reward_to_risk_estimate"))
    if reward_to_risk is None:
        risk_reward = 10 if action in {"watch_breakout", "watch_breakdown", "watch_reversal", "wait_pullback"} else 8
    elif reward_to_risk >= 2.2:
        risk_reward = 20
    elif reward_to_risk >= 1.8:
        risk_reward = 18
    elif reward_to_risk >= 1.5:
        risk_reward = 16
    elif reward_to_risk >= 1.2:
        risk_reward = 13
    elif reward_to_risk >= 1.0:
        risk_reward = 10
    elif reward_to_risk >= 0.8:
        risk_reward = 7
    else:
        risk_reward = 4

    timing_state = decision.get("entry_timing")
    timing_quality_map = {
        "immediate": 10,
        "wait_retest": 8,
        "monitor": 6,
        "blocked": 2,
        "timing_wait": 4,
    }
    timing_quality = timing_quality_map.get(timing_state, 5)

    higher_context = 5
    higher_bias = context.get("higher_timeframe_bias")
    if direction == "long":
        if context.get("higher_bearish_pressure"):
            higher_context = 3
        elif higher_bias in {"bullish", "strong_bullish"}:
            higher_context = 10
        elif higher_bias == "neutral":
            higher_context = 6
        else:
            higher_context = 7
    elif direction == "short":
        if context.get("higher_bullish_pressure"):
            higher_context = 3
        elif higher_bias in {"bearish", "strong_bearish"}:
            higher_context = 10
        elif higher_bias == "neutral":
            higher_context = 6
        else:
            higher_context = 7

    components = {
        "trigger_alignment": int(trigger_alignment),
        "structure_quality": int(structure_quality),
        "risk_reward": int(risk_reward),
        "timing_quality": int(timing_quality),
        "higher_timeframe_context": int(higher_context),
    }
    score = int(sum(components.values()))

    if score >= 82:
        tier = "execute"
    elif score >= 68:
        tier = "ready"
    elif score >= 55:
        tier = "watchlist"
    elif score >= 40:
        tier = "exploratory"
    else:
        tier = "no_trade"

    notes: list[str] = []
    if setup_directional_support and not setup_continuation_support:
        notes.append("Directional support exists, but the setup timeframe is not yet continuation-qualified.")
    if trigger_indicator_confirmed and band_break_required and band_break_confirmed is False:
        notes.append("Indicator breakout is active, but price has not cleared the trigger band yet.")
    elif components["trigger_alignment"] < 18:
        notes.append("Trigger alignment is still incomplete, so promotion to execution should wait.")
    if components["risk_reward"] < 10:
        notes.append("Reward-to-risk remains thin, so the trade should be treated tactically.")
    if components["higher_timeframe_context"] <= 4:
        notes.append("Higher timeframe context is leaning against the active idea.")
    if components["timing_quality"] <= 4:
        notes.append("Timing is inefficient right now, so avoid forcing the entry.")

    confidence_modifier = 0
    if score >= 82:
        confidence_modifier = 4
    elif score >= 72:
        confidence_modifier = 2
    elif score >= 60:
        confidence_modifier = 1
    elif score < 40:
        confidence_modifier = -4
    elif score < 50:
        confidence_modifier = -2

    return {
        "score": score,
        "tier": tier,
        "components": components,
        "notes": notes[:3],
        "confidence_modifier": confidence_modifier,
    }
def strategy_is_memory_eligible(strategy: dict, mode_name: str) -> bool:
    if strategy.get("strategy_bias") not in {"long", "short"}:
        return False
    if strategy.get("action") not in {"enter_long", "enter_short", "watch_breakout", "watch_breakdown", "wait_pullback"}:
        return False
    if strategy.get("state") in {"trap", "failed_move", "standby"}:
        return False
    return int(strategy.get("confidence", 0) or 0) >= get_bias_memory_min_confidence(mode_name)



def invalidate_bias_memory_record(record: dict, context: dict) -> bool:
    if not record:
        return True

    mode_name = context.get("mode", DEFAULT_STRATEGY_MODE)
    now_ts = current_utc_timestamp()
    age_seconds = max(0.0, now_ts - float(record.get("stored_at", now_ts)))
    if age_seconds > min(BIAS_MEMORY_MAX_AGE_SECONDS, get_bias_memory_ttl_seconds(mode_name)):
        return True

    results = context["results"]
    bias = record.get("bias")
    signal_5m = results.get("5m", {}).get("signal")
    signal_15m = results.get("15m", {}).get("signal")
    trend_5m = results.get("5m", {}).get("signal_details", {}).get("trend_bias")
    trend_15m = results.get("15m", {}).get("signal_details", {}).get("trend_bias")
    momentum_5m = results.get("5m", {}).get("signal_details", {}).get("momentum_state")
    momentum_15m = results.get("15m", {}).get("signal_details", {}).get("momentum_state")

    if context.get("failure_risk_count", 0) >= 1:
        return True

    if bias == "short":
        if context.get("bear_trap_count", 0) >= 1:
            return True
        if signal_15m in BULLISH_SIGNALS and signal_5m in BULLISH_SIGNALS:
            return True
        if trend_15m == "bullish" and trend_5m == "bullish":
            return True
        if context.get("higher_bullish_pressure") and momentum_5m == "rising" and momentum_15m == "rising":
            return True
    elif bias == "long":
        if context.get("bull_trap_count", 0) >= 1:
            return True
        if signal_15m in BEARISH_SIGNALS and signal_5m in BEARISH_SIGNALS:
            return True
        if trend_15m == "bearish" and trend_5m == "bearish":
            return True
        if context.get("higher_bearish_pressure") and momentum_5m == "falling" and momentum_15m == "falling":
            return True

    return False



def build_bias_persistence_meta(record: dict | None, *, active: bool, source: str | None = None) -> dict | None:
    if not record:
        return None
    now_ts = current_utc_timestamp()
    return {
        "active": active,
        "tracked_bias": record.get("bias"),
        "source": source or record.get("source", "memory_tracking"),
        "age_seconds": int(max(0.0, now_ts - float(record.get("stored_at", now_ts)))),
        "stored_confidence": int(record.get("confidence", 0) or 0),
        "stored_confirmation_level": int(record.get("confirmation_level", 0) or 0),
    }



def store_directional_bias_memory(strategy: dict, context: dict, *, source: str = "live") -> dict | None:
    mode_name = context.get("mode", DEFAULT_STRATEGY_MODE)
    if not strategy_is_memory_eligible(strategy, mode_name):
        return None

    key = build_bias_memory_key(context["symbol"], mode_name)
    trigger_context = strategy.get("trigger_context") or {}
    entry_score = strategy.get("entry_score") or {}
    record = {
        "bias": strategy.get("strategy_bias"),
        "state": strategy.get("state"),
        "action": strategy.get("action"),
        "confidence": int(strategy.get("confidence", 0) or 0),
        "confirmation_level": int(trigger_context.get("confirmation_level", 0) or 0),
        "entry_score": int(entry_score.get("score", 0) or 0),
        "stored_at": current_utc_timestamp(),
        "source": source,
    }
    STRATEGY_BIAS_MEMORY[key] = record
    return record



def build_persisted_bias_payload(strategy: dict, context: dict) -> tuple[dict | None, dict | None]:
    mode_name = context.get("mode", DEFAULT_STRATEGY_MODE)
    key = build_bias_memory_key(context["symbol"], mode_name)
    record = STRATEGY_BIAS_MEMORY.get(key)
    if not record:
        return None, None

    if invalidate_bias_memory_record(record, context):
        STRATEGY_BIAS_MEMORY.pop(key, None)
        return None, None

    if strategy.get("strategy_bias") in {"long", "short"}:
        return None, build_bias_persistence_meta(record, active=False)

    if strategy.get("state") not in {"compression", "standby"}:
        return None, build_bias_persistence_meta(record, active=False)

    results = context["results"]
    signal_15m = results.get("15m", {}).get("signal")
    trend_15m = results.get("15m", {}).get("signal_details", {}).get("trend_bias")
    momentum_15m = results.get("15m", {}).get("signal_details", {}).get("momentum_state")
    momentum_5m = results.get("5m", {}).get("signal_details", {}).get("momentum_state")

    bias = record.get("bias")
    age_meta = build_bias_persistence_meta(record, active=True, source="memory_override")

    if bias == "short":
        still_supported = (
            signal_15m in BEARISH_SIGNALS
            or trend_15m == "bearish"
            or momentum_15m in {"falling", "flat"}
            or momentum_5m in {"falling", "flat"}
        )
        if not still_supported:
            return None, build_bias_persistence_meta(record, active=False)

        payload = {
            "strategy_bias": "short",
            "state": "pre_breakdown",
            "action": "watch_breakdown",
            "setup_timeframes": ["15m"],
            "trigger_timeframes": ["5m"],
            "risk_timeframes": ["5m", "15m"],
            "risk_state": "mixed",
            "summary": (
                f'{context["symbol"]} is still carrying a bearish directional bias from the prior setup cycle, even though the current snapshot has compressed. '
                f'Treat this as a persistent lead-short watch and require a weak bounce failure or fresh 5m breakdown before promoting execution.'
            ),
            "bias_origin": "persisted_directional_bias",
            "bias_score": round(max(5.2, (record.get("confirmation_level", 0) or 0) * 0.8), 1),
            "bias_reasons": [
                "A prior bearish setup cycle is still active in memory.",
                "The current pullback has not invalidated the 15m downside idea.",
                "Wait for the 5m trigger to fail the bounce before reactivating the short.",
            ],
            "bias_confidence": max(50, min(78, int(record.get("confidence", 0) or 0) - 4)),
            "bias_edge": 1.0,
        }
        return payload, age_meta

    if bias == "long":
        still_supported = (
            signal_15m in BULLISH_SIGNALS
            or trend_15m == "bullish"
            or momentum_15m in {"rising", "flat"}
            or momentum_5m in {"rising", "flat"}
        )
        if not still_supported:
            return None, build_bias_persistence_meta(record, active=False)

        payload = {
            "strategy_bias": "long",
            "state": "pre_breakout",
            "action": "watch_breakout",
            "setup_timeframes": ["15m"],
            "trigger_timeframes": ["5m"],
            "risk_timeframes": ["5m", "15m"],
            "risk_state": "mixed",
            "summary": (
                f'{context["symbol"]} is still carrying a bullish directional bias from the prior setup cycle, even though the current snapshot has compressed. '
                f'Treat this as a persistent lead-long watch and require a pullback hold or fresh 5m breakout before promoting execution.'
            ),
            "bias_origin": "persisted_directional_bias",
            "bias_score": round(max(5.2, (record.get("confirmation_level", 0) or 0) * 0.8), 1),
            "bias_reasons": [
                "A prior bullish setup cycle is still active in memory.",
                "The current pullback has not invalidated the 15m upside idea.",
                "Wait for the 5m trigger to hold the pullback before reactivating the long.",
            ],
            "bias_confidence": max(50, min(78, int(record.get("confidence", 0) or 0) - 4)),
            "bias_edge": 1.0,
        }
        return payload, age_meta

    return None, build_bias_persistence_meta(record, active=False)





def build_trigger_context(
    *,
    decision: dict,
    context: dict,
    entry_validation: dict,
    execution_plan: dict | None = None,
) -> dict:
    results = context["results"]
    direction = decision.get("strategy_bias")
    action = decision.get("action")

    trigger_tf = decision.get("trigger_timeframes", ["5m"])[0] if decision.get("trigger_timeframes") else "5m"
    setup_tf = decision.get("setup_timeframes", ["15m"])[0] if decision.get("setup_timeframes") else "15m"

    trigger = results.get(trigger_tf, {})
    setup = results.get(setup_tf, {})
    micro = results.get("1m", {})

    trigger_signal = trigger.get("signal")
    setup_signal = setup.get("signal")
    setup_name = setup.get("setup")
    trigger_structure = trigger.get("structure")
    setup_structure = setup.get("structure")
    trigger_momentum = trigger.get("signal_details", {}).get("momentum_state")
    setup_details = setup.get("signal_details", {})
    setup_momentum = setup_details.get("momentum_state")
    micro_signal = micro.get("signal")
    micro_momentum = micro.get("signal_details", {}).get("momentum_state")
    trigger_price = first_valid_number(trigger.get("price"), setup.get("price"))

    entry_zone = execution_plan.get("entry_zone") if execution_plan else None
    zone_low = first_valid_number((entry_zone or {}).get("low"))
    zone_high = first_valid_number((entry_zone or {}).get("high"))
    in_entry_zone = False
    if trigger_price is not None and zone_low is not None and zone_high is not None:
        in_entry_zone = zone_low <= trigger_price <= zone_high

    if direction not in {"long", "short"}:
        ladder = compute_trigger_confirmation_profile(
            directional_support=False,
            continuation_qualified=False,
            indicator_confirmed=False,
            band_break_required=False,
            band_break_confirmed=None,
            momentum_ready=False,
            micro_ready=False,
            higher_guardrail_clear=False,
            trigger_structure=trigger_structure,
            in_entry_zone=in_entry_zone,
        )
        return {
            "setup_timeframe": setup_tf,
            "trigger_timeframe": trigger_tf,
            "state": "neutral",
            "confirmation_level": 0,
            "confirmation_score": 0.0,
            "confirmation_max": 10,
            "ladder_stage": "inactive",
            "activation_ready": False,
            "setup_support_directional": False,
            "setup_support_continuation": False,
            "trigger_indicator_confirmed": False,
            "trigger_band_break_required": False,
            "trigger_band_break_confirmed": None,
            "blocking_flags": ["neutral_bias"],
            "components": ladder["components"],
            "reasons": ["Directional trigger context is inactive while the strategy bias is neutral."],
        }

    setup_directional_support = directional_setup_supports(
        direction=direction,
        setup_signal=setup_signal,
        setup_name=setup_name,
        structure=setup_structure,
        trend_bias=setup_details.get("trend_bias"),
        momentum_state=setup_momentum,
        rsi_state=setup_details.get("rsi_state"),
    )
    setup_continuation_support = continuation_setup_qualified(
        direction=direction,
        setup_signal=setup_signal,
        setup_name=setup_name,
        structure=setup_structure,
        trend_bias=setup_details.get("trend_bias"),
        momentum_state=setup_momentum,
    )
    trigger_indicator_confirmed = trigger_indicator_confirms_direction(
        direction=direction,
        trigger_signal=trigger_signal,
        activation_source=decision.get("activation_source"),
        compression_breakout_confirmed=bool(decision.get("compression_breakout_confirmed")),
        compression_break_direction=decision.get("compression_break_direction"),
    )

    band_break = detect_trigger_band_break(
        direction=direction,
        trigger=trigger,
        setup=setup,
        entry_zone=entry_zone,
        entry_model=(execution_plan or {}).get("entry_model"),
    )
    band_break_required = bool(band_break.get("required"))
    band_break_confirmed = band_break.get("confirmed")

    if direction == "long":
        momentum_ready = trigger_momentum in {"rising", "flat"}
        micro_ready = micro_signal in BULLISH_SIGNALS or micro_momentum == "rising"
        higher_guardrail_clear = not context.get("higher_bearish_pressure", False)
        side_label = "bullish"
    else:
        momentum_ready = trigger_momentum in {"falling", "flat"}
        micro_ready = micro_signal in BEARISH_SIGNALS or micro_momentum == "falling"
        higher_guardrail_clear = not context.get("higher_bullish_pressure", False)
        side_label = "bearish"

    ladder = compute_trigger_confirmation_profile(
        directional_support=setup_directional_support,
        continuation_qualified=setup_continuation_support,
        indicator_confirmed=trigger_indicator_confirmed,
        band_break_required=band_break_required,
        band_break_confirmed=band_break_confirmed,
        momentum_ready=momentum_ready,
        micro_ready=micro_ready,
        higher_guardrail_clear=higher_guardrail_clear,
        trigger_structure=trigger_structure,
        in_entry_zone=in_entry_zone,
    )

    blocking_flags: list[str] = []
    if not setup_directional_support:
        blocking_flags.append("no_directional_setup_support")
    if setup_directional_support and not setup_continuation_support:
        blocking_flags.append("setup_not_continuation_qualified")
    if not trigger_indicator_confirmed:
        blocking_flags.append("trigger_indicator_unconfirmed")
    if trigger_indicator_confirmed and band_break_required and band_break_confirmed is False:
        blocking_flags.append("indicator_without_band_break")
    if not higher_guardrail_clear:
        blocking_flags.append("higher_timeframe_guardrail_opposition")
    if entry_zone and not in_entry_zone and not (band_break_required and band_break_confirmed):
        blocking_flags.append("outside_entry_zone")

    strategy_state = decision.get("state")
    if action in {"enter_long", "enter_short"} and entry_validation.get("allowed"):
        state = "confirmed"
    elif strategy_state == "exhaustion_rejection":
        state = "exhaustion_rejection_watch"
    elif action == "wait_pullback":
        state = "retest_wait"
    elif strategy_state in {"countertrend_long_watch", "countertrend_short_watch"}:
        state = strategy_state
    elif not setup_directional_support:
        state = "no_directional_setup_support"
    elif setup_directional_support and not setup_continuation_support:
        state = "direction_supported_but_setup_unqualified"
    elif trigger_indicator_confirmed and band_break_required and band_break_confirmed is False:
        state = "indicator_break_without_band_break"
    elif setup_continuation_support and not trigger_indicator_confirmed:
        state = "setup_qualified_waiting_for_trigger"
    elif setup_continuation_support and trigger_indicator_confirmed and not entry_validation.get("allowed"):
        state = "execution_blocked"
    elif setup_continuation_support and trigger_indicator_confirmed:
        state = "confirmed"
    else:
        state = "trigger_alignment_incomplete"

    reasons: list[str] = []
    if state == "exhaustion_rejection_watch":
        reasons.append(f"{setup_tf} is monitoring a failed upside push; this is a tactical exhaustion fade, not a normal trend short.")
    elif state == "countertrend_short_watch":
        reasons.append(f"{setup_tf} is still structurally bullish, so the short idea is countertrend until sellers prove control.")
    elif state == "countertrend_long_watch":
        reasons.append(f"{setup_tf} is still structurally bearish, so the long idea is countertrend until buyers prove control.")
    elif setup_continuation_support:
        reasons.append(f"{setup_tf} is continuation-qualified on the {side_label} side.")
    elif setup_directional_support:
        reasons.append(f"{setup_tf} supports the {side_label} direction, but it is not yet continuation-qualified.")
    else:
        reasons.append(f"{setup_tf} does not yet support the {side_label} direction.")

    if trigger_indicator_confirmed and band_break_required and band_break_confirmed is False:
        reasons.append(f"{trigger_tf} has the {side_label} indicator trigger, but price is still inside the compression trigger band.")
    elif trigger_indicator_confirmed:
        reasons.append(f"{trigger_tf} has confirmed the {side_label} indicator trigger.")
    else:
        reasons.append(f"{trigger_tf} has not confirmed the {side_label} trigger yet.")

    if band_break_required:
        if band_break_confirmed:
            reasons.append(f"Price has already cleared the monitored trigger band on the {side_label} side.")
        else:
            reasons.append(f"A clean break outside the monitored trigger band is still required before promotion.")

    if direction == "long":
        if momentum_ready:
            reasons.append(f"{trigger_tf} momentum is {trigger_momentum or 'unknown'}, so buyers still have upside pressure.")
        else:
            reasons.append(f"{trigger_tf} momentum is {trigger_momentum or 'unknown'}, so buyers still need cleaner follow-through.")
    else:
        if momentum_ready:
            reasons.append(f"{trigger_tf} momentum is {trigger_momentum or 'unknown'}, so sellers still have downside pressure.")
        else:
            reasons.append(f"{trigger_tf} momentum is {trigger_momentum or 'unknown'}, so sellers still need cleaner follow-through.")

    if not higher_guardrail_clear:
        reasons.append("Higher timeframe pressure is still leaning against the active directional idea.")
    elif entry_zone and zone_low is not None and zone_high is not None:
        reasons.append(f"Watch the monitored trigger band for the next activation test.")

    return {
        "setup_timeframe": setup_tf,
        "trigger_timeframe": trigger_tf,
        "setup_signal": setup_signal,
        "setup": setup_name,
        "trigger_signal": trigger_signal,
        "trigger_structure": trigger_structure,
        "state": state,
        "confirmation_level": ladder["level"],
        "confirmation_score": ladder["score"],
        "confirmation_max": ladder["max_score"],
        "ladder_stage": ladder["stage"],
        "activation_ready": bool(entry_validation.get("allowed")) and action in {"enter_long", "enter_short"},
        "setup_support_directional": setup_directional_support,
        "setup_support_continuation": setup_continuation_support,
        "trigger_indicator_confirmed": trigger_indicator_confirmed,
        "trigger_band_break_required": band_break_required,
        "trigger_band_break_confirmed": band_break_confirmed,
        "trigger_band_break_method": band_break.get("method"),
        "blocking_flags": blocking_flags,
        "activation_blockers": blocking_flags,
        "primary_blocker": blocking_flags[0] if blocking_flags else None,
        "components": ladder["components"],
        "in_entry_zone": in_entry_zone,
        "reasons": reasons[:5],
    }


def direction_matches_bias(direction: str | None, bias: str | None) -> bool:
    if direction == "long":
        return bias in {"bullish", "strong_bullish", "long"}
    if direction == "short":
        return bias in {"bearish", "strong_bearish", "short"}
    return False


def direction_opposes_bias(direction: str | None, bias: str | None) -> bool:
    if direction == "long":
        return bias in {"bearish", "strong_bearish", "short"}
    if direction == "short":
        return bias in {"bullish", "strong_bullish", "long"}
    return False


def clamp_lens_score(value: float) -> int:
    return int(max(0, min(100, round(value))))


def build_execution_lens(
    *,
    decision: dict,
    context: dict,
    entry_validation: dict,
    execution_plan: dict,
    trigger_context: dict,
    entry_score: dict,
    confidence: int,
    trade_quality: str,
) -> dict:
    direction = execution_plan.get("direction")
    if direction not in {"long", "short"}:
        direction = decision.get("strategy_bias") if decision.get("strategy_bias") in {"long", "short"} else "neutral"

    entry_score_value = first_valid_number(entry_score.get("score")) or 0
    trigger_level = first_valid_number(trigger_context.get("confirmation_level")) or 0
    trigger_max = first_valid_number(trigger_context.get("confirmation_max")) or 10
    trigger_ratio = trigger_level / trigger_max if trigger_max else 0
    hard_fail_count = int(entry_validation.get("hard_fail_count", 0) or 0)
    entry_allowed = bool(entry_validation.get("allowed"))
    execution_ready = bool(execution_plan.get("execution_ready"))
    trigger_confirmed = bool(trigger_context.get("trigger_indicator_confirmed"))
    band_required = bool(trigger_context.get("trigger_band_break_required"))
    band_confirmed = trigger_context.get("trigger_band_break_confirmed")
    band_ok = (not band_required) or bool(band_confirmed)

    higher_bias = context.get("higher_timeframe_bias")
    overall_bias = context.get("overall_bias")
    short_bias = context.get("short_term_bias")
    higher_aligned = direction_matches_bias(direction, higher_bias)
    higher_opposed = direction_opposes_bias(direction, higher_bias)
    overall_aligned = direction_matches_bias(direction, overall_bias)
    short_aligned = direction_matches_bias(direction, short_bias)

    state = decision.get("state")
    action = decision.get("action")
    weak_scalp_states = {"compression", "continuation_watch", "reversal", "pullback", "exhaustion_rejection"}
    continuation_ready_states = {"continuation", "pre_breakout", "pre_breakdown"}

    scalp_score = (
        entry_score_value * 0.45
        + confidence * 0.20
        + trigger_ratio * 18
        + (18 if entry_allowed else 0)
        + (10 if trigger_confirmed else 0)
        + (6 if band_ok else -10)
        + (6 if short_aligned else 0)
        - hard_fail_count * 7
    )
    if state in weak_scalp_states and not entry_allowed:
        scalp_score -= 10
    if higher_opposed:
        scalp_score -= 8
    scalp_score = clamp_lens_score(scalp_score)

    if direction == "neutral":
        scalp_state = "standby"
        scalp_label = "No Scalp"
        scalp_instruction = "No directional scalp is active."
    elif execution_ready or (entry_allowed and trigger_confirmed and entry_score_value >= 55 and band_ok):
        scalp_state = "ready"
        scalp_label = "Scalp Ready"
        scalp_instruction = "Short-term trigger is qualified; use the execution plan instead of chasing."
    elif entry_allowed and entry_score_value >= 50:
        scalp_state = "limit_wait"
        scalp_label = "Limit Wait"
        scalp_instruction = "Entry is permitted, but the better trade is still the planned limit band."
    elif action in {"watch_breakout", "watch_breakdown", "watch_reversal", "wait_pullback"} and entry_score_value >= 45:
        scalp_state = "wait_confirmation"
        scalp_label = "No Scalp Yet"
        scalp_instruction = "Direction is forming, but the short-term trigger is not clean enough for immediate execution."
    else:
        scalp_state = "blocked"
        scalp_label = "No Scalp"
        scalp_instruction = "Short-term timing is blocked; do not treat the directional read as an entry."

    bias_confidence = first_valid_number(decision.get("bias_confidence")) or confidence
    directional_score = (
        confidence * 0.30
        + bias_confidence * 0.25
        + entry_score_value * 0.20
        + (14 if higher_aligned else 0)
        + (10 if overall_aligned else 0)
        + (8 if state in continuation_ready_states else 0)
        + (6 if state in {"exhaustion_rejection", "capitulation_bounce"} else 0)
        - (18 if higher_opposed else 0)
    )
    directional_score = clamp_lens_score(directional_score)

    if direction == "neutral" or directional_score < 40:
        directional_state = "neutral"
        directional_label = "No Directional Edge"
        directional_instruction = "Directional edge is not strong enough to plan around."
    elif directional_score >= 65:
        directional_state = "active_bias"
        directional_label = "Directional Bias"
        directional_instruction = "Directional read is usable, but execution still depends on the scalp lane."
    else:
        directional_state = "watch"
        directional_label = "Directional Watch"
        directional_instruction = "Directional read is worth tracking while waiting for cleaner short-term timing."

    if scalp_state in {"ready", "limit_wait"}:
        primary_lane = "scalp"
        headline = f"{scalp_label.upper()} - {direction.upper()}"
        summary = scalp_instruction
    elif directional_state in {"active_bias", "watch"}:
        primary_lane = "directional_watch"
        headline = f"{directional_label.upper()} - {direction.upper()}"
        summary = directional_instruction
    else:
        primary_lane = "stand_aside"
        headline = "STAND ASIDE"
        summary = "Neither short-term execution nor longer-horizon direction is strong enough."

    notes = [
        scalp_instruction,
        directional_instruction,
    ]
    if state in weak_scalp_states and not entry_allowed:
        notes.append("Recent v36 behavior favors treating this state as a watch lane until the trigger confirms.")
    if higher_opposed:
        notes.append("Higher timeframe pressure is leaning against the active direction.")

    return {
        "primary_lane": primary_lane,
        "headline": headline,
        "summary": summary,
        "direction": direction,
        "timeframes": {
            "scalp": ["5m", "15m"],
            "directional": ["1h", "4h"],
        },
        "scalp": {
            "state": scalp_state,
            "label": scalp_label,
            "score": scalp_score,
            "window": "15-60m",
            "instruction": scalp_instruction,
            "ready": scalp_state in {"ready", "limit_wait"},
            "blockers": list(entry_validation.get("blocking_reasons", []) or [])[:4],
        },
        "directional": {
            "state": directional_state,
            "label": directional_label,
            "score": directional_score,
            "window": "4h watch",
            "instruction": directional_instruction,
            "higher_timeframe_aligned": higher_aligned,
            "overall_aligned": overall_aligned,
        },
        "evidence": {
            "entry_score": entry_score_value,
            "confidence": confidence,
            "trade_quality": trade_quality,
            "trigger_level": trigger_level,
            "hard_fail_count": hard_fail_count,
            "higher_timeframe_bias": higher_bias,
            "overall_bias": overall_bias,
            "state": state,
            "action": action,
        },
        "notes": notes[:4],
    }


def finalize_strategy_decision(decision: dict, context: dict) -> dict:
    mode_config = context.get("mode_config", get_strategy_mode_config(DEFAULT_STRATEGY_MODE))
    raw_strategy_action = decision["action"]
    raw_strategy_state = decision["state"]
    downgrade_reasons: list[str] = []

    def evaluate_current_entry() -> dict:
        return evaluate_entry_permission(
            action=decision["action"],
            strategy_bias=decision["strategy_bias"],
            results=context["results"],
            short_breakdown_count=context["short_breakdown_count"],
            short_breakout_count=context["short_breakout_count"],
            higher_bullish_pressure=context["higher_bullish_pressure"],
            higher_bearish_pressure=context["higher_bearish_pressure"],
            bull_trap_count=context["bull_trap_count"],
            bear_trap_count=context["bear_trap_count"],
            failure_risk_count=context["failure_risk_count"],
            transition_cluster=context["transition_cluster"],
            activation_source=decision.get("activation_source"),
            compression_breakout_confirmed=bool(decision.get("compression_breakout_confirmed")),
            mode_config=mode_config,
        )

    entry_validation = evaluate_current_entry()

    if decision["action"] == "enter_long" and not entry_validation["allowed"]:
        decision["action"] = "watch_breakout"
        decision["state"] = "continuation_watch"
        decision["risk_state"] = "mixed"
        downgrade_reasons.append("entry_validation")
    elif decision["action"] == "enter_short" and not entry_validation["allowed"]:
        decision["action"] = "watch_breakdown"
        decision["state"] = "continuation_watch"
        decision["risk_state"] = "mixed"
        downgrade_reasons.append("entry_validation")

    entry_validation = evaluate_current_entry()
    execution_plan = build_execution_plan(decision, context, entry_validation)

    activation_decision = detect_compression_breakout_activation(decision, context, execution_plan)
    if activation_decision:
        decision.update(activation_decision)
        entry_validation = evaluate_current_entry()

        if decision["action"] == "enter_long" and not entry_validation["allowed"]:
            decision["action"] = "watch_breakout"
            decision["state"] = "continuation_watch"
            decision["risk_state"] = "mixed"
            if "entry_validation" not in downgrade_reasons:
                downgrade_reasons.append("entry_validation")
        elif decision["action"] == "enter_short" and not entry_validation["allowed"]:
            decision["action"] = "watch_breakdown"
            decision["state"] = "continuation_watch"
            decision["risk_state"] = "mixed"
            if "entry_validation" not in downgrade_reasons:
                downgrade_reasons.append("entry_validation")

        entry_validation = evaluate_current_entry()
        execution_plan = build_execution_plan(decision, context, entry_validation)

    candidate_action = decision["action"]
    candidate_state = decision["state"]

    timing_assessment = assess_entry_timing(
        decision=decision,
        context=context,
        entry_validation=entry_validation,
        execution_plan=execution_plan,
    )
    decision["entry_timing"] = timing_assessment["timing_state"]

    if timing_assessment["should_wait"] and decision["action"] in {"enter_long", "enter_short"}:
        decision["action"] = "wait_pullback"
        decision["state"] = "pullback"
        decision["risk_state"] = "mixed"
        if "timing_wait" not in downgrade_reasons:
            downgrade_reasons.append("timing_wait")

        entry_validation = evaluate_current_entry()
        entry_validation = apply_timing_wait_to_entry_validation(entry_validation, timing_assessment)
        execution_plan = build_execution_plan(decision, context, entry_validation)
    else:
        if decision["action"] == "wait_pullback":
            entry_validation = apply_timing_wait_to_entry_validation(entry_validation, timing_assessment)
        execution_plan = build_execution_plan(decision, context, entry_validation)

    execution_plan["execution_management"] = build_execution_management(
        decision=decision,
        execution_plan=execution_plan,
        context=context,
    )

    trigger_context = build_trigger_context(
        decision=decision,
        context=context,
        entry_validation=entry_validation,
        execution_plan=execution_plan,
    )
    entry_validation = {
        **entry_validation,
        "status_detail": trigger_context.get("state"),
        "qualification": {
            **(entry_validation.get("qualification") or {}),
            "setup_support_directional": trigger_context.get("setup_support_directional"),
            "setup_support_continuation": trigger_context.get("setup_support_continuation"),
            "trigger_indicator_confirmed": trigger_context.get("trigger_indicator_confirmed"),
            "trigger_band_break_required": trigger_context.get("trigger_band_break_required"),
            "trigger_band_break_confirmed": trigger_context.get("trigger_band_break_confirmed"),
        },
    }

    components = build_strategy_confidence_components(
        strategy_bias=decision["strategy_bias"],
        action=decision["action"],
        state=decision.get("state"),
        short_term_bias=context["short_term_bias"],
        higher_timeframe_bias=context["higher_timeframe_bias"],
        overall_bias=context["overall_bias"],
        short_breakdown_count=context["short_breakdown_count"],
        short_breakout_count=context["short_breakout_count"],
        compression_cluster=context["compression_cluster"],
        transition_cluster=context["transition_cluster"],
        reversal_setup_count=context["reversal_setup_count"],
        bull_trap_count=context["bull_trap_count"],
        bear_trap_count=context["bear_trap_count"],
        failure_risk_count=context["failure_risk_count"],
        higher_bullish_pressure=context["higher_bullish_pressure"],
        higher_bearish_pressure=context["higher_bearish_pressure"],
        results=context["results"],
    )
    components = apply_blocked_component_caps(
        components=components,
        entry_validation=entry_validation,
        decision=decision,
        mode_config=mode_config,
    )

    confidence = combine_strategy_confidence(
        components=components,
        action=decision["action"],
        strategy_bias=decision["strategy_bias"],
        entry_permission=entry_validation["allowed"],
        short_breakdown_count=context["short_breakdown_count"],
        short_breakout_count=context["short_breakout_count"],
        compression_cluster=context["compression_cluster"],
        transition_cluster=context["transition_cluster"],
        reversal_setup_count=context["reversal_setup_count"],
        bull_trap_count=context["bull_trap_count"],
        bear_trap_count=context["bear_trap_count"],
        failure_risk_count=context["failure_risk_count"],
        mode_config=mode_config,
    )
    confidence = apply_blocked_confidence_cap(
        confidence=confidence,
        entry_validation=entry_validation,
        decision=decision,
        mode_config=mode_config,
    )

    entry_score = build_entry_score(
        decision=decision,
        context=context,
        entry_validation=entry_validation,
        execution_plan=execution_plan,
        trigger_context=trigger_context,
    )
    confidence = clamp_confidence(confidence + int(entry_score.get("confidence_modifier", 0) or 0))

    market_phase = infer_market_phase(
        state=decision["state"],
        action=decision["action"],
        strategy_bias=decision["strategy_bias"],
        overall_bias=context["overall_bias"],
        compression_cluster=context["compression_cluster"],
        transition_cluster=context["transition_cluster"],
        reversal_setup_count=context["reversal_setup_count"],
        bull_trap_count=context["bull_trap_count"],
        bear_trap_count=context["bear_trap_count"],
    )

    decision["summary"] = build_actionable_strategy_summary(decision, context)

    execution_notes = execution_plan.get("notes", []) if isinstance(execution_plan, dict) else []
    extension_risk = any("extension" in str(note).lower() or "bounce" in str(note).lower() for note in execution_notes)

    trade_quality = infer_trade_quality(
        action=decision["action"],
        state=decision["state"],
        strategy_bias=decision["strategy_bias"],
        confidence=confidence,
        entry_permission=entry_validation["allowed"],
        reward_to_risk_estimate=execution_plan.get("reward_to_risk_estimate"),
        extension_risk=extension_risk,
    )

    entry_score_value = int(entry_score.get("score", 0) or 0)
    if decision["action"] in {"enter_long", "enter_short"}:
        if entry_score_value < 45:
            trade_quality = "D"
        elif entry_score_value < 55 and trade_quality in {"A", "B"}:
            trade_quality = "C"
    elif decision["action"] in {"watch_breakout", "watch_breakdown", "watch_reversal"} and entry_score_value < 42:
        trade_quality = "D"

    execution_plan = {
        **execution_plan,
        **infer_execution_playbook(
            decision=decision,
            execution_plan=execution_plan,
            confidence=confidence,
            trade_quality=trade_quality,
            market_phase=market_phase,
        ),
        "entry_score": entry_score,
    }
    execution_plan["execution_management"] = build_execution_management(
        decision=decision,
        execution_plan=execution_plan,
        context=context,
    )
    execution_review = build_execution_review(
        decision=decision,
        context=context,
        execution_plan=execution_plan,
    )
    limit_order_plan = build_limit_order_plan(
        decision=decision,
        context=context,
        execution_plan=execution_plan,
        execution_review=execution_review,
        entry_validation=entry_validation,
        entry_score=entry_score,
    )
    execution_plan["execution_review"] = execution_review
    execution_plan["limit_order_plan"] = limit_order_plan
    execution_lens = build_execution_lens(
        decision=decision,
        context=context,
        entry_validation=entry_validation,
        execution_plan=execution_plan,
        trigger_context=trigger_context,
        entry_score=entry_score,
        confidence=confidence,
        trade_quality=trade_quality,
    )
    execution_plan["execution_lens"] = execution_lens

    return {
        "mode": mode_config["name"],
        "strategy_bias": decision["strategy_bias"],
        "bias_origin": decision.get("bias_origin"),
        "bias_score": decision.get("bias_score"),
        "bias_reasons": decision.get("bias_reasons", []),
        "bias_confidence": decision.get("bias_confidence"),
        "bias_edge": decision.get("bias_edge"),
        "market_phase": market_phase,
        "state": decision["state"],
        "action": decision["action"],
        "raw_strategy_action": raw_strategy_action,
        "raw_strategy_state": raw_strategy_state,
        "candidate_action": candidate_action,
        "candidate_state": candidate_state,
        "downgraded_from": candidate_action if candidate_action != decision["action"] else None,
        "downgrade_reasons": downgrade_reasons,
        "entry_timing": decision.get("entry_timing", "monitor"),
        "trade_quality": trade_quality,
        "setup_timeframes": decision["setup_timeframes"],
        "trigger_timeframes": decision["trigger_timeframes"],
        "risk_timeframes": decision["risk_timeframes"],
        "risk_state": decision["risk_state"],
        "summary": decision["summary"],
        "trigger_context": trigger_context,
        "entry_score": entry_score,
        "confidence": confidence,
        "execution_confidence": confidence,
        "confidence_components": components,
        "entry_permission": entry_validation["allowed"],
        "entry_validation": entry_validation,
        "execution_lens": execution_lens,
        "execution_review": execution_review,
        "limit_order_plan": limit_order_plan,
        "execution_plan": execution_plan,
    }



def build_strategy_summary(

    short_term_bias: str,
    higher_timeframe_bias: str,
    overall_bias: str,
    symbol: str,
    results: dict,
    mode: str = DEFAULT_STRATEGY_MODE,
) -> dict:
    mode_config = get_strategy_mode_config(mode)
    signal_5m = results.get("5m", {}).get("signal")
    signal_15m = results.get("15m", {}).get("signal")
    signal_1h = results.get("1h", {}).get("signal")
    signal_4h = results.get("4h", {}).get("signal")
    signal_1d = results.get("1d", {}).get("signal")

    structure_5m = results.get("5m", {}).get("structure")
    structure_15m = results.get("15m", {}).get("structure")
    structure_1h = results.get("1h", {}).get("structure")
    structure_4h = results.get("4h", {}).get("structure")

    setup_5m = results.get("5m", {}).get("setup")
    setup_15m = results.get("15m", {}).get("setup")
    setup_1h = results.get("1h", {}).get("setup")
    setup_4h = results.get("4h", {}).get("setup")
    setup_1d = results.get("1d", {}).get("setup")

    momentum_5m = results.get("5m", {}).get("signal_details", {}).get("momentum_state")
    momentum_15m = results.get("15m", {}).get("signal_details", {}).get("momentum_state")

    active_traps = {tf: results.get(tf, {}).get("trap_risk") for tf in ACTIVE_TIMEFRAMES}

    bull_trap_count = sum(1 for t in active_traps.values() if t == "bull_trap_risk")
    bear_trap_count = sum(1 for t in active_traps.values() if t == "bear_trap_risk")
    failure_risk_count = sum(1 for t in active_traps.values() if t in FAILURE_RISKS)

    short_breakdown_count = sum(1 for s in [signal_5m, signal_15m] if s == "bearish_breakdown")
    short_breakout_count = sum(1 for s in [signal_5m, signal_15m] if s == "bullish_breakout")
    transition_cluster = sum(1 for s in [structure_5m, structure_15m, structure_1h] if s == "transition")
    compression_cluster = sum(
        1 for s in [structure_5m, structure_15m, structure_1h, structure_4h] if s == "compression"
    )
    reversal_setup_count = sum(
        1 for s in [setup_5m, setup_15m, setup_1h, setup_4h, setup_1d] if s in REVERSAL_SETUPS
    )

    higher_bearish_pressure = (
        signal_1d in BEARISH_SIGNALS
        or signal_4h in BEARISH_SIGNALS
        or setup_4h in {"Overbought Reversal Risk", "Early Bearish Reversal"}
        or (
            structure_1h == "transition"
            and signal_1h == "overbought_pullback_watch"
            and results.get("1h", {}).get("signal_details", {}).get("trend_bias") in {"bearish", "mixed"}
        )
    )

    higher_bullish_pressure = (
        (
            signal_1h in BULLISH_SIGNALS
            and setup_4h not in {"Overbought Reversal Risk", "Early Bearish Reversal"}
            and signal_1d not in {"bearish_breakdown"}
        )
        or (
            structure_1h == "transition"
            and signal_1h == "bearish_but_bouncing"
            and results.get("1h", {}).get("signal_details", {}).get("trend_bias") in {"bullish", "mixed"}
        )
    )

    active_trap_weights = {"5m": 1.5, "15m": 2.0, "1h": 2.0}
    bull_trap_weight = sum(
        active_trap_weights.get(tf, 1.0)
        for tf, trap in active_traps.items()
        if trap == "bull_trap_risk"
    )
    bear_trap_weight = sum(
        active_trap_weights.get(tf, 1.0)
        for tf, trap in active_traps.items()
        if trap == "bear_trap_risk"
    )

    early_bias = infer_early_directional_bias(
        results=results,
        mode_config=mode_config,
        compression_cluster=compression_cluster,
        transition_cluster=transition_cluster,
        reversal_setup_count=reversal_setup_count,
        higher_bullish_pressure=higher_bullish_pressure,
        higher_bearish_pressure=higher_bearish_pressure,
        bull_trap_count=bull_trap_count,
        bear_trap_count=bear_trap_count,
        short_breakout_count=short_breakout_count,
        short_breakdown_count=short_breakdown_count,
    )

    context = {
        "symbol": symbol,
        "mode": mode_config["name"],
        "mode_config": mode_config,
        "short_term_bias": short_term_bias,
        "higher_timeframe_bias": higher_timeframe_bias,
        "overall_bias": overall_bias,
        "short_breakdown_count": short_breakdown_count,
        "short_breakout_count": short_breakout_count,
        "compression_cluster": compression_cluster,
        "transition_cluster": transition_cluster,
        "reversal_setup_count": reversal_setup_count,
        "bull_trap_count": bull_trap_count,
        "bear_trap_count": bear_trap_count,
        "failure_risk_count": failure_risk_count,
        "higher_bearish_pressure": higher_bearish_pressure,
        "higher_bullish_pressure": higher_bullish_pressure,
        "results": results,
    }

    trend_bias_15m = results.get("15m", {}).get("signal_details", {}).get("trend_bias")
    trend_bias_4h = results.get("4h", {}).get("signal_details", {}).get("trend_bias")

    fifteen_bullish_structure = (
        signal_15m in BULLISH_SIGNALS
        or setup_15m in {"Bullish Pullback", "Trend Continuation", "Bullish Pressure Build", "Early Bullish Reversal", "Oversold Reversal", "Bullish Exhaustion Reversal"}
        or trend_bias_15m == "bullish"
    )
    fifteen_bearish_structure = (
        signal_15m in BEARISH_SIGNALS
        or setup_15m in {"Bearish Bounce Attempt", "Trend Continuation", "Early Bearish Reversal", "Overbought Reversal Risk", "Bearish Exhaustion"}
        or trend_bias_15m == "bearish"
    )
    higher_bullish_structure = (
        signal_4h in BULLISH_SIGNALS
        or setup_4h in {"Trend Continuation", "Bullish Pullback", "Bullish Pressure Build"}
        or trend_bias_4h == "bullish"
    )
    higher_bearish_structure = (
        signal_4h in BEARISH_SIGNALS
        or setup_4h in {"Trend Continuation", "Bearish Bounce Attempt", "Early Bearish Reversal", "Overbought Reversal Risk"}
        or trend_bias_4h == "bearish"
    )
    real_bullish_trigger = signal_5m in BULLISH_SIGNALS or short_breakout_count >= 1
    real_bearish_trigger = signal_5m in BEARISH_SIGNALS or short_breakdown_count >= 1

    def decision(
        *,
        strategy_bias: str,
        state: str,
        action: str,
        setup_timeframes: list[str],
        trigger_timeframes: list[str],
        risk_timeframes: list[str],
        risk_state: str,
        summary: str,
        **extra_fields,
    ) -> dict:
        payload = {
            "strategy_bias": strategy_bias,
            "state": state,
            "action": action,
            "setup_timeframes": setup_timeframes,
            "trigger_timeframes": trigger_timeframes,
            "risk_timeframes": risk_timeframes,
            "risk_state": risk_state,
            "summary": summary,
        }
        payload.update(extra_fields)
        finalized = finalize_strategy_decision(payload, context)

        persisted_payload, persistence_meta = build_persisted_bias_payload(finalized, context)
        if persisted_payload is not None:
            persisted_finalized = finalize_strategy_decision(persisted_payload, context)
            persisted_finalized["bias_persistence"] = persistence_meta
            store_directional_bias_memory(persisted_finalized, context, source="memory_override")
            return persisted_finalized

        stored_record = store_directional_bias_memory(finalized, context)
        persistence_snapshot = build_bias_persistence_meta(stored_record, active=False) if stored_record else None
        if persistence_snapshot is not None:
            finalized["bias_persistence"] = persistence_snapshot
        return finalized

    trap_block_weight = float(mode_config.get("strategy_trap_block_weight", 1.5))
    failure_hard_count = int(mode_config.get("failure_risk_hard_count", 1))
    compression_activation_min_cluster = int(mode_config.get("compression_activation_min_cluster", 2))
    reversal_activation_min_transition = int(mode_config.get("reversal_activation_min_transition", 2))
    reversal_activation_min_setups = int(mode_config.get("reversal_activation_min_setups", 2))

    exhaustion_rejection = detect_exhaustion_rejection_candidate(context)
    if exhaustion_rejection is not None:
        return decision(**exhaustion_rejection)

    if bull_trap_weight >= trap_block_weight and overall_bias in {"bullish", "neutral"}:
        return decision(
            strategy_bias="neutral",
            state="trap",
            action="avoid_long_trap",
            setup_timeframes=[tf for tf in ["15m", "1h"] if results.get(tf, {}).get("signal") in BULLISH_SIGNALS],
            trigger_timeframes=[],
            risk_timeframes=[tf for tf in ACTIVE_TIMEFRAMES if results.get(tf, {}).get("trap_risk") == "bull_trap_risk"],
            risk_state="high",
            summary=(
                f"{symbol} is showing bullish pressure, but trap risk is present on active trading timeframes. "
                f"The better move is patience rather than chasing upside continuation."
            ),
        )

    if bear_trap_weight >= trap_block_weight and overall_bias in {"bearish", "neutral"}:
        return decision(
            strategy_bias="neutral",
            state="trap",
            action="avoid_short_trap",
            setup_timeframes=[tf for tf in ["15m", "1h"] if results.get(tf, {}).get("signal") in BEARISH_SIGNALS],
            trigger_timeframes=[],
            risk_timeframes=[tf for tf in ACTIVE_TIMEFRAMES if results.get(tf, {}).get("trap_risk") == "bear_trap_risk"],
            risk_state="high",
            summary=(
                f"{symbol} is showing bearish pressure, but bear trap risk is present on active trading timeframes. "
                f"The better move is patience rather than forcing downside continuation."
            ),
        )

    if failure_risk_count >= failure_hard_count:
        return decision(
            strategy_bias="neutral",
            state="failed_move",
            action="wait",
            setup_timeframes=[],
            trigger_timeframes=[],
            risk_timeframes=[tf for tf in ACTIVE_TIMEFRAMES if results.get(tf, {}).get("trap_risk") in FAILURE_RISKS],
            risk_state="high",
            summary=(
                f"{symbol} is showing unstable follow-through, with failure risk present on recent directional moves. "
                f"This is a lower-quality environment for fresh entries."
            ),
        )

    if short_breakdown_count == 2 and short_term_bias in {"bearish", "strong_bearish"} and higher_bearish_pressure:
        summary = (
            f"{symbol} is showing confirmed bearish breakdown pressure on the 15m setup timeframe, "
            f"with 5m weakness available as the trigger and the broader backdrop still leaning bearish. "
            f"This favors short continuation setups over aggressive countertrend longs."
        )
        if higher_timeframe_bias == "neutral":
            summary = (
                f"{symbol} is showing confirmed bearish breakdown pressure on the active short-term timeframes. "
                f"Higher timeframes are not fully aligned yet, but the broader backdrop still leans bearish, "
                f"keeping downside continuation in play."
            )
        return decision(
            strategy_bias="short",
            state="continuation",
            action="enter_short",
            setup_timeframes=["15m"],
            trigger_timeframes=["5m"],
            risk_timeframes=["5m"],
            risk_state="favorable" if higher_timeframe_bias in {"bearish", "strong_bearish"} else "mixed",
            summary=summary,
        )

    if (
        (signal_15m in BEARISH_SIGNALS or setup_15m in BEARISH_BREAKOUT_SETUPS or (structure_15m == "expansion" and momentum_15m in {"falling", "flat"}))
        and signal_5m not in BEARISH_SIGNALS
        and momentum_5m in {"falling", "flat"}
        and overall_bias in {"neutral", "bearish", "strong_bearish"}
        and not higher_bullish_pressure
    ):
        return decision(
            strategy_bias="short",
            state="pre_breakdown",
            action="watch_breakdown",
            setup_timeframes=["15m"],
            trigger_timeframes=["5m"],
            risk_timeframes=["5m", "15m"],
            risk_state="mixed",
            summary=(
                f"{symbol} already has bearish pressure on the 15m setup timeframe, but the 5m trigger has not fully confirmed the move yet. "
                f"Treat this as a lead-short setup and wait for a failed bounce or fresh 5m breakdown before activating the trade."
            ),
            bias_origin="setup_trigger_preconfirmation",
            bias_score=6.4 if signal_15m in BEARISH_SIGNALS else 5.8,
            bias_reasons=[
                "15m is already leaning bearish on the setup timeframe.",
                "5m momentum is leaning lower, but the trigger is not confirmed yet.",
                "Higher timeframes are not strongly opposing the short idea.",
            ],
            bias_confidence=58,
            bias_edge=1.2,
        )

    if short_breakdown_count >= 1 and overall_bias in {"neutral", "bearish", "strong_bearish"} and higher_bearish_pressure:
        return decision(
            strategy_bias="short",
            state="continuation_watch",
            action="watch_breakdown",
            setup_timeframes=["15m"] if signal_15m == "bearish_breakdown" else [],
            trigger_timeframes=[tf for tf in ["5m", "15m"] if results.get(tf, {}).get("signal") == "bearish_breakdown"],
            risk_timeframes=[tf for tf in ["5m", "15m"] if results.get(tf, {}).get("trap_risk") in FAILURE_RISKS],
            risk_state="mixed",
            summary=(
                f"{symbol} is starting to show bearish breakdown pressure on the active trading timeframes while the broader backdrop remains weak. "
                f"This is a watch-for-follow-through environment rather than a force-entry environment."
            ),
        )

    if (
        short_breakout_count == 2
        and short_term_bias in {"bullish", "strong_bullish"}
        and higher_bullish_pressure
        and not higher_bearish_pressure
    ):
        return decision(
            strategy_bias="long",
            state="continuation",
            action="enter_long",
            setup_timeframes=["15m"],
            trigger_timeframes=["5m"],
            risk_timeframes=["5m"],
            risk_state="favorable" if higher_timeframe_bias in {"bullish", "strong_bullish"} else "mixed",
            summary=(
                f"{symbol} is showing confirmed breakout pressure on the 15m setup timeframe, "
                f"with 5m strength supporting continuation. This favors long continuation setups."
            ),
        )

    if (
        (signal_15m in BULLISH_SIGNALS or setup_15m in BULLISH_BREAKOUT_SETUPS or (structure_15m == "expansion" and momentum_15m in {"rising", "flat"}))
        and signal_5m not in BULLISH_SIGNALS
        and momentum_5m in {"rising", "flat"}
        and overall_bias in {"neutral", "bullish", "strong_bullish"}
        and not higher_bearish_pressure
    ):
        return decision(
            strategy_bias="long",
            state="pre_breakout",
            action="watch_breakout",
            setup_timeframes=["15m"],
            trigger_timeframes=["5m"],
            risk_timeframes=["5m", "15m"],
            risk_state="mixed",
            summary=(
                f"{symbol} already has bullish pressure on the 15m setup timeframe, but the 5m trigger has not fully confirmed the move yet. "
                f"Treat this as a lead-long setup and wait for a pullback hold or fresh 5m breakout before activating the trade."
            ),
            bias_origin="setup_trigger_preconfirmation",
            bias_score=6.4 if signal_15m in BULLISH_SIGNALS else 5.8,
            bias_reasons=[
                "15m is already leaning bullish on the setup timeframe.",
                "5m momentum is leaning higher, but the trigger is not confirmed yet.",
                "Higher timeframes are not strongly opposing the long idea.",
            ],
            bias_confidence=58,
            bias_edge=1.2,
        )

    if (
        short_breakout_count >= 1
        and overall_bias in {"neutral", "bullish", "strong_bullish"}
        and higher_bullish_pressure
        and not higher_bearish_pressure
    ):
        return decision(
            strategy_bias="long",
            state="continuation_watch",
            action="watch_breakout",
            setup_timeframes=["5m"],
            trigger_timeframes=["5m"],
            risk_timeframes=["15m", "1h"],
            risk_state="mixed",
            summary=(
                f"{symbol} is building bullish breakout pressure on the 5m timeframe, "
                f"with short-term momentum aligned upward. However, higher timeframes are not fully confirmed, "
                f"so this is a watch-for-confirmation environment rather than an immediate entry."
            ),
        )

    if compression_cluster >= compression_activation_min_cluster:
        if early_bias and early_bias["state"] == "compression":
            direction = early_bias["strategy_bias"]

            if direction == "short" and fifteen_bullish_structure and not real_bearish_trigger and not higher_bearish_pressure:
                if mode_config.get("name") == "aggressive":
                    return decision(
                        strategy_bias="short",
                        state="countertrend_short_watch",
                        action="watch_reversal",
                        setup_timeframes=[tf for tf in ["15m", "1h"] if tf in results],
                        trigger_timeframes=["5m"],
                        risk_timeframes=["5m", "15m"],
                        risk_state="high",
                        summary=(
                            f"{symbol} is still holding a bullish 15m structure, but short-term pullback pressure is starting to build inside compression. "
                            f"Treat this only as a countertrend short watch until 5m confirms real bearish follow-through."
                        ),
                        bias_origin="countertrend_compression_bias",
                        bias_score=round(max(4.2, early_bias["score"] - 1.5), 2),
                        bias_reasons=[
                            "Short-term pullback pressure is building inside compression.",
                            "The 15m structure is still broadly bullish, so this is countertrend.",
                            "Require a real 5m bearish trigger before promoting the short.",
                        ],
                        bias_confidence=max(48, int((early_bias.get("confidence") or 56) - 8)),
                        bias_edge=round(max(0.6, (early_bias.get("edge") or 1.0) - 0.8), 2),
                    )

                return decision(
                    strategy_bias="neutral",
                    state="compression",
                    action="watch_breakout",
                    setup_timeframes=["15m", "1h"],
                    trigger_timeframes=["5m"],
                    risk_timeframes=["5m"],
                    risk_state="neutral",
                    summary=(
                        f"{symbol} is still holding a bullish structure while price pulls back inside compression. "
                        f"That keeps bearish continuation unqualified for now, so stay patient until the range resolves more cleanly."
                    ),
                )

            if direction == "long" and fifteen_bearish_structure and not real_bullish_trigger and not higher_bullish_pressure:
                if mode_config.get("name") == "aggressive":
                    return decision(
                        strategy_bias="long",
                        state="countertrend_long_watch",
                        action="watch_reversal",
                        setup_timeframes=[tf for tf in ["15m", "1h"] if tf in results],
                        trigger_timeframes=["5m"],
                        risk_timeframes=["5m", "15m"],
                        risk_state="high",
                        summary=(
                            f"{symbol} is still holding a bearish 15m structure, but short-term bounce pressure is starting to build inside compression. "
                            f"Treat this only as a countertrend long watch until 5m confirms real bullish follow-through."
                        ),
                        bias_origin="countertrend_compression_bias",
                        bias_score=round(max(4.2, early_bias["score"] - 1.5), 2),
                        bias_reasons=[
                            "Short-term bounce pressure is building inside compression.",
                            "The 15m structure is still broadly bearish, so this is countertrend.",
                            "Require a real 5m bullish trigger before promoting the long.",
                        ],
                        bias_confidence=max(48, int((early_bias.get("confidence") or 56) - 8)),
                        bias_edge=round(max(0.6, (early_bias.get("edge") or 1.0) - 0.8), 2),
                    )

                return decision(
                    strategy_bias="neutral",
                    state="compression",
                    action="watch_breakout",
                    setup_timeframes=["15m", "1h"],
                    trigger_timeframes=["5m"],
                    risk_timeframes=["5m"],
                    risk_state="neutral",
                    summary=(
                        f"{symbol} is still holding a bearish structure while price bounces inside compression. "
                        f"That keeps bullish continuation unqualified for now, so stay patient until the range resolves more cleanly."
                    ),
                )

            if direction == "long":
                if overall_bias in {"bullish", "strong_bullish"} or short_term_bias in {"bullish", "strong_bullish"} or higher_bullish_structure:
                    summary = (
                        f"{symbol} remains broadly constructive, but momentum is pausing inside a compression pocket. "
                        f"Treat this as a bullish breakout watch until 5m follow-through confirms the next push higher."
                    )
                else:
                    summary = (
                        f"{symbol} is compressing across the active timeframes, but early upside pressure is starting to build inside the range. "
                        f"Treat this as a bullish breakout watch until 5m follow-through confirms the break."
                    )
            else:
                if overall_bias in {"bearish", "strong_bearish"} or short_term_bias in {"bearish", "strong_bearish"} or higher_bearish_structure:
                    summary = (
                        f"{symbol} remains broadly soft, but momentum is pausing inside a compression pocket. "
                        f"Treat this as a bearish breakdown watch until 5m follow-through confirms the next leg lower."
                    )
                else:
                    summary = (
                        f"{symbol} is compressing across the active timeframes, but early downside pressure is starting to build inside the range. "
                        f"Treat this as a bearish breakdown watch until 5m follow-through confirms the break."
                    )
            return decision(
                strategy_bias=direction,
                state="compression",
                action=early_bias["action"],
                setup_timeframes=[tf for tf in ["15m", "1h"] if tf in results],
                trigger_timeframes=["5m"],
                risk_timeframes=["5m"],
                risk_state="mixed",
                summary=summary,
                bias_origin=early_bias["origin"],
                bias_score=early_bias["score"],
                bias_reasons=early_bias["reasons"],
                bias_confidence=early_bias.get("confidence"),
                bias_edge=early_bias.get("edge"),
            )

        if mode_config.get("name") == "aggressive":
            if (
                fifteen_bullish_structure
                and not real_bearish_trigger
                and momentum_5m in {"falling", "flat"}
                and (setup_5m == "Early Bearish Reversal" or signal_1m in BEARISH_SIGNALS)
                and not higher_bearish_pressure
            ):
                return decision(
                    strategy_bias="short",
                    state="countertrend_short_watch",
                    action="watch_reversal",
                    setup_timeframes=[tf for tf in ["15m", "1h"] if tf in results],
                    trigger_timeframes=["5m"],
                    risk_timeframes=["5m", "15m"],
                    risk_state="high",
                    summary=(
                        f"{symbol} is still holding a bullish 15m structure, but short-term pullback pressure is trying to develop inside compression. "
                        f"Treat this only as a countertrend short watch until sellers prove control with a real 5m breakdown."
                    ),
                    bias_origin="countertrend_compression_probe",
                    bias_score=4.8,
                    bias_reasons=[
                        "5m pullback pressure is building against a still-bullish 15m structure.",
                        "The fade idea is countertrend, so it needs stronger confirmation than a normal continuation setup.",
                        "Require a real 5m bearish trigger before promoting the short.",
                    ],
                    bias_confidence=52,
                    bias_edge=0.8,
                )

            if (
                fifteen_bearish_structure
                and not real_bullish_trigger
                and momentum_5m in {"rising", "flat"}
                and (setup_5m == "Early Bullish Reversal" or signal_1m in BULLISH_SIGNALS)
                and not higher_bullish_pressure
            ):
                return decision(
                    strategy_bias="long",
                    state="countertrend_long_watch",
                    action="watch_reversal",
                    setup_timeframes=[tf for tf in ["15m", "1h"] if tf in results],
                    trigger_timeframes=["5m"],
                    risk_timeframes=["5m", "15m"],
                    risk_state="high",
                    summary=(
                        f"{symbol} is still holding a bearish 15m structure, but short-term bounce pressure is trying to develop inside compression. "
                        f"Treat this only as a countertrend long watch until buyers prove control with a real 5m breakout."
                    ),
                    bias_origin="countertrend_compression_probe",
                    bias_score=4.8,
                    bias_reasons=[
                        "5m bounce pressure is building against a still-bearish 15m structure.",
                        "The bounce idea is countertrend, so it needs stronger confirmation than a normal continuation setup.",
                        "Require a real 5m bullish trigger before promoting the long.",
                    ],
                    bias_confidence=52,
                    bias_edge=0.8,
                )

        if overall_bias == "neutral":
            summary = (
                f"{symbol} is compressing across multiple active timeframes, which suggests the market is coiling. "
                f"This is a watchlist environment for the next confirmed breakout or breakdown rather than a force-entry environment."
            )
            if results.get("1h", {}).get("signal") in BULLISH_SIGNALS and results.get("1d", {}).get("signal") in BEARISH_SIGNALS:
                summary = (
                    f"{symbol} is compressing across multiple timeframes while higher timeframe signals remain split between bullish pressure and bearish trend context. "
                    f"This is a watchlist environment until one side gains clear follow-through."
                )
            return decision(
                strategy_bias="neutral",
                state="compression",
                action="watch_breakout",
                setup_timeframes=["15m", "1h"],
                trigger_timeframes=["5m"],
                risk_timeframes=["5m"],
                risk_state="neutral",
                summary=summary,
            )

    if transition_cluster >= reversal_activation_min_transition or reversal_setup_count >= reversal_activation_min_setups:
        if early_bias and early_bias["state"] == "reversal":
            direction = early_bias["strategy_bias"]
            if direction == "long":
                summary = (
                    f"{symbol} is moving through a reversal environment, and early bullish pressure is starting to build against the recent downside move. "
                    f"Treat this as a bullish reversal watch until the 5m trigger confirms the turn."
                )
            else:
                summary = (
                    f"{symbol} is moving through a reversal environment, and early bearish pressure is starting to build against the recent upside move. "
                    f"Treat this as a bearish reversal watch until the 5m trigger confirms the turn."
                )
            return decision(
                strategy_bias=direction,
                state="reversal",
                action="watch_reversal",
                setup_timeframes=["15m", "1h", "1d"] if setup_1d in REVERSAL_SETUPS else ["15m", "1h"],
                trigger_timeframes=["5m"],
                risk_timeframes=["5m", "15m"],
                risk_state="mixed",
                summary=summary,
                bias_origin=early_bias["origin"],
                bias_score=early_bias["score"],
                bias_reasons=early_bias["reasons"],
                bias_confidence=early_bias.get("confidence"),
                bias_edge=early_bias.get("edge"),
            )

        if mode_config.get("name") == "balanced":
            higher_bullish_support = signal_4h in BULLISH_SIGNALS or (
                results.get("4h", {}).get("structure") == "trend"
                and results.get("4h", {}).get("signal_details", {}).get("trend_bias") == "bullish"
            )
            higher_bearish_support = signal_4h in BEARISH_SIGNALS or (
                results.get("4h", {}).get("structure") == "trend"
                and results.get("4h", {}).get("signal_details", {}).get("trend_bias") == "bearish"
            )

            if (
                overall_bias in {"bullish", "strong_bullish"}
                and higher_bullish_support
                and signal_1d not in BEARISH_SIGNALS
                and setup_1d not in {"Overbought Reversal Risk", "Early Bearish Reversal", "Bearish Exhaustion"}
                and signal_5m not in BEARISH_SIGNALS
                and short_breakdown_count == 0
            ):
                return decision(
                    strategy_bias="long",
                    state="pullback",
                    action="wait_pullback" if bull_trap_count >= 1 else "watch_breakout",
                    setup_timeframes=["15m", "1h"],
                    trigger_timeframes=["5m"],
                    risk_timeframes=["5m", "15m"],
                    risk_state="mixed",
                    summary=(
                        f"{symbol} retains a bullish directional lean, but the market is still moving through a noisy transition pocket. "
                        f"Treat this as a cautious pullback-or-breakout continuation watch rather than an early bearish reversal call."
                    ),
                    bias_origin="balanced_context_bias",
                    bias_score=6.2 if bull_trap_count >= 1 else 5.8,
                    bias_reasons=[
                        "4h structure is still supportive on the upside.",
                        "Daily context is not confirming a bearish turn.",
                        "5m has not confirmed downside continuation.",
                    ],
                    bias_confidence=56,
                    bias_edge=1.1,
                )

            if (
                overall_bias in {"bearish", "strong_bearish"}
                and higher_bearish_support
                and signal_1d not in BULLISH_SIGNALS
                and setup_1d not in {"Oversold Reversal", "Early Bullish Reversal", "Bullish Exhaustion Reversal"}
                and signal_5m not in BULLISH_SIGNALS
                and short_breakout_count == 0
            ):
                return decision(
                    strategy_bias="short",
                    state="pullback",
                    action="wait_pullback" if bear_trap_count >= 1 else "watch_breakdown",
                    setup_timeframes=["15m", "1h"],
                    trigger_timeframes=["5m"],
                    risk_timeframes=["5m", "15m"],
                    risk_state="mixed",
                    summary=(
                        f"{symbol} retains a bearish directional lean, but the market is still moving through a noisy transition pocket. "
                        f"Treat this as a cautious bounce-or-breakdown continuation watch rather than an early bullish reversal call."
                    ),
                    bias_origin="balanced_context_bias",
                    bias_score=6.2 if bear_trap_count >= 1 else 5.8,
                    bias_reasons=[
                        "4h structure is still supportive on the downside.",
                        "Daily context is not confirming a bullish turn.",
                        "5m has not confirmed upside continuation.",
                    ],
                    bias_confidence=56,
                    bias_edge=1.1,
                )

        return decision(
            strategy_bias="neutral",
            state="reversal",
            action="watch_reversal",
            setup_timeframes=["15m", "1h", "1d"] if setup_1d in REVERSAL_SETUPS else ["15m", "1h"],
            trigger_timeframes=["5m"],
            risk_timeframes=["5m", "15m"],
            risk_state="mixed",
            summary=(
                f"{symbol} is moving through a transition phase across multiple timeframes. "
                f"Direction is becoming less stable, so this is better treated as a reversal-watch environment than a clean continuation setup."
            ),
        )

    if overall_bias == "bearish" and higher_bearish_pressure:
        return decision(
            strategy_bias="short",
            state="pullback",
            action="wait_pullback",
            setup_timeframes=["15m", "1h"],
            trigger_timeframes=["5m"],
            risk_timeframes=["5m", "15m"],
            risk_state="mixed",
            summary=(
                f"{symbol} still leans bearish overall, but entry quality is not ideal right now. "
                f"Short-term compression suggests momentum has paused, so favor short setups on weak bounces "
                f"or failed recoveries rather than chasing downside after extension."
            ),
        )

    if overall_bias == "bullish" and higher_bullish_pressure and not higher_bearish_pressure:
        return decision(
            strategy_bias="long",
            state="pullback",
            action="wait_pullback",
            setup_timeframes=["15m", "1h"],
            trigger_timeframes=["5m"],
            risk_timeframes=["5m"],
            risk_state="mixed",
            summary=(
                f"{symbol} still leans bullish overall, but entry quality is not ideal right now. "
                f"Favor long setups on controlled pullbacks rather than chasing extension."
            ),
        )

    if mode_config.get("name") in {"balanced", "aggressive"}:
        higher_bullish_support = signal_4h in BULLISH_SIGNALS or (results.get("4h", {}).get("structure") == "trend" and results.get("4h", {}).get("signal_details", {}).get("trend_bias") == "bullish")
        higher_bearish_support = signal_4h in BEARISH_SIGNALS or (results.get("4h", {}).get("structure") == "trend" and results.get("4h", {}).get("signal_details", {}).get("trend_bias") == "bearish")

        if overall_bias in {"bullish", "strong_bullish"} and short_term_bias in {"bullish", "strong_bullish"} and higher_bullish_support and signal_15m in BULLISH_SIGNALS:
            return decision(
                strategy_bias="long",
                state="pullback",
                action="wait_pullback" if bull_trap_count >= 1 else "watch_breakout",
                setup_timeframes=["15m", "1h"],
                trigger_timeframes=["5m"],
                risk_timeframes=["5m"],
                risk_state="mixed",
                summary=(
                    f"{symbol} retains a bullish directional lean, but the setup is not clean enough for an immediate chase entry. "
                    f"Treat this as a pullback-or-breakout continuation watch rather than a neutral wait state."
                ),
                bias_origin="directional_pullback_bias",
                bias_score=6.5 if bull_trap_count >= 1 else 6.0,
                bias_reasons=[
                    "15m trend structure remains bullish.",
                    "4h context is still supportive on the upside.",
                    "Short-term structure is constructive even though timing is not clean yet.",
                ],
                bias_confidence=58,
                bias_edge=1.3,
            )

        if overall_bias in {"bearish", "strong_bearish"} and short_term_bias in {"bearish", "strong_bearish"} and higher_bearish_support and signal_15m in BEARISH_SIGNALS:
            return decision(
                strategy_bias="short",
                state="pullback",
                action="wait_pullback" if bear_trap_count >= 1 else "watch_breakdown",
                setup_timeframes=["15m", "1h"],
                trigger_timeframes=["5m"],
                risk_timeframes=["5m"],
                risk_state="mixed",
                summary=(
                    f"{symbol} retains a bearish directional lean, but the setup is not clean enough for an immediate chase entry. "
                    f"Treat this as a bounce-or-breakdown continuation watch rather than a neutral wait state."
                ),
                bias_origin="directional_pullback_bias",
                bias_score=6.5 if bear_trap_count >= 1 else 6.0,
                bias_reasons=[
                    "15m trend structure remains bearish.",
                    "4h context is still supportive on the downside.",
                    "Short-term structure is weak even though timing is not clean yet.",
                ],
                bias_confidence=58,
                bias_edge=1.3,
            )

    return decision(
        strategy_bias="neutral",
        state="standby",
        action="wait",
        setup_timeframes=[],
        trigger_timeframes=[],
        risk_timeframes=[],
        risk_state="mixed",
        summary=(
            f"{symbol} does not currently show enough multi-timeframe alignment for a high-quality directional entry. "
            f"The best move right now is to wait for cleaner confirmation."
        ),
    )





def detect_setup(rsi, momentum, bb_state, trend_bias):
    setup = None
    confidence = 0

    # 1) Trap / failure risk setups first
    if (
        trend_bias == "bullish"
        and momentum < 0
        and rsi >= 65
        and bb_state in ["above_upper_band", "near_upper_band"]
    ):
        setup = "Bull Trap Risk"
        confidence = 72

    elif (
        trend_bias == "bearish"
        and momentum > 0
        and rsi <= 35
        and bb_state in ["below_lower_band", "near_lower_band"]
    ):
        setup = "Bear Trap Risk"
        confidence = 72

    # 2) Expansion setups
    elif (
        trend_bias == "bullish"
        and momentum > 0
        and rsi >= 65
        and bb_state == "above_upper_band"
    ):
        setup = "Volatility Expansion Bullish"
        confidence = 78

    elif (
        trend_bias == "bearish"
        and momentum < 0
        and rsi <= 35
        and bb_state == "below_lower_band"
    ):
        setup = "Volatility Expansion Bearish"
        confidence = 78

    # 3) Breakout / breakdown
    elif (
        trend_bias == "bullish"
        and momentum > 0
        and rsi >= 55
        and bb_state in ["near_upper_band", "above_upper_band"]
    ):
        setup = "Bullish Breakout"
        confidence = 75

    elif (
        trend_bias == "bearish"
        and momentum < 0
        and rsi <= 45
        and bb_state in ["near_lower_band", "below_lower_band"]
    ):
        setup = "Bearish Breakdown"
        confidence = 75

    # 4) Reversal setups
    elif trend_bias in ["bearish", "mixed"] and rsi <= 35 and momentum > 0:
        setup = "Oversold Reversal"
        confidence = 70

    elif trend_bias in ["bullish", "mixed"] and rsi > 70 and momentum < 0:
        setup = "Overbought Reversal Risk"
        confidence = 70

    elif (
        trend_bias == "mixed"
        and momentum > 0
        and 35 <= rsi <= 50
        and bb_state in ["near_lower_band", "inside_bands"]
    ):
        setup = "Early Bullish Reversal"
        confidence = 63

    elif (
        trend_bias == "mixed"
        and momentum < 0
        and 50 <= rsi <= 65
        and bb_state in ["near_upper_band", "inside_bands"]
    ):
        setup = "Early Bearish Reversal"
        confidence = 63

    # 5) Continuation / pullback
    elif (
        trend_bias == "bullish"
        and momentum > 0
        and bb_state == "inside_bands"
        and rsi >= 55
    ):
        setup = "Trend Continuation"
        confidence = 65

    elif (
        trend_bias == "bullish"
        and momentum < 0
        and bb_state in ["near_upper_band", "above_upper_band", "inside_bands"]
        and rsi >= 50
    ):
        setup = "Bullish Pullback"
        confidence = 60

    elif (
        trend_bias == "bearish"
        and momentum > 0
        and bb_state in ["near_lower_band", "below_lower_band"]
        and 40 <= rsi <= 55
    ):
        setup = "Bearish Bounce Attempt"
        confidence = 60

    elif (
        trend_bias == "bullish"
        and momentum > 0
        and bb_state == "near_upper_band"
        and 50 <= rsi <= 60
    ):
        setup = "Bullish Pressure Build"
        confidence = 62

    # 6) Exhaustion / compression
    elif rsi > 75 and momentum < 0:
        setup = "Bearish Exhaustion"
        confidence = 70

    elif rsi < 25 and momentum > 0:
        setup = "Bullish Exhaustion Reversal"
        confidence = 70

    elif 45 <= rsi <= 55 and bb_state == "inside_bands":
        setup = "Range Compression"
        confidence = 55

    return setup, confidence



def detect_market_structure(signal: str, setup: str | None, signal_details: dict) -> str:
    bb = signal_details.get("bollinger_state")

    # Expansion
    if setup in ["Volatility Expansion Bullish", "Volatility Expansion Bearish"]:
        return "expansion"

    if signal in ["bullish_breakout", "bearish_breakdown"]:
        return "expansion"

    # Trend
    if signal in ["bullish_trend", "bearish_trend"]:
        return "trend"

    # Transition
    if setup in REVERSAL_SETUPS:
        return "transition"

    if signal in [
        "bullish_but_pulling_back",
        "bearish_but_bouncing",
        "oversold_reversal_watch",
        "overbought_pullback_watch",
    ]:
        return "transition"

    # Compression
    if setup == "Range Compression":
        return "compression"

    if signal == "range_neutral" and bb == "inside_bands":
        return "compression"

    return "neutral"



def detect_trap_risk(signal: str, structure: str, signal_details: dict) -> str | None:
    trend = signal_details.get("trend_bias")
    momentum = signal_details.get("momentum_state")
    bb = signal_details.get("bollinger_state")

    # Bull trap risk
    if signal == "bullish_breakout" and momentum == "falling":
        return "breakout_failure_risk"

    if trend == "bullish" and momentum == "falling" and bb in ["above_upper_band", "near_upper_band"]:
        return "bull_trap_risk"

    # Bear trap risk
    if signal == "bearish_breakdown" and momentum == "rising":
        return "breakdown_failure_risk"

    if trend == "bearish" and momentum == "rising" and bb in ["below_lower_band", "near_lower_band"]:
        return "bear_trap_risk"

    # Failed move risk in transition states
    if signal == "bullish_but_pulling_back" and structure == "transition" and bb in ["above_upper_band", "near_upper_band"]:
        return "bull_trap_risk"

    if signal == "bearish_but_bouncing" and structure == "transition" and bb in ["below_lower_band", "near_lower_band"]:
        return "bear_trap_risk"

    return None





def analyze_timeframe(product_id: str, timeframe: str) -> dict:
    cfg = TIMEFRAME_CONFIG[timeframe]

    df = fetch_candles(product_id=product_id, granularity=cfg["granularity"], limit=cfg["limit"])

    if cfg["resample"]:
        df = resample_candles(df, cfg["resample"])

    df = add_indicators(df)

    if df.empty:
        raise HTTPException(status_code=502, detail="No processed data available.")

    history = df["close"].tail(30).astype(float).tolist()

    last = df.iloc[-1]
    prev = df.iloc[-2] if len(df) > 1 else last
    fib = add_fibonacci(df, lookback=50)
    signal = generate_signal(df)
    signal_score = get_signal_score(signal)
    signal_details = get_signal_details(df)

    rsi_value = float(last["rsi_14"]) if pd.notna(last["rsi_14"]) else 0.0
    momentum_value = float(last["momentum"]) if pd.notna(last["momentum"]) else 0.0

    setup, setup_confidence = detect_setup(
        rsi=rsi_value,
        momentum=momentum_value,
        bb_state=signal_details.get("bollinger_state"),
        trend_bias=signal_details.get("trend_bias"),
    )

    structure = detect_market_structure(signal=signal, setup=setup, signal_details=signal_details)

    trap_risk = detect_trap_risk(signal=signal, structure=structure, signal_details=signal_details)

    summary = build_summary(
        signal=signal,
        signal_details=signal_details,
        timeframe=timeframe,
        symbol=product_id,
        setup=setup,
        setup_confidence=setup_confidence,
        structure=structure,
        trap_risk=trap_risk,
    )

    return {
        "symbol": product_id,
        "timeframe": timeframe,
        "price": round_or_none(last["close"]),
        "signal": signal,
        "signal_score": signal_score,
        "setup": setup,
        "setup_confidence": setup_confidence,
        "structure": structure,
        "trap_risk": trap_risk,
        "history": history,
        "signal_details": signal_details,
        "summary": summary,
        "status": "live",
        "candle": {
            "time": last["time"].isoformat(),
            "open": round_or_none(last["open"]),
            "high": round_or_none(last["high"]),
            "low": round_or_none(last["low"]),
            "close": round_or_none(last["close"]),
            "volume": round_or_none(last["volume"]),
        },
        "previous_candle": {
            "time": prev["time"].isoformat(),
            "open": round_or_none(prev["open"]),
            "high": round_or_none(prev["high"]),
            "low": round_or_none(prev["low"]),
            "close": round_or_none(prev["close"]),
            "volume": round_or_none(prev["volume"]),
        },
        "indicators": {
            "sma_5": round_or_none(last["sma_5"]),
            "sma_20": round_or_none(last["sma_20"]),
            "ema_9": round_or_none(last["ema_9"]),
            "rsi_14": round_or_none(last["rsi_14"]),
            "momentum": round_or_none(last["momentum"]),
            "bb_mid": round_or_none(last["bb_mid"]),
            "bb_upper": round_or_none(last["bb_upper"]),
            "bb_lower": round_or_none(last["bb_lower"]),
            "atr_14": round_or_none(last["atr_14"]),
            "obv": round_or_none(last["obv"]),
            "history_points": int(len(df)),
        },
        "fibonacci": fib,
    }



def ensure_log_dir() -> None:
    LOG_DIR.mkdir(parents=True, exist_ok=True)


def clamp_log_limit(limit: int) -> int:
    if limit < 1:
        return 1
    return min(limit, MAX_LOG_LIMIT)


def compact_price_zone(zone: dict | None) -> dict | None:
    if not zone:
        return None
    return {
        "low": first_valid_number(zone.get("low")),
        "high": first_valid_number(zone.get("high")),
        "mid": first_valid_number(zone.get("mid")),
        "label": zone.get("label"),
        "description": zone.get("description"),
        "reference_timeframes": list(zone.get("reference_timeframes", []) or []),
    }


def compact_take_profit_zone(zone: dict | None) -> dict | None:
    if not zone:
        return None
    return {
        "tp1": first_valid_number(zone.get("tp1")),
        "tp2": first_valid_number(zone.get("tp2")),
        "tp3": first_valid_number(zone.get("tp3")),
        "description": zone.get("description"),
        "reference_timeframes": list(zone.get("reference_timeframes", []) or []),
    }


def compact_timeframe_log_view(results: dict) -> dict:
    compact: dict[str, dict] = {}
    for tf, item in results.items():
        candle = item.get("candle", {}) or {}
        compact[tf] = {
            "price": first_valid_number(item.get("price")),
            "signal": item.get("signal"),
            "signal_score": item.get("signal_score"),
            "setup": item.get("setup"),
            "setup_confidence": item.get("setup_confidence"),
            "structure": item.get("structure"),
            "trap_risk": item.get("trap_risk"),
            "candle_time": candle.get("time"),
            "candle_open": first_valid_number(candle.get("open")),
            "candle_high": first_valid_number(candle.get("high")),
            "candle_low": first_valid_number(candle.get("low")),
            "candle_close": first_valid_number(candle.get("close")),
        }
    return compact


def classify_snapshot_signal(record: dict) -> dict:
    strategy = record.get("strategy", {}) or {}
    trigger_context = record.get("trigger_context", {}) or {}
    execution_plan = record.get("execution_plan", {}) or {}

    bias = strategy.get("bias")
    action = strategy.get("action")
    state = strategy.get("state")
    direction = execution_plan.get("direction") if execution_plan.get("direction") in {"long", "short"} else bias if bias in {"long", "short"} else "neutral"
    entry_score = int(strategy.get("entry_score", 0) or 0)
    entry_score_tier = str(strategy.get("entry_score_tier") or "unknown")
    entry_permission = bool(strategy.get("entry_permission"))
    execution_ready = bool(execution_plan.get("execution_ready"))

    reasons: list[str] = []
    label = "context_only"
    detail = "neutral_bias"
    is_trade_candidate = False
    candidate_type = "inactive"
    candidate_strength = "none"

    if bias not in {"long", "short"} or direction == "neutral":
        reasons.append("Strategy bias is neutral, so this snapshot is context only.")
    elif action in {"avoid_long_trap", "avoid_short_trap", "wait"} or state in {"trap", "standby", "failed_move"}:
        detail = "blocked_state"
        reasons.append("The strategy is intentionally blocked by a trap, wait, or standby state.")
    elif entry_score_tier == "no_trade" or entry_score < 40:
        detail = "low_entry_score"
        reasons.append("Entry score is too low to treat this as a tradable candidate.")
    elif trigger_context.get("state") in {"neutral", "no_directional_setup_support"}:
        detail = "inactive_trigger_context"
        reasons.append("Trigger context is inactive, so this snapshot should be treated as context only.")
    else:
        label = "trade_candidate"
        detail = "candidate"
        is_trade_candidate = True
        if execution_ready or entry_permission:
            candidate_type = "execution_ready"
        elif action == "wait_pullback":
            candidate_type = "pullback_candidate"
        elif action in {"watch_breakout", "watch_breakdown", "watch_reversal"}:
            candidate_type = "watch_candidate"
        else:
            candidate_type = "planned_trade"

        if entry_score >= 80:
            candidate_strength = "high"
        elif entry_score >= 60:
            candidate_strength = "medium"
        else:
            candidate_strength = "low"

        if not trigger_context.get("setup_support_continuation", True):
            reasons.append("Directional support exists, but the setup timeframe is not continuation-qualified yet.")
        if not trigger_context.get("trigger_indicator_confirmed", True):
            reasons.append("Trigger confirmation is incomplete, so execution should still wait.")
        if execution_plan.get("reward_to_risk_estimate") is None:
            reasons.append("Reward-to-risk is not fully mapped yet, so the candidate is still early.")
        if not reasons:
            reasons.append("The snapshot has directional bias and enough structure to be tracked as a trade candidate.")

    return {
        "label": label,
        "detail": detail,
        "is_trade_candidate": is_trade_candidate,
        "direction": direction,
        "candidate_type": candidate_type,
        "candidate_strength": candidate_strength,
        "entry_score": entry_score,
        "entry_score_tier": entry_score_tier,
        "reasons": reasons[:3],
    }


def enrich_snapshot_record(record: dict) -> dict:
    if not isinstance(record, dict):
        return record
    signal_meta = classify_snapshot_signal(record)
    record["signal_meta"] = signal_meta
    record["signal_label"] = signal_meta["label"]
    record["is_trade_candidate"] = signal_meta["is_trade_candidate"]
    return record


def clamp_forward_window(value: int) -> int:
    if value < 1:
        return 1
    return min(int(value), 250)


def extract_record_price_range(record: dict) -> dict:
    market_price = first_valid_number(record.get("market", {}).get("price"))
    timeframe_priority = ["5m", "15m", "1m", "1h"]
    for tf in timeframe_priority:
        item = (record.get("timeframes", {}) or {}).get(tf, {}) or {}
        price = first_valid_number(item.get("price"), market_price)
        high = first_valid_number(item.get("candle_high"), price)
        low = first_valid_number(item.get("candle_low"), price)
        close = first_valid_number(item.get("candle_close"), price)
        if price is not None or high is not None or low is not None:
            ref_price = first_valid_number(close, price, market_price)
            ref_high = first_valid_number(high, ref_price, market_price)
            ref_low = first_valid_number(low, ref_price, market_price)
            return {
                "price": ref_price,
                "high": ref_high if ref_high is not None else ref_price,
                "low": ref_low if ref_low is not None else ref_price,
            }
    return {"price": market_price, "high": market_price, "low": market_price}


def compute_raw_move_pct(entry_price: float | None, future_price: float | None) -> float | None:
    if entry_price in (None, 0.0) or future_price is None:
        return None
    return round(((future_price - entry_price) / entry_price) * 100, 4)


def compute_directional_move_pct(direction: str, entry_price: float | None, future_price: float | None) -> float | None:
    raw_move = compute_raw_move_pct(entry_price, future_price)
    if raw_move is None or direction not in {"long", "short"}:
        return None
    return raw_move if direction == "long" else round(-raw_move, 4)


def classify_directional_outcome(move_pct: float | None) -> str | None:
    if move_pct is None:
        return None
    if move_pct > 0:
        return "favorable"
    if move_pct < 0:
        return "unfavorable"
    return "flat"


def extract_outcome_reference_levels(record: dict) -> dict:
    signal_meta = record.get("signal_meta", {}) or {}
    direction = signal_meta.get("direction")
    plan = record.get("execution_plan", {}) or {}
    take_profit = plan.get("take_profit_zone", {}) or {}
    invalidation = plan.get("invalidation_zone", {}) or {}

    tp1 = first_valid_number(take_profit.get("tp1"))
    if direction == "long":
        invalidation_touch = first_valid_number(invalidation.get("high"), invalidation.get("mid"), invalidation.get("low"))
    elif direction == "short":
        invalidation_touch = first_valid_number(invalidation.get("low"), invalidation.get("mid"), invalidation.get("high"))
    else:
        invalidation_touch = None

    return {
        "direction": direction,
        "tp1": tp1,
        "invalidation_touch": invalidation_touch,
    }


def evaluate_touch_outcomes(record: dict, future_records: list[dict]) -> dict:
    refs = extract_outcome_reference_levels(record)
    direction = refs.get("direction")
    tp1 = refs.get("tp1")
    invalidation_touch = refs.get("invalidation_touch")

    tp1_touched = False
    invalidation_touched = False
    first_touch = None
    first_touch_snapshot_id = None
    first_touch_logged_at = None

    if direction not in {"long", "short"}:
        return {
            "tp1_touched": False,
            "invalidation_touched": False,
            "both_touched": False,
            "first_touch": None,
            "first_touch_snapshot_id": None,
            "first_touch_logged_at": None,
        }

    for future in future_records:
        future_range = extract_record_price_range(future)
        high = future_range.get("high")
        low = future_range.get("low")

        if direction == "long":
            touched_tp = tp1 is not None and high is not None and high >= tp1
            touched_invalidation = invalidation_touch is not None and low is not None and low <= invalidation_touch
        else:
            touched_tp = tp1 is not None and low is not None and low <= tp1
            touched_invalidation = invalidation_touch is not None and high is not None and high >= invalidation_touch

        if touched_tp:
            tp1_touched = True
        if touched_invalidation:
            invalidation_touched = True

        if first_touch is None and (touched_tp or touched_invalidation):
            if touched_tp and touched_invalidation:
                first_touch = "both_same_snapshot"
            elif touched_tp:
                first_touch = "tp1"
            else:
                first_touch = "invalidation"
            first_touch_snapshot_id = future.get("snapshot_id")
            first_touch_logged_at = future.get("logged_at")

    return {
        "tp1_touched": tp1_touched,
        "invalidation_touched": invalidation_touched,
        "both_touched": tp1_touched and invalidation_touched,
        "first_touch": first_touch,
        "first_touch_snapshot_id": first_touch_snapshot_id,
        "first_touch_logged_at": first_touch_logged_at,
    }


def build_outcome_summary(rows: list[dict]) -> dict:
    if not rows:
        return {
            "row_count": 0,
            "signal_labels": {},
            "trade_candidates": 0,
            "context_only": 0,
            "tp1_touched": 0,
            "tp2_touched": 0,
            "tp3_touched": 0,
            "invalidation_touched": 0,
            "protected_after_tp1": 0,
            "avg_mfe_pct": None,
            "avg_mae_pct": None,
            "avg_mark_to_market_pnl_pct": None,
            "favorable_h1": 0,
            "favorable_h3": 0,
            "favorable_h6": 0,
            "favorable_h12": 0,
        }

    label_counts = Counter(row.get("signal_label") or "unknown" for row in rows)
    candidate_rows = [row for row in rows if row.get("is_trade_candidate")]
    tracking_rows = [row.get("outcome_tracking") or {} for row in candidate_rows]

    favorable_counts = {}
    for horizon in DEFAULT_OUTCOME_HORIZONS:
        key = str(horizon)
        favorable_counts[key] = sum(
            1
            for row in candidate_rows
            if ((row.get("horizons", {}) or {}).get(key, {}) or {}).get("directional_outcome") == "favorable"
        )

    return {
        "row_count": len(rows),
        "signal_labels": dict(label_counts),
        "trade_candidates": len(candidate_rows),
        "context_only": len(rows) - len(candidate_rows),
        "tp1_touched": sum(1 for item in tracking_rows if item.get("tp1_touched")),
        "tp2_touched": sum(1 for item in tracking_rows if item.get("tp2_touched")),
        "tp3_touched": sum(1 for item in tracking_rows if item.get("tp3_touched")),
        "invalidation_touched": sum(1 for item in tracking_rows if item.get("invalidation_touched")),
        "protected_after_tp1": sum(1 for item in tracking_rows if item.get("breakeven_touched_after_tp1")),
        "tp1_rate": pct_rate(sum(1 for item in tracking_rows if item.get("tp1_touched")), len(tracking_rows)),
        "tp2_rate": pct_rate(sum(1 for item in tracking_rows if item.get("tp2_touched")), len(tracking_rows)),
        "tp3_rate": pct_rate(sum(1 for item in tracking_rows if item.get("tp3_touched")), len(tracking_rows)),
        "invalidation_rate": pct_rate(sum(1 for item in tracking_rows if item.get("invalidation_touched")), len(tracking_rows)),
        "avg_mfe_pct": average_numeric([item.get("mfe_pct") for item in tracking_rows], 5),
        "avg_mae_pct": average_numeric([item.get("mae_pct") for item in tracking_rows], 5),
        "avg_mark_to_market_pnl_pct": average_numeric([item.get("mark_to_market_pnl_pct") for item in tracking_rows], 5),
        "avg_realized_pnl_pct": average_numeric([item.get("realized_pnl_pct") for item in tracking_rows], 5),
        "favorable_h1": favorable_counts["1"],
        "favorable_h3": favorable_counts["3"],
        "favorable_h6": favorable_counts["6"],
        "favorable_h12": favorable_counts["12"],
    }




def compact_execution_review(review: dict | None) -> dict | None:
    if not review:
        return None
    return {
        "planned_entry_price": first_valid_number(review.get("planned_entry_price")),
        "ideal_entry_price": first_valid_number(review.get("ideal_entry_price")),
        "entry_location": review.get("entry_location"),
        "execution_efficiency_score": first_valid_number(review.get("execution_efficiency_score")),
        "execution_efficiency_label": review.get("execution_efficiency_label"),
        "quality": review.get("quality"),
        "planned_edge_vs_market_latest_pct": first_valid_number(review.get("planned_edge_vs_market_latest_pct")),
        "notes": list(review.get("notes", []) or []),
    }




def compact_limit_order_plan(plan: dict | None) -> dict | None:
    if not plan:
        return None
    return {
        "enabled": bool(plan.get("enabled")),
        "planner_state": plan.get("planner_state"),
        "preference": plan.get("preference"),
        "side": plan.get("side"),
        "order_type": plan.get("order_type"),
        "limit_price": first_valid_number(plan.get("limit_price")),
        "backup_limit_price": first_valid_number(plan.get("backup_limit_price")),
        "invalidation_price": first_valid_number(plan.get("invalidation_price")),
        "take_profit_reference": first_valid_number(plan.get("take_profit_reference")),
        "market_entry_allowed": bool(plan.get("market_entry_allowed")),
        "expiry_snapshots": int(plan.get("expiry_snapshots", 0) or 0),
        "expiry_minutes_estimate": int(plan.get("expiry_minutes_estimate", 0) or 0),
        "reason": plan.get("reason"),
        "historical_hint": plan.get("historical_hint"),
        "cancel_if": list(plan.get("cancel_if", []) or []),
        "promotion_rules": list(plan.get("promotion_rules", []) or []),
        "notes": list(plan.get("notes", []) or []),
    }


def build_compact_snapshot_record(payload: dict) -> dict:

    consensus = payload.get("consensus", {}) or {}
    strategy = consensus.get("strategy", {}) or {}
    trigger_context = strategy.get("trigger_context", {}) or {}
    entry_validation = strategy.get("entry_validation", {}) or {}
    execution_lens = strategy.get("execution_lens", {}) or {}
    execution_plan = strategy.get("execution_plan", {}) or {}
    execution_review = strategy.get("execution_review") or execution_plan.get("execution_review") or {}
    limit_order_plan = strategy.get("limit_order_plan") or execution_plan.get("limit_order_plan") or {}
    mode_meta = payload.get("mode", {}) or {}
    mode_name = mode_meta.get("active") or normalize_strategy_mode(None)
    market_price = first_valid_number(payload.get("timeframes", {}).get("5m", {}).get("price"))
    if market_price is None:
        market_price = first_valid_number(payload.get("timeframes", {}).get("15m", {}).get("price"))
    snapshot_ts = datetime.now(timezone.utc)
    snapshot_id = f"{mode_name}-{int(snapshot_ts.timestamp() * 1000)}"

    record = {
        "snapshot_id": snapshot_id,
        "logged_at": snapshot_ts.isoformat(),
        "symbol": payload.get("symbol"),
        "engine_version": payload.get("engine_version"),
        "mode": mode_name,
        "market": {
            "price": market_price,
            "short_term_bias": consensus.get("short_term", {}).get("bias"),
            "higher_timeframe_bias": consensus.get("higher_timeframes", {}).get("bias"),
            "overall_bias": consensus.get("overall", {}).get("bias"),
            "dashboard_summary": consensus.get("dashboard_summary"),
        },
        "strategy": {
            "bias": strategy.get("strategy_bias"),
            "state": strategy.get("state"),
            "action": strategy.get("action"),
            "market_phase": strategy.get("market_phase"),
            "confidence": strategy.get("confidence"),
            "execution_confidence": strategy.get("execution_confidence"),
            "trade_quality": strategy.get("trade_quality"),
            "entry_permission": strategy.get("entry_permission"),
            "entry_timing": strategy.get("entry_timing"),
            "risk_state": strategy.get("risk_state"),
            "summary": strategy.get("summary"),
            "entry_score": strategy.get("entry_score", {}).get("score"),
            "entry_score_tier": strategy.get("entry_score", {}).get("tier"),
            "execution_lane": execution_lens.get("primary_lane"),
            "scalp_state": (execution_lens.get("scalp") or {}).get("state"),
            "directional_state": (execution_lens.get("directional") or {}).get("state"),
        },
        "execution_lens": execution_lens,
        "trigger_context": {
            "state": trigger_context.get("state"),
            "confirmation_level": trigger_context.get("confirmation_level"),
            "confirmation_score": trigger_context.get("confirmation_score"),
            "ladder_stage": trigger_context.get("ladder_stage"),
            "setup_support_directional": trigger_context.get("setup_support_directional"),
            "setup_support_continuation": trigger_context.get("setup_support_continuation"),
            "trigger_indicator_confirmed": trigger_context.get("trigger_indicator_confirmed"),
            "trigger_band_break_required": trigger_context.get("trigger_band_break_required"),
            "trigger_band_break_confirmed": trigger_context.get("trigger_band_break_confirmed"),
            "trigger_band_break_method": trigger_context.get("trigger_band_break_method"),
            "activation_ready": trigger_context.get("activation_ready"),
            "activation_blockers": list(trigger_context.get("activation_blockers", []) or []),
            "primary_blocker": trigger_context.get("primary_blocker"),
        },
        "entry_validation": {
            "allowed": entry_validation.get("allowed"),
            "status": entry_validation.get("status"),
            "status_detail": entry_validation.get("status_detail"),
            "hard_fail_count": entry_validation.get("hard_fail_count"),
            "soft_fail_count": entry_validation.get("soft_fail_count"),
            "blocking_reasons": list(entry_validation.get("blocking_reasons", []) or []),
        },
        "execution_plan": {
            "execution_ready": execution_plan.get("execution_ready"),
            "direction": execution_plan.get("direction"),
            "entry_model": execution_plan.get("entry_model"),
            "timing_state": execution_plan.get("timing_state"),
            "entry_style": execution_plan.get("entry_style"),
            "position_sizing": execution_plan.get("position_sizing"),
            "aggressiveness": execution_plan.get("aggressiveness"),
            "reward_to_risk_estimate": first_valid_number(execution_plan.get("reward_to_risk_estimate")),
            "entry_zone": compact_price_zone(execution_plan.get("entry_zone")),
            "invalidation_zone": compact_price_zone(execution_plan.get("invalidation_zone")),
            "take_profit_zone": compact_take_profit_zone(execution_plan.get("take_profit_zone")),
            "execution_management": execution_plan.get("execution_management"),
            "limit_order_plan": compact_limit_order_plan(limit_order_plan),
            "notes": list(execution_plan.get("notes", []) or []),
        },
        "execution_review": compact_execution_review(execution_review),
        "limit_order_plan": compact_limit_order_plan(limit_order_plan),
        "timeframes": compact_timeframe_log_view(payload.get("timeframes", {}) or {}),
    }
    return enrich_snapshot_record(record)


def append_snapshot_record(record: dict) -> None:
    ensure_log_dir()
    with SNAPSHOT_LOG_LOCK:
        with SNAPSHOT_LOG_PATH.open("a", encoding="utf-8") as f:
            f.write(json.dumps(record, separators=(",", ":"), ensure_ascii=False) + "\n")


def parse_logged_at(value: str | None) -> datetime | None:
    if not value:
        return None
    try:
        return datetime.fromisoformat(value.replace("Z", "+00:00"))
    except ValueError:
        return None


def read_snapshot_records(
    *,
    mode: str | None = None,
    limit: int | None = 100,
    action: str | None = None,
    state: str | None = None,
    since_minutes: int | None = None,
) -> list[dict]:
    ensure_log_dir()
    if not SNAPSHOT_LOG_PATH.exists():
        return []

    normalized_mode = None
    if mode and str(mode).strip().lower() != "all":
        normalized_mode = normalize_strategy_mode(mode)

    action_filter = str(action).strip().lower() if action else None
    state_filter = str(state).strip().lower() if state else None
    threshold_dt = None
    if since_minutes is not None:
        threshold_dt = datetime.now(timezone.utc) - pd.Timedelta(minutes=max(0, int(since_minutes)))

    records: list[dict] = []
    with SNAPSHOT_LOG_PATH.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                record = json.loads(line)
            except json.JSONDecodeError:
                continue

            if normalized_mode and record.get("mode") != normalized_mode:
                continue
            if action_filter and str(record.get("strategy", {}).get("action", "")).lower() != action_filter:
                continue
            if state_filter and str(record.get("strategy", {}).get("state", "")).lower() != state_filter:
                continue
            if threshold_dt is not None:
                logged_at = parse_logged_at(record.get("logged_at"))
                if logged_at is None or logged_at < threshold_dt:
                    continue
            records.append(enrich_snapshot_record(record))

    records.sort(key=lambda item: item.get("logged_at", ""), reverse=True)
    if limit is None:
        return records
    return records[:clamp_log_limit(limit)]


def build_log_summary(records: list[dict]) -> dict:
    if not records:
        return {
            "count": 0,
            "actions": {},
            "states": {},
            "biases": {},
            "signal_labels": {},
            "trade_candidates": 0,
            "context_only": 0,
            "logged_from": None,
            "logged_to": None,
            "entry_permission_true": 0,
            "execution_ready_true": 0,
        }

    action_counts = Counter(record.get("strategy", {}).get("action") or "unknown" for record in records)
    state_counts = Counter(record.get("strategy", {}).get("state") or "unknown" for record in records)
    bias_counts = Counter(record.get("strategy", {}).get("bias") or "neutral" for record in records)
    signal_label_counts = Counter(record.get("signal_label") or "unknown" for record in records)
    ordered = sorted(records, key=lambda item: item.get("logged_at", ""))

    return {
        "count": len(records),
        "actions": dict(action_counts),
        "states": dict(state_counts),
        "biases": dict(bias_counts),
        "signal_labels": dict(signal_label_counts),
        "trade_candidates": sum(1 for record in records if record.get("is_trade_candidate")),
        "context_only": sum(1 for record in records if not record.get("is_trade_candidate")),
        "logged_from": ordered[0].get("logged_at"),
        "logged_to": ordered[-1].get("logged_at"),
        "entry_permission_true": sum(1 for record in records if record.get("strategy", {}).get("entry_permission")),
        "execution_ready_true": sum(1 for record in records if record.get("execution_plan", {}).get("execution_ready")),
    }


def build_replay_rows(records: list[dict], *, max_forward_snapshots: int = 12) -> list[dict]:
    ordered = sorted((enrich_snapshot_record(record) for record in records), key=lambda item: item.get("logged_at", ""))
    if not ordered:
        return []

    latest_price = first_valid_number(ordered[-1].get("market", {}).get("price"))
    forward_window = clamp_forward_window(max_forward_snapshots)
    rows: list[dict] = []

    for index, record in enumerate(ordered):
        current_price = first_valid_number(record.get("market", {}).get("price"))
        next_price = first_valid_number(ordered[index + 1].get("market", {}).get("price")) if index + 1 < len(ordered) else None
        move_to_next_pct = compute_raw_move_pct(current_price, next_price)
        move_to_latest_pct = compute_raw_move_pct(current_price, latest_price)

        signal_meta = record.get("signal_meta", {}) or {}
        direction = signal_meta.get("direction", "neutral")
        future_records = ordered[index + 1:index + 1 + forward_window]

        horizons: dict[str, dict] = {}
        for horizon in DEFAULT_OUTCOME_HORIZONS:
            future_index = index + horizon
            if future_index < len(ordered):
                future_record = ordered[future_index]
                future_price = first_valid_number(future_record.get("market", {}).get("price"))
                directional_move = compute_directional_move_pct(direction, current_price, future_price)
                horizons[str(horizon)] = {
                    "snapshot_id": future_record.get("snapshot_id"),
                    "logged_at": future_record.get("logged_at"),
                    "price": future_price,
                    "raw_move_pct": compute_raw_move_pct(current_price, future_price),
                    "directional_move_pct": directional_move,
                    "directional_outcome": classify_directional_outcome(directional_move),
                }
            else:
                horizons[str(horizon)] = {
                    "snapshot_id": None,
                    "logged_at": None,
                    "price": None,
                    "raw_move_pct": None,
                    "directional_move_pct": None,
                    "directional_outcome": None,
                }

        raw_best_move_pct = None
        raw_worst_move_pct = None
        mfe_pct = None
        mae_pct = None
        if current_price not in (None, 0.0) and future_records:
            raw_best_candidates: list[float] = []
            raw_worst_candidates: list[float] = []
            directional_favorable: list[float] = []
            directional_adverse: list[float] = []

            for future in future_records:
                future_range = extract_record_price_range(future)
                future_high = future_range.get("high")
                future_low = future_range.get("low")
                if future_high is not None:
                    raw_best_candidates.append(((future_high - current_price) / current_price) * 100)
                if future_low is not None:
                    raw_worst_candidates.append(((future_low - current_price) / current_price) * 100)

                if direction == "long":
                    if future_high is not None:
                        directional_favorable.append(((future_high - current_price) / current_price) * 100)
                    if future_low is not None:
                        directional_adverse.append(((future_low - current_price) / current_price) * 100)
                elif direction == "short":
                    if future_low is not None:
                        directional_favorable.append(((current_price - future_low) / current_price) * 100)
                    if future_high is not None:
                        directional_adverse.append(((current_price - future_high) / current_price) * 100)

            if raw_best_candidates:
                raw_best_move_pct = round(max(raw_best_candidates), 4)
            if raw_worst_candidates:
                raw_worst_move_pct = round(min(raw_worst_candidates), 4)
            if directional_favorable and direction in {"long", "short"}:
                mfe_pct = round(max(directional_favorable), 4)
            if directional_adverse and direction in {"long", "short"}:
                mae_pct = round(min(directional_adverse), 4)

        touch_outcomes = evaluate_touch_outcomes(record, future_records)
        outcome_tracking = evaluate_outcome_tracking(record, future_records)
        pattern_profile = outcome_tracking.get("pattern") or classify_pattern_profile(record)

        rows.append({
            "snapshot_id": record.get("snapshot_id"),
            "logged_at": record.get("logged_at"),
            "mode": record.get("mode"),
            "price": current_price,
            "overall_bias": record.get("market", {}).get("overall_bias"),
            "action": record.get("strategy", {}).get("action"),
            "state": record.get("strategy", {}).get("state"),
            "strategy_bias": record.get("strategy", {}).get("bias"),
            "confidence": record.get("strategy", {}).get("confidence"),
            "entry_permission": record.get("strategy", {}).get("entry_permission"),
            "entry_score": record.get("strategy", {}).get("entry_score"),
            "entry_score_tier": record.get("strategy", {}).get("entry_score_tier"),
            "signal_label": signal_meta.get("label"),
            "signal_detail": signal_meta.get("detail"),
            "is_trade_candidate": signal_meta.get("is_trade_candidate"),
            "candidate_type": signal_meta.get("candidate_type"),
            "candidate_strength": signal_meta.get("candidate_strength"),
            "signal_reasons": list(signal_meta.get("reasons", []) or []),
            "direction": direction,
            "evaluation_window_snapshots": len(future_records),
            "next_price": next_price,
            "move_to_next_pct": move_to_next_pct,
            "latest_price": latest_price,
            "move_to_latest_pct": move_to_latest_pct,
            "raw_best_move_pct": raw_best_move_pct,
            "raw_worst_move_pct": raw_worst_move_pct,
            "mfe_pct": mfe_pct,
            "mae_pct": mae_pct,
            "horizons": horizons,
            **touch_outcomes,
            "tp2_touched": outcome_tracking.get("tp2_touched"),
            "tp3_touched": outcome_tracking.get("tp3_touched"),
            "breakeven_touched_after_tp1": outcome_tracking.get("breakeven_touched_after_tp1"),
            "outcome_tracking": outcome_tracking,
            "pattern": pattern_profile,
            "pattern_key": pattern_profile.get("pattern_key"),
            "trigger_context": record.get("trigger_context", {}),
            "execution_review": record.get("execution_review", {}),
            "limit_order_plan": record.get("limit_order_plan") or (record.get("execution_plan", {}) or {}).get("limit_order_plan"),
            "execution_plan": {
                "direction": record.get("execution_plan", {}).get("direction"),
                "entry_model": record.get("execution_plan", {}).get("entry_model"),
                "entry_zone": record.get("execution_plan", {}).get("entry_zone"),
                "invalidation_zone": record.get("execution_plan", {}).get("invalidation_zone"),
                "take_profit_zone": record.get("execution_plan", {}).get("take_profit_zone"),
                "reward_to_risk_estimate": record.get("execution_plan", {}).get("reward_to_risk_estimate"),
                "limit_order_plan": record.get("limit_order_plan") or (record.get("execution_plan", {}) or {}).get("limit_order_plan"),
            },
        })
    return rows


def reset_snapshot_logs(archive: bool = True) -> dict:
    ensure_log_dir()
    if not SNAPSHOT_LOG_PATH.exists():
        return {"archived": False, "archive_path": None, "cleared": False}

    archive_path = None
    if archive:
        timestamp = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
        archive_path = LOG_DIR / f"engine_snapshots_{timestamp}.jsonl"
        SNAPSHOT_LOG_PATH.replace(archive_path)
    else:
        SNAPSHOT_LOG_PATH.unlink(missing_ok=True)

    return {
        "archived": archive,
        "archive_path": str(archive_path.name) if archive_path else None,
        "cleared": True,
    }




def classify_trade_outcome(row: dict) -> tuple[str, str, str | None]:
    if int(row.get("evaluation_window_snapshots", 0) or 0) < 1:
        return "open", "still_open", "Still Open"

    first_touch = row.get("first_touch")
    tp1_hit = bool(row.get("tp1_touched"))
    invalidation_hit = bool(row.get("invalidation_touched"))

    if tp1_hit and invalidation_hit:
        if first_touch == "tp1":
            return "closed", "win", "Win"
        if first_touch == "invalidation":
            return "closed", "loss", "Loss"
        return "closed", "mixed", "Mixed"
    if tp1_hit:
        return "closed", "win", "Win"
    if invalidation_hit:
        return "closed", "loss", "Loss"

    h3 = (((row.get("horizons") or {}).get("3") or {}).get("directional_move_pct"))
    if h3 is None:
        h1 = (((row.get("horizons") or {}).get("1") or {}).get("directional_move_pct"))
        if h1 is None:
            return "closed", "flat", "Flat"
        if h1 > 0:
            return "closed", "win", "Win"
        if h1 < 0:
            return "closed", "loss", "Loss"
        return "closed", "flat", "Flat"
    if h3 > 0:
        return "closed", "win", "Win"
    if h3 < 0:
        return "closed", "loss", "Loss"
    return "closed", "flat", "Flat"





def parse_record_datetime(value: str | None) -> datetime | None:
    if not value:
        return None
    try:
        parsed = datetime.fromisoformat(str(value).replace("Z", "+00:00"))
        if parsed.tzinfo is None:
            return parsed.replace(tzinfo=timezone.utc)
        return parsed.astimezone(timezone.utc)
    except (TypeError, ValueError):
        return None


def seconds_between(start_value: str | None, end_value: str | None) -> int | None:
    start = parse_record_datetime(start_value)
    end = parse_record_datetime(end_value)
    if start is None or end is None:
        return None
    return int(max(0.0, (end - start).total_seconds()))


def direction_from_record(record: dict) -> str:
    signal_meta = record.get("signal_meta", {}) or {}
    direction = signal_meta.get("direction")
    if direction in {"long", "short"}:
        return direction
    plan_direction = (record.get("execution_plan", {}) or {}).get("direction")
    if plan_direction in {"long", "short"}:
        return plan_direction
    strategy_bias = (record.get("strategy", {}) or {}).get("bias")
    if strategy_bias in {"long", "short"}:
        return strategy_bias
    return "neutral"


def directional_price_move_pct(direction: str, entry_price: float | None, exit_price: float | None) -> float | None:
    if direction not in {"long", "short"} or entry_price in (None, 0.0) or exit_price is None:
        return None
    raw = ((float(exit_price) - float(entry_price)) / float(entry_price)) * 100.0
    return round(raw if direction == "long" else -raw, 5)


def extract_ordered_targets(record: dict, entry_price: float | None = None) -> list[dict]:
    plan = record.get("execution_plan", {}) or {}
    zone = plan.get("take_profit_zone") or {}
    raw_targets = [
        {"label": "tp1", "price": first_valid_number(zone.get("tp1")), "default_size_pct": 60.0},
        {"label": "tp2", "price": first_valid_number(zone.get("tp2")), "default_size_pct": 25.0},
        {"label": "tp3", "price": first_valid_number(zone.get("tp3")), "default_size_pct": 15.0},
    ]

    management = plan.get("execution_management") or {}
    scale_plan = management.get("scale_out_plan") or []
    scale_by_label = {}
    for item in scale_plan:
        label = str(item.get("target") or "").lower()
        size = first_valid_number(item.get("size_pct"))
        price = first_valid_number(item.get("price"))
        if label:
            scale_by_label[label] = {"size_pct": size, "price": price}

    direction = direction_from_record(record)
    enriched = []
    for target in raw_targets:
        label = target["label"]
        override = scale_by_label.get(label, {})
        price = first_valid_number(override.get("price"), target.get("price"))
        size = first_valid_number(override.get("size_pct"), target.get("default_size_pct"))
        if price is None or size is None:
            continue
        if entry_price is not None and direction == "long" and price <= entry_price:
            continue
        if entry_price is not None and direction == "short" and price >= entry_price:
            continue
        enriched.append({"label": label, "price": float(price), "size_pct": float(size)})

    if direction == "long":
        enriched.sort(key=lambda item: item["price"])
    elif direction == "short":
        enriched.sort(key=lambda item: item["price"], reverse=True)
    return enriched


def extract_invalidation_stop(record: dict) -> float | None:
    direction = direction_from_record(record)
    invalidation = ((record.get("execution_plan", {}) or {}).get("invalidation_zone") or {})
    if direction == "long":
        return first_valid_number(invalidation.get("high"), invalidation.get("mid"), invalidation.get("low"))
    if direction == "short":
        return first_valid_number(invalidation.get("low"), invalidation.get("mid"), invalidation.get("high"))
    return None


def touched_favorable_target(direction: str, target_price: float | None, high: float | None, low: float | None) -> bool:
    if direction == "long":
        return target_price is not None and high is not None and high >= target_price
    if direction == "short":
        return target_price is not None and low is not None and low <= target_price
    return False


def touched_adverse_stop(direction: str, stop_price: float | None, high: float | None, low: float | None) -> bool:
    if direction == "long":
        return stop_price is not None and low is not None and low <= stop_price
    if direction == "short":
        return stop_price is not None and high is not None and high >= stop_price
    return False


def classify_pattern_profile(record: dict) -> dict:
    strategy = record.get("strategy", {}) or {}
    trigger = record.get("trigger_context", {}) or {}
    execution_plan = record.get("execution_plan", {}) or {}
    timeframes = record.get("timeframes", {}) or {}
    direction = direction_from_record(record)

    trap_tfs = [tf for tf, item in timeframes.items() if (item or {}).get("trap_risk")]
    setup_5m = (timeframes.get("5m", {}) or {}).get("setup")
    setup_15m = (timeframes.get("15m", {}) or {}).get("setup")
    signal_5m = (timeframes.get("5m", {}) or {}).get("signal")
    signal_15m = (timeframes.get("15m", {}) or {}).get("signal")
    structure_5m = (timeframes.get("5m", {}) or {}).get("structure")
    structure_15m = (timeframes.get("15m", {}) or {}).get("structure")

    confirmation_level = int(first_valid_number(trigger.get("confirmation_level"), 0) or 0)
    if confirmation_level >= 9:
        confirmation_bucket = "confirmed_9_10"
    elif confirmation_level >= 7:
        confirmation_bucket = "ready_7_8"
    elif confirmation_level >= 5:
        confirmation_bucket = "armed_5_6"
    elif confirmation_level >= 3:
        confirmation_bucket = "early_3_4"
    else:
        confirmation_bucket = "inactive_0_2"

    entry_score = int(first_valid_number(strategy.get("entry_score"), 0) or 0)
    if entry_score >= 82:
        score_bucket = "execute_82_plus"
    elif entry_score >= 68:
        score_bucket = "ready_68_81"
    elif entry_score >= 55:
        score_bucket = "watchlist_55_67"
    elif entry_score >= 40:
        score_bucket = "exploratory_40_54"
    else:
        score_bucket = "no_trade_below_40"

    blockers = list(trigger.get("activation_blockers", []) or [])
    primary_blocker = trigger.get("primary_blocker") or (blockers[0] if blockers else "none")

    profile = {
        "direction": direction,
        "action": strategy.get("action"),
        "state": strategy.get("state"),
        "market_phase": strategy.get("market_phase"),
        "entry_model": execution_plan.get("entry_model"),
        "entry_score_bucket": score_bucket,
        "confirmation_bucket": confirmation_bucket,
        "trigger_state": trigger.get("state"),
        "primary_blocker": primary_blocker,
        "has_trap_risk": bool(trap_tfs),
        "trap_timeframes": trap_tfs,
        "setup_5m": setup_5m,
        "setup_15m": setup_15m,
        "signal_5m": signal_5m,
        "signal_15m": signal_15m,
        "structure_5m": structure_5m,
        "structure_15m": structure_15m,
        "higher_timeframe_bias": (record.get("market", {}) or {}).get("higher_timeframe_bias"),
        "overall_bias": (record.get("market", {}) or {}).get("overall_bias"),
    }
    profile["pattern_key"] = "|".join(
        str(profile.get(key) or "none")
        for key in [
            "direction",
            "action",
            "state",
            "entry_model",
            "entry_score_bucket",
            "confirmation_bucket",
            "trigger_state",
            "primary_blocker",
            "has_trap_risk",
        ]
    )
    return profile


def evaluate_outcome_tracking(record: dict, future_records: list[dict]) -> dict:
    direction = direction_from_record(record)
    entry_price = first_valid_number((record.get("market", {}) or {}).get("price"))
    opened_at = record.get("logged_at")
    pattern = classify_pattern_profile(record)

    base = {
        "direction": direction,
        "entry_price": entry_price,
        "status": "open" if not future_records else "evaluated",
        "outcome_status": "not_applicable",
        "outcome_label": "Not Applicable",
        "close_reason": None,
        "closed_at": None,
        "closed_snapshot_id": None,
        "snapshots_to_close": None,
        "seconds_to_close": None,
        "tp1_touched": False,
        "tp2_touched": False,
        "tp3_touched": False,
        "invalidation_touched": False,
        "breakeven_touched_after_tp1": False,
        "first_touch": None,
        "first_touch_logged_at": None,
        "first_touch_snapshot_id": None,
        "time_to_tp1_seconds": None,
        "time_to_tp2_seconds": None,
        "time_to_tp3_seconds": None,
        "time_to_invalidation_seconds": None,
        "mfe_pct": None,
        "mae_pct": None,
        "realized_pnl_pct": None,
        "mark_to_market_pnl_pct": None,
        "remaining_size_pct": 100.0,
        "scale_outs": [],
        "protection_state": "unprotected",
        "sequence_uncertain": False,
        "pattern": pattern,
        "notes": [],
    }

    if direction not in {"long", "short"} or entry_price in (None, 0.0):
        base["status"] = "context_only"
        base["notes"].append("Outcome tracking is inactive because there is no directional entry reference.")
        return base

    targets = extract_ordered_targets(record, entry_price)
    stop_price = extract_invalidation_stop(record)
    stop_move_pct = directional_price_move_pct(direction, entry_price, stop_price)
    target_index = 0
    realized_pnl = 0.0
    remaining_size = 100.0
    protected_after_tp1 = False
    close_reason = None
    closed_at = None
    closed_snapshot_id = None
    snapshots_to_close = None
    first_touch = None
    first_touch_logged_at = None
    first_touch_snapshot_id = None
    mfe_values: list[float] = []
    mae_values: list[float] = []
    notes: list[str] = []
    tp_touch_map: dict[str, dict] = {}

    def register_first_touch(touch_type: str, future: dict) -> None:
        nonlocal first_touch, first_touch_logged_at, first_touch_snapshot_id
        if first_touch is None:
            first_touch = touch_type
            first_touch_logged_at = future.get("logged_at")
            first_touch_snapshot_id = future.get("snapshot_id")

    def close_trade(reason: str, future: dict, index: int) -> None:
        nonlocal close_reason, closed_at, closed_snapshot_id, snapshots_to_close
        close_reason = reason
        closed_at = future.get("logged_at")
        closed_snapshot_id = future.get("snapshot_id")
        snapshots_to_close = index + 1

    for idx, future in enumerate(future_records):
        future_range = extract_record_price_range(future)
        high = first_valid_number(future_range.get("high"))
        low = first_valid_number(future_range.get("low"))

        if direction == "long":
            favorable = directional_price_move_pct(direction, entry_price, high) if high is not None else None
            adverse = directional_price_move_pct(direction, entry_price, low) if low is not None else None
        else:
            favorable = directional_price_move_pct(direction, entry_price, low) if low is not None else None
            adverse = directional_price_move_pct(direction, entry_price, high) if high is not None else None
        if favorable is not None:
            mfe_values.append(favorable)
        if adverse is not None:
            mae_values.append(adverse)

        stop_touched = touched_adverse_stop(direction, stop_price, high, low)
        breakeven_touched = protected_after_tp1 and touched_adverse_stop(direction, entry_price, high, low)

        target_touched_this_snapshot = False
        while target_index < len(targets):
            target = targets[target_index]
            if not touched_favorable_target(direction, target["price"], high, low):
                break

            target_label = target["label"]
            move_pct = directional_price_move_pct(direction, entry_price, target["price"]) or 0.0
            size_pct = min(float(target.get("size_pct") or 0.0), remaining_size)
            if size_pct <= 0:
                target_index += 1
                continue

            realized_pnl += (size_pct / 100.0) * move_pct
            remaining_size = max(0.0, remaining_size - size_pct)
            target_touched_this_snapshot = True
            tp_touch_map[target_label] = {
                "price": target["price"],
                "size_pct": size_pct,
                "move_pct": round(move_pct, 5),
                "logged_at": future.get("logged_at"),
                "snapshot_id": future.get("snapshot_id"),
                "seconds_from_open": seconds_between(opened_at, future.get("logged_at")),
                "snapshots_from_open": idx + 1,
            }
            register_first_touch(target_label, future)

            if target_label == "tp1":
                protected_after_tp1 = True
                base["protection_state"] = "breakeven_after_tp1"
            if target_label == "tp2":
                base["protection_state"] = "trailing_after_tp2"
            if target_label == "tp3" or remaining_size <= 0:
                close_trade("tp3_or_full_scaleout", future, idx)
                break
            target_index += 1

        if close_reason:
            break

        if stop_touched and target_touched_this_snapshot:
            base["sequence_uncertain"] = True
            notes.append("A target and invalidation/breakeven were both touched inside the same snapshot; exact intrabar sequence is unknown.")

        if remaining_size > 0 and breakeven_touched:
            register_first_touch("breakeven_after_tp1", future)
            base["breakeven_touched_after_tp1"] = True
            close_trade("breakeven_after_tp1", future, idx)
            remaining_size = 0.0
            break

        if remaining_size > 0 and stop_touched:
            register_first_touch("invalidation", future)
            base["invalidation_touched"] = True
            if base["time_to_invalidation_seconds"] is None:
                base["time_to_invalidation_seconds"] = seconds_between(opened_at, future.get("logged_at"))
            realized_pnl += (remaining_size / 100.0) * (stop_move_pct or 0.0)
            remaining_size = 0.0
            close_trade("invalidation", future, idx)
            break

    scale_outs = []
    for target in ["tp1", "tp2", "tp3"]:
        touch = tp_touch_map.get(target)
        base[f"{target}_touched"] = bool(touch)
        if touch:
            base[f"time_to_{target}_seconds"] = touch.get("seconds_from_open")
            scale_outs.append({"target": target, **touch})

    base["scale_outs"] = scale_outs
    base["first_touch"] = first_touch
    base["first_touch_logged_at"] = first_touch_logged_at
    base["first_touch_snapshot_id"] = first_touch_snapshot_id
    base["mfe_pct"] = round(max(mfe_values), 5) if mfe_values else None
    base["mae_pct"] = round(min(mae_values), 5) if mae_values else None
    base["remaining_size_pct"] = round(remaining_size, 2)

    latest_future = future_records[-1] if future_records else None
    final_price = first_valid_number((latest_future or {}).get("market", {}).get("price"))
    final_move_pct = directional_price_move_pct(direction, entry_price, final_price)
    mark_to_market = realized_pnl + ((remaining_size / 100.0) * (final_move_pct or 0.0))

    if close_reason is None:
        if future_records:
            close_reason = "evaluation_window_expired"
            closed_at = (latest_future or {}).get("logged_at")
            closed_snapshot_id = (latest_future or {}).get("snapshot_id")
            snapshots_to_close = len(future_records)
        else:
            base["status"] = "open"
            base["outcome_status"] = "still_open"
            base["outcome_label"] = "Still Open"
            base["notes"] = ["No future snapshots are available yet, so the outcome is still open."]
            return base

    base["close_reason"] = close_reason
    base["closed_at"] = closed_at
    base["closed_snapshot_id"] = closed_snapshot_id
    base["snapshots_to_close"] = snapshots_to_close
    base["seconds_to_close"] = seconds_between(opened_at, closed_at)
    base["realized_pnl_pct"] = round(realized_pnl, 5) if remaining_size == 0 else None
    base["mark_to_market_pnl_pct"] = round(mark_to_market, 5)

    if base["tp1_touched"] and close_reason == "breakeven_after_tp1":
        outcome_status = "protected_win" if mark_to_market > 0 else "flat"
        outcome_label = "Protected Win" if mark_to_market > 0 else "Flat"
        notes.append("TP1 was hit first, so the remaining size is modeled as protected at breakeven.")
    elif base["tp3_touched"]:
        outcome_status = "runner_win"
        outcome_label = "Runner Win"
    elif base["tp2_touched"]:
        outcome_status = "partial_win"
        outcome_label = "Partial Win"
    elif base["tp1_touched"] and not base["invalidation_touched"]:
        outcome_status = "tp1_win"
        outcome_label = "TP1 Win"
    elif base["invalidation_touched"] and not base["tp1_touched"]:
        outcome_status = "loss"
        outcome_label = "Loss"
    elif base["tp1_touched"] and base["invalidation_touched"]:
        outcome_status = "mixed"
        outcome_label = "Mixed"
    elif mark_to_market > 0:
        outcome_status = "favorable_timeout"
        outcome_label = "Favorable Timeout"
    elif mark_to_market < 0:
        outcome_status = "unfavorable_timeout"
        outcome_label = "Unfavorable Timeout"
    else:
        outcome_status = "flat"
        outcome_label = "Flat"

    if base["tp1_touched"] and base["mark_to_market_pnl_pct"] is not None and base["mark_to_market_pnl_pct"] < 0:
        notes.append("TP1 was touched but mark-to-market P/L still went negative; this flags a profit-protection issue worth reviewing.")
    if not targets:
        notes.append("No valid TP ladder was available for this candidate.")
    if stop_price is None:
        notes.append("No valid invalidation level was available for this candidate.")

    base["status"] = "closed"
    base["outcome_status"] = outcome_status
    base["outcome_label"] = outcome_label
    base["notes"] = notes[:5]
    return base


def average_numeric(values: list[float | int | None], digits: int = 4) -> float | None:
    clean = [float(value) for value in values if value is not None]
    if not clean:
        return None
    return round(sum(clean) / len(clean), digits)


def pct_rate(count: int, total: int, digits: int = 1) -> float | None:
    if total <= 0:
        return None
    return round((count / total) * 100, digits)


def build_pattern_outcome_summary(rows: list[dict], *, min_count: int = 1, limit: int = 50) -> list[dict]:
    groups: dict[str, list[dict]] = {}
    for row in rows:
        if not row.get("is_trade_candidate"):
            continue
        tracking = row.get("outcome_tracking") or {}
        pattern = tracking.get("pattern") or row.get("pattern") or {}
        key = pattern.get("pattern_key") or "unknown"
        groups.setdefault(key, []).append(row)

    summaries: list[dict] = []
    for key, group_rows in groups.items():
        if len(group_rows) < min_count:
            continue
        tracking_rows = [row.get("outcome_tracking") or {} for row in group_rows]
        closed = [item for item in tracking_rows if item.get("status") == "closed"]
        wins = [item for item in closed if item.get("outcome_status") in {"tp1_win", "partial_win", "runner_win", "protected_win", "favorable_timeout"}]
        losses = [item for item in closed if item.get("outcome_status") in {"loss", "unfavorable_timeout"}]
        sample_pattern = (tracking_rows[0].get("pattern") or {}) if tracking_rows else {}
        summaries.append({
            "pattern_key": key,
            "pattern": sample_pattern,
            "count": len(group_rows),
            "closed": len(closed),
            "wins": len(wins),
            "losses": len(losses),
            "win_rate": pct_rate(len(wins), len(closed)),
            "loss_rate": pct_rate(len(losses), len(closed)),
            "tp1_rate": pct_rate(sum(1 for item in tracking_rows if item.get("tp1_touched")), len(tracking_rows)),
            "tp2_rate": pct_rate(sum(1 for item in tracking_rows if item.get("tp2_touched")), len(tracking_rows)),
            "tp3_rate": pct_rate(sum(1 for item in tracking_rows if item.get("tp3_touched")), len(tracking_rows)),
            "invalidation_rate": pct_rate(sum(1 for item in tracking_rows if item.get("invalidation_touched")), len(tracking_rows)),
            "avg_mark_to_market_pnl_pct": average_numeric([item.get("mark_to_market_pnl_pct") for item in tracking_rows], 5),
            "avg_mfe_pct": average_numeric([item.get("mfe_pct") for item in tracking_rows], 5),
            "avg_mae_pct": average_numeric([item.get("mae_pct") for item in tracking_rows], 5),
            "avg_seconds_to_tp1": average_numeric([item.get("time_to_tp1_seconds") for item in tracking_rows], 1),
            "avg_seconds_to_close": average_numeric([item.get("seconds_to_close") for item in closed], 1),
        })

    summaries.sort(
        key=lambda item: (
            item.get("count", 0),
            item.get("avg_mark_to_market_pnl_pct") if item.get("avg_mark_to_market_pnl_pct") is not None else -999,
        ),
        reverse=True,
    )
    return summaries[:clamp_log_limit(limit)]

def build_trade_records_from_rows(rows: list[dict], *, limit: int = 50) -> list[dict]:
    trade_rows = [row for row in rows if row.get("is_trade_candidate")]
    records: list[dict] = []
    for row in trade_rows:
        tracking = row.get("outcome_tracking") or {}
        execution_plan = row.get("execution_plan", {}) or {}
        execution_review = row.get("execution_review", {}) or {}
        limit_order_plan = row.get("limit_order_plan") or execution_plan.get("limit_order_plan") or {}
        entry_zone = execution_plan.get("entry_zone") or {}
        pattern = tracking.get("pattern") or row.get("pattern") or {}

        outcome_status = tracking.get("outcome_status") or "still_open"
        status = "open" if outcome_status == "still_open" else (tracking.get("status") or "closed")

        records.append(
            {
                "trade_id": row.get("snapshot_id"),
                "direction": row.get("direction"),
                "status": status,
                "outcome_status": outcome_status,
                "outcome_label": tracking.get("outcome_label") or "Still Open",
                "opened_at": row.get("logged_at"),
                "closed_at": tracking.get("closed_at"),
                "opening_action": row.get("action"),
                "opening_market_phase": row.get("state"),
                "confidence_best": row.get("confidence"),
                "entry_score_best": row.get("entry_score"),
                "entry_score_tier": row.get("entry_score_tier"),
                "entry_model_family": execution_plan.get("entry_model"),
                "reward_to_risk_estimate": execution_plan.get("reward_to_risk_estimate"),
                "entry_market_price": row.get("price"),
                "planned_entry_price": execution_review.get("planned_entry_price") or first_valid_number((entry_zone or {}).get("mid")),
                "ideal_entry_price": execution_review.get("ideal_entry_price"),
                "limit_order_enabled": bool(limit_order_plan.get("enabled")),
                "limit_order_preference": limit_order_plan.get("preference"),
                "limit_order_side": limit_order_plan.get("side"),
                "limit_price": limit_order_plan.get("limit_price"),
                "backup_limit_price": limit_order_plan.get("backup_limit_price"),
                "limit_market_entry_allowed": bool(limit_order_plan.get("market_entry_allowed")),
                "execution_efficiency_score": execution_review.get("execution_efficiency_score"),
                "execution_efficiency_label": execution_review.get("execution_efficiency_label"),
                "entry_location": execution_review.get("entry_location"),
                "tp1_hit": bool(tracking.get("tp1_touched")),
                "tp2_hit": bool(tracking.get("tp2_touched")),
                "tp3_hit": bool(tracking.get("tp3_touched")),
                "invalidation_hit": bool(tracking.get("invalidation_touched")),
                "breakeven_hit": bool(tracking.get("breakeven_touched_after_tp1")),
                "first_touch": tracking.get("first_touch"),
                "close_reason": tracking.get("close_reason"),
                "snapshots_held": tracking.get("snapshots_to_close") or row.get("evaluation_window_snapshots"),
                "seconds_held": tracking.get("seconds_to_close"),
                "time_to_tp1_seconds": tracking.get("time_to_tp1_seconds"),
                "mfe_pct": tracking.get("mfe_pct"),
                "mae_pct": tracking.get("mae_pct"),
                "market_entry_resolved_pct": tracking.get("mark_to_market_pnl_pct"),
                "realized_pnl_pct": tracking.get("realized_pnl_pct"),
                "remaining_size_pct": tracking.get("remaining_size_pct"),
                "planned_edge_vs_market_resolved_pct": execution_review.get("planned_edge_vs_market_latest_pct"),
                "pattern_key": pattern.get("pattern_key"),
                "pattern": pattern,
                "scale_outs": list(tracking.get("scale_outs", []) or []),
                "notes": list(tracking.get("notes", []) or execution_review.get("notes", []) or row.get("signal_reasons", []) or []),
            }
        )
    records.sort(key=lambda item: item.get("opened_at") or "", reverse=True)
    return records[:clamp_log_limit(limit)]

def build_trade_summary(trades: list[dict]) -> dict:
    total = len(trades)
    if total == 0:
        return {
            "total_trades": 0,
            "open_trades": 0,
            "closed_trades": 0,
            "wins": 0,
            "losses": 0,
            "flats": 0,
            "mixed": 0,
            "protected_wins": 0,
            "runner_wins": 0,
            "win_rate": None,
            "avg_snapshots_held": None,
            "avg_seconds_held": None,
            "avg_market_entry_resolved_pct": None,
            "avg_realized_pnl_pct": None,
            "avg_mfe_pct": None,
            "avg_mae_pct": None,
            "tp1_rate": None,
            "tp2_rate": None,
            "tp3_rate": None,
            "invalidation_rate": None,
            "avg_planned_edge_vs_market_resolved_pct": None,
            "avg_execution_efficiency_score": None,
        }

    open_trades = [trade for trade in trades if trade.get("status") == "open"]
    closed_trades = [trade for trade in trades if trade.get("status") != "open"]
    win_statuses = {"tp1_win", "partial_win", "runner_win", "protected_win", "favorable_timeout", "win"}
    loss_statuses = {"loss", "unfavorable_timeout"}
    wins = [trade for trade in closed_trades if trade.get("outcome_status") in win_statuses]
    losses = [trade for trade in closed_trades if trade.get("outcome_status") in loss_statuses]
    flats = [trade for trade in closed_trades if trade.get("outcome_status") == "flat"]
    mixed = [trade for trade in closed_trades if trade.get("outcome_status") == "mixed"]
    protected_wins = [trade for trade in closed_trades if trade.get("outcome_status") == "protected_win"]
    runner_wins = [trade for trade in closed_trades if trade.get("outcome_status") == "runner_win"]

    closed_count = len(closed_trades)
    return {
        "total_trades": total,
        "open_trades": len(open_trades),
        "closed_trades": closed_count,
        "wins": len(wins),
        "losses": len(losses),
        "flats": len(flats),
        "mixed": len(mixed),
        "protected_wins": len(protected_wins),
        "runner_wins": len(runner_wins),
        "win_rate": pct_rate(len(wins), closed_count),
        "loss_rate": pct_rate(len(losses), closed_count),
        "tp1_rate": pct_rate(sum(1 for trade in trades if trade.get("tp1_hit")), total),
        "tp2_rate": pct_rate(sum(1 for trade in trades if trade.get("tp2_hit")), total),
        "tp3_rate": pct_rate(sum(1 for trade in trades if trade.get("tp3_hit")), total),
        "invalidation_rate": pct_rate(sum(1 for trade in trades if trade.get("invalidation_hit")), total),
        "avg_snapshots_held": average_numeric([trade.get("snapshots_held") for trade in closed_trades]),
        "avg_seconds_held": average_numeric([trade.get("seconds_held") for trade in closed_trades], 1),
        "avg_market_entry_resolved_pct": average_numeric([trade.get("market_entry_resolved_pct") for trade in closed_trades], 5),
        "avg_realized_pnl_pct": average_numeric([trade.get("realized_pnl_pct") for trade in closed_trades], 5),
        "avg_mfe_pct": average_numeric([trade.get("mfe_pct") for trade in trades], 5),
        "avg_mae_pct": average_numeric([trade.get("mae_pct") for trade in trades], 5),
        "avg_planned_edge_vs_market_resolved_pct": average_numeric([trade.get("planned_edge_vs_market_resolved_pct") for trade in closed_trades], 5),
        "avg_execution_efficiency_score": average_numeric([trade.get("execution_efficiency_score") for trade in closed_trades], 2),
    }




@app.get("/engine/trades/summary")
def engine_trades_summary(
    mode: str | None = None,
    scope: str = "canonical",
    since_minutes: int | None = None,
    max_forward_snapshots: int = 12,
):
    records = read_snapshot_records(mode=mode, limit=None, since_minutes=since_minutes)
    rows = build_replay_rows(records, max_forward_snapshots=max_forward_snapshots)
    trades = build_trade_records_from_rows(rows, limit=1000)
    return {"summary": build_trade_summary(trades)}


@app.get("/health")
def health_check():
    return {
        "status": "ok",
        "engine_version": ENGINE_VERSION,
        "symbol": PRODUCT_ID,
    }



@app.get("/engine/trades")
def engine_trades(
    mode: str | None = None,
    scope: str = "canonical",
    limit: int = 50,
    since_minutes: int | None = None,
    max_forward_snapshots: int = 12,
):
    records = read_snapshot_records(mode=mode, limit=None, since_minutes=since_minutes)
    rows = build_replay_rows(records, max_forward_snapshots=max_forward_snapshots)
    trades = build_trade_records_from_rows(rows, limit=limit)
    return {
        "summary": build_trade_summary(trades),
        "trades": trades,
    }



def build_multi_timeframe_consensus(results: dict, symbol: str, mode: str = DEFAULT_STRATEGY_MODE) -> dict:

    normalized_mode = normalize_strategy_mode(mode)
    short_term_keys = ["1m", "5m", "15m"]
    higher_timeframe_keys = ["1h", "4h", "1d"]

    short_scores = [results[tf]["signal_score"] for tf in short_term_keys if tf in results]
    higher_scores = [results[tf]["signal_score"] for tf in higher_timeframe_keys if tf in results]
    all_scores = [results[tf]["signal_score"] for tf in results]

    short_avg = sum(short_scores) / len(short_scores) if short_scores else 0
    higher_avg = sum(higher_scores) / len(higher_scores) if higher_scores else 0
    overall_avg = sum(all_scores) / len(all_scores) if all_scores else 0

    short_term_bias = classify_bias_from_average(short_avg)
    higher_timeframe_bias = classify_bias_from_average(higher_avg)
    overall_bias = classify_bias_from_average(overall_avg)

    strategy = build_strategy_summary(
        short_term_bias=short_term_bias,
        higher_timeframe_bias=higher_timeframe_bias,
        overall_bias=overall_bias,
        symbol=symbol,
        results=results,
        mode=normalized_mode,
    )

    dashboard_summary = build_dashboard_summary(
        short_term_bias=short_term_bias,
        higher_timeframe_bias=higher_timeframe_bias,
        overall_bias=overall_bias,
        symbol=symbol,
        results=results,
        strategy=strategy,
    )

    return {
        "mode": build_strategy_mode_metadata(normalized_mode),
        "short_term": {
            "timeframes": short_term_keys,
            "average_score": round(short_avg, 3),
            "bias": short_term_bias,
        },
        "higher_timeframes": {
            "timeframes": higher_timeframe_keys,
            "average_score": round(higher_avg, 3),
            "bias": higher_timeframe_bias,
        },
        "overall": {
            "average_score": round(overall_avg, 3),
            "bias": overall_bias,
        },
        "dashboard_summary": dashboard_summary,
        "strategy": strategy,
    }


@app.get("/engine/multi-state")
def engine_multi_state(mode: str = DEFAULT_STRATEGY_MODE, log_snapshot: bool = True):
    try:
        normalized_mode = normalize_strategy_mode(mode)
        selected_timeframes = ["1m", "5m", "15m", "1h", "4h", "1d"]

        results = {}
        for tf in selected_timeframes:
            results[tf] = analyze_timeframe(PRODUCT_ID, tf)

        consensus = build_multi_timeframe_consensus(results, PRODUCT_ID, mode=normalized_mode)

        response_payload = {
            "symbol": PRODUCT_ID,
            "status": "live",
            "engine_version": ENGINE_VERSION,
            "generated_at": datetime.now(timezone.utc).isoformat(),
            "mode": build_strategy_mode_metadata(normalized_mode),
            "consensus": consensus,
            "timeframes": results,
        }

        if log_snapshot:
            record = build_compact_snapshot_record(response_payload)
            append_snapshot_record(record)
            response_payload["log_meta"] = {
                "logged": True,
                "snapshot_id": record["snapshot_id"],
                "storage": "jsonl",
                "filename": SNAPSHOT_LOG_PATH.name,
            }
        else:
            response_payload["log_meta"] = {
                "logged": False,
                "snapshot_id": None,
                "storage": "jsonl",
                "filename": SNAPSHOT_LOG_PATH.name,
            }

        return response_payload

    except requests.RequestException as e:
        raise HTTPException(status_code=502, detail=f"Coinbase request failed: {str(e)}")
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/engine/logs")
def engine_logs(
    mode: str | None = None,
    limit: int = 100,
    action: str | None = None,
    state: str | None = None,
    since_minutes: int | None = None,
):
    records = read_snapshot_records(
        mode=mode,
        limit=limit,
        action=action,
        state=state,
        since_minutes=since_minutes,
    )
    return {
        "storage": "jsonl",
        "filename": SNAPSHOT_LOG_PATH.name,
        "filters": {
            "mode": mode,
            "limit": clamp_log_limit(limit),
            "action": action,
            "state": state,
            "since_minutes": since_minutes,
        },
        "summary": build_log_summary(records),
        "records": records,
    }


@app.get("/engine/replay")
def engine_replay(
    mode: str | None = None,
    limit: int = 50,
    action: str | None = None,
    state: str | None = None,
    since_minutes: int | None = None,
    only_trade_candidates: bool = False,
    max_forward_snapshots: int = 12,
):
    records = read_snapshot_records(
        mode=mode,
        limit=None,
        action=action,
        state=state,
        since_minutes=since_minutes,
    )
    replay_rows = build_replay_rows(records, max_forward_snapshots=max_forward_snapshots)
    if only_trade_candidates:
        replay_rows = [row for row in replay_rows if row.get("is_trade_candidate")]
    replay_rows = replay_rows[-clamp_log_limit(limit):]
    return {
        "storage": "jsonl",
        "filename": SNAPSHOT_LOG_PATH.name,
        "filters": {
            "mode": mode,
            "limit": clamp_log_limit(limit),
            "action": action,
            "state": state,
            "since_minutes": since_minutes,
            "only_trade_candidates": only_trade_candidates,
            "max_forward_snapshots": clamp_forward_window(max_forward_snapshots),
        },
        "summary": build_log_summary(records),
        "evaluation_summary": build_outcome_summary(replay_rows),
        "rows": replay_rows,
    }


@app.get("/engine/outcome-tracking")
def engine_outcome_tracking(
    mode: str | None = None,
    limit: int = 100,
    action: str | None = None,
    state: str | None = None,
    since_minutes: int | None = None,
    only_trade_candidates: bool = True,
    max_forward_snapshots: int = 12,
):
    records = read_snapshot_records(
        mode=mode,
        limit=None,
        action=action,
        state=state,
        since_minutes=since_minutes,
    )
    rows = build_replay_rows(records, max_forward_snapshots=max_forward_snapshots)
    if only_trade_candidates:
        rows = [row for row in rows if row.get("is_trade_candidate")]
    rows = rows[-clamp_log_limit(limit):]
    return {
        "storage": "jsonl",
        "filename": SNAPSHOT_LOG_PATH.name,
        "filters": {
            "mode": mode,
            "limit": clamp_log_limit(limit),
            "action": action,
            "state": state,
            "since_minutes": since_minutes,
            "only_trade_candidates": only_trade_candidates,
            "max_forward_snapshots": clamp_forward_window(max_forward_snapshots),
        },
        "summary": build_log_summary(records),
        "evaluation_summary": build_outcome_summary(rows),
        "rows": rows,
    }


@app.get("/engine/outcome-patterns")
def engine_outcome_patterns(
    mode: str | None = None,
    limit: int = 50,
    action: str | None = None,
    state: str | None = None,
    since_minutes: int | None = None,
    min_count: int = 2,
    max_forward_snapshots: int = 12,
):
    records = read_snapshot_records(
        mode=mode,
        limit=None,
        action=action,
        state=state,
        since_minutes=since_minutes,
    )
    rows = build_replay_rows(records, max_forward_snapshots=max_forward_snapshots)
    candidate_rows = [row for row in rows if row.get("is_trade_candidate")]
    patterns = build_pattern_outcome_summary(
        candidate_rows,
        min_count=max(1, int(min_count)),
        limit=limit,
    )
    return {
        "storage": "jsonl",
        "filename": SNAPSHOT_LOG_PATH.name,
        "filters": {
            "mode": mode,
            "limit": clamp_log_limit(limit),
            "action": action,
            "state": state,
            "since_minutes": since_minutes,
            "min_count": max(1, int(min_count)),
            "max_forward_snapshots": clamp_forward_window(max_forward_snapshots),
        },
        "summary": build_log_summary(records),
        "evaluation_summary": build_outcome_summary(candidate_rows),
        "patterns": patterns,
    }



@app.get("/engine/outcomes")
def engine_outcomes(
    mode: str | None = None,
    limit: int = 50,
    action: str | None = None,
    state: str | None = None,
    since_minutes: int | None = None,
    only_trade_candidates: bool = True,
    max_forward_snapshots: int = 12,
):
    records = read_snapshot_records(
        mode=mode,
        limit=None,
        action=action,
        state=state,
        since_minutes=since_minutes,
    )
    outcome_rows = build_replay_rows(records, max_forward_snapshots=max_forward_snapshots)
    if only_trade_candidates:
        outcome_rows = [row for row in outcome_rows if row.get("is_trade_candidate")]
    outcome_rows = outcome_rows[-clamp_log_limit(limit):]
    return {
        "storage": "jsonl",
        "filename": SNAPSHOT_LOG_PATH.name,
        "filters": {
            "mode": mode,
            "limit": clamp_log_limit(limit),
            "action": action,
            "state": state,
            "since_minutes": since_minutes,
            "only_trade_candidates": only_trade_candidates,
            "max_forward_snapshots": clamp_forward_window(max_forward_snapshots),
        },
        "summary": build_log_summary(records),
        "evaluation_summary": build_outcome_summary(outcome_rows),
        "rows": outcome_rows,
    }


@app.post("/engine/logs/reset")
def engine_logs_reset(archive: bool = True):
    result = reset_snapshot_logs(archive=archive)
    return {
        "storage": "jsonl",
        "filename": SNAPSHOT_LOG_PATH.name,
        **result,
    }
