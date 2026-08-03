from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Dict, List, Optional

import pandas as pd

from risk_engine import RiskEngine, calculate_rr


@dataclass
class Signal:
    side: str
    score: int
    grade: str
    quality: str
    entry_low: float
    entry_high: float
    stop_loss: float
    tp1: float
    tp2: float
    lot_size: float
    rr: float
    reasons: List[str]
    invalidation: str


def ema(series: pd.Series, period: int) -> pd.Series:
    return series.ewm(span=period, adjust=False).mean()


def rsi(series: pd.Series, period: int = 14) -> pd.Series:
    delta = series.diff()
    gain = delta.clip(lower=0)
    loss = -delta.clip(upper=0)
    avg_gain = gain.ewm(alpha=1 / period, adjust=False).mean()
    avg_loss = loss.ewm(alpha=1 / period, adjust=False).mean()
    rs = avg_gain / avg_loss.replace(0, math.nan)
    return 100 - (100 / (1 + rs))


def atr(df: pd.DataFrame, period: int = 14) -> pd.Series:
    high_low = df["high"] - df["low"]
    high_close = (df["high"] - df["close"].shift()).abs()
    low_close = (df["low"] - df["close"].shift()).abs()
    tr = pd.concat([high_low, high_close, low_close], axis=1).max(axis=1)
    return tr.ewm(alpha=1 / period, adjust=False).mean()


def adx(df: pd.DataFrame, period: int = 14) -> pd.Series:
    up_move = df["high"].diff()
    down_move = -df["low"].diff()
    plus_dm = up_move.where((up_move > down_move) & (up_move > 0), 0.0)
    minus_dm = down_move.where((down_move > up_move) & (down_move > 0), 0.0)
    tr = atr(df, period)
    plus_di = 100 * plus_dm.ewm(alpha=1 / period, adjust=False).mean() / tr
    minus_di = 100 * minus_dm.ewm(alpha=1 / period, adjust=False).mean() / tr
    dx = ((plus_di - minus_di).abs() / (plus_di + minus_di).replace(0, math.nan)) * 100
    return dx.ewm(alpha=1 / period, adjust=False).mean()


def bollinger_width(series: pd.Series, period: int = 20) -> pd.Series:
    mid = series.rolling(period).mean()
    std = series.rolling(period).std()
    upper = mid + 2 * std
    lower = mid - 2 * std
    return (upper - lower) / mid


def enrich(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    for col in ["open", "high", "low", "close"]:
        df[col] = pd.to_numeric(df[col], errors="coerce")
    df = df.dropna(subset=["open", "high", "low", "close"])
    df["ema50"] = ema(df["close"], 50)
    df["ema200"] = ema(df["close"], 200)
    df["rsi"] = rsi(df["close"], 14)
    df["atr"] = atr(df, 14)
    df["adx"] = adx(df, 14)
    df["bb_width"] = bollinger_width(df["close"], 20)
    df["ema50_slope"] = df["ema50"].diff(5)
    return df.dropna()


class GerkoXAUModel:
    """Gerko-inspired retail quant scoring model for XAUUSD.

    The model is deliberately strict. It prioritises no-trade decisions over
    low-quality signals.
    """

    def __init__(self, min_score: int = 80) -> None:
        self.risk_engine = RiskEngine(min_score=min_score)

    def analyse(self, candles: Dict[str, pd.DataFrame]) -> Optional[Signal]:
        required = {"h1", "m15", "m5"}
        if not required.issubset(candles):
            raise ValueError("Candles must include h1, m15 and m5 dataframes")

        h1 = enrich(candles["h1"])
        m15 = enrich(candles["m15"])
        m5 = enrich(candles["m5"])
        if min(len(h1), len(m15), len(m5)) < 50:
            return None

        h1_last = h1.iloc[-1]
        m15_last = m15.iloc[-1]
        m5_last = m5.iloc[-1]
        price = float(m5_last["close"])
        current_atr = float(m15_last["atr"])

        bullish_trend = h1_last["close"] > h1_last["ema50"] > h1_last["ema200"] and h1_last["ema50_slope"] > 0
        bearish_trend = h1_last["close"] < h1_last["ema50"] < h1_last["ema200"] and h1_last["ema50_slope"] < 0

        side = "WAIT"
        if bullish_trend:
            side = "BUY"
        elif bearish_trend:
            side = "SELL"
        else:
            return None

        score = 0
        reasons: List[str] = []

        # Layer 1: Regime / trend, 30 points
        if side == "BUY" and bullish_trend:
            score += 24
            reasons.append("H1 bullish regime: price above EMA50 and EMA200")
        elif side == "SELL" and bearish_trend:
            score += 24
            reasons.append("H1 bearish regime: price below EMA50 and EMA200")

        if h1_last["adx"] >= 20:
            score += 6
            reasons.append(f"ADX confirms active trend: {h1_last['adx']:.1f}")

        # Layer 2: Rejection / liquidity behaviour, 20 points
        recent = m15.tail(12)
        last = m15.iloc[-1]
        prev_swing_high = recent["high"].iloc[:-1].max()
        prev_swing_low = recent["low"].iloc[:-1].min()
        body = abs(last["close"] - last["open"])
        candle_range = max(last["high"] - last["low"], 0.01)
        rejection_ratio = 1 - (body / candle_range)

        if side == "BUY" and last["low"] <= prev_swing_low and last["close"] > last["open"]:
            score += 14
            reasons.append("M15 swept sell-side liquidity and closed bullish")
        elif side == "SELL" and last["high"] >= prev_swing_high and last["close"] < last["open"]:
            score += 14
            reasons.append("M15 swept buy-side liquidity and closed bearish")

        if rejection_ratio >= 0.45:
            score += 6
            reasons.append("Strong rejection wick detected")

        # Layer 3: Momentum, 20 points
        if side == "BUY" and m15_last["rsi"] > 50 and m5_last["rsi"] > 50:
            score += 10
            reasons.append("M15/M5 RSI aligned above 50")
        elif side == "SELL" and m15_last["rsi"] < 50 and m5_last["rsi"] < 50:
            score += 10
            reasons.append("M15/M5 RSI aligned below 50")

        if side == "BUY" and m5_last["close"] > m5_last["ema50"]:
            score += 5
            reasons.append("M5 price reclaimed EMA50")
        elif side == "SELL" and m5_last["close"] < m5_last["ema50"]:
            score += 5
            reasons.append("M5 price rejected below EMA50")

        if side == "BUY" and m15_last["ema50_slope"] > 0:
            score += 5
            reasons.append("M15 EMA slope supports buy continuation")
        elif side == "SELL" and m15_last["ema50_slope"] < 0:
            score += 5
            reasons.append("M15 EMA slope supports sell continuation")

        # Layer 4: Volatility, 15 points
        atr_median = float(m15["atr"].tail(80).median())
        if current_atr >= atr_median * 0.80:
            score += 10
            reasons.append("ATR supports enough movement for TP")
        if float(m15_last["bb_width"]) >= float(m15["bb_width"].tail(80).median()) * 0.75:
            score += 5
            reasons.append("Bollinger width confirms market is not too compressed")

        # Layer 5: Execution quality, 15 points
        if side == "BUY":
            entry_low = price - current_atr * 0.15
            entry_high = price + current_atr * 0.10
            stop_loss = min(float(last["low"]), price - current_atr * 0.90)
            tp1 = price + current_atr * 1.00
            tp2 = price + current_atr * 2.00
        else:
            entry_low = price - current_atr * 0.10
            entry_high = price + current_atr * 0.15
            stop_loss = max(float(last["high"]), price + current_atr * 0.90)
            tp1 = price - current_atr * 1.00
            tp2 = price - current_atr * 2.00

        rr = calculate_rr(price, stop_loss, tp2, side)
        if rr >= 1.8:
            score += 10
            reasons.append(f"Risk reward acceptable: {rr}:1")
        if abs(price - float(m5_last["ema50"])) <= current_atr * 0.75:
            score += 5
            reasons.append("Entry is not overextended from M5 EMA50")

        score = min(int(round(score)), 100)
        decision = self.risk_engine.decide(score)
        if not decision.trade_allowed:
            return None

        invalidation = (
            f"Close below {stop_loss:.2f} cancels the buy setup."
            if side == "BUY"
            else f"Close above {stop_loss:.2f} cancels the sell setup."
        )

        return Signal(
            side=side,
            score=score,
            grade=decision.grade,
            quality=decision.quality,
            entry_low=round(float(entry_low), 2),
            entry_high=round(float(entry_high), 2),
            stop_loss=round(float(stop_loss), 2),
            tp1=round(float(tp1), 2),
            tp2=round(float(tp2), 2),
            lot_size=decision.lot_size,
            rr=rr,
            reasons=reasons[:7],
            invalidation=invalidation,
        )


def format_signal(signal: Signal) -> str:
    reason_text = "\n".join(f"- {r}" for r in signal.reasons)
    return f"""🧠 <b>GERKO-XAU QUANT SIGNAL</b>

<b>Best Scenario:</b> {signal.side}
<b>Grade:</b> {signal.grade}
<b>Score:</b> {signal.score}/100
<b>Quality:</b> {signal.quality}

<b>Entry:</b>
{signal.entry_low:.2f} - {signal.entry_high:.2f} after confirmed rejection/continuation close

<b>Take Profit:</b>
TP1: {signal.tp1:.2f}
TP2: {signal.tp2:.2f}

<b>Stop Loss:</b>
{signal.stop_loss:.2f}

<b>Best Lot Size:</b>
{signal.lot_size:.2f}

<b>Risk Reward:</b>
{signal.rr}:1

<b>Model Reason:</b>
{reason_text}

<b>Invalidation:</b>
{signal.invalidation}
"""
