from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class RiskDecision:
    lot_size: float
    grade: str
    quality: str
    trade_allowed: bool
    reason: str


class RiskEngine:
    """Retail risk engine for XAUUSD signal sizing.

    This bot does not place trades. It only recommends one of the user's
    allowed lot sizes based on model confidence.
    """

    def __init__(self, min_score: int = 80):
        self.min_score = min_score

    def decide(self, score: int) -> RiskDecision:
        if score >= 90:
            return RiskDecision(0.50, "A+", "Premium Execution", True, "Very strong statistical alignment")
        if score >= 85:
            return RiskDecision(0.25, "A", "Execution Grade", True, "Strong statistical alignment")
        if score >= self.min_score:
            return RiskDecision(0.15, "A-", "Execution Grade", True, "Acceptable high-quality setup")
        if score >= 75:
            return RiskDecision(0.00, "B+", "Wait", False, "Close, but not enough edge")
        return RiskDecision(0.00, "C", "No Trade", False, "Weak or mixed setup")


def calculate_rr(entry: float, stop_loss: float, tp2: float, side: str) -> float:
    side = side.upper()
    if side == "BUY":
        risk = entry - stop_loss
        reward = tp2 - entry
    else:
        risk = stop_loss - entry
        reward = entry - tp2
    if risk <= 0:
        return 0.0
    return round(reward / risk, 2)
