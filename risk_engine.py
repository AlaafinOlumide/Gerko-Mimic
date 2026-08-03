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
    """
    Confidence-based recommendation engine.

    The bot does not place trades automatically. It only provides
    a suggested lot size based on the setup score.

    The suggested figures must still be checked against:
    - Account balance
    - Stop-loss distance
    - Maximum permitted account risk
    - Prop-firm drawdown rules
    """

    def __init__(
        self,
        min_score: int = 80,
    ) -> None:
        if not 0 <= min_score <= 100:
            raise ValueError(
                "min_score must be between 0 and 100."
            )

        self.min_score = min_score

    def decide(
        self,
        score: int,
    ) -> RiskDecision:
        score = max(
            0,
            min(
                int(score),
                100,
            ),
        )

        if score >= 95:
            return RiskDecision(
                lot_size=0.50,
                grade="A+",
                quality="Elite Gerko Alignment",
                trade_allowed=True,
                reason=(
                    "Exceptional multi-layer alignment."
                ),
            )

        if score >= 90:
            return RiskDecision(
                lot_size=0.25,
                grade="A",
                quality="Premium Gerko Execution",
                trade_allowed=True,
                reason=(
                    "Very strong probability alignment."
                ),
            )

        if score >= 85:
            return RiskDecision(
                lot_size=0.15,
                grade="A-",
                quality="Strong Gerko Execution",
                trade_allowed=True,
                reason=(
                    "Strong trend, momentum and execution alignment."
                ),
            )

        if score >= self.min_score:
            return RiskDecision(
                lot_size=0.15,
                grade="B+",
                quality="Qualified Gerko Setup",
                trade_allowed=True,
                reason=(
                    "Setup meets the minimum Gerko threshold."
                ),
            )

        if score >= 75:
            return RiskDecision(
                lot_size=0.00,
                grade="B",
                quality="Gerko Watchlist",
                trade_allowed=False,
                reason=(
                    "Setup is close but lacks sufficient confirmation."
                ),
            )

        return RiskDecision(
            lot_size=0.00,
            grade="C",
            quality="No Trade",
            trade_allowed=False,
            reason=(
                "Weak, mixed or incomplete setup."
            ),
        )


def calculate_rr(
    entry: float,
    stop_loss: float,
    take_profit: float,
    side: str,
) -> float:
    side = side.upper().strip()

    if side == "BUY":
        risk = entry - stop_loss
        reward = take_profit - entry

    elif side == "SELL":
        risk = stop_loss - entry
        reward = entry - take_profit

    else:
        raise ValueError(
            "side must be BUY or SELL."
        )

    if risk <= 0 or reward <= 0:
        return 0.0

    return round(
        reward / risk,
        2,
    )