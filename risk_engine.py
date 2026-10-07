from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class RiskDecision:
    allowed: bool
    grade: str
    quality: str
    lot_size: float


def classify_signal(
    score: int,
    minimum: int,
) -> RiskDecision:

    score = max(
        0,
        min(
            int(score),
            100,
        ),
    )

    minimum = max(
        0,
        min(
            int(minimum),
            100,
        ),
    )

    if score < minimum:

        return RiskDecision(
            False,
            "NO TRADE",
            "Insufficient Confirmation",
            0.00,
        )

    if score >= 94:

        return RiskDecision(
            True,
            "A+",
            "Elite LumiAj Setup",
            0.25,
        )

    if score >= 88:

        return RiskDecision(
            True,
            "A",
            "Premium LumiAj Setup",
            0.20,
        )

    if score >= 83:

        return RiskDecision(
            True,
            "A-",
            "Strong LumiAj Setup",
            0.15,
        )

    if score >= 78:

        return RiskDecision(
            True,
            "B+",
            "Qualified LumiAj Setup",
            0.10,
        )

    return RiskDecision(
        True,
        "B",
        "Early LumiAj Setup",
        0.10,
    )