# Gerko-XAU Quant Bot

Rule-based XAUUSD (XAU/USD) Telegram signal bot. Analysis and signal generation only.
It does not guarantee profit.

## How it works

- Timeframes: H4 and H1 for trend, M15 for confirmation, M5 for entries.
- Bars are evaluated only after they close. The still-forming bar is never used.
- Data refreshes when a new bar closes (about 15 seconds after the close). The loop
  wakes at each M5 close, capped at `CHECK_INTERVAL_SECONDS` as a fallback.
- Setups: trend continuation, support/resistance bounce, break-and-retest, and range
  edge bounce. Each setup has a minimum score.
- Trend entries are blocked when price is too far from M5 EMA20 (anti-chase).

## Sessions (London time)

- Trend setups: 23:00–03:00 and 07:00–17:30
- Support/resistance setups: 04:00–22:00
- Market closed: Friday 22:00 until Sunday 22:00

## Grades and lot sizes

| Score   | Grade | Lot  |
|---------|-------|------|
| 94–100  | A+    | 0.25 |
| 88–93   | A     | 0.20 |
| 83–87   | A-    | 0.15 |
| below the setup minimum | no signal | — |

The setup minimum is the setup's own threshold (for example `TREND_MIN_SCORE`), so a
signal can be rejected before the grade is applied.

## Environment variables

Secrets (set in the Render dashboard or a local `.env`, never committed):

- `TWELVE_DATA_API_KEY`
- `TELEGRAM_BOT_TOKEN`
- `TELEGRAM_CHAT_ID`

Settings (defaults shown; `render.yaml` sets the deployed values):

| Variable | Default | Deployed |
|---|---|---|
| `SYMBOL` | XAU/USD | XAU/USD |
| `BOT_TIMEZONE` | Europe/London | Europe/London |
| `CHECK_INTERVAL_SECONDS` | 180 | 180 |
| `BAR_CLOSE_FETCH_DELAY_SECONDS` | 15 | — |
| `MAX_SIGNALS_PER_DAY` | 20 | 12 |
| `SIGNAL_COOLDOWN_MINUTES` | 20 | 20 |
| `TREND_MIN_SCORE` | 76 | 76 |
| `SR_MIN_SCORE` | 74 | 74 |
| `COUNTERTREND_SR_MIN_SCORE` | 84 | 84 |
| `BREAK_RETEST_MIN_SCORE` | 76 | 76 |
| `RANGE_MIN_SCORE` | 76 | — |
| `MAX_TREND_CHASE_ATR` | 0.65 | 0.65 |
| `MAX_SR_CHASE_ATR` | 0.45 | 0.45 |
| `SEND_STARTUP_MESSAGE` | — | true |

## Local run

```bash
pip install -r requirements.txt
# create a .env file with the three secrets above
python main.py
```

## Render deployment

Use `render.yaml` to create a Worker service, then add the three secrets in the Render
dashboard. The bot runs as a background worker and sends signals to the Telegram chat.

## Logs

Logs never print API keys or bot tokens. URLs and error messages are masked.
