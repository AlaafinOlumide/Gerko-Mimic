# Gerko-XAU Quant Bot

Standalone XAUUSD Telegram signal bot inspired by quantitative market-scoring principles.

It does not copy or replicate Alexander Gerko, XTX Markets, or any proprietary model. It is a retail, rule-based quant-style bot for XAUUSD only.

## Features

- XAUUSD only
- H1, M15 and M5 multi-timeframe scoring
- EMA trend regime filter
- RSI momentum filter
- ATR/Bollinger volatility filter
- Liquidity sweep/rejection logic
- Confidence score out of 100
- Lot size recommendations: 0.15, 0.25, 0.50
- Telegram alerts
- Render worker deployment
- Sleeps from Friday 22:00 until Sunday 22:00
- Scouting windows:
  - 23:00–03:00
  - 07:00–11:00
  - 12:00–17:00

## Setup

1. Create a new folder/repo for this bot.
2. Add these files.
3. Create environment variables:

```bash
TWELVE_DATA_API_KEY=your_key
TELEGRAM_BOT_TOKEN=your_bot_token
TELEGRAM_CHAT_ID=your_chat_id
SYMBOL=XAU/USD
MIN_SCORE=80
MAX_SIGNALS_PER_DAY=8
CHECK_INTERVAL_SECONDS=300
TZ=Europe/London
```

4. Install requirements:

```bash
pip install -r requirements.txt
```

5. Run locally:

```bash
python main.py
```

## Render deployment

Use `render.yaml` and create a new Blueprint or Worker service.

Make sure to add your secret environment variables manually on Render:

- `TWELVE_DATA_API_KEY`
- `TELEGRAM_BOT_TOKEN`
- `TELEGRAM_CHAT_ID`

## Signal threshold

- 90–100: A+ / 0.50 lot
- 85–89: A / 0.25 lot
- 80–84: A- / 0.15 lot
- Below 80: no signal

## Important warning

This bot is for analysis and signal generation only. It does not guarantee profit. Always test on demo first and use responsible risk management.
