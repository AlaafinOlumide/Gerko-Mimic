from __future__ import annotations

import logging
import os

import requests


logging.basicConfig(
    level=logging.INFO
)

logger = logging.getLogger(
    __name__
)


class TelegramBot:

    def __init__(self):

        self.token = os.getenv(
            "TELEGRAM_BOT_TOKEN",
            "",
        ).strip()

        self.chat_id = os.getenv(
            "TELEGRAM_CHAT_ID",
            "",
        ).strip()

    def send(
        self,
        message: str,
    ) -> bool:

        if not self.token:

            logger.error(
                "TELEGRAM_BOT_TOKEN missing."
            )

            return False

        if not self.chat_id:

            logger.error(
                "TELEGRAM_CHAT_ID missing."
            )

            return False

        if any(
            ch.isspace()
            for ch in self.token
        ):

            logger.error(
                "Telegram token contains whitespace."
            )

            return False

        url = (
            "https://api.telegram.org/"
            f"bot{self.token}/sendMessage"
        )

        payload = {
            "chat_id": self.chat_id,
            "text": message,
            "parse_mode": "HTML",
            "disable_web_page_preview": True,
        }

        try:

            response = requests.post(
                url,
                json=payload,
                timeout=20,
            )

            if response.status_code == 401:

                logger.error(
                    "Telegram token unauthorized."
                )

                return False

            if response.status_code == 404:

                logger.error(
                    "Telegram endpoint not found. "
                    "Check bot token."
                )

                return False

            if response.status_code == 403:

                logger.error(
                    "Telegram bot does not have "
                    "permission to send."
                )

                return False

            if response.status_code == 400:

                logger.error(
                    "Telegram Bad Request: %s",
                    response.text,
                )

                return False

            response.raise_for_status()

            result = response.json()

            if not result.get("ok"):

                logger.error(
                    "Telegram API error: %s",
                    result,
                )

                return False

            logger.info(
                "Telegram signal sent."
            )

            return True

        except requests.Timeout:

            logger.error(
                "Telegram timeout."
            )

            return False

        except requests.RequestException as exc:

            logger.exception(
                "Telegram error: %s",
                exc,
            )

            return False