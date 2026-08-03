from __future__ import annotations

import logging
import os
from typing import Any, Dict

import requests

logger = logging.getLogger(
    __name__
)


class TelegramBot:
    def __init__(self) -> None:
        self.token = os.getenv(
            "TELEGRAM_BOT_TOKEN",
            "",
        ).strip()

        self.chat_id = os.getenv(
            "TELEGRAM_CHAT_ID",
            "",
        ).strip()

        self.timeout_seconds = int(
            os.getenv(
                "TELEGRAM_TIMEOUT_SECONDS",
                "20",
            )
        )

    @property
    def enabled(self) -> bool:
        return bool(
            self.token
            and self.chat_id
        )

    def configuration_status(
        self,
    ) -> Dict[str, bool]:
        return {
            "token_present": bool(
                self.token
            ),
            "chat_id_present": bool(
                self.chat_id
            ),
        }

    def send(
        self,
        message: str,
    ) -> bool:
        if not self.enabled:
            status = self.configuration_status()

            logger.error(
                "Telegram is not configured. "
                "token_present=%s, chat_id_present=%s",
                status["token_present"],
                status["chat_id_present"],
            )

            print(
                "Telegram not configured. "
                "Message output follows:\n"
            )

            print(message)

            return False

        if any(
            character.isspace()
            for character in self.token
        ):
            logger.error(
                "TELEGRAM_BOT_TOKEN contains whitespace. "
                "Paste the token as one uninterrupted string."
            )

            return False

        url = (
            "https://api.telegram.org/"
            f"bot{self.token}/sendMessage"
        )

        payload: Dict[str, Any] = {
            "chat_id": self.chat_id,
            "text": message,
            "parse_mode": "HTML",
            "disable_web_page_preview": True,
        }

        try:
            response = requests.post(
                url,
                json=payload,
                timeout=self.timeout_seconds,
            )

            if response.status_code == 401:
                logger.error(
                    "Telegram returned 401 Unauthorized. "
                    "The bot token is invalid or revoked."
                )

                return False

            if response.status_code == 404:
                logger.error(
                    "Telegram returned 404 Not Found. "
                    "Check the token for missing characters "
                    "or spaces."
                )

                return False

            if response.status_code == 400:
                logger.error(
                    "Telegram returned 400 Bad Request: %s",
                    response.text,
                )

                return False

            if response.status_code == 403:
                logger.error(
                    "Telegram returned 403 Forbidden. "
                    "The bot may be blocked or may not have "
                    "permission to post in the destination."
                )

                return False

            response.raise_for_status()

            response_data = response.json()

            if not response_data.get(
                "ok",
                False,
            ):
                logger.error(
                    "Telegram API returned an unsuccessful "
                    "response: %s",
                    response_data,
                )

                return False

            logger.info(
                "Telegram message sent successfully."
            )

            return True

        except requests.Timeout:
            logger.error(
                "Telegram request timed out."
            )

            return False

        except requests.RequestException as exc:
            logger.exception(
                "Telegram request failed: %s",
                exc,
            )

            return False

        except ValueError as exc:
            logger.error(
                "Telegram returned invalid JSON: %s",
                exc,
            )

            return False