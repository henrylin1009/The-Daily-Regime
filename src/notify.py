"""Email and Telegram delivery for daily briefs."""

from __future__ import annotations

import os
import smtplib
from email.mime.multipart import MIMEMultipart
from email.mime.text import MIMEText
from pathlib import Path

import requests
from dotenv import load_dotenv

from src.config import OUTPUT_DIR, PROJECT_ROOT

load_dotenv(PROJECT_ROOT / ".env")


def send_email(subject: str, html_body: str, text_body: str | None = None) -> None:
    """Send brief via SMTP (configure SMTP_* and NOTIFY_EMAIL_TO in .env)."""
    host = os.getenv("SMTP_HOST")
    port = int(os.getenv("SMTP_PORT", "587"))
    user = os.getenv("SMTP_USER")
    password = os.getenv("SMTP_PASSWORD")
    to_addr = os.getenv("NOTIFY_EMAIL_TO")
    from_addr = os.getenv("SMTP_FROM", user)

    if not all([host, user, password, to_addr]):
        raise EnvironmentError(
            "Email not configured. Set SMTP_HOST, SMTP_USER, SMTP_PASSWORD, NOTIFY_EMAIL_TO in .env"
        )

    msg = MIMEMultipart("alternative")
    msg["Subject"] = subject
    msg["From"] = from_addr or user
    msg["To"] = to_addr
    if text_body:
        msg.attach(MIMEText(text_body, "plain"))
    msg.attach(MIMEText(html_body, "html"))

    with smtplib.SMTP(host, port) as server:
        server.starttls()
        server.login(user, password)
        server.sendmail(msg["From"], [to_addr], msg.as_string())


def send_telegram(message: str) -> None:
    """Send plain-text summary via Telegram bot."""
    token = os.getenv("TELEGRAM_BOT_TOKEN")
    chat_id = os.getenv("TELEGRAM_CHAT_ID")
    if not token or not chat_id:
        raise EnvironmentError("Set TELEGRAM_BOT_TOKEN and TELEGRAM_CHAT_ID in .env")

    url = f"https://api.telegram.org/bot{token}/sendMessage"
    resp = requests.post(url, json={"chat_id": chat_id, "text": message[:4000]}, timeout=30)
    resp.raise_for_status()


def notify_brief(brief_path: Path, signals: dict | None = None) -> None:
    """Deliver brief via configured channels (EMAIL, TELEGRAM, or both)."""
    html = brief_path.read_text(encoding="utf-8")
    subject = f"Macro Brief — {brief_path.stem.replace('brief_', '')}"

    channels = os.getenv("NOTIFY_CHANNELS", "email").lower().split(",")
    summary = ""
    if signals:
        h = signals.get("headline", {})
        summary = " | ".join(
            f"{k}: {v.get('signal', '')}" for k, v in h.items()
        )

    if "email" in channels:
        send_email(subject, html, text_body=summary or "See HTML brief.")

    if "telegram" in channels:
        send_telegram(f"{subject}\n{summary}\nFile: {brief_path}")
