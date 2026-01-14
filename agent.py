import os
import requests
from datetime import date
from openai import OpenAI

TELEGRAM_BOT_TOKEN = os.getenv("TELEGRAM_BOT_TOKEN")
TELEGRAM_CHAT_ID = os.getenv("TELEGRAM_CHAT_ID")
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")

# WAŻNE: pusty OPENAI_MODEL ma nie psuć działania
OPENAI_MODEL = (os.getenv("OPENAI_MODEL") or "").strip() or "gpt-5"

COUNTRY = os.getenv("AGENT_COUNTRY", "PL")
TIMEZONE = os.getenv("AGENT_TIMEZONE", "Europe/Warsaw")

TELEGRAM_MAX_LEN = int(os.getenv("TELEGRAM_MAX_LEN", "3800"))
ALLOWED_DOMAINS_CSV = (os.getenv("ALLOWED_DOMAINS") or "").strip()

TOPIC = """
IT & DATA JOB MARKET (EU / PL / Remote)

Focus areas:
- hiring trends (IT, Data, Analytics, AI)
- salary ranges and changes
- in-demand roles and skills
- layoffs / hiring freezes / expansions
- market signals (demand ↑ / ↓)
"""


def build_prompt() -> str:
    today = date.today().isoformat()
    return f"""
Jesteś analitykiem rynku pracy (IT/Data/AI). Przygotuj DZIEŃNY BRIEF RYNKU PRACY dla daty: {today}.

TEMAT:
{TOPIC}

TWARDE WYMAGANIA (bez dyskusji):
- Użyj web search, aby znaleźć WIARYGODNE aktualizacje z ostatnich 48 godzin. Jeśli brak istotnych informacji, rozszerz do ostatnich 7 dni.
- Każda informacja o: zatrudnieniach / zwolnieniach / hiring freeze / ekspansjach / nowych widełkach płacowych MUSI mieć ŹRÓDŁO.
- Maksymalnie 5 pozycji. Preferuj EU/PL/Remote, źródła reputacyjne. Zero spekulacji.
- Jeśli nic sensownego nie ma, zwróć tylko sekcję TLDR z informacją, że brak istotnych zmian.

FORMAT WYJŚCIA (ściśle):
TLDR:
- <max 3 punkty, po polsku>

ITEMS:
- Tytuł (PL): ...
  Co się stało: <1–2 zdania po polsku>
  Dlaczego ważne: <1 zdanie po polsku>
  Źródło: <publisher> | <YYYY-MM-DD>
  URL: <bezpośredni link>

KONIEC.
Ogranicz długość całości do ~2000–2500 znaków.
""".strip()


def _parse_allowed_domains(csv_text: str):
    if not csv_text:
        return []
    parts = [p.strip() for p in csv_text.split(",")]
    return [p for p in parts if p]


def call_openai(prompt: str) -> str:
    if not OPENAI_API_KEY:
        raise RuntimeError("Missing OPENAI_API_KEY environment variable.")

    client = OpenAI(api_key=OPENAI_API_KEY)

    tool_cfg = {
        "type": "web_search",
        "user_location": {
            "type": "approximate",
            "country": COUNTRY,
            "timezone": TIMEZONE,
        },
    }

    allowed_domains = _parse_allowed_domains(ALLOWED_DOMAINS_CSV)
    if allowed_domains:
        tool_cfg["filters"] = {"allowed_domains": allowed_domains}

    response = client.responses.create(
        model=OPENAI_MODEL,
        tools=[tool_cfg],
        tool_choice="auto",
        include=["web_search_call.action.sources"],
        input=prompt,
    )

    text = (getattr(response, "output_text", "") or "").strip()
    if not text:
        raise RuntimeError("Empty response from OpenAI (no output_text).")

    return text


def split_message(text: str, max_len: int = TELEGRAM_MAX_LEN):
    chunks = []
    text = (text or "").strip()
    while len(text) > max_len:
        cut = text.rfind("\n", 0, max_len)
        if cut == -1:
            cut = max_len
        chunks.append(text[:cut].strip())
        text = text[cut:].strip()
    if text:
        chunks.append(text)
    return chunks


def send_to_telegram(message: str):
    if not TELEGRAM_BOT_TOKEN or not TELEGRAM_CHAT_ID:
        raise RuntimeError("Missing TELEGRAM_BOT_TOKEN or TELEGRAM_CHAT_ID environment variable.")

    url = f"https://api.telegram.org/bot{TELEGRAM_BOT_TOKEN}/sendMessage"

    for part in split_message(message):
        payload = {"chat_id": TELEGRAM_CHAT_ID, "text": part}
        r = requests.post(url, json=payload, timeout=20)
        r.raise_for_status()


def is_verifiable(text: str) -> bool:
    t = (text or "").lower()
    return ("url:" in t) and ("http://" in t or "https://" in t)


def main():
    prompt = build_prompt()
    brief = call_openai(prompt)

    if not is_verifiable(brief):
        brief = (
            "TLDR:\n"
            "- Brak wiarygodnych, weryfikowalnych aktualizacji w ostatnich 48h (fallback do 7 dni bez istotnych zmian).\n\n"
            "ITEMS:\n"
            "- (brak)\n\n"
            "KONIEC."
        )

    send_to_telegram(brief)


if __name__ == "__main__":
    main()
