from __future__ import annotations

from pathlib import Path

from .ollama_client import OllamaVisionClient


def answer_question_over_events(
    events_path: str,
    question: str,
    client: OllamaVisionClient,
) -> str:
    try:
        events_json = Path(events_path).read_text(encoding="utf-8")
    except OSError as exc:
        return f"Could not read events JSON: {exc}"

    prompt = f"""You are answering questions about a video based only on the following analyzed event JSON.
Do not invent details.
If the information is not available, say so.

Event JSON:
{events_json}

User question:
{question}

Answer in Vietnamese, clearly and briefly.
Include timestamps when relevant."""

    return client.generate_text(prompt)
