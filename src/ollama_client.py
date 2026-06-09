from __future__ import annotations

import base64
from pathlib import Path

import requests


class OllamaVisionClient:
    def __init__(self, base_url: str, model: str, timeout_sec: float = 60.0):
        self.base_url = base_url.rstrip("/")
        self.model = model
        self.timeout_sec = timeout_sec

    def analyze_image(self, image_path: str, question: str) -> str:
        try:
            image_bytes = Path(image_path).read_bytes()
        except OSError as exc:
            return f"Error reading image: {exc}"

        encoded_image = base64.b64encode(image_bytes).decode("utf-8")
        payload = {
            "model": self.model,
            "messages": [
                {
                    "role": "user",
                    "content": question,
                    "images": [encoded_image],
                }
            ],
            "stream": False,
        }

        try:
            response = requests.post(
                f"{self.base_url}/api/chat",
                json=payload,
                timeout=self.timeout_sec,
            )
            response.raise_for_status()
            data = response.json()
        except requests.RequestException as exc:
            return f"Ollama request failed: {exc}"
        except ValueError as exc:
            return f"Ollama returned invalid JSON: {exc}"

        try:
            return data["message"]["content"].strip()
        except (KeyError, TypeError):
            return f"Ollama response missing message content: {data}"
