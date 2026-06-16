from __future__ import annotations

import base64
from pathlib import Path

import requests


class OllamaVisionClient:
    def __init__(self, base_url: str, model: str, timeout_sec: float = 180.0):
        self.base_url = base_url.rstrip("/")
        self.model = model
        self.timeout_sec = timeout_sec

    def analyze_images(self, image_paths: list[str], prompt: str) -> str:
        encoded_images = []
        for image_path in image_paths:
            try:
                image_bytes = Path(image_path).read_bytes()
            except OSError as exc:
                return f"Error reading image {image_path}: {exc}"
            encoded_images.append(base64.b64encode(image_bytes).decode("utf-8"))

        payload = {
            "model": self.model,
            "messages": [
                {
                    "role": "user",
                    "content": prompt,
                    "images": encoded_images,
                }
            ],
            "format": "json",
            "options": {
                "temperature": 0,
            },
            "stream": False,
        }

        return self._post_chat(payload)

    def generate_text(self, prompt: str) -> str:
        payload = {
            "model": self.model,
            "messages": [
                {
                    "role": "user",
                    "content": prompt,
                }
            ],
            "stream": False,
        }

        return self._post_chat(payload)

    def _post_chat(self, payload: dict) -> str:
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
