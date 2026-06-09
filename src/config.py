from dataclasses import dataclass
from os import getenv

from dotenv import load_dotenv


DEFAULT_QUESTION = (
    "You are a factory CCTV visual assistant. Analyze this camera frame. "
    "Describe visible people, machines, vehicles, and possible safety risks. "
    "Answer briefly."
)


@dataclass(frozen=True)
class Config:
    ollama_base_url: str
    ollama_model: str
    video_source: str
    sample_interval_sec: float
    question: str
    enable_danger_zone: bool
    danger_zone_points: str


def _get_bool(name: str, default: bool = False) -> bool:
    value = getenv(name)
    if value is None:
        return default
    return value.strip().lower() in {"1", "true", "yes", "y", "on"}


def _get_float(name: str, default: float) -> float:
    value = getenv(name)
    if value is None or not value.strip():
        return default
    try:
        return float(value)
    except ValueError:
        print(f"Invalid {name}={value!r}; using default {default}.")
        return default


def load_config() -> Config:
    load_dotenv()

    return Config(
        ollama_base_url=getenv("OLLAMA_BASE_URL", "http://localhost:11434").rstrip("/"),
        ollama_model=getenv("OLLAMA_MODEL", "qwen3-vl:8b"),
        video_source=getenv("VIDEO_SOURCE", "0"),
        sample_interval_sec=_get_float("SAMPLE_INTERVAL_SEC", 2.0),
        question=getenv("QUESTION", DEFAULT_QUESTION),
        enable_danger_zone=_get_bool("ENABLE_DANGER_ZONE", False),
        danger_zone_points=getenv("DANGER_ZONE_POINTS", ""),
    )
