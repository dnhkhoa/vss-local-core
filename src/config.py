from dataclasses import dataclass
from os import getenv

from dotenv import load_dotenv


@dataclass(frozen=True)
class Config:
    ollama_base_url: str
    ollama_model: str
    video_source: str
    segment_seconds: float
    frames_per_segment: int
    output_events_path: str
    memory_db_path: str
    question: str


def _get_float(name: str, default: float) -> float:
    value = getenv(name)
    if value is None or not value.strip():
        return default
    try:
        return float(value)
    except ValueError:
        print(f"Invalid {name}={value!r}; using default {default}.")
        return default


def _get_int(name: str, default: int) -> int:
    value = getenv(name)
    if value is None or not value.strip():
        return default
    try:
        parsed = int(value)
    except ValueError:
        print(f"Invalid {name}={value!r}; using default {default}.")
        return default
    return max(1, parsed)


def load_config() -> Config:
    load_dotenv()

    return Config(
        ollama_base_url=getenv("OLLAMA_BASE_URL", "http://localhost:11434").rstrip("/"),
        ollama_model=getenv("OLLAMA_MODEL", "qwen3-vl:8b"),
        video_source=getenv("VIDEO_SOURCE", "sample.mp4"),
        segment_seconds=_get_float("SEGMENT_SECONDS", 5.0),
        frames_per_segment=_get_int("FRAMES_PER_SEGMENT", 3),
        output_events_path=getenv("OUTPUT_EVENTS_PATH", "outputs/events.json"),
        memory_db_path=getenv("MEMORY_DB_PATH", "outputs/video_memory.sqlite"),
        question=getenv("QUESTION", ""),
    )
