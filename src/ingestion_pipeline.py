from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Callable

from .config import Config
from .event_analyzer import VideoEventAnalyzer, save_events_json
from .ollama_client import OllamaVisionClient
from .person_tracker import build_tracking_summary
from .video_memory import VideoMemoryStore, index_events_to_memory
from .video_sampler import sample_video_segments


ProgressCallback = Callable[[dict], None]


def ingest_video(
    config: Config,
    video_path: str,
    *,
    root_dir: str | Path = ".",
    force: bool = False,
    progress_callback: ProgressCallback | None = None,
) -> dict:
    root = Path(root_dir)
    resolved_video = (root / video_path).resolve()
    fingerprint = build_video_fingerprint(config, resolved_video)

    _emit(progress_callback, stage="cache_check", message="Checking indexed video memory.")
    cached_events = None if force else _load_cached_events(config.memory_db_path, video_path, fingerprint)
    if cached_events:
        _emit(progress_callback, stage="cache_hit", message="Using existing indexed memory.", percent=100)
        save_events_json(
            output_path=config.output_events_path,
            video_path=video_path,
            segment_seconds=config.segment_seconds,
            frames_per_segment=config.frames_per_segment,
            segment_results=cached_events.get("segments") or [],
            extra_fields={
                "analysis_summary": cached_events.get("analysis_summary"),
                "tracking_summary": cached_events.get("tracking_summary"),
                "ingestion": cached_events.get("ingestion"),
            },
        )
        return {
            "cached": True,
            "events": cached_events,
            "memory_stats": _memory_stats(config.memory_db_path),
        }

    _emit(progress_callback, stage="sampling", message="Sampling video segments.", percent=5)
    segments = sample_video_segments(
        video_path=video_path,
        segment_seconds=config.segment_seconds,
        frames_per_segment=config.frames_per_segment,
    )
    _emit(
        progress_callback,
        stage="sampled",
        message=f"Sampled {len(segments)} segment(s).",
        total_segments=len(segments),
        completed_segments=0,
        percent=10,
    )

    client = OllamaVisionClient(config.ollama_base_url, config.ollama_model)
    analyzer = VideoEventAnalyzer(client)
    segment_results = []
    total_segments = len(segments)
    for index, segment in enumerate(segments, start=1):
        _emit(
            progress_callback,
            stage="analyzing",
            message=f"Analyzing {segment['segment_id']} ({index}/{total_segments}).",
            total_segments=total_segments,
            completed_segments=index - 1,
            current_segment=segment["segment_id"],
            percent=10 + int(70 * (index - 1) / max(1, total_segments)),
        )
        segment_results.append(analyzer.analyze_segment(segment))
        _emit(
            progress_callback,
            stage="analyzing",
            message=f"Analyzed {segment['segment_id']} ({index}/{total_segments}).",
            total_segments=total_segments,
            completed_segments=index,
            current_segment=segment["segment_id"],
            percent=10 + int(70 * index / max(1, total_segments)),
        )

    ingestion_meta = {
        "fingerprint": fingerprint,
        "model": config.ollama_model,
        "segment_seconds": config.segment_seconds,
        "frames_per_segment": config.frames_per_segment,
    }
    events_data = save_events_json(
        output_path=config.output_events_path,
        video_path=video_path,
        segment_seconds=config.segment_seconds,
        frames_per_segment=config.frames_per_segment,
        segment_results=segment_results,
        extra_fields={"ingestion": ingestion_meta},
    )

    _emit(progress_callback, stage="tracking", message="Tracking people in sampled frames.", percent=84)
    tracking_summary = build_tracking_summary(events_data, root_dir=root)
    events_data["tracking_summary"] = tracking_summary
    events_data["ingestion"] = ingestion_meta
    save_events_json(
        output_path=config.output_events_path,
        video_path=video_path,
        segment_seconds=config.segment_seconds,
        frames_per_segment=config.frames_per_segment,
        segment_results=segment_results,
        extra_fields={
            "tracking_summary": tracking_summary,
            "ingestion": ingestion_meta,
        },
    )

    _emit(progress_callback, stage="indexing", message="Indexing video memory.", percent=92)
    memory_stats = index_events_to_memory(config.memory_db_path, events_data)
    _emit(progress_callback, stage="completed", message="Ingestion completed.", percent=100)
    return {
        "cached": False,
        "events": events_data,
        "memory_stats": memory_stats,
    }


def build_video_fingerprint(config: Config, video_path: Path) -> str:
    stat = video_path.stat()
    payload = {
        "path": str(video_path.resolve()),
        "size": stat.st_size,
        "mtime_ns": stat.st_mtime_ns,
        "model": config.ollama_model,
        "segment_seconds": config.segment_seconds,
        "frames_per_segment": config.frames_per_segment,
    }
    return hashlib.sha256(json.dumps(payload, sort_keys=True).encode("utf-8")).hexdigest()


def _load_cached_events(db_path: str, video_path: str, fingerprint: str) -> dict | None:
    path = Path(db_path)
    if not path.exists():
        return None
    store = VideoMemoryStore(str(path))
    try:
        return store.get_events_by_fingerprint(video_path, fingerprint)
    finally:
        store.close()


def _memory_stats(db_path: str) -> dict:
    store = VideoMemoryStore(db_path)
    try:
        return store.stats()
    finally:
        store.close()


def _emit(callback: ProgressCallback | None, **payload) -> None:
    if callback:
        callback(payload)
