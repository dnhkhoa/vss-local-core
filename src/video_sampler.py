from __future__ import annotations

from pathlib import Path
import math

import cv2

from .video_reader import VideoReader


def format_time(seconds: float) -> str:
    total_seconds = int(round(seconds))
    hours = total_seconds // 3600
    minutes = (total_seconds % 3600) // 60
    secs = total_seconds % 60
    return f"{hours:02d}:{minutes:02d}:{secs:02d}"


def sample_video_segments(
    video_path: str,
    segment_seconds: float,
    frames_per_segment: int,
    output_dir: str = "outputs/frames",
) -> list[dict]:
    if segment_seconds <= 0:
        raise ValueError("segment_seconds must be greater than 0.")
    if frames_per_segment <= 0:
        raise ValueError("frames_per_segment must be greater than 0.")

    frame_dir = Path(output_dir)
    frame_dir.mkdir(parents=True, exist_ok=True)

    reader = VideoReader(video_path)
    segments: list[dict] = []

    try:
        segment_count = int(math.ceil(reader.duration_seconds / segment_seconds))
        for index in range(segment_count):
            start_time_sec = index * segment_seconds
            end_time_sec = min(start_time_sec + segment_seconds, reader.duration_seconds)
            segment_id = f"segment_{index + 1:04d}"
            frame_paths: list[str] = []

            sample_times = _sample_times(
                start_time_sec,
                end_time_sec,
                frames_per_segment,
            )
            for frame_index, timestamp_sec in enumerate(sample_times, start=1):
                ok, frame = reader.read_frame_at(timestamp_sec)
                if not ok:
                    print(
                        f"Skipped failed frame at {format_time(timestamp_sec)} "
                        f"for {segment_id}."
                    )
                    continue

                frame_path = frame_dir / f"{segment_id}_frame_{frame_index:04d}.jpg"
                if cv2.imwrite(str(frame_path), frame):
                    frame_paths.append(str(frame_path))
                else:
                    print(f"Failed to save sampled frame: {frame_path}")

            segments.append(
                {
                    "segment_id": segment_id,
                    "start_time_sec": start_time_sec,
                    "end_time_sec": end_time_sec,
                    "start_time_text": format_time(start_time_sec),
                    "end_time_text": format_time(end_time_sec),
                    "frame_paths": frame_paths,
                }
            )
    finally:
        reader.release()

    return segments


def _sample_times(
    start_time_sec: float,
    end_time_sec: float,
    frames_per_segment: int,
) -> list[float]:
    duration = max(0.0, end_time_sec - start_time_sec)
    if duration == 0:
        return [start_time_sec]

    step = duration / (frames_per_segment + 1)
    return [start_time_sec + step * i for i in range(1, frames_per_segment + 1)]
