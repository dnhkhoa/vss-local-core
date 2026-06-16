from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

import cv2


@dataclass
class Track:
    track_id: int
    last_center: tuple[float, float]
    last_timestamp_sec: float
    observations: list[dict] = field(default_factory=list)


def build_tracking_summary(events_data: dict, *, root_dir: str | Path = ".") -> dict:
    root = Path(root_dir)
    detections_by_frame = _detect_people_in_sampled_frames(events_data, root)
    tracks = _link_detections(detections_by_frame)
    return {
        "method": "opencv_hog_centroid_tracker",
        "note": "Lightweight baseline tracking over sampled frames only; not production-grade identity tracking.",
        "track_count": len(tracks),
        "tracks": [_track_to_dict(track) for track in tracks],
    }


def _detect_people_in_sampled_frames(events_data: dict, root: Path) -> list[dict]:
    hog = cv2.HOGDescriptor()
    hog.setSVMDetector(cv2.HOGDescriptor_getDefaultPeopleDetector())
    detections_by_frame = []

    for segment in events_data.get("segments") or []:
        for frame in segment.get("sampled_frames") or []:
            frame_path = root / str(frame.get("path") or "")
            image = cv2.imread(str(frame_path))
            if image is None:
                continue

            original_height, original_width = image.shape[:2]
            resized_width = 640
            scale = resized_width / max(1, original_width)
            resized_height = int(original_height * scale)
            resized = cv2.resize(image, (resized_width, resized_height))
            rects, weights = hog.detectMultiScale(
                resized,
                winStride=(8, 8),
                padding=(8, 8),
                scale=1.05,
            )

            detections = []
            for rect, weight in zip(rects, weights):
                x, y, w, h = [float(value) for value in rect]
                confidence = float(weight)
                if confidence < 0.2:
                    continue
                bbox = [
                    int(x / scale),
                    int(y / scale),
                    int(w / scale),
                    int(h / scale),
                ]
                center = (bbox[0] + bbox[2] / 2, bbox[1] + bbox[3] / 2)
                detections.append(
                    {
                        "bbox": bbox,
                        "center": center,
                        "confidence": confidence,
                        "segment_id": segment.get("segment_id"),
                        "timestamp": frame.get("timestamp") or segment.get("start_time"),
                        "timestamp_sec": float(frame.get("timestamp_sec") or segment.get("start_time_sec") or 0),
                        "frame_path": str(frame_path),
                    }
                )

            detections_by_frame.append(
                {
                    "segment_id": segment.get("segment_id"),
                    "timestamp_sec": float(frame.get("timestamp_sec") or segment.get("start_time_sec") or 0),
                    "detections": detections,
                }
            )

    return detections_by_frame


def _link_detections(detections_by_frame: list[dict]) -> list[Track]:
    tracks: list[Track] = []
    next_track_id = 1
    max_distance = 180.0

    for frame in sorted(detections_by_frame, key=lambda item: item["timestamp_sec"]):
        assigned_track_ids: set[int] = set()
        for detection in frame["detections"]:
            best_track = None
            best_distance = None
            for track in tracks:
                if track.track_id in assigned_track_ids:
                    continue
                distance = _distance(track.last_center, detection["center"])
                if distance <= max_distance and (best_distance is None or distance < best_distance):
                    best_track = track
                    best_distance = distance

            if best_track is None:
                best_track = Track(
                    track_id=next_track_id,
                    last_center=detection["center"],
                    last_timestamp_sec=detection["timestamp_sec"],
                )
                tracks.append(best_track)
                next_track_id += 1

            best_track.last_center = detection["center"]
            best_track.last_timestamp_sec = detection["timestamp_sec"]
            best_track.observations.append(detection)
            assigned_track_ids.add(best_track.track_id)

    return tracks


def _track_to_dict(track: Track) -> dict:
    observations = [
        {
            "segment_id": observation["segment_id"],
            "timestamp": observation["timestamp"],
            "timestamp_sec": observation["timestamp_sec"],
            "frame_path": observation["frame_path"],
            "bbox": observation["bbox"],
            "confidence": round(observation["confidence"], 4),
        }
        for observation in track.observations
    ]
    return {
        "track_label": f"track_{track.track_id:04d}",
        "first_time": observations[0]["timestamp"] if observations else None,
        "last_time": observations[-1]["timestamp"] if observations else None,
        "observation_count": len(observations),
        "observations": observations,
    }


def _distance(a: tuple[float, float], b: tuple[float, float]) -> float:
    return ((a[0] - b[0]) ** 2 + (a[1] - b[1]) ** 2) ** 0.5
