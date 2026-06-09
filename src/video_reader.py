from __future__ import annotations

from pathlib import Path

import cv2


class VideoReader:
    def __init__(self, video_path: str):
        self.video_path = video_path
        path = Path(video_path)
        if not path.exists():
            raise RuntimeError(f"Video file does not exist: {video_path}")

        self.capture = cv2.VideoCapture(str(path))
        if not self.capture.isOpened():
            raise RuntimeError(f"Could not open video file: {video_path}")

        self.fps = float(self.capture.get(cv2.CAP_PROP_FPS) or 0)
        self.total_frame_count = int(self.capture.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
        if self.fps <= 0:
            self.release()
            raise RuntimeError(f"Could not read FPS from video file: {video_path}")

        self.duration_seconds = (
            self.total_frame_count / self.fps if self.total_frame_count > 0 else 0.0
        )
        if self.duration_seconds <= 0:
            self.release()
            raise RuntimeError(f"Could not read duration from video file: {video_path}")

    def read_frame_at(self, timestamp_sec: float):
        timestamp_sec = max(0.0, min(timestamp_sec, self.duration_seconds))
        self.capture.set(cv2.CAP_PROP_POS_MSEC, timestamp_sec * 1000)
        ok, frame = self.capture.read()
        if not ok or frame is None:
            return False, None
        return True, frame

    def release(self) -> None:
        if self.capture is not None:
            self.capture.release()
