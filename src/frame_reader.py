from __future__ import annotations

import cv2


class VideoStreamReader:
    def __init__(self, source: str):
        self.source = self._normalize_source(source)
        self.capture = cv2.VideoCapture(self.source)
        if not self.capture.isOpened():
            raise RuntimeError(f"Could not open video source: {source}")

    @staticmethod
    def _normalize_source(source: str):
        text = str(source).strip()
        if text.isdigit():
            return int(text)
        return text

    def read_frame(self):
        if not self.capture.isOpened():
            return False, None

        ok, frame = self.capture.read()
        if not ok or frame is None:
            return False, None

        return True, frame

    def release(self) -> None:
        if self.capture is not None:
            self.capture.release()
