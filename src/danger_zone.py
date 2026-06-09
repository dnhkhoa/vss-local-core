from __future__ import annotations

import cv2
import numpy as np


def parse_polygon_points(text: str) -> list[tuple[int, int]]:
    if not text or not text.strip():
        return []

    points: list[tuple[int, int]] = []
    for item in text.split(";"):
        raw_pair = item.strip()
        if not raw_pair:
            continue

        try:
            raw_x, raw_y = raw_pair.split(",", maxsplit=1)
            points.append((int(raw_x.strip()), int(raw_y.strip())))
        except ValueError as exc:
            raise ValueError(
                "DANGER_ZONE_POINTS must use format 'x,y;x,y;x,y', "
                f"got invalid point {raw_pair!r}."
            ) from exc

    if len(points) < 3:
        raise ValueError("DANGER_ZONE_POINTS must contain at least 3 points.")

    return points


def draw_danger_zone(frame, points: list[tuple[int, int]]):
    if not points:
        return frame

    polygon = np.array(points, dtype=np.int32)
    overlay = frame.copy()

    cv2.fillPoly(overlay, [polygon], color=(0, 0, 255))
    cv2.addWeighted(overlay, 0.25, frame, 0.75, 0, frame)
    cv2.polylines(frame, [polygon], isClosed=True, color=(0, 0, 255), thickness=2)

    label_x, label_y = points[0]
    label_y = max(20, label_y - 10)
    cv2.putText(
        frame,
        "DANGER ZONE",
        (label_x, label_y),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.7,
        (0, 0, 255),
        2,
        cv2.LINE_AA,
    )

    return frame
