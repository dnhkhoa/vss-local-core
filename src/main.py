from __future__ import annotations

from datetime import datetime
from pathlib import Path
import time

import cv2

from .config import load_config
from .danger_zone import draw_danger_zone, parse_polygon_points
from .frame_reader import VideoStreamReader
from .ollama_client import OllamaVisionClient


OUTPUT_DIR = Path("outputs")
LATEST_FRAME_PATH = OUTPUT_DIR / "latest_frame.jpg"


def main() -> int:
    config = load_config()
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    danger_zone_points = []
    if config.enable_danger_zone:
        try:
            danger_zone_points = parse_polygon_points(config.danger_zone_points)
        except ValueError as exc:
            print(f"Danger zone disabled: {exc}")

    try:
        reader = VideoStreamReader(config.video_source)
    except RuntimeError as exc:
        print(exc)
        return 1

    client = OllamaVisionClient(config.ollama_base_url, config.ollama_model)
    last_sample_time = 0.0

    print("VSS local core started. Press 'q' in the preview window to quit.")
    print(f"Video source: {config.video_source}")
    print(f"Ollama: {config.ollama_base_url} model={config.ollama_model}")

    try:
        while True:
            ok, frame = reader.read_frame()
            if not ok:
                print("Could not read frame from video source. Retrying...")
                time.sleep(1.0)
                continue

            if danger_zone_points:
                frame = draw_danger_zone(frame, danger_zone_points)

            cv2.imshow("vss-local-core", frame)

            now = time.monotonic()
            if now - last_sample_time >= config.sample_interval_sec:
                last_sample_time = now
                if cv2.imwrite(str(LATEST_FRAME_PATH), frame):
                    answer = client.analyze_image(str(LATEST_FRAME_PATH), config.question)
                else:
                    answer = f"Failed to write frame to {LATEST_FRAME_PATH}"

                timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                print(f"\n[{timestamp}]")
                print(answer)

            if cv2.waitKey(1) & 0xFF == ord("q"):
                break
    finally:
        reader.release()
        cv2.destroyAllWindows()

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
