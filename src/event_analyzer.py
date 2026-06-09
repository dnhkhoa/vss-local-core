from __future__ import annotations

import json
from pathlib import Path

from .ollama_client import OllamaVisionClient


class VideoEventAnalyzer:
    def __init__(self, client: OllamaVisionClient):
        self.client = client

    def analyze_segments(self, segments: list[dict]) -> list[dict]:
        results = []
        for segment in segments:
            print(
                f"Analyzing {segment['segment_id']} "
                f"({segment['start_time_text']} - {segment['end_time_text']}) "
                f"with {len(segment['frame_paths'])} frame(s)..."
            )
            results.append(self.analyze_segment(segment))
        return results

    def analyze_segment(self, segment: dict) -> dict:
        if not segment["frame_paths"]:
            return self._fallback_segment(segment, "No readable sampled frames.")

        prompt = self._build_prompt(segment)
        response_text = self.client.analyze_images(segment["frame_paths"], prompt)
        parsed = _parse_json_response(response_text)
        if parsed is None:
            return self._fallback_segment(
                segment,
                f"Could not parse model response as JSON: {response_text}",
            )

        return _normalize_segment_result(segment, parsed)

    def _build_prompt(self, segment: dict) -> str:
        return f"""You are a factory CCTV video analysis assistant.
You are given sampled frames from one video segment.

Segment time:

* start: {segment["start_time_text"]}
* end: {segment["end_time_text"]}

Analyze only this segment.
Identify visible people, vehicles, machines, movements, unsafe behavior, and important events.

Return JSON only with this schema:
{{
"segment_id": "{segment["segment_id"]}",
"start_time": "{segment["start_time_text"]}",
"end_time": "{segment["end_time_text"]}",
"summary": "short description of what happens in this segment",
"events": [
{{
"event_type": "person_visible | vehicle_visible | machine_activity | danger_zone_violation | unsafe_behavior | normal_activity | unclear",
"description": "short event description",
"approx_time": "approximate timestamp inside this segment, or segment start time if unclear",
"risk_level": "safe | warning | danger | unclear"
}}
],
"has_person": true/false,
"has_vehicle": true/false,
"has_unsafe_behavior": true/false,
"risk_level": "safe | warning | danger | unclear"
}}

* If nothing important happens, return one event with event_type "normal_activity".
* If the frames are unclear, use risk_level "unclear".
* Do not hallucinate. Only describe what is visible."""

    @staticmethod
    def _fallback_segment(segment: dict, reason: str) -> dict:
        return {
            "segment_id": segment["segment_id"],
            "start_time": segment["start_time_text"],
            "end_time": segment["end_time_text"],
            "summary": reason,
            "events": [
                {
                    "event_type": "unclear",
                    "description": reason,
                    "approx_time": segment["start_time_text"],
                    "risk_level": "unclear",
                }
            ],
            "has_person": False,
            "has_vehicle": False,
            "has_unsafe_behavior": False,
            "risk_level": "unclear",
        }


def save_events_json(
    output_path: str,
    video_path: str,
    segment_seconds: float,
    frames_per_segment: int,
    segment_results: list[dict],
) -> dict:
    data = {
        "video_path": video_path,
        "segment_seconds": segment_seconds,
        "frames_per_segment": frames_per_segment,
        "segments": segment_results,
    }

    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")
    return data


def _parse_json_response(text: str) -> dict | None:
    cleaned = text.strip()
    if cleaned.startswith("```"):
        lines = cleaned.splitlines()
        if lines and lines[0].startswith("```"):
            lines = lines[1:]
        if lines and lines[-1].startswith("```"):
            lines = lines[:-1]
        cleaned = "\n".join(lines).strip()

    try:
        parsed = json.loads(cleaned)
    except json.JSONDecodeError:
        return None

    return parsed if isinstance(parsed, dict) else None


def _normalize_segment_result(segment: dict, parsed: dict) -> dict:
    events = parsed.get("events")
    if not isinstance(events, list) or not events:
        events = [
            {
                "event_type": "normal_activity",
                "description": parsed.get("summary", "No important activity detected."),
                "approx_time": segment["start_time_text"],
                "risk_level": parsed.get("risk_level", "safe"),
            }
        ]

    return {
        "segment_id": str(parsed.get("segment_id") or segment["segment_id"]),
        "start_time": str(parsed.get("start_time") or segment["start_time_text"]),
        "end_time": str(parsed.get("end_time") or segment["end_time_text"]),
        "summary": str(parsed.get("summary") or "No summary returned."),
        "events": events,
        "has_person": bool(parsed.get("has_person", False)),
        "has_vehicle": bool(parsed.get("has_vehicle", False)),
        "has_unsafe_behavior": bool(parsed.get("has_unsafe_behavior", False)),
        "risk_level": str(parsed.get("risk_level") or "unclear"),
    }
