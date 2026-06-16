from __future__ import annotations

import json
import re
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
            retry_prompt = self._build_retry_prompt(segment)
            response_text = self.client.analyze_images(segment["frame_paths"], retry_prompt)
            parsed = _parse_json_response(response_text)
        if parsed is None:
            return self._fallback_segment(segment, "Model response could not be parsed as JSON.")

        return _normalize_segment_result(segment, parsed)

    def _build_prompt(self, segment: dict) -> str:
        return f"""/no_think
Analyze these sampled factory CCTV frames for segment {segment["segment_id"]} from {segment["start_time_text"]} to {segment["end_time_text"]}.
Return one compact JSON object only. No explanation.
Fields: segment_id,start_time,end_time,scene,location,summary,person_count,people,objects,activities,visible_text,events,has_person,has_vehicle,has_unsafe_behavior,risk_level.
people is an array of visible people with clothing_colors and position. Example: [{{"person_label":"person_1","clothing_colors":["white shirt","black pants"],"position":"left"}}].
objects is an array of visible objects with name,count,position. Include furniture, tools, machines, doors, computers, vehicles, signs.
activities is an array of short visible actions. Example: ["person using laptop", "person walking near door"].
visible_text is an array of readable text strings in the frames.
events is an array of objects with event_type,description,approx_time,risk_level.
person_count is the number of visible people in the sampled frames, or null if unclear.
Risk levels must be one of: safe, warning, danger, unclear.
Use only visible evidence."""

    def _build_retry_prompt(self, segment: dict) -> str:
        return f"""/no_think
Return one small JSON object only for sampled CCTV segment {segment["segment_id"]}, {segment["start_time_text"]} to {segment["end_time_text"]}.
Fields: scene,location,summary,person_count,people,objects,activities,visible_text,risk_level.
Keep it short. Max 3 people, max 6 objects, max 4 activities. Use only visible evidence."""

    @staticmethod
    def _fallback_segment(segment: dict, reason: str) -> dict:
        return {
            "segment_id": segment["segment_id"],
            "start_time": segment["start_time_text"],
            "end_time": segment["end_time_text"],
            "scene": "unclear",
            "location": "unclear",
            "summary": reason,
            "objects": [],
            "activities": [],
            "visible_text": [],
            "people": [],
            "events": [
                {
                    "event_type": "unclear",
                    "description": reason,
                    "approx_time": segment["start_time_text"],
                    "risk_level": "unclear",
                }
            ],
            "has_person": False,
            "person_count": 0,
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
    extra_fields: dict | None = None,
) -> dict:
    data = {
        "video_path": video_path,
        "segment_seconds": segment_seconds,
        "frames_per_segment": frames_per_segment,
        "analysis_summary": _build_analysis_summary(segment_results),
        "segments": segment_results,
    }
    if extra_fields:
        data.update(extra_fields)

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
    events = _normalize_events(parsed.get("events"), segment, parsed)
    people = _normalize_people(parsed.get("people"))
    objects = _normalize_objects(parsed.get("objects"))
    activities = _normalize_string_list(parsed.get("activities"))
    visible_text = _normalize_string_list(parsed.get("visible_text"))
    person_count = _parse_person_count(parsed.get("person_count"))
    if person_count is None:
        person_count = len(people) if people else _infer_person_count(parsed)

    risk_level = _normalize_risk_level(parsed.get("risk_level"))
    has_person = bool(parsed.get("has_person", False)) or person_count > 0

    return {
        "segment_id": str(parsed.get("segment_id") or segment["segment_id"]),
        "start_time": str(parsed.get("start_time") or segment["start_time_text"]),
        "end_time": str(parsed.get("end_time") or segment["end_time_text"]),
        "sampled_frames": segment.get("sampled_frames", []),
        "scene": str(parsed.get("scene") or "unclear"),
        "location": str(parsed.get("location") or "unclear"),
        "summary": str(parsed.get("summary") or "No summary returned."),
        "person_count": person_count,
        "people": people,
        "objects": objects,
        "activities": activities,
        "visible_text": visible_text,
        "events": events,
        "has_person": has_person,
        "has_vehicle": bool(parsed.get("has_vehicle", False)),
        "has_unsafe_behavior": bool(parsed.get("has_unsafe_behavior", False)),
        "risk_level": risk_level,
        "searchable_text": _build_segment_searchable_text(parsed, people, objects, activities, visible_text, events),
    }


def _normalize_events(raw_events, segment: dict, parsed: dict) -> list[dict]:
    if not isinstance(raw_events, list) or not raw_events:
        events = [
            {
                "event_type": "normal_activity",
                "description": parsed.get("summary", "No important activity detected."),
                "approx_time": segment["start_time_text"],
                "risk_level": _normalize_risk_level(parsed.get("risk_level", "safe")),
            }
        ]
    else:
        events = []
        for raw_event in raw_events:
            if isinstance(raw_event, dict):
                description = str(raw_event.get("description") or raw_event.get("event_type") or "")
                event_type = str(raw_event.get("event_type") or "normal_activity")
                approx_time = str(raw_event.get("approx_time") or segment["start_time_text"])
                risk_level = _normalize_risk_level(raw_event.get("risk_level", "unclear"))
            else:
                description = str(raw_event)
                event_type = "normal_activity"
                approx_time = segment["start_time_text"]
                risk_level = _normalize_risk_level(parsed.get("risk_level", "safe"))

            events.append(
                {
                    "event_type": event_type,
                    "description": description,
                    "approx_time": approx_time,
                    "risk_level": risk_level,
                }
            )

    return events


def _normalize_people(raw_people) -> list[dict]:
    if not isinstance(raw_people, list):
        return []

    people = []
    for index, raw_person in enumerate(raw_people, start=1):
        if isinstance(raw_person, dict):
            colors = raw_person.get("clothing_colors") or raw_person.get("clothes") or raw_person.get("clothing")
            if isinstance(colors, str):
                clothing_colors = [colors]
            elif isinstance(colors, list):
                clothing_colors = [str(color) for color in colors if str(color).strip()]
            else:
                clothing_colors = []

            people.append(
                {
                    "person_label": str(raw_person.get("person_label") or f"person_{index}"),
                    "clothing_colors": clothing_colors,
                    "position": str(raw_person.get("position") or "unknown"),
                }
            )
        else:
            people.append(
                {
                    "person_label": f"person_{index}",
                    "clothing_colors": [str(raw_person)],
                    "position": "unknown",
                }
            )

    return people


def _normalize_objects(raw_objects) -> list[dict]:
    if not isinstance(raw_objects, list):
        return []

    objects = []
    for raw_object in raw_objects:
        if isinstance(raw_object, dict):
            name = str(raw_object.get("name") or raw_object.get("object") or "").strip()
            if not name:
                continue
            count = _parse_person_count(raw_object.get("count"))
            objects.append(
                {
                    "name": name,
                    "count": count,
                    "position": str(raw_object.get("position") or "unknown"),
                        "attributes": _normalize_string_list(raw_object.get("attributes")),
                    }
                )
        elif isinstance(raw_object, list) and raw_object:
            name = str(raw_object[0]).strip()
            if not name:
                continue
            count = _parse_person_count(raw_object[1]) if len(raw_object) > 1 else None
            position = str(raw_object[2]) if len(raw_object) > 2 else "unknown"
            attributes = _normalize_string_list(raw_object[3]) if len(raw_object) > 3 else []
            objects.append(
                {
                    "name": name,
                    "count": count,
                    "position": position,
                    "attributes": attributes,
                }
            )
        else:
            name = str(raw_object).strip()
            if name:
                objects.append(
                    {
                        "name": name,
                        "count": None,
                        "position": "unknown",
                        "attributes": [],
                    }
                )

    return objects


def _normalize_string_list(value) -> list[str]:
    if value is None:
        return []
    if isinstance(value, str):
        return [value] if value.strip() else []
    if isinstance(value, list):
        return [str(item) for item in value if str(item).strip()]
    return [str(value)]


def _build_segment_searchable_text(
    parsed: dict,
    people: list[dict],
    objects: list[dict],
    activities: list[str],
    visible_text: list[str],
    events: list[dict],
) -> str:
    parts = [
        str(parsed.get("scene") or ""),
        str(parsed.get("location") or ""),
        str(parsed.get("summary") or ""),
        " ".join(activities),
        " ".join(visible_text),
    ]
    for person in people:
        parts.append(str(person.get("position") or ""))
        parts.extend(person.get("clothing_colors") or [])
    for obj in objects:
        parts.append(str(obj.get("name") or ""))
        parts.append(str(obj.get("position") or ""))
        parts.extend(obj.get("attributes") or [])
    for event in events:
        parts.append(str(event.get("event_type") or ""))
        parts.append(str(event.get("description") or ""))
    return " ".join(part for part in parts if part).strip()


def _parse_person_count(value) -> int | None:
    if value is None:
        return None
    if isinstance(value, bool):
        return None
    if isinstance(value, int):
        return max(0, value)
    if isinstance(value, float):
        return max(0, int(value))
    match = re.search(r"\d+", str(value))
    return int(match.group(0)) if match else None


def _infer_person_count(parsed: dict) -> int:
    text_parts = [str(parsed.get("summary") or "")]
    for event in parsed.get("events") or []:
        if isinstance(event, dict):
            text_parts.append(str(event.get("description") or ""))
            text_parts.append(str(event.get("event_type") or ""))
        else:
            text_parts.append(str(event))
    text = " ".join(text_parts).lower()

    word_counts = {
        "one": 1,
        "a person": 1,
        "an individual": 1,
        "single": 1,
        "two": 2,
        "couple": 2,
        "three": 3,
        "four": 4,
        "five": 5,
        "một": 1,
        "hai": 2,
        "ba": 3,
        "bốn": 4,
        "nam": 5,
    }
    for word, count in sorted(word_counts.items(), key=lambda item: len(item[0]), reverse=True):
        if word in text:
            return count

    digit_match = re.search(r"\b(\d+)\s+(person|people|individual|individuals|employee|employees|worker|workers|người)", text)
    if digit_match:
        return int(digit_match.group(1))

    person_terms = ("person", "people", "individual", "employee", "worker", "staff", "người")
    if parsed.get("has_person") or any(term in text for term in person_terms):
        return 1
    return 0


def _normalize_risk_level(value) -> str:
    risk_level = str(value or "unclear").strip().lower()
    if risk_level in {"low", "normal"}:
        return "safe"
    if risk_level in {"medium", "caution", "high"}:
        return "warning"
    if risk_level in {"critical", "unsafe"}:
        return "danger"
    return risk_level if risk_level in {"safe", "warning", "danger", "unclear"} else "unclear"


def _build_analysis_summary(segments: list[dict]) -> dict:
    unsafe_segments = [
        {
            "segment_id": segment["segment_id"],
            "start_time": segment["start_time"],
            "end_time": segment["end_time"],
            "risk_level": segment["risk_level"],
            "summary": segment["summary"],
        }
        for segment in segments
        if segment.get("has_unsafe_behavior") or segment.get("risk_level") in {"warning", "danger"}
    ]
    person_counts = [int(segment.get("person_count", 0) or 0) for segment in segments]
    segments_with_people = [
        {
            "segment_id": segment["segment_id"],
            "start_time": segment["start_time"],
            "end_time": segment["end_time"],
            "person_count": int(segment.get("person_count", 0) or 0),
        }
        for segment in segments
        if int(segment.get("person_count", 0) or 0) > 0 or segment.get("has_person")
    ]
    clothing_observations = [
        {
            "segment_id": segment["segment_id"],
            "start_time": segment["start_time"],
            "end_time": segment["end_time"],
            "people": segment.get("people", []),
        }
        for segment in segments
        if segment.get("people")
    ]
    object_observations = [
        {
            "segment_id": segment["segment_id"],
            "start_time": segment["start_time"],
            "end_time": segment["end_time"],
            "objects": segment.get("objects", []),
        }
        for segment in segments
        if segment.get("objects")
    ]
    activity_observations = [
        {
            "segment_id": segment["segment_id"],
            "start_time": segment["start_time"],
            "end_time": segment["end_time"],
            "activities": segment.get("activities", []),
        }
        for segment in segments
        if segment.get("activities")
    ]
    scene_observations = [
        {
            "segment_id": segment["segment_id"],
            "start_time": segment["start_time"],
            "end_time": segment["end_time"],
            "scene": segment.get("scene", "unclear"),
            "location": segment.get("location", "unclear"),
            "summary": segment.get("summary", ""),
        }
        for segment in segments
    ]
    text_observations = [
        {
            "segment_id": segment["segment_id"],
            "start_time": segment["start_time"],
            "end_time": segment["end_time"],
            "visible_text": segment.get("visible_text", []),
        }
        for segment in segments
        if segment.get("visible_text")
    ]

    return {
        "segment_count": len(segments),
        "max_visible_person_count": max(person_counts, default=0),
        "segments_with_people": segments_with_people,
        "clothing_observations": clothing_observations,
        "object_observations": object_observations,
        "activity_observations": activity_observations,
        "scene_observations": scene_observations,
        "text_observations": text_observations,
        "unsafe_segments": unsafe_segments,
        "unique_person_count": None,
        "unique_person_count_note": (
            "Not available without person detection and tracking across frames."
        ),
    }
