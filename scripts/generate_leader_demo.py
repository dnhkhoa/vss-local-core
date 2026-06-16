from __future__ import annotations

import json
import textwrap
from pathlib import Path

import cv2
import numpy as np


ROOT = Path(__file__).resolve().parents[1]
REPORTS = ROOT / "reports"
EVENTS_PATH = ROOT / "outputs" / "events.json"


DEMO_QUESTIONS = [
    {
        "question": "Video nay o dau, boi canh la gi?",
        "answer": "Office meeting room / I-Soft Meeting room. Video co nhieu nguoi lam viec quanh ban, laptop, hop, quat va cay trang tri.",
        "evidence": ["00:00:00-00:00:14", "scene/location fields in events.json"],
    },
    {
        "question": "Co bao nhieu nguoi trong video?",
        "answer": "He thong thay toi da khoang 3 nguoi trong cung mot segment. Chua ket luan unique person count vi chua tracking identity.",
        "evidence": ["analysis_summary.max_visible_person_count = 3"],
    },
    {
        "question": "Nhung nguoi do mac ao mau gi?",
        "answer": "Co ao trang, ao vang/beige, ao xanh nhat va nguoi mac vest xanh/helmet o mot frame.",
        "evidence": ["people[].clothing_colors by segment"],
    },
    {
        "question": "Trong video co laptop khong?",
        "answer": "Co. Laptop xuat hien o nhieu segment tren ban lam viec.",
        "evidence": ["objects contains laptop"],
    },
    {
        "question": "Nguoi ao vang co dang cam gi khong?",
        "answer": "Co. Visual re-check tren frame lien quan thay nguoi ao vang dang cam mot vat trang/xanh.",
        "evidence": ["visual re-check over retrieved frames"],
    },
]


def main() -> None:
    REPORTS.mkdir(exist_ok=True)
    events = json.loads(EVENTS_PATH.read_text(encoding="utf-8"))
    frames = _select_frames(events)
    _write_demo_questions()
    _write_contact_sheet(frames, events)
    _write_demo_video(frames, events)
    _write_markdown_report(events)
    _write_html_report(events)
    print(REPORTS / "leader_report.md")
    print(REPORTS / "leader_demo.html")
    print(REPORTS / "vss_mini_demo.mp4")
    print(REPORTS / "evidence_contact_sheet.jpg")


def _select_frames(events: dict) -> list[dict]:
    selected = []
    for segment in events.get("segments", []):
        sampled_frames = segment.get("sampled_frames") or []
        if not sampled_frames:
            continue
        frame_path = ROOT / sampled_frames[0]["path"]
        if not frame_path.exists():
            continue
        selected.append(
            {
                "path": frame_path,
                "segment_id": segment.get("segment_id"),
                "time": f"{segment.get('start_time')}-{segment.get('end_time')}",
                "summary": segment.get("summary") or "",
                "person_count": segment.get("person_count", 0),
                "scene": segment.get("scene", "unclear"),
                "activities": segment.get("activities") or [],
                "objects": [obj.get("name") for obj in segment.get("objects", [])[:5]],
                "people": segment.get("people") or [],
            }
        )
    return selected


def _write_demo_questions() -> None:
    (REPORTS / "demo_questions.json").write_text(
        json.dumps(DEMO_QUESTIONS, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )


def _write_contact_sheet(frames: list[dict], events: dict) -> None:
    thumbs = []
    for item in frames:
        frame = cv2.imread(str(item["path"]))
        if frame is None:
            continue
        frame = cv2.resize(frame, (420, 236))
        _overlay_text(
            frame,
            [
                f"{item['segment_id']} | {item['time']}",
                f"people={item['person_count']} | {item['scene']}",
            ],
            top=10,
        )
        thumbs.append(frame)

    if not thumbs:
        return

    cols = 2
    rows = int(np.ceil(len(thumbs) / cols))
    sheet = np.full((rows * 236, cols * 420, 3), 245, dtype=np.uint8)
    for index, thumb in enumerate(thumbs):
        row = index // cols
        col = index % cols
        sheet[row * 236 : (row + 1) * 236, col * 420 : (col + 1) * 420] = thumb
    cv2.imwrite(str(REPORTS / "evidence_contact_sheet.jpg"), sheet)


def _write_demo_video(frames: list[dict], events: dict) -> None:
    out_path = REPORTS / "vss_mini_demo.mp4"
    width, height = 1280, 720
    writer = cv2.VideoWriter(str(out_path), cv2.VideoWriter_fourcc(*"mp4v"), 1.0, (width, height))
    if not writer.isOpened():
        raise RuntimeError("Could not create demo video")

    slides = [_title_slide(width, height, events)]
    for item in frames[:7]:
        slides.append(_frame_slide(width, height, item))
    slides.extend(_qa_slides(width, height))
    slides.append(_closing_slide(width, height))

    for slide in slides:
        for _ in range(2):
            writer.write(slide)
    writer.release()


def _title_slide(width: int, height: int, events: dict) -> np.ndarray:
    frame = _blank(width, height)
    summary = events.get("analysis_summary") or {}
    lines = [
        "VSS-mini local demo",
        f"Video: {events.get('video_path')}",
        f"Segments: {summary.get('segment_count')} | Max visible people: {summary.get('max_visible_person_count')}",
        "Added: video memory, structured Q&A, visual re-check, web UI",
    ]
    _overlay_text(frame, lines, top=90, scale=1.0, thickness=2)
    return frame


def _frame_slide(width: int, height: int, item: dict) -> np.ndarray:
    frame = _blank(width, height)
    image = cv2.imread(str(item["path"]))
    if image is None:
        return frame
    image = _fit(image, 760, 520)
    y0 = 110
    x0 = 40
    frame[y0 : y0 + image.shape[0], x0 : x0 + image.shape[1]] = image
    text_lines = [
        f"{item['segment_id']} | {item['time']}",
        f"Scene: {item['scene']}",
        f"People: {item['person_count']}",
        "Activities: " + ", ".join(item["activities"][:3]),
        "Objects: " + ", ".join([obj for obj in item["objects"] if obj][:5]),
    ]
    _overlay_text(frame, text_lines, left=840, top=130, width=380, scale=0.62)
    return frame


def _qa_slides(width: int, height: int) -> list[np.ndarray]:
    slides = []
    for item in DEMO_QUESTIONS:
        frame = _blank(width, height)
        lines = [
            "Demo Q&A",
            "Q: " + item["question"],
            "A: " + item["answer"],
            "Evidence: " + "; ".join(item["evidence"]),
        ]
        _overlay_text(frame, lines, top=90, scale=0.78, thickness=2)
        slides.append(frame)
    return slides


def _closing_slide(width: int, height: int) -> np.ndarray:
    frame = _blank(width, height)
    lines = [
        "Current capability",
        "- Fast answers from outputs/events.json for common questions",
        "- Hard questions retrieve relevant frames and re-check with VLM",
        "- Next AIM: person/object tracking for identity across time",
    ]
    _overlay_text(frame, lines, top=120, scale=0.82, thickness=2)
    return frame


def _write_markdown_report(events: dict) -> None:
    summary = events.get("analysis_summary") or {}
    content = f"""# VSS-mini Progress Report

## What was added

- Local web Q&A UI: `http://127.0.0.1:7860`
- Structured video memory in `outputs/events.json`
- Segment evidence: scene, location, people, clothing, objects, activities, visible text, risk, frame paths
- Fast Q&A from structured memory
- Visual re-check for hard questions: retrieve relevant evidence frames and ask Qwen3-VL to inspect again
- AIM document: `AIM.md`

## Demo video

- `reports/vss_mini_demo.mp4`
- `reports/evidence_contact_sheet.jpg`
- `reports/leader_demo.html`

## Current demo data

- Video: `{events.get("video_path")}`
- Segments analyzed: `{summary.get("segment_count")}`
- Max visible people in a segment: `{summary.get("max_visible_person_count")}`
- Unsafe segments detected: `{len(summary.get("unsafe_segments") or [])}`

## Example questions now supported

1. Video nay o dau, boi canh la gi?
2. Co bao nhieu nguoi trong video?
3. Nhung nguoi do mac ao mau gi?
4. Trong video co laptop khong?
5. Co chu gi nhin thay trong video?
6. Luc nao co nguoi di gan cua?
7. Nguoi ao vang co dang cam gi khong?
8. Co hanh vi nguy hiem nao khong?

## Limitation

The system does not yet perform true person identity tracking across time. It can report segment-level observations and visually re-check relevant frames, but cannot prove that a person in one segment is the same person in another segment.

## Next AIM

Add person/object tracking to support identity questions such as "nguoi ao vang o dau video co phai cung nguoi o cuoi video khong?" and "nguoi do di tu dau den dau?"
"""
    (REPORTS / "leader_report.md").write_text(content, encoding="utf-8")


def _write_html_report(events: dict) -> None:
    summary = events.get("analysis_summary") or {}
    html = f"""<!doctype html>
<html lang="vi">
<head>
  <meta charset="utf-8">
  <title>VSS-mini Leader Demo</title>
  <style>
    body {{ font-family: Arial, sans-serif; margin: 32px; background: #f6f7f9; color: #172033; }}
    section {{ background: white; border: 1px solid #d8dee8; border-radius: 8px; padding: 18px; margin-bottom: 18px; }}
    h1, h2 {{ margin-top: 0; }}
    video, img {{ max-width: 100%; border: 1px solid #d8dee8; border-radius: 8px; background: #111; }}
    li {{ margin: 8px 0; }}
    code {{ background: #eef2f6; padding: 2px 5px; border-radius: 4px; }}
  </style>
</head>
<body>
  <h1>VSS-mini Progress Demo</h1>
  <section>
    <h2>Summary</h2>
    <ul>
      <li>Video: <code>{events.get("video_path")}</code></li>
      <li>Segments analyzed: <strong>{summary.get("segment_count")}</strong></li>
      <li>Max visible people in a segment: <strong>{summary.get("max_visible_person_count")}</strong></li>
      <li>Unsafe segments: <strong>{len(summary.get("unsafe_segments") or [])}</strong></li>
    </ul>
  </section>
  <section>
    <h2>Demo Video</h2>
    <video src="vss_mini_demo.mp4" controls></video>
  </section>
  <section>
    <h2>Evidence Contact Sheet</h2>
    <img src="evidence_contact_sheet.jpg" alt="Evidence frames">
  </section>
  <section>
    <h2>Implemented</h2>
    <ul>
      <li>Web Q&A UI</li>
      <li>Structured video memory in <code>outputs/events.json</code></li>
      <li>Fast Q&A for common questions</li>
      <li>Visual re-check for harder questions using retrieved frames</li>
    </ul>
  </section>
  <section>
    <h2>Next AIM</h2>
    <p>Add person/object tracking to answer identity and path questions across time.</p>
  </section>
</body>
</html>
"""
    (REPORTS / "leader_demo.html").write_text(html, encoding="utf-8")


def _blank(width: int, height: int) -> np.ndarray:
    frame = np.full((height, width, 3), (246, 247, 249), dtype=np.uint8)
    cv2.rectangle(frame, (24, 24), (width - 24, height - 24), (255, 255, 255), -1)
    cv2.rectangle(frame, (24, 24), (width - 24, height - 24), (220, 225, 235), 2)
    return frame


def _fit(image: np.ndarray, max_width: int, max_height: int) -> np.ndarray:
    height, width = image.shape[:2]
    scale = min(max_width / width, max_height / height)
    return cv2.resize(image, (int(width * scale), int(height * scale)))


def _overlay_text(
    frame: np.ndarray,
    lines: list[str],
    *,
    left: int = 70,
    top: int = 70,
    width: int = 1080,
    scale: float = 0.72,
    thickness: int = 1,
) -> None:
    y = top
    for line in lines:
        wrapped = textwrap.wrap(line, width=max(20, int(width / 16)))
        if not wrapped:
            wrapped = [""]
        for part in wrapped:
            cv2.putText(
                frame,
                part,
                (left, y),
                cv2.FONT_HERSHEY_SIMPLEX,
                scale,
                (24, 35, 48),
                thickness,
                cv2.LINE_AA,
            )
            y += int(34 * scale) + 10
        y += 8


if __name__ == "__main__":
    main()
