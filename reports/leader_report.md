# VSS-mini Progress Report

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

- Video: `danger_zone_checking_camera-1_1780970612_org.mp4`
- Segments analyzed: `7`
- Max visible people in a segment: `3`
- Unsafe segments detected: `0`

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
