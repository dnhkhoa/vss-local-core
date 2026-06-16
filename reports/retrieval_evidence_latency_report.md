# Retrieval/Evidence Latency Report

Date: 2026-06-12

Scope: measure response time for the local web API `/api/ask` after video has already been imported/indexed. This does not include video upload/import time.

## Test Environment

- UI/API: `http://127.0.0.1:7860`
- Model: `qwen3-vl:8b` via Ollama
- Current video: `danger_zone_checking_camera-1_1780970612_org.mp4`
- Evidence returned per question: 4 images/timestamps
- Retrieval report output: `outputs/retrieval_report.json`

## Results

### Video Import/Indexing

| Scenario | Video | Segments | Latency |
| --- | --- | ---: | ---: |
| Re-open already indexed video | `danger_zone_checking_camera-1_1780970612_org.mp4` | 7 | 0.55s |
| Fresh import with `--force` | `danger_zone_checking_camera-1_1780970612_org.mp4` | 7 | 290.65s |

Fresh import average:

- 41.52s per segment
- 20.76s per sampled frame, based on current 2 images per segment sent to the VLM during ingest
- About 20.76x slower than realtime for this 14s video

### Question Answering

| Question | Flow Used | Evidence | Latency |
| --- | --- | ---: | ---: |
| `co bao nhieu nguoi trong video` | Structured video memory + evidence | 4 | 0.33s |
| `co bao nhieu laptop trong video` | Retrieval + VLM visual reasoning | 4 | 59.47s |
| `co quat nao dang xoay khong va mau gi` | Temporal retrieval + VLM visual reasoning | 4 | 73.88s |

## Interpretation

- Video import is the expensive stage because it calls the VLM for each sampled segment and writes the indexed video memory.
- Re-opening an already indexed video is fast because the fingerprint cache skips VLM analysis and reloads memory.
- Fast path is working: questions that can be answered from indexed structured memory, such as visible person count, return in under 1 second.
- VLM path is the bottleneck: questions that require visual reasoning over evidence images take around 1 minute on the current local setup.
- Temporal/motion questions are slower because the system extracts ordered evidence frames and asks the VLM to reason across multiple moments.
- The current local Ollama/Qwen3-VL setup is acceptable for prototype/demo, but not yet production latency for broad customer Q&A.

## Current Production Gap

- The system still uses image evidence batches, not a fully optimized video-native inference backend.
- Ollama local inference has high latency for multiple images.
- Object/action precision depends on the VLM and indexed captions; there is no production-grade detector/tracker yet.
- Retrieval is now traceable and debuggable, but vector retrieval/embedding is not implemented yet.

## Recommended Next Steps

1. Add response mode:
   - Fast answer from structured memory when reliable.
   - Visual verification only when needed.
2. Reduce VLM image calls:
   - First rank evidence more aggressively.
   - Send top 1-2 images for simple object/color questions.
   - Use 4 images only for temporal/action questions.
3. Add detector/tracker:
   - Person/object detection for count, color, bounding boxes.
   - Action/motion tracking for fight/fall/running/fan movement.
4. Add vector retrieval:
   - Store text/image embeddings per segment.
   - Retrieve semantically relevant segments instead of relying mainly on lexical/structured scoring.
5. Benchmark again after optimization:
   - Target local demo: simple questions under 2s, visual questions under 15-25s.
   - Target production GPU server: simple questions under 1s, visual questions under 5-10s depending on model/backend.
