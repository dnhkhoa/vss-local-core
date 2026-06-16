# VSS-mini Aim

This project should behave like a small local Video Search and Summarization system, not just a fixed event classifier.

## Target

Given a video, the system should:

1. Build a reusable video memory from sampled frames.
2. Store structured evidence per segment:
   - time range
   - evidence frame paths
   - scene and location
   - people, clothing, positions
   - objects
   - activities
   - visible text
   - risk and events
   - searchable text
3. Answer common questions directly from structured memory.
4. For harder questions, retrieve relevant segments and ask the vision model to inspect the evidence frames again.
5. Avoid pretending to know things the pipeline cannot prove, especially unique identity tracking across time.

## Current Level

Implemented:

- Segment-level video memory in `outputs/events.json`.
- Indexed video memory in `outputs/video_memory.sqlite`.
- Fast structured Q&A for people count, clothing, objects, activities, visible text, scene, timing, and safety.
- Visual re-check for hard questions by retrieving evidence frames and asking the VLM to inspect them again.
- Lightweight sampled-frame person tracking baseline.
- Web UI for analyze and ask.

Next:

- Upgrade lightweight tracking to detector-based production tracking with stable identity IDs.
- Add vector search when keyword FTS is not enough for longer videos.
- Add background job queue, progress reporting, and API auth for deployment.
