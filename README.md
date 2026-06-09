# vss-local-core

Minimal local video event analysis prototype for MP4/video files. It samples short video segments, sends selected frames to a local Ollama vision-language model, saves timestamped event summaries to JSON, and can answer questions from those saved events.

## Architecture

Video -> segment sampler -> sampled frames -> Qwen3-VL via Ollama -> timestamped events -> Q&A

The system does not send the full video directly to the model. It reads the video with OpenCV, splits it into short segments, samples a few frames from each segment, asks Qwen3-VL what is visible in that segment, and writes the result to `outputs/events.json`.

Timestamps are approximate because the model analyzes sampled frames, not every frame.

## Requirements

- Python 3.10+
- OpenCV-readable video file
- Ollama running locally
- Ollama model `qwen3-vl:8b`

The code does not download models and the repository does not include model files.

## Install

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
cp .env.example .env
```

On Windows PowerShell:

```powershell
.\.venv\Scripts\Activate.ps1
```

## Ollama Setup

```bash
ollama pull qwen3-vl:8b
ollama run qwen3-vl:8b
```

Default endpoint:

```text
http://localhost:11434
```

## Configuration

Copy `.env.example` to `.env` and adjust values:

```env
OLLAMA_BASE_URL=http://localhost:11434
OLLAMA_MODEL=qwen3-vl:8b
VIDEO_SOURCE=sample.mp4
SEGMENT_SECONDS=5
FRAMES_PER_SEGMENT=3
OUTPUT_EVENTS_PATH=outputs/events.json
QUESTION=
```

Increasing `FRAMES_PER_SEGMENT` gives the model more visual context, but inference becomes slower because more images are sent per segment.

## Run

### Analyze Video

```bash
python -m src.main analyze --video samples/factory.mp4
```

This samples frames into `outputs/frames/`, analyzes each segment with Ollama, and saves:

```text
outputs/events.json
```

### Ask About Saved Events

```bash
python -m src.main ask --question "Trong video co nhung su kien gi?"
```

The QA mode uses only `outputs/events.json`. It does not send video frames to the model.

### Analyze And Ask

```bash
python -m src.main analyze-ask --video samples/factory.mp4 --question "Co hanh vi nguy hiem nao khong? Xay ra luc nao?"
```

This analyzes the video, saves the event JSON, then answers the question from the generated events.

## Output Format

`outputs/events.json` has this shape:

```json
{
  "video_path": "samples/factory.mp4",
  "segment_seconds": 5,
  "frames_per_segment": 3,
  "segments": [
    {
      "segment_id": "segment_0001",
      "start_time": "00:00:00",
      "end_time": "00:00:05",
      "summary": "short description of what happens in this segment",
      "events": [
        {
          "event_type": "person_visible",
          "description": "short event description",
          "approx_time": "00:00:02",
          "risk_level": "safe"
        }
      ],
      "has_person": true,
      "has_vehicle": false,
      "has_unsafe_behavior": false,
      "risk_level": "safe"
    }
  ]
}
```

Generated frames and event JSON are ignored by git. Only `outputs/.gitkeep` is tracked.

## Future Extensions

- Danger zone overlay
- Object detection
- Tracking
- Safety rule engine
- Clip-based VLM analysis
- Report generation
- Dashboard
