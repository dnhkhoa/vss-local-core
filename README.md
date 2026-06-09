# vss-local-core

Minimal local VSS-core prototype for sampling camera frames and asking a local Ollama vision-language model for a brief safety-oriented description.

## Architecture

Camera / RTSP / MP4 -> OpenCV -> frame sampler -> Qwen3-VL via Ollama -> terminal answer

This prototype uses sampled frames first, not full continuous video inference. It is intended as a simple local baseline before adding richer video understanding, tracking, rules, and alert workflows.

## Requirements

- Python 3.10+
- OpenCV-compatible webcam, RTSP stream, or MP4 file
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

On Windows PowerShell, activate the virtual environment with:

```powershell
.\.venv\Scripts\Activate.ps1
```

## Ollama Setup

```bash
curl -fsSL https://ollama.com/install.sh | sh
ollama pull qwen3-vl:8b
ollama run qwen3-vl:8b
```

The default Ollama endpoint is:

```text
http://localhost:11434
```

## Run

```bash
python -m src.main
```

Press `q` in the OpenCV preview window to quit.

## Configuration

Copy `.env.example` to `.env`, then adjust values as needed.

### Webcam Example

```env
OLLAMA_BASE_URL=http://localhost:11434
OLLAMA_MODEL=qwen3-vl:8b
VIDEO_SOURCE=0
SAMPLE_INTERVAL_SEC=2
QUESTION=You are a factory CCTV visual assistant. Analyze this camera frame. Describe visible people, machines, vehicles, and possible safety risks. Answer briefly.
ENABLE_DANGER_ZONE=false
```

### MP4 Example

```env
OLLAMA_BASE_URL=http://localhost:11434
OLLAMA_MODEL=qwen3-vl:8b
VIDEO_SOURCE=D:\videos\factory-test.mp4
SAMPLE_INTERVAL_SEC=3
ENABLE_DANGER_ZONE=false
```

### RTSP Example

```env
OLLAMA_BASE_URL=http://localhost:11434
OLLAMA_MODEL=qwen3-vl:8b
VIDEO_SOURCE=rtsp://user:password@192.168.1.20:554/stream1
SAMPLE_INTERVAL_SEC=2
ENABLE_DANGER_ZONE=false
```

### Danger Zone Example

```env
ENABLE_DANGER_ZONE=true
DANGER_ZONE_POINTS=100,200;500,200;520,420;80,420
```

Danger-zone points use `x,y` pairs separated by semicolons. When enabled, the app draws a red polygon overlay and labels it `DANGER ZONE` before saving and sending the sampled frame to Ollama.

## Environment Variables

| Name | Default | Description |
| --- | --- | --- |
| `OLLAMA_BASE_URL` | `http://localhost:11434` | Local Ollama server URL. |
| `OLLAMA_MODEL` | `qwen3-vl:8b` | Vision-language model name in Ollama. |
| `VIDEO_SOURCE` | `0` | Webcam index, RTSP URL, or video file path. |
| `SAMPLE_INTERVAL_SEC` | `2` | Seconds between sampled frames sent to Ollama. |
| `QUESTION` | Safety assistant prompt | Prompt sent with each sampled frame. |
| `ENABLE_DANGER_ZONE` | `false` | Draw a danger-zone overlay when true. |
| `DANGER_ZONE_POINTS` | empty | Polygon points like `100,200;500,200;520,420;80,420`. |

## Outputs

The latest sampled frame is written to:

```text
outputs/latest_frame.jpg
```

Generated output files are ignored by git. Only `outputs/.gitkeep` is tracked.

## Future Extensions

- Object/person tracking
- Safety rule engine
- Clip-based VLM analysis
- Alert verification
- Report generation
- Web dashboard
