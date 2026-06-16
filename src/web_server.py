from __future__ import annotations

import argparse
import json
import mimetypes
import os
import sys
import threading
import time
import uuid
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import unquote, urlparse

from .config import load_config
from .ingestion_pipeline import ingest_video
from .ollama_client import OllamaVisionClient
from .qa_over_events import answer_question_with_evidence
from .video_memory import VideoMemoryStore


ROOT_DIR = Path(__file__).resolve().parents[1]
WEB_DIR = ROOT_DIR / "web"
VIDEO_EXTENSIONS = {".mp4", ".mov", ".avi", ".mkv", ".webm"}

JOBS: dict[str, dict] = {}
JOBS_LOCK = threading.Lock()


def run_server(host: str, port: int) -> None:
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")
        sys.stderr.reconfigure(encoding="utf-8")

    os.chdir(ROOT_DIR)
    server = ThreadingHTTPServer((host, port), VssRequestHandler)
    print(f"VSS local UI running at http://{host}:{port}")
    server.serve_forever()


class VssRequestHandler(BaseHTTPRequestHandler):
    server_version = "VSSLocalCore/0.2"

    def do_GET(self) -> None:
        parsed = urlparse(self.path)
        if parsed.path == "/":
            self._send_file(WEB_DIR / "index.html")
            return
        if parsed.path == "/api/state":
            self._send_json(_get_state())
            return
        if parsed.path.startswith("/api/jobs/"):
            self._handle_get_job(parsed.path.removeprefix("/api/jobs/"))
            return
        if parsed.path.startswith("/assets/"):
            self._send_file(WEB_DIR / unquote(parsed.path.removeprefix("/assets/")))
            return
        if parsed.path.startswith("/outputs/"):
            self._send_output_file(ROOT_DIR / unquote(parsed.path.removeprefix("/")))
            return
        self._send_error(404, "Not found")

    def do_POST(self) -> None:
        parsed = urlparse(self.path)
        if parsed.path == "/api/ask":
            self._handle_ask()
            return
        if parsed.path == "/api/analyze":
            self._handle_analyze()
            return
        self._send_error(404, "Not found")

    def log_message(self, format: str, *args) -> None:
        print(f"{self.address_string()} - {format % args}")

    def _handle_ask(self) -> None:
        payload = self._read_json()
        question = str(payload.get("question") or "").strip()
        if not question:
            self._send_error(400, "Question is required.")
            return

        config = load_config()
        client = OllamaVisionClient(config.ollama_base_url, config.ollama_model)
        result = answer_question_with_evidence(
            config.output_events_path,
            question,
            client,
            memory_db_path=config.memory_db_path,
        )
        self._send_json(
            {
                "answer": result.get("answer"),
                "evidence": result.get("evidence") or [],
                "state": _get_state(),
            }
        )

    def _handle_analyze(self) -> None:
        payload = self._read_json()
        video_path = str(payload.get("video_path") or "").strip()
        force = bool(payload.get("force"))
        if not video_path:
            self._send_error(400, "video_path is required.")
            return

        resolved_video = _resolve_video_path(video_path)
        if resolved_video is None:
            self._send_error(400, "Video path is invalid or outside the project.")
            return
        if not resolved_video.exists():
            self._send_error(404, f"Video file does not exist: {video_path}")
            return
        if _has_active_job():
            self._send_error(409, "Another analysis job is already running on this local worker.")
            return

        job_id = uuid.uuid4().hex
        rel_video = str(resolved_video.relative_to(ROOT_DIR))
        now = time.time()
        job = {
            "id": job_id,
            "status": "queued",
            "video_path": rel_video,
            "force": force,
            "stage": "queued",
            "message": "Queued for local ingestion.",
            "percent": 0,
            "total_segments": None,
            "completed_segments": 0,
            "started_at": now,
            "updated_at": now,
            "finished_at": None,
            "result": None,
            "error": None,
        }
        with JOBS_LOCK:
            JOBS[job_id] = job

        worker = threading.Thread(
            target=_run_analyze_job,
            args=(job_id, rel_video, force),
            daemon=True,
        )
        worker.start()
        self._send_json({"job_id": job_id, "job": _job_snapshot(job)}, status=202)

    def _handle_get_job(self, job_id: str) -> None:
        with JOBS_LOCK:
            job = JOBS.get(job_id)
            if not job:
                self._send_error(404, "Job not found.")
                return
            snapshot = _job_snapshot(job)
        self._send_json({"job": snapshot})

    def _read_json(self) -> dict:
        length = int(self.headers.get("Content-Length") or 0)
        raw_body = self.rfile.read(length).decode("utf-8") if length else "{}"
        try:
            data = json.loads(raw_body)
        except json.JSONDecodeError:
            return {}
        return data if isinstance(data, dict) else {}

    def _send_json(self, data: dict, status: int = 200) -> None:
        body = json.dumps(data, ensure_ascii=False, indent=2).encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def _send_file(self, path: Path) -> None:
        resolved_path = path.resolve()
        try:
            resolved_path.relative_to(WEB_DIR.resolve())
        except ValueError:
            self._send_error(403, "Forbidden")
            return
        if not resolved_path.exists() or not resolved_path.is_file():
            self._send_error(404, "Not found")
            return

        content = resolved_path.read_bytes()
        content_type = mimetypes.guess_type(str(resolved_path))[0] or "application/octet-stream"
        self.send_response(200)
        self.send_header("Content-Type", content_type)
        self.send_header("Content-Length", str(len(content)))
        self.end_headers()
        self.wfile.write(content)

    def _send_output_file(self, path: Path) -> None:
        resolved_path = path.resolve()
        outputs_dir = (ROOT_DIR / "outputs").resolve()
        try:
            resolved_path.relative_to(outputs_dir)
        except ValueError:
            self._send_error(403, "Forbidden")
            return
        if not resolved_path.exists() or not resolved_path.is_file():
            self._send_error(404, "Not found")
            return

        content = resolved_path.read_bytes()
        content_type = mimetypes.guess_type(str(resolved_path))[0] or "application/octet-stream"
        self.send_response(200)
        self.send_header("Content-Type", content_type)
        self.send_header("Content-Length", str(len(content)))
        self.end_headers()
        self.wfile.write(content)

    def _send_error(self, status: int, message: str) -> None:
        self._send_json({"error": message}, status=status)


def _run_analyze_job(job_id: str, video_path: str, force: bool) -> None:
    config = load_config()

    def on_progress(progress: dict) -> None:
        _update_job(job_id, status="running", **progress)

    try:
        _update_job(
            job_id,
            status="running",
            stage="starting",
            message="Starting local video ingestion.",
            percent=1,
        )
        result = ingest_video(
            config,
            video_path,
            root_dir=ROOT_DIR,
            force=force,
            progress_callback=on_progress,
        )
        message = "Loaded from indexed memory." if result.get("cached") else "Analysis and indexing completed."
        _update_job(
            job_id,
            status="completed",
            stage="completed",
            message=message,
            percent=100,
            finished_at=time.time(),
            result={
                "cached": bool(result.get("cached")),
                "memory_stats": result.get("memory_stats"),
                "state": _get_state(),
            },
        )
    except Exception as exc:  # Keep background failures visible to the UI.
        _update_job(
            job_id,
            status="failed",
            stage="failed",
            message="Analysis failed.",
            error=str(exc),
            finished_at=time.time(),
        )


def _update_job(job_id: str, **updates) -> None:
    with JOBS_LOCK:
        job = JOBS.get(job_id)
        if not job:
            return
        job.update({key: value for key, value in updates.items() if value is not None})
        job["updated_at"] = time.time()


def _has_active_job() -> bool:
    with JOBS_LOCK:
        return any(job.get("status") in {"queued", "running"} for job in JOBS.values())


def _job_snapshot(job: dict) -> dict:
    return {
        "id": job.get("id"),
        "status": job.get("status"),
        "video_path": job.get("video_path"),
        "force": job.get("force"),
        "stage": job.get("stage"),
        "message": job.get("message"),
        "percent": job.get("percent"),
        "total_segments": job.get("total_segments"),
        "completed_segments": job.get("completed_segments"),
        "current_segment": job.get("current_segment"),
        "started_at": job.get("started_at"),
        "updated_at": job.get("updated_at"),
        "finished_at": job.get("finished_at"),
        "result": job.get("result"),
        "error": job.get("error"),
    }


def _get_state() -> dict:
    config = load_config()
    events_path = ROOT_DIR / config.output_events_path
    events = _read_events(events_path)
    memory_stats = _read_memory_stats(config.memory_db_path)
    with JOBS_LOCK:
        jobs = [_job_snapshot(job) for job in JOBS.values()]
    jobs.sort(key=lambda item: item.get("updated_at") or 0, reverse=True)
    return {
        "videos": _list_videos(),
        "events_path": str(events_path),
        "memory_stats": memory_stats,
        "jobs": jobs[:10],
        "events": events,
        "config": {
            "model": config.ollama_model,
            "segment_seconds": config.segment_seconds,
            "frames_per_segment": config.frames_per_segment,
            "memory_db_path": config.memory_db_path,
        },
    }


def _read_memory_stats(memory_db_path: str) -> dict | None:
    path = ROOT_DIR / memory_db_path
    if not path.exists():
        return None
    store = VideoMemoryStore(str(path))
    try:
        return store.stats()
    finally:
        store.close()


def _read_events(events_path: Path) -> dict | None:
    try:
        return json.loads(events_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return None


def _list_videos() -> list[dict]:
    videos = []
    for path in ROOT_DIR.rglob("*"):
        if not path.is_file() or path.suffix.lower() not in VIDEO_EXTENSIONS:
            continue
        if any(part in {".venv", ".git", "outputs"} for part in path.relative_to(ROOT_DIR).parts):
            continue
        videos.append(
            {
                "path": str(path.relative_to(ROOT_DIR)),
                "size_bytes": path.stat().st_size,
                "modified": path.stat().st_mtime,
            }
        )
    return sorted(videos, key=lambda item: item["modified"], reverse=True)


def _resolve_video_path(video_path: str) -> Path | None:
    candidate = (ROOT_DIR / video_path).resolve()
    try:
        candidate.relative_to(ROOT_DIR)
    except ValueError:
        return None
    if candidate.suffix.lower() not in VIDEO_EXTENSIONS:
        return None
    return candidate


def main() -> int:
    parser = argparse.ArgumentParser(description="Run the local VSS Q&A web UI.")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=7860)
    args = parser.parse_args()
    run_server(args.host, args.port)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
