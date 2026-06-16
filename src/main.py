from __future__ import annotations

import argparse
import sys

from .config import load_config
from .ollama_client import OllamaVisionClient
from .ingestion_pipeline import ingest_video
from .qa_over_events import answer_question_over_events
from .video_memory import VideoMemoryStore


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Analyze video-file events with sampled frames and Ollama VLM."
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    analyze = subparsers.add_parser("analyze", help="Analyze a video and save events JSON.")
    analyze.add_argument("--video", help="Path to input video file.")
    analyze.add_argument("--force", action="store_true", help="Ignore cached indexed memory.")

    ask = subparsers.add_parser("ask", help="Ask a question over saved events JSON.")
    ask.add_argument("--question", help="Question to answer from analyzed events.")

    search = subparsers.add_parser("search", help="Search indexed video memory.")
    search.add_argument("--query", required=True, help="Search query.")
    search.add_argument("--limit", type=int, default=8, help="Maximum result count.")

    memory_stats = subparsers.add_parser(
        "memory-stats",
        help="Print indexed video memory statistics.",
    )

    analyze_ask = subparsers.add_parser(
        "analyze-ask",
        help="Analyze a video, save events JSON, then answer a question.",
    )
    analyze_ask.add_argument("--video", help="Path to input video file.")
    analyze_ask.add_argument("--question", help="Question to answer from analyzed events.")

    return parser


def analyze_video(config, video_path: str) -> int:
    return analyze_video_with_options(config, video_path, force=False)


def analyze_video_with_options(config, video_path: str, *, force: bool) -> int:
    def report(progress: dict) -> None:
        message = progress.get("message")
        percent = progress.get("percent")
        if percent is not None:
            print(f"[{percent:3}%] {message}")
        elif message:
            print(message)

    try:
        result = ingest_video(
            config,
            video_path,
            force=force,
            progress_callback=report,
        )
    except (RuntimeError, ValueError, OSError) as exc:
        print(f"Video analysis failed: {exc}")
        return 1

    cache_text = "cache hit" if result.get("cached") else "fresh ingest"
    print(f"Saved events to {config.output_events_path} ({cache_text}).")
    print(f"Indexed video memory: {result.get('memory_stats')}")
    return 0


def ask_question(config, question: str) -> int:
    client = OllamaVisionClient(config.ollama_base_url, config.ollama_model)
    answer = answer_question_over_events(
        config.output_events_path,
        question,
        client,
        memory_db_path=config.memory_db_path,
    )
    print(answer)
    return 0


def search_memory(config, query: str, limit: int) -> int:
    store = VideoMemoryStore(config.memory_db_path)
    try:
        results = store.search_segments(query, limit=limit)
    finally:
        store.close()

    if not results:
        print("No indexed memory results found.")
        return 0

    for segment in results:
        print(
            f"{segment.get('segment_id')} "
            f"{segment.get('start_time')}-{segment.get('end_time')}: "
            f"{segment.get('summary')}"
        )
    return 0


def print_memory_stats(config) -> int:
    store = VideoMemoryStore(config.memory_db_path)
    try:
        stats = store.stats()
    finally:
        store.close()
    for key, value in stats.items():
        print(f"{key}: {value}")
    return 0


def main() -> int:
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")
        sys.stderr.reconfigure(encoding="utf-8")

    args = build_parser().parse_args()
    config = load_config()

    if args.command == "analyze":
        video_path = args.video or config.video_source
        return analyze_video_with_options(config, video_path, force=args.force)

    if args.command == "ask":
        question = args.question or config.question
        if not question:
            print("A question is required. Use --question or set QUESTION in .env.")
            return 1
        return ask_question(config, question)

    if args.command == "search":
        return search_memory(config, args.query, args.limit)

    if args.command == "memory-stats":
        return print_memory_stats(config)

    if args.command == "analyze-ask":
        video_path = args.video or config.video_source
        question = args.question or config.question
        if not question:
            print("A question is required. Use --question or set QUESTION in .env.")
            return 1

        analyze_status = analyze_video(config, video_path)
        if analyze_status != 0:
            return analyze_status
        return ask_question(config, question)

    return 1


if __name__ == "__main__":
    raise SystemExit(main())
