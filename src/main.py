from __future__ import annotations

import argparse

from .config import load_config
from .event_analyzer import VideoEventAnalyzer, save_events_json
from .ollama_client import OllamaVisionClient
from .qa_over_events import answer_question_over_events
from .video_sampler import sample_video_segments


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Analyze video-file events with sampled frames and Ollama VLM."
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    analyze = subparsers.add_parser("analyze", help="Analyze a video and save events JSON.")
    analyze.add_argument("--video", help="Path to input video file.")

    ask = subparsers.add_parser("ask", help="Ask a question over saved events JSON.")
    ask.add_argument("--question", help="Question to answer from analyzed events.")

    analyze_ask = subparsers.add_parser(
        "analyze-ask",
        help="Analyze a video, save events JSON, then answer a question.",
    )
    analyze_ask.add_argument("--video", help="Path to input video file.")
    analyze_ask.add_argument("--question", help="Question to answer from analyzed events.")

    return parser


def analyze_video(config, video_path: str) -> int:
    print(f"Sampling video: {video_path}")
    try:
        segments = sample_video_segments(
            video_path=video_path,
            segment_seconds=config.segment_seconds,
            frames_per_segment=config.frames_per_segment,
        )
    except (RuntimeError, ValueError) as exc:
        print(f"Video analysis failed: {exc}")
        return 1

    print(f"Sampled {len(segments)} segment(s).")

    client = OllamaVisionClient(config.ollama_base_url, config.ollama_model)
    analyzer = VideoEventAnalyzer(client)
    segment_results = analyzer.analyze_segments(segments)
    save_events_json(
        output_path=config.output_events_path,
        video_path=video_path,
        segment_seconds=config.segment_seconds,
        frames_per_segment=config.frames_per_segment,
        segment_results=segment_results,
    )

    print(f"Saved events to {config.output_events_path}")
    return 0


def ask_question(config, question: str) -> int:
    client = OllamaVisionClient(config.ollama_base_url, config.ollama_model)
    answer = answer_question_over_events(config.output_events_path, question, client)
    print(answer)
    return 0


def main() -> int:
    args = build_parser().parse_args()
    config = load_config()

    if args.command == "analyze":
        video_path = args.video or config.video_source
        return analyze_video(config, video_path)

    if args.command == "ask":
        question = args.question or config.question
        if not question:
            print("A question is required. Use --question or set QUESTION in .env.")
            return 1
        return ask_question(config, question)

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
