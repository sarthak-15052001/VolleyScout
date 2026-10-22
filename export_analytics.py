# -*- coding: utf-8 -*-
"""
VolleyScout analytics export (Event Classification & Player Association).

Usage (from project root):
  python export_analytics.py --max-seconds 90 --limit-videos 1   # smoke
  python export_analytics.py --full                                # full match
  python export_analytics.py --video /path/to/match.mp4
  python export_analytics.py --full --video-dir holdout/          # custom folder

Output per video (v7: clean video stem names, no split/smoke/holdout prefix)
-------------------------------------------------------------------------------
  output/<video_stem>_analytics.mp4   annotated video (H.264 MP4 + audio when available)
  output/<video_stem>_events.csv      single clean CSV (incl. timestamp_ms)
  output/<video_stem>_jerseys.json    track_id -> jersey_number map

Merged CSV (output/all_events.csv) is written only with --write-merged-csv
or when processing more than one video.
"""
from __future__ import annotations

import argparse
import os
import re
import sys
from pathlib import Path

os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")
os.environ.setdefault("OMP_NUM_THREADS", "1")

import cv2
import pandas as pd
import torch

from volleyball_analytics.pipeline import (
    OUTPUT_CSV_COLUMNS,
    PipelineConfig,
    VolleyballAnalyticsPipeline,
    process_video,
)

PROJECT_ROOT = Path(__file__).resolve().parent
TRAIN_DIR = PROJECT_ROOT / "train"
TEST_DIR = PROJECT_ROOT / "holdout"
OUTPUT_DIR = PROJECT_ROOT / "output"
VIDEO_EXTS = {".mp4", ".avi", ".mov", ".mkv", ".webm", ".MP4"}


def list_videos(folder: Path):
    if not folder.exists():
        return []
    return sorted(p for p in folder.iterdir() if p.is_file() and p.suffix in VIDEO_EXTS)


def readable_videos(paths):
    out = []
    for v in paths:
        cap = cv2.VideoCapture(str(v))
        ok = cap.isOpened() and int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0) > 0
        cap.release()
        if ok:
            out.append(v)
        else:
            print("SKIP unreadable:", v.name)
    return out


def main():
    parser = argparse.ArgumentParser(
        description="VolleyScout - run event detection & jersey OCR on match videos"
    )
    parser.add_argument(
        "--max-seconds", type=float, default=90.0,
        help="Seconds per video (default 90 smoke test). Use 0 or --full for entire file.",
    )
    parser.add_argument("--full", action="store_true", help="Process entire video(s)")
    parser.add_argument(
        "--net-y-ratio", type=float, default=0.42,
        help="Net height as a fraction of frame height (0=top, 1=bottom). Depends on the camera: "
             "~0.42 for the Villadoro end-view, ~0.32 for the Cremonese raised side-view. "
             "The thin blue line in the annotated video should lie along the net.",
    )
    parser.add_argument(
        "--start-sec", type=float, default=0.0,
        help="Skip the first N seconds (warm-up/ceremony). With --max-seconds N the run "
             "covers N seconds FROM here. CSV times stay in original-video time; outputs get a "
             "_from<N>s suffix.",
    )
    parser.add_argument(
        "--finetuned", action="store_true",
        help="Force models/volleyball_player_best.pt",
    )
    parser.add_argument(
        "--pretrained-only", action="store_true",
        help="Ignore fine-tuned weights; use yolo26n COCO person only",
    )
    parser.add_argument("--limit-videos", type=int, default=0)
    parser.add_argument("--video", type=str, default="", help="Process a single video path")
    parser.add_argument(
        "--video-dir", type=str, default="",
        help="Process all videos in this folder (overrides holdout/ auto-discovery)",
    )
    parser.add_argument("--player-weights", type=str, default="yolo26n.pt")
    parser.add_argument("--ball-weights", type=str, default="")
    parser.add_argument("--split", type=str, default="holdout", help="Split label")
    parser.add_argument(
        "--max-jersey-number", type=int, default=30,
        help="Max valid jersey number to filter OCR false positives (default 30; use --roster when known)",
    )
    parser.add_argument(
        "--debug-overlay", action="store_true",
        help="Draw ball/tracker/contact/rally debug state on the analytics MP4",
    )
    parser.add_argument(
        "--roster", type=str, default="",
        help="Comma-separated valid jersey numbers for this match, e.g. 1,3,6,12,18",
    )
    parser.add_argument(
        "--jersey-export-min-conf", type=float, default=0.50,
        help="Blank jersey numbers below this confidence in CSV (default 0.50)",
    )
    parser.add_argument(
        "--write-merged-csv", action="store_true",
        help="Also write output/all_events.csv (always on when processing >1 video)",
    )
    parser.add_argument(
        "--device", type=str, default="auto",
        help="Compute device: auto (prefer CUDA:0), 0 / cuda / cuda:0, or cpu",
    )
    parser.add_argument(
        "--require-gpu", action="store_true",
        help="Exit with error if CUDA GPU is not available (do not fall back to CPU)",
    )
    args = parser.parse_args()

    # ── Resolve device (GPU preferred) ──────────────────────────────────────
    def _resolve_device(spec: str):
        s = (spec or "auto").strip().lower()
        cuda_ok = torch.cuda.is_available()
        if s in ("auto",):
            return 0 if cuda_ok else "cpu"
        if s in ("cpu",):
            return "cpu"
        if s in ("0", "cuda", "cuda:0", "gpu"):
            return 0 if cuda_ok else "cpu"
        if s.startswith("cuda:"):
            try:
                idx = int(s.split(":", 1)[1])
                return idx if cuda_ok else "cpu"
            except ValueError:
                return 0 if cuda_ok else "cpu"
        try:
            return int(s) if cuda_ok else "cpu"
        except ValueError:
            return 0 if cuda_ok else "cpu"

    device = _resolve_device(args.device)
    want_gpu = args.require_gpu or args.device.strip().lower() in (
        "0", "cuda", "cuda:0", "gpu",
    )
    if want_gpu and not torch.cuda.is_available():
        print(
            "ERROR: CUDA GPU required but torch.cuda.is_available() is False.\n"
            "  In this same terminal: conda activate py312 && nvidia-smi\n"
            "  Then: python -c \"import torch; print(torch.cuda.is_available(), "
            "torch.cuda.get_device_name(0) if torch.cuda.is_available() else '')\""
        )
        sys.exit(1)
    if args.require_gpu and device == "cpu":
        print("ERROR: --require-gpu set but resolved device is CPU.")
        sys.exit(1)

    max_seconds: float | None = None if (args.full or args.max_seconds <= 0) else args.max_seconds
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    local_player = PROJECT_ROOT / "models" / "volleyball_player_best.pt"
    local_ball   = PROJECT_ROOT / "models" / "volleyball_ball_best.pt"
    run_player   = PROJECT_ROOT / "runs" / "volleyball_player_detection" / "weights" / "best.pt"
    run_ball     = PROJECT_ROOT / "runs" / "volleyball_ball_detection"   / "weights" / "best.pt"

    import shutil
    for src, dst in [(run_player, local_player), (run_ball, local_ball)]:
        if not dst.exists() and src.exists():
            dst.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(src, dst)
            print("Copied training best ->", dst)

    use_finetuned = not args.pretrained_only and (args.finetuned or local_player.exists())
    roster_numbers = None
    if args.roster.strip():
        roster_numbers = {
            int(x.strip())
            for x in args.roster.split(",")
            if x.strip().isdigit()
        }

    cfg = PipelineConfig(
        project_root=PROJECT_ROOT,
        net_y_ratio=args.net_y_ratio,
        player_weights=args.player_weights,
        ball_weights=args.ball_weights or None,
        prefer_finetuned_player=use_finetuned,
        prefer_finetuned_ball=local_ball.exists(),
        finetuned_player=local_player,
        finetuned_ball=local_ball,
        device=device,
        tracker="trackers/botsort_volley.yaml",
        infer_stride=1,
        player_infer_stride=2,
        ocr_interval_sec=2.0,   # denser OCR for jersey fill on long matches
        use_motion_fallback=True,
        ocr_on_cpu=(device == "cpu"),
        crops_dir=OUTPUT_DIR / "jersey_crops",
        max_jersey_number=args.max_jersey_number,
        roster_numbers=roster_numbers,
        debug_overlay=args.debug_overlay,
        jersey_export_min_conf=args.jersey_export_min_conf,
        jersey_event_min_conf=0.45,
        ocr_min_bbox_area=5000,
        ocr_blur_threshold=60.0,
        ocr_brightness_min=40.0,
        ocr_brightness_max=230.0,
        ocr_skip_confirmed=True,
        ocr_min_interval_sec=0.4,
        jersey_vote_min=3,
        confirm_min_avg_conf=0.60,
        ocr_min_vote_conf=0.55,
        cluster_max_time_gap_sec=6.0,
        cluster_max_bbox_gap_px=150.0,
        max_rally_sec=25.0,
        serve_grace_sec=2.8,
        point_dead_ball_sec=1.0,
        max_point_after_touch_sec=1.5,
        reject_same_player_serve_reception=True,
        enforce_unique_jersey_per_frame=True,
        cluster_inherit_unconfirmed=True,
        ocr_min_player_height_px=100,
        serve_to_reception_window_sec=5.0,
        contact_cooldown_sec=0.80,
        min_ball_speed=25.0,
        contact_min_frames=2,
        contact_min_frames_set_block=2,
        contact_dist_px=160.0,
        contact_dist_mid_rally_px=180.0,
        inter_rally_lockout_sec=2.0,
        rally_gap_sec=4.5,
        enable_team_color_jersey=True,
        # v13 - YOLO-only contacts + tighter Set/Block spam guards
        rally_start_min_ball_frames=2,
        rally_start_window_sec=1.5,
        min_rally_ball_frames=3,
        min_ball_conf_yolo_contact=0.18,
        min_ball_conf_pred_contact=0.28,
        ball_predict_frames=10,
        mid_rally_ball_gap_sec=7.0,
        allow_pred_contacts=False,
        max_consecutive_same_touch=1,
        max_receptions_per_rally=1,
        jersey_dominance_max_frac=0.25,
        # v19: stricter 10↔18 lock + unlock on strong conflict OCR
        jersey_confusion_extra_conf=0.20,
        jersey_top2_conflict_ratio=0.65,
        jersey_confusion_vote_min=5,
        jersey_confusion_min_height_px=110,
        jersey_unlock_conflict_conf=0.72,
        timeout_suppress_sec=8.0,
        gap_watchdog_sec=30.0,
        soft_reopen_min_ball_conf=0.32,
        long_idle_reopen_sec=15.0,
        serve_only_max_sec=1.6,
        post_touch_dead_sec=3.5,
        max_digs_per_rally=2,
        duplicate_touch_collapse_sec=0.70,
        max_touches_per_rally=12,
        attack_to_attack_cooldown_sec=0.85,
        set_same_track_cooldown_sec=0.70,
        block_same_track_cooldown_sec=0.70,
        set_chain_max_in_window=1,
        track_expire_sec=120.0,
        max_jersey_track_entries=250,
    )
    pipeline = VolleyballAnalyticsPipeline(cfg)
    print("Player weights:", local_player if use_finetuned and local_player.exists() else cfg.player_weights)
    print("Ball weights  :", local_ball if local_ball.exists() else (cfg.ball_weights or cfg.player_weights))
    if torch.cuda.is_available() and cfg.device != "cpu":
        gpu_name = torch.cuda.get_device_name(int(cfg.device) if str(cfg.device).isdigit() else 0)
        mem_gb = torch.cuda.get_device_properties(0).total_memory / 1024**3
        print(f"Device        : CUDA:{cfg.device} ({gpu_name}, {mem_gb:.1f} GiB) | Tracker: {cfg.tracker}")
        print(f"OCR on GPU    : {not cfg.ocr_on_cpu}")
    else:
        print("Device        : CPU  (CUDA not active) | Tracker:", cfg.tracker)
        print("  Tip: run with  --require-gpu  after `nvidia-smi` works in this same shell.")
    print(f"min_ball_speed: {cfg.min_ball_speed} px/s  (Set/Attack/Block/Dig gate)")
    print(f"max_jersey_num: {cfg.max_jersey_number} (range 1-{cfg.max_jersey_number})")
    if cfg.roster_numbers:
        print("roster filter :", ",".join(str(n) for n in sorted(cfg.roster_numbers)))
    print(f"debug_overlay : {cfg.debug_overlay}")
    print(f"net_y_ratio   : {cfg.net_y_ratio}  (net line at {cfg.net_y_ratio:.0%} of frame height)")
    print(f"jersey_min_conf: {cfg.jersey_export_min_conf}")
    print(f"start_sec     : {args.start_sec}  (skip warm-up; CSV times stay original-video)")
    print(f"max_seconds   : {max_seconds}  (duration from start_sec)")

    # ── Resolve video list ─────────────────────────────────────────────────
    if args.video:
        vpath = Path(args.video).expanduser().resolve()
        if not vpath.exists():
            print("Video not found:", vpath)
            sys.exit(1)
        videos = [(vpath, args.split)]
    elif args.video_dir:
        vdir = Path(args.video_dir).expanduser().resolve()
        videos = [(v, args.split) for v in readable_videos(list_videos(vdir))]
    else:
        test_vids = readable_videos(list_videos(TEST_DIR))
        if not test_vids:
            test_vids = readable_videos(list_videos(TRAIN_DIR))
        if args.limit_videos > 0:
            test_vids = test_vids[: args.limit_videos]
        videos = [(v, args.split) for v in test_vids]

    if not videos:
        print("No readable videos found. Use --video /path/to/file.mp4 or put files in holdout/")
        sys.exit(1)

    print(f"\nProcessing {len(videos)} video(s) | max_seconds={max_seconds}\n")

    all_output_rows: list[dict] = []

    for vpath, split in videos:
        # v5: output filename = <video_stem>_analytics.mp4 (no split/smoke prefix)
        # process_video() writes intermediate .writing.avi, then converts to H.264 MP4 and unlinks the AVI.
        out_vid = OUTPUT_DIR / f"{vpath.stem}_analytics.mp4"
        print(f">>> {vpath.name}\n    -> {out_vid}")

        info = process_video(pipeline, vpath, out_vid, max_seconds=max_seconds, split_label=split,
                            start_seconds=args.start_sec)

        all_output_rows.extend(info["output_rows"])

        # Summary
        event_counts = {}
        for row in info["output_rows"]:
            et = row["event_type"]
            event_counts[et] = event_counts.get(et, 0) + 1
        print(f"    events={info['n_events']}  event types={event_counts}")
        print(f"    false rallies discarded={info.get('discarded_rallies', 0)}  (details: output/{vpath.stem}_events_debug.csv)")
        jersey_filled = sum(1 for r in info["output_rows"] if r.get("jersey_number"))
        print(f"    jerseys filled={jersey_filled}/{info['n_events']}")
        print(f"    CSV -> {info['out_csv']}")
        # Phase 5 light: short accuracy report next to CSV
        try:
            from volleyball_analytics.evaluation import analyze_rallies, _print_rally_summary
            dur = float(max_seconds) if max_seconds else 0.0
            rally_rep = analyze_rallies(Path(info["out_csv"]))
            print("\n----- Accuracy summary -----")
            _print_rally_summary(rally_rep)
            if dur:
                print(f"(window duration_sec={dur})")
        except Exception as exc:
            print(f"    [eval] skipped accuracy summary: {exc}")

    # ── Merged CSV: only when requested or multi-video batch ──────────────
    write_merged = args.write_merged_csv or len(videos) > 1
    print("\n===== DONE =====")
    if write_merged:
        merged_csv = OUTPUT_DIR / "all_events.csv"
        pd.DataFrame(all_output_rows, columns=OUTPUT_CSV_COLUMNS).to_csv(
            merged_csv, index=False
        )
        print(f"Merged events CSV : {merged_csv}  ({len(all_output_rows)} rows)")
    else:
        print("Merged events CSV : skipped (single video; pass --write-merged-csv to enable)")
    print(f"Output folder     : {OUTPUT_DIR}")


if __name__ == "__main__":
    main()
