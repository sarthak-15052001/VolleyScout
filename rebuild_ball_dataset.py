#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Rebuild a dense, trainable volleyball-ball YOLO dataset.

Why the previous notebook ball train failed quality:
  - COCO sports-ball pseudo-labels were extremely sparse
  - train had ~89 positives vs ~1661 empty backgrounds
  - the model mostly learned "no ball"

This script uses trajectory + physics auto-labelling (independent of COCO),
keeps mostly POSITIVE frames, and writes into dataset_ball/ for fine-tuning.

Usage (conda py312, project root):
  python rebuild_ball_dataset.py --max-seconds 600
  python rebuild_ball_dataset.py --max-seconds 900 --videos-limit 0
  python rebuild_ball_dataset.py --max-seconds 600 --use-coco-candidates

Then train:
  python -c "from ultralytics import YOLO; YOLO('yolo26n.pt').train(
      data='data_ball.yaml', epochs=40, imgsz=1280, batch=2, device=0,
      project='runs', name='volleyball_ball_detection', exist_ok=True,
      patience=15, workers=0, cos_lr=True, close_mosaic=8)"
"""
from __future__ import annotations

import argparse
import json
import shutil
from collections import defaultdict
from pathlib import Path

import yaml

from volleyball_analytics.ball_autolabel import (
    AutoLabelConfig,
    autolabel_video,
    default_ignore_rects,
    export_dataset,
    make_yolo_detector,
    write_review_pack,
)

VIDEO_EXTS = {".mp4", ".avi", ".mov", ".mkv", ".webm"}
IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}


def _clear_split_dirs(root: Path) -> None:
    for split in ("train", "val", "test"):
        for kind in ("images", "labels"):
            d = root / kind / split
            if d.exists():
                for p in d.iterdir():
                    if p.is_file():
                        p.unlink()
            d.mkdir(parents=True, exist_ok=True)


def _count_split(root: Path, split: str) -> dict:
    img_dir = root / "images" / split
    lbl_dir = root / "labels" / split
    imgs = [p for p in img_dir.iterdir() if p.suffix.lower() in IMAGE_EXTS]
    pos = emp = boxes = 0
    for img in imgs:
        lab = lbl_dir / f"{img.stem}.txt"
        text = lab.read_text(encoding="utf-8").strip() if lab.exists() else ""
        if text:
            pos += 1
            boxes += len(text.splitlines())
        else:
            emp += 1
    return {
        "images": len(imgs),
        "positives": pos,
        "empties": emp,
        "boxes": boxes,
        "pos_ratio": round(pos / max(len(imgs), 1), 3),
    }


def _copy_tree_files(src: Path, dst: Path) -> int:
    dst.mkdir(parents=True, exist_ok=True)
    n = 0
    if not src.exists():
        return 0
    for p in src.iterdir():
        if p.is_file():
            shutil.copy2(p, dst / p.name)
            n += 1
    return n


def main() -> None:
    ap = argparse.ArgumentParser(description="Rebuild dense ball YOLO dataset")
    ap.add_argument("--project-root", type=str, default=".")
    ap.add_argument("--video-dir", type=str, default="train",
                    help="Folder of match videos (never holdout/ for training labels)")
    ap.add_argument("--out-staging", type=str, default="label_auto_ball",
                    help="Staging folder for trajectory exports")
    ap.add_argument("--dataset-root", type=str, default="dataset_ball",
                    help="Final YOLO dataset folder used by data_ball.yaml")
    ap.add_argument("--start", type=float, default=60.0,
                    help="Skip warm-up seconds at the start of each video")
    ap.add_argument("--max-seconds", type=float, default=900.0,
                    help="Seconds of each video to label (0 = full video); default 900")
    ap.add_argument("--label-stride", type=int, default=1,
                    help="Keep every N-th labelled frame (1 = densest)")
    ap.add_argument("--max-pos-per-video", type=int, default=2000)
    ap.add_argument("--neg-ratio", type=float, default=0.0,
                    help="Empty frames kept per positive (0 = positives only; recommended)")
    ap.add_argument("--videos-limit", type=int, default=0,
                    help="Process only first N videos (0 = all)")
    ap.add_argument("--use-coco-candidates", action="store_true",
                    help="Also feed low-conf YOLO boxes as candidates")
    ap.add_argument("--pretrained", type=str, default="yolo26n.pt",
                    help="Weights for --use-coco-candidates (prefer models/volleyball_ball_best.pt)")
    ap.add_argument("--no-scoreboard-mask", action="store_true",
                    help="Do not ignore top/bottom scoreboard strips")
    ap.add_argument("--review-top", type=int, default=200,
                    help="After rebuild, write this many uncertain frames to review pack (0=skip)")
    ap.add_argument("--keep-staging", action="store_true")
    ap.add_argument(
        "--resume",
        action="store_true",
        help="Do not wipe staging/dataset; skip videos that already have exports in staging",
    )
    ap.add_argument(
        "--exclude",
        type=str,
        default="",
        help="Comma-separated substrings; skip matching video names (e.g. Modena)",
    )
    args = ap.parse_args()

    root = Path(args.project_root).resolve()
    video_dir = (root / args.video_dir).resolve()
    staging = (root / args.out_staging).resolve()
    dataset = (root / args.dataset_root).resolve()

    if "holdout" in {p.lower() for p in video_dir.parts}:
        raise SystemExit("REFUSED: do not build training labels from holdout/")

    videos = sorted(
        p for p in video_dir.iterdir() if p.is_file() and p.suffix.lower() in VIDEO_EXTS
    )
    excludes = [x.strip() for x in args.exclude.split(",") if x.strip()]
    if excludes:
        videos = [v for v in videos if not any(x.lower() in v.name.lower() for x in excludes)]
    if args.videos_limit > 0:
        videos = videos[: args.videos_limit]
    if not videos:
        raise SystemExit(f"No videos found in {video_dir}")

    print(f"Project     : {root}")
    print(f"Videos      : {len(videos)} from {video_dir}")
    print(f"Per video   : start={args.start}s, max_seconds={args.max_seconds or 'FULL'}")
    print(f"Neg ratio   : {args.neg_ratio} (empties per positive)")
    print(f"Staging     : {staging}")
    print(f"Dataset     : {dataset}")
    print(f"Resume      : {args.resume}")

    if args.resume:
        staging.mkdir(parents=True, exist_ok=True)
        for sp in ("train", "val"):
            (staging / "images" / sp).mkdir(parents=True, exist_ok=True)
            (staging / "labels" / sp).mkdir(parents=True, exist_ok=True)
        existing = []
        for sp in ("train", "val"):
            existing.extend((staging / "images" / sp).glob("*.jpg"))
        done_prefixes = {p.name.rsplit("_f", 1)[0] for p in existing}
    else:
        if staging.exists():
            shutil.rmtree(staging)
        staging.mkdir(parents=True, exist_ok=True)
        _clear_split_dirs(dataset)
        done_prefixes = set()
        # Fresh rebuild → fresh manifest (export appends per video).
        man_staging = staging / "review_manifest.csv"
        if man_staging.exists():
            man_staging.unlink()

    detector = None
    if args.use_coco_candidates:
        # Prefer fine-tuned ball weights when the arg still points at yolo26n
        cand = Path(args.pretrained)
        if not cand.is_file():
            cand = root / args.pretrained
        local_ball = root / "models" / "volleyball_ball_best.pt"
        if (not cand.is_file() or cand.name == "yolo26n.pt") and local_ball.is_file():
            cand = local_ball
            print(f"Using fine-tuned ball weights for candidates: {cand}")
        if cand.is_file():
            print(f"Detector candidates from {cand}")
            detector = make_yolo_detector(str(cand), conf=0.08, imgsz=1280)
        else:
            print(f"WARNING: {args.pretrained} not found — continuing without detector candidates")

    ignore = () if args.no_scoreboard_mask else default_ignore_rects()
    cfg = AutoLabelConfig(ignore_rects=ignore)
    print(f"Scoreboard mask: {'OFF' if args.no_scoreboard_mask else 'ON (top+bottom strips)'}")
    print(f"Label stride   : {args.label_stride} | max_pos/video={args.max_pos_per_video}")
    per_video = []
    for i, video in enumerate(videos, 1):
        end = None if args.max_seconds <= 0 else args.start + args.max_seconds
        print(f"\n[{i}/{len(videos)}] {video.name}")
        print(f"  window: {args.start}s -> {end if end else 'end'}")
        stem_key = video.stem
        if any(
            p.name.startswith(stem_key + "_f")
            for sp in ("train", "val")
            for p in (staging / "images" / sp).glob("*.jpg")
        ):
            print("  SKIP: already present in staging (--resume)")
            per_video.append({"video": video.name, "labels": -1, "exported": {"skipped": "resume"}})
            continue
        try:
            labels, fps, size = autolabel_video(
                video, cfg, start_sec=args.start, end_sec=end, detector=detector
            )
        except Exception as exc:
            print(f"  SKIP (cannot process video): {exc}")
            per_video.append({"video": video.name, "labels": 0, "exported": {}, "error": str(exc)})
            continue
        det = sum(1 for b in labels.values() if b.source == "detected")
        print(f"  labelled frames: {len(labels)} (detected={det}, interp={len(labels) - det})")
        if not labels:
            print("  SKIP: no ballistic ball tracks found")
            per_video.append({"video": video.name, "labels": 0, "exported": {}})
            continue
        # Keep positives only by default; small neg_ratio is allowed without events CSV
        # by leaving neg_ranges=None when neg_ratio==0.
        neg_ranges = None
        neg_ratio = 0.0 if args.neg_ratio <= 0 else 0.0  # force 0 without dead-ball ranges
        if args.neg_ratio > 0:
            print("  NOTE: neg_ratio requested but no events CSV — keeping positives only "
                  "(safer; random empties can hide unlabelled balls)")
        counts = export_dataset(
            video,
            labels,
            fps,
            size,
            staging,
            label_stride=args.label_stride,
            max_pos=args.max_pos_per_video,
            neg_ratio=neg_ratio,
            neg_ranges=neg_ranges,
            cvat=False,
        )
        print(f"  exported: {counts}")
        per_video.append({"video": video.name, "labels": len(labels), "exported": counts})

    # Move staging train/val into dataset_ball (replace previous contents)
    _clear_split_dirs(dataset)
    copied = {
        "train_images": _copy_tree_files(staging / "images" / "train", dataset / "images" / "train"),
        "train_labels": _copy_tree_files(staging / "labels" / "train", dataset / "labels" / "train"),
        "val_images": _copy_tree_files(staging / "images" / "val", dataset / "images" / "val"),
        "val_labels": _copy_tree_files(staging / "labels" / "val", dataset / "labels" / "val"),
    }
    # Optional small test = copy of val (or leave empty). Prefer independent review later.
    for kind in ("images", "labels"):
        src = dataset / kind / "val"
        dst = dataset / kind / "test"
        _copy_tree_files(src, dst)

    yaml_path = root / "data_ball.yaml"
    ball_yaml = {
        "path": str(dataset).replace("\\", "/"),
        "train": "images/train",
        "val": "images/val",
        "test": "images/test",
        "names": {0: "ball"},
    }
    yaml_path.write_text(yaml.safe_dump(ball_yaml, sort_keys=False), encoding="utf-8")

    stats = {split: _count_split(dataset, split) for split in ("train", "val", "test")}
    report = {
        "videos": per_video,
        "copied": copied,
        "stats": stats,
        "data_yaml": str(yaml_path),
        "pretrained": args.pretrained,
    }
    report_path = root / "output" / "ball_dataset_rebuild_report.json"
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(json.dumps(report, indent=2), encoding="utf-8")

    print("\n========== BALL DATASET REBUILD SUMMARY ==========")
    for split, s in stats.items():
        print(
            f"  {split:5s}: images={s['images']:5d}  positives={s['positives']:5d}  "
            f"empties={s['empties']:5d}  boxes={s['boxes']:5d}  pos_ratio={s['pos_ratio']:.3f}"
        )
    print(f"  data_ball.yaml -> {yaml_path}")
    print(f"  report         -> {report_path}")

    train_pos = stats["train"]["positives"]
    if train_pos < 500:
        print(
            f"\nWARNING: only {train_pos} positive train frames "
            "(aim ≥1000). Re-run with --max-seconds 0 or more train/ videos."
        )
    else:
        print(f"\nTrain positives: {train_pos} (OK to review + train).")

    man = staging / "review_manifest.csv"
    man_out = root / "output" / "ball_review_manifest.csv"
    if man.exists():
        man_out.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(man, man_out)
        print(f"  review manifest -> {man_out}")

    if args.review_top > 0 and man_out.exists():
        pack = root / "output" / "ball_review_pack"
        try:
            rstats = write_review_pack(dataset, man_out, pack, top_n=args.review_top)
            print(
                f"  review pack    -> {pack} "
                f"({rstats['overlays']} overlays / {rstats['listed']} listed)"
            )
            print(f"  checklist      -> {pack / 'CHECKLIST.md'}")
        except Exception as exc:
            print(f"  review pack skipped: {exc}")

    print(
        "\nNext:\n"
        "  1) Review overlays:  output/ball_review_pack/overlays/\n"
        "     Guide:           BALL_LABEL_REVIEW.md\n"
        "  2) Fix bad labels under dataset_ball/labels/\n"
        "  3) Train YOLO on data_ball.yaml (epochs=50, imgsz=1280, device=0)\n"
        "  4) Copy best.pt -> models/volleyball_ball_best.pt\n"
    )

    if not args.keep_staging:
        shutil.rmtree(staging, ignore_errors=True)
        print("Staging folder removed (manifest + review pack kept under output/).")


if __name__ == "__main__":
    main()
