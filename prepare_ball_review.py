#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Build a human-review pack from ball auto-label manifests.

After rebuild_ball_dataset.py / autolabel_balls.py, run:

  conda activate py312
  python prepare_ball_review.py --top 200

Opens:
  output/ball_review_pack/overlays/   — frames with yellow boxes (fix / blank / OK)
  output/ball_review_pack/CHECKLIST.md

Then fix labels under dataset_ball/labels/{train,val}/ for bad frames
(empty .txt = negative; YOLO line = corrected box), and retrain.
"""
from __future__ import annotations

import argparse
from pathlib import Path

from volleyball_analytics.ball_autolabel import write_review_pack


def main() -> None:
    ap = argparse.ArgumentParser(description="Prepare ball label review pack")
    ap.add_argument("--project-root", type=str, default=".")
    ap.add_argument("--dataset-root", type=str, default="dataset_ball")
    ap.add_argument(
        "--manifest",
        type=str,
        default="",
        help="review_manifest.csv (default: output/ball_review_manifest.csv)",
    )
    ap.add_argument("--out", type=str, default="output/ball_review_pack")
    ap.add_argument("--top", type=int, default=200, help="How many uncertain frames to export")
    args = ap.parse_args()

    root = Path(args.project_root).resolve()
    dataset = (root / args.dataset_root).resolve()
    manifest = (
        Path(args.manifest).expanduser().resolve()
        if args.manifest
        else (root / "output" / "ball_review_manifest.csv")
    )
    out = (root / args.out).resolve()

    if not manifest.exists():
        # Fall back to staging / label_auto locations
        for alt in (
            root / "label_auto_ball" / "review_manifest.csv",
            root / "label_auto" / "review_manifest.csv",
            dataset / "review_manifest.csv",
        ):
            if alt.exists():
                manifest = alt
                break
    if not manifest.exists():
        raise SystemExit(
            f"No review_manifest.csv found.\n"
            f"  Expected: {root / 'output' / 'ball_review_manifest.csv'}\n"
            f"  Run: python rebuild_ball_dataset.py --max-seconds 900 --keep-staging"
        )

    stats = write_review_pack(dataset, manifest, out, top_n=args.top)
    print(f"Manifest : {manifest}")
    print(f"Dataset  : {dataset}")
    print(f"Pack     : {out}")
    print(f"Copied   : {stats['copied']} images, {stats['overlays']} overlays, "
          f"{stats['listed']} checklist rows")
    print(f"Open     : {out / 'CHECKLIST.md'}")
    print(f"Overlays : {out / 'overlays'}")
    print(
        "\nReview steps:\n"
        "  1. Browse overlays/ — yellow box should hug the ball\n"
        "  2. Bad label → edit dataset_ball/labels/... matching .txt "
        "(blank file = no ball; YOLO line = fix box)\n"
        "  3. Retrain ball YOLO on cleaned dataset_ball\n"
    )


if __name__ == "__main__":
    main()
