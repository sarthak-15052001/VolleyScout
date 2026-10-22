#!/usr/bin/env python3
"""Export a lightweight ground-truth review sheet from an events CSV.

Helps Tier-2 human review: open the analytics MP4, check each Serve/Point, and
mark OK / miss / wrong_label / wrong_jersey. Does not invent labels.

Usage:
  python tools/gt_review_export.py \\
    --csv output/..._from330s_events.csv \\
    --out output/gt_review_cremonese_from330s.csv
"""
from __future__ import annotations

import argparse
import csv
from pathlib import Path


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--csv", required=True, help="events CSV from export_analytics.py")
    ap.add_argument("--out", required=True, help="review sheet path")
    args = ap.parse_args()

    rows = list(csv.DictReader(open(args.csv)))
    ecol = "event_type" if "event_type" in (rows[0] or {}) else "event"
    tcol = "time_start_sec"
    out_rows = []
    for r in rows:
        ev = r.get(ecol, "")
        if ev not in ("Serve", "Reception", "Set", "Attack", "Block", "Dig", "Point"):
            continue
        out_rows.append({
            "rally_id": r.get("rally_id", ""),
            "event": ev,
            "time_start_sec": r.get(tcol, ""),
            "timestamp": r.get("timestamp", ""),
            "jersey_number": r.get("jersey_number", ""),
            "jersey_confidence": r.get("jersey_confidence") or r.get("jersey_conf", ""),
            "player_track_id": r.get("player_track_id", ""),
            "review_ok": "",
            "review_note": "",
            "correct_event": "",
            "correct_jersey": "",
        })

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    with open(out, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(out_rows[0].keys()) if out_rows else [])
        w.writeheader()
        w.writerows(out_rows)
    print(f"Wrote {len(out_rows)} review rows -> {out}")
    print("Fill review_ok=yes/no, correct_event, correct_jersey while watching the MP4.")


if __name__ == "__main__":
    main()
