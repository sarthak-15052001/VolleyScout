# -*- coding: utf-8 -*-
"""Phase 0 / Phase 5 — smoke, holdout, gold scoring, and rally-level reports."""
from __future__ import annotations

import argparse
import json
import statistics
from pathlib import Path
from typing import Any, Dict, List, Optional

import pandas as pd

EVENT_NAMES = [
    "Serve",
    "Reception",
    "Set",
    "Attack",
    "Block",
    "Dig",
    "Rally",
    "Point",
]

TOUCH_TYPES = {"Serve", "Reception", "Set", "Attack", "Block", "Dig"}

# Fabio review clip list — priority rallies from the Villadoro client feedback video.
FABIO_REVIEW_RALLIES = [1, 2, 4, 6, 7, 10]


def _event_col(df: pd.DataFrame) -> str:
    if "event_type" in df.columns:
        return "event_type"
    return "event"


def _safe_read_csv(path: Path) -> pd.DataFrame:
    if not path.exists():
        return pd.DataFrame()
    try:
        return pd.read_csv(path)
    except pd.errors.EmptyDataError:
        return pd.DataFrame()


def _jersey_nonempty_mask(series: pd.Series) -> pd.Series:
    """True only for real jersey values — blanks / NaN / None must not count as filled.

    Note: pandas often loads empty CSV jersey cells as float NaN. ``astype(str)`` can
    leave those as NA rather than the literal ``\"nan\"``, so we gate with ``isna`` first.
    """
    present = series.notna()
    text = series.astype("string").fillna("").str.strip()
    # Float CSV values become \"20.0\" — normalize trailing .0 for emptiness checks only.
    text = text.str.replace(r"\.0$", "", regex=True)
    lower = text.str.lower()
    bad = lower.isin(["", "nan", "none", "<na>", "nat"])
    return present & ~bad


def analyze_events_csv(events_csv: Path, duration_sec: float = 90.0) -> Dict[str, Any]:
    df = _safe_read_csv(events_csv)
    if df.empty:
        return {
            "path": str(events_csv), "n_events": 0, "events_per_10s": 0.0,
            "event_counts": {}, "n_rallies": 0, "jerseys_filled": 0,
            "jersey_fill_rate_all": 0.0,
            "ball_conf_median": None, "ball_speed_median": None,
            "gap_median_sec": None, "single_frame_events": 0,
            "flags": ["no_confirmed_events"],
        }

    ecol = _event_col(df)
    counts = df[ecol].value_counts().to_dict() if ecol in df.columns else {}
    n_rallies = df["rally_id"].nunique() if "rally_id" in df.columns else 0
    jersey_filled = 0
    jersey_fill_rate_all = 0.0
    if "jersey_number" in df.columns and len(df):
        filled_mask = _jersey_nonempty_mask(df["jersey_number"])
        jersey_filled = int(filled_mask.sum())
        jersey_fill_rate_all = float(filled_mask.mean())

    ball_conf = pd.to_numeric(df.get("ball_conf", pd.Series(dtype=float)), errors="coerce")
    speeds = pd.to_numeric(df.get("ball_speed_peak", pd.Series(dtype=float)), errors="coerce")

    same_frame = 0
    if "frame_start" in df.columns and "frame_end" in df.columns:
        same_frame = int((df["frame_start"] == df["frame_end"]).sum())

    gaps = []
    if "time_start_sec" in df.columns:
        times = sorted(pd.to_numeric(df["time_start_sec"], errors="coerce").dropna())
        gaps = [times[i + 1] - times[i] for i in range(len(times) - 1)]

    missing_events = sorted(set(EVENT_NAMES) - set(counts.keys()))
    events_per_10s = len(df) / max(duration_sec, 1) * 10

    flags: List[str] = []
    if events_per_10s > 8:
        flags.append("too_many_events_per_10s")
    if n_rallies <= 1 and duration_sec >= 60:
        flags.append("rally_segmentation_stuck")
    if jersey_filled == 0:
        flags.append("no_jerseys_read")
    if missing_events:
        flags.append(f"missing_event_types:{','.join(missing_events)}")
    if same_frame == len(df) and len(df) > 0:
        flags.append("all_single_frame_events")

    return {
        "path": str(events_csv),
        "n_events": len(df),
        "events_per_10s": round(events_per_10s, 2),
        "event_counts": counts,
        "n_rallies": int(n_rallies),
        "jerseys_filled": jersey_filled,
        "jersey_fill_rate_all": round(jersey_fill_rate_all, 3),
        "ball_conf_median": round(float(ball_conf.median()), 3) if ball_conf.notna().any() else None,
        "ball_speed_median": round(float(speeds.median()), 1) if speeds.notna().any() else None,
        "gap_median_sec": round(float(sorted(gaps)[len(gaps) // 2]), 2) if gaps else None,
        "single_frame_events": same_frame,
        "flags": flags,
    }


def analyze_rallies(events_csv: Path) -> Dict[str, Any]:
    """Rally-level report (no ground truth) for regression / Fabio review."""
    df = _safe_read_csv(events_csv)
    if df.empty or "rally_id" not in df.columns:
        return {
            "path": str(events_csv),
            "n_rallies": 0,
            "serve_only_rallies": 0,
            "serve_only_pct": 0.0,
            "reception_serve_ratio": 0.0,
            "median_touches_per_rally": 0.0,
            "median_serve_duration_sec": None,
            "point_to_next_rally_gaps": {},
            "long_silence_gaps": [],
            "set_block_share": 0.0,
            "mean_touch_rate_per_sec": None,
            "inferred_contact_rate": None,
            "jersey_dominance_max_frac": 0.0,
            "jersey_fill_rate_touches": 0.0,
            "same_player_serve_reception": 0,
            "late_point_rallies": [],
            "fabio_review": [],
            "rallies": [],
            "summary_flags": ["empty_csv"],
            "acceptance_gates": {},
        }

    ecol = _event_col(df)
    df = df.copy()
    df["time_start_sec"] = pd.to_numeric(df["time_start_sec"], errors="coerce")
    df["time_end_sec"] = pd.to_numeric(df["time_end_sec"], errors="coerce")
    df["rally_id"] = pd.to_numeric(df["rally_id"], errors="coerce").astype("Int64")

    rallies: List[Dict[str, Any]] = []
    serve_durs: List[float] = []
    touch_counts: List[int] = []
    serve_only = 0
    same_player_sr = 0
    late_points: List[Dict[str, Any]] = []
    point_end_by_rally: Dict[int, float] = {}
    rally_start_by_id: Dict[int, float] = {}

    for rid, g in df.groupby("rally_id"):
        if pd.isna(rid):
            continue
        rid_i = int(rid)
        g = g.sort_values("time_start_sec")
        seq = [str(x) for x in g[ecol].tolist()]
        touches = g[g[ecol].isin(TOUCH_TYPES)]
        n_touches = len(touches)
        touch_counts.append(n_touches)

        serve_rows = g[g[ecol] == "Serve"]
        serve_dur = None
        if not serve_rows.empty:
            serve_dur = float(
                serve_rows.iloc[0]["time_end_sec"] - serve_rows.iloc[0]["time_start_sec"]
            )
            serve_durs.append(serve_dur)

        mid = set(touches[ecol].tolist()) - {"Serve"}
        is_serve_only = ("Serve" in seq) and (len(mid) == 0)
        if is_serve_only:
            serve_only += 1

        jersey_filled = 0
        if "jersey_number" in touches.columns and n_touches > 0:
            jersey_filled = int(_jersey_nonempty_mask(touches["jersey_number"]).sum())

        # P0 diagnostic: Serve + first Reception share track or jersey.
        recv_rows = g[g[ecol] == "Reception"]
        same_player = False
        if not serve_rows.empty and not recv_rows.empty:
            s = serve_rows.iloc[0]
            r = recv_rows.iloc[0]
            st = str(s.get("player_track_id", "") or "").strip()
            rt = str(r.get("player_track_id", "") or "").strip()
            sj = str(s.get("jersey_number", "") or "").strip()
            rj = str(r.get("jersey_number", "") or "").strip()
            if sj.lower() in ("nan", "none"):
                sj = ""
            if rj.lower() in ("nan", "none"):
                rj = ""
            if (st and rt and st == rt) or (sj and rj and sj == rj):
                same_player = True
                same_player_sr += 1

        point_rows = g[g[ecol] == "Point"]
        point_end = float(point_rows.iloc[0]["time_end_sec"]) if not point_rows.empty else None
        if point_end is not None:
            point_end_by_rally[rid_i] = point_end

        rally_start = float(g["time_start_sec"].min())
        rally_start_by_id[rid_i] = rally_start

        gap_after_last_touch = None
        if point_end is not None and n_touches > 0:
            last_touch_end = float(touches.iloc[-1]["time_end_sec"])
            gap_after_last_touch = point_end - last_touch_end
            if gap_after_last_touch > 2.0:
                late_points.append({
                    "rally_id": rid_i,
                    "gap_sec": round(gap_after_last_touch, 2),
                    "sequence": " → ".join(seq),
                })

        flags: List[str] = []
        if is_serve_only:
            flags.append("serve_only")
        if same_player:
            flags.append("same_player_serve_reception")
        if "Serve" in seq and "Reception" not in seq and len(mid) >= 1:
            flags.append("missing_reception")
        if serve_dur is not None and serve_dur > 5.0:
            flags.append("long_serve_segment")
        if gap_after_last_touch is not None and gap_after_last_touch > 2.0:
            flags.append("late_point")
        if point_end is not None and n_touches > 0:
            last_touch_end = float(touches.iloc[-1]["time_end_sec"])
            if point_end < last_touch_end - 0.05:
                flags.append("invalid_point_timing")

        touch_seq = [str(x) for x in touches[ecol].tolist()]
        if (
            len(touch_seq) >= 2
            and touch_seq[0] == "Serve"
            and touch_seq[1] == "Attack"
        ):
            flags.append("serve_to_attack")

        rallies.append({
            "rally_id": rid_i,
            "sequence": " → ".join(seq),
            "n_touches": n_touches,
            "serve_duration_sec": round(serve_dur, 3) if serve_dur is not None else None,
            "jersey_fill_rate": round(jersey_filled / n_touches, 3) if n_touches else 0.0,
            "time_start_sec": round(rally_start, 3),
            "time_end_sec": round(float(g["time_end_sec"].max()), 3),
            "flags": flags,
            "fabio_priority": rid_i in FABIO_REVIEW_RALLIES,
        })

    sorted_rids = sorted(rally_start_by_id.keys())
    gaps_pnr: List[float] = []
    long_silence_gaps: List[Dict[str, Any]] = []
    # Gaps ≥ this are treated as timeouts / between-point idle (not missed play).
    # Cremonese R10→R11 ~38s was idle milling (low motion), not a live rally.
    timeout_silence_sec = 35.0
    for i in range(len(sorted_rids) - 1):
        a, b = sorted_rids[i], sorted_rids[i + 1]
        if a in point_end_by_rally:
            gap = rally_start_by_id[b] - point_end_by_rally[a]
            gaps_pnr.append(gap)
            if gap > 30.0:
                likely_timeout = gap >= timeout_silence_sec
                long_silence_gaps.append({
                    "after_rally_id": a,
                    "before_rally_id": b,
                    "gap_sec": round(gap, 2),
                    "silence_start_sec": round(point_end_by_rally[a], 3),
                    "silence_end_sec": round(rally_start_by_id[b], 3),
                    "likely_timeout": likely_timeout,
                })

    n_serve = int((df[ecol] == "Serve").sum())
    n_recv = int((df[ecol] == "Reception").sum())
    n_rallies = len(rallies)

    touch_df = df[df[ecol].isin(TOUCH_TYPES)].copy()
    jersey_fill_rate = 0.0
    if len(touch_df) and "jersey_number" in touch_df.columns:
        jersey_fill_rate = float(_jersey_nonempty_mask(touch_df["jersey_number"]).mean())

    # Set+Block share of in-rally touches (exclude Serve)
    in_rally = touch_df[touch_df[ecol] != "Serve"] if len(touch_df) else touch_df
    n_in = len(in_rally)
    n_set_block = int(in_rally[ecol].isin(["Set", "Block"]).sum()) if n_in else 0
    set_block_share = (n_set_block / n_in) if n_in else 0.0

    # Mean touch rate while rally active (touches / sum of rally durations)
    active_sec = 0.0
    for r in rallies:
        active_sec += max(0.0, float(r["time_end_sec"]) - float(r["time_start_sec"]))
    mean_touch_rate = (len(touch_df) / active_sec) if active_sec > 0 else None

    # Inferred-contact rate: client CSV flags, else sibling *_events_debug.csv
    inferred_rate = None
    debug_df = None
    dbg_path = Path(str(events_csv).replace("_events.csv", "_events_debug.csv"))
    if dbg_path.exists() and dbg_path != events_csv:
        try:
            debug_df = _safe_read_csv(dbg_path)
        except Exception:
            debug_df = None
    src_for_inferred = touch_df
    if debug_df is not None and not debug_df.empty:
        dcol = _event_col(debug_df)
        src_for_inferred = debug_df[debug_df[dcol].isin(TOUCH_TYPES)]
    if len(src_for_inferred):
        if "inferred_contact" in src_for_inferred.columns:
            inferred_rate = float(
                src_for_inferred["inferred_contact"].fillna(False).astype(bool).mean()
            )
        elif "ball_src" in src_for_inferred.columns:
            inferred_rate = float(
                src_for_inferred["ball_src"].astype(str).isin(["pred", "motion"]).mean()
            )
        elif "inferred" in src_for_inferred.columns:
            inferred_rate = float(
                src_for_inferred["inferred"].fillna(False).astype(bool).mean()
            )
        elif "ball_conf" in src_for_inferred.columns:
            bc = pd.to_numeric(src_for_inferred["ball_conf"], errors="coerce")
            inferred_rate = float((bc.fillna(0) < 0.25).mean())

    # Jersey dominance: max fraction of one number among filled touch rows
    jersey_dom = 0.0
    if len(touch_df) and "jersey_number" in touch_df.columns:
        filled_mask = _jersey_nonempty_mask(touch_df["jersey_number"])
        filled_vals = (
            touch_df.loc[filled_mask, "jersey_number"]
            .astype("string")
            .str.replace(r"\.0$", "", regex=True)
            .str.strip()
        )
        if len(filled_vals):
            counts = filled_vals.value_counts()
            jersey_dom = float(counts.iloc[0] / len(filled_vals))

    invalid_point = sum(1 for r in rallies if "invalid_point_timing" in r["flags"])
    serve_to_attack = sum(1 for r in rallies if "serve_to_attack" in r["flags"])
    block_before_attack = 0
    for r in rallies:
        seq = [x.strip() for x in r["sequence"].split("→")]
        touches = [x for x in seq if x in TOUCH_TYPES]
        seen_attack = False
        for ev in touches:
            if ev == "Attack":
                seen_attack = True
            elif ev == "Block" and not seen_attack:
                block_before_attack += 1
                break

    serve_only_pct = 100.0 * serve_only / max(n_rallies, 1)
    recv_serve_ratio = n_recv / max(n_serve, 1)

    summary_flags: List[str] = []
    if n_rallies and serve_only / n_rallies > 0.25:
        summary_flags.append("high_serve_only_rate")
    if n_serve and recv_serve_ratio < 0.4:
        summary_flags.append("low_reception_serve_ratio")
    if gaps_pnr and sum(1 for g in gaps_pnr if g < 1.0) / len(gaps_pnr) > 0.15:
        summary_flags.append("many_fast_restarts")
    if invalid_point:
        summary_flags.append("invalid_point_timing")
    if serve_to_attack / max(n_rallies, 1) > 0.15:
        summary_flags.append("many_serve_to_attack")
    if same_player_sr / max(n_rallies, 1) > 0.25:
        summary_flags.append("many_same_player_serve_reception")
    if set_block_share > 0.40:
        summary_flags.append("set_block_share_high")
    if mean_touch_rate is not None and mean_touch_rate > 1.2:
        summary_flags.append("touch_rate_spam")
    unexplained_gaps = [g for g in long_silence_gaps if not g.get("likely_timeout")]
    if unexplained_gaps:
        summary_flags.append("long_silence_gaps")
    elif long_silence_gaps:
        summary_flags.append("timeout_silences_ok")
    if jersey_dom > 0.25:
        summary_flags.append("jersey_dominance")
    if inferred_rate is not None and inferred_rate > 0.35:
        summary_flags.append("inferred_contact_high")
    if block_before_attack:
        summary_flags.append("block_before_attack")

    # §2.1 / §2.2 acceptance gates (True = pass)
    acceptance_gates = {
        "serve_only_lt_10pct": serve_only_pct < 10.0,
        "reception_serve_ratio_ok": 0.85 <= recv_serve_ratio <= 1.30 if n_serve else True,
        "same_player_serve_reception_zero": same_player_sr == 0,
        "block_before_attack_zero": block_before_attack == 0,
        "set_block_share_lt_40pct": set_block_share < 0.40,
        "touch_rate_in_band": (
            mean_touch_rate is None or 0.4 <= mean_touch_rate <= 1.2
        ),
        "inferred_contact_lt_35pct": (
            inferred_rate is None or inferred_rate < 0.35
        ),
        # Pass when every >30s gap is a likely timeout (≥45s). Short 30–45s
        # silences still fail (possible missed live play).
        "no_long_silence_gaps": all(
            bool(g.get("likely_timeout")) for g in long_silence_gaps
        ) if long_silence_gaps else True,
        "jersey_dominance_lt_25pct": jersey_dom <= 0.25,
    }

    fabio_review = [r for r in rallies if r["fabio_priority"]]

    def _med(vals: List[float]) -> Optional[float]:
        return round(float(statistics.median(vals)), 3) if vals else None

    return {
        "path": str(events_csv),
        "n_rallies": n_rallies,
        "serve_only_rallies": serve_only,
        "serve_only_pct": round(serve_only_pct, 1),
        "reception_serve_ratio": round(recv_serve_ratio, 3),
        "median_touches_per_rally": _med([float(x) for x in touch_counts]) or 0.0,
        "median_serve_duration_sec": _med(serve_durs),
        "invalid_point_timing": invalid_point,
        "serve_to_attack_rallies": serve_to_attack,
        "same_player_serve_reception": same_player_sr,
        "block_before_attack_rallies": block_before_attack,
        "point_to_next_rally_gaps": {
            "median_sec": _med(gaps_pnr),
            "count_lt_1s": int(sum(1 for g in gaps_pnr if g < 1.0)),
            "count_lt_3s": int(sum(1 for g in gaps_pnr if g < 3.0)),
            "count_gt_30s": len(long_silence_gaps),
            "n": len(gaps_pnr),
        },
        "long_silence_gaps": long_silence_gaps,
        "set_block_share": round(set_block_share, 3),
        "mean_touch_rate_per_sec": (
            round(mean_touch_rate, 3) if mean_touch_rate is not None else None
        ),
        "inferred_contact_rate": (
            round(inferred_rate, 3) if inferred_rate is not None else None
        ),
        "jersey_dominance_max_frac": round(jersey_dom, 3),
        "jersey_fill_rate_touches": round(jersey_fill_rate, 3),
        "late_point_rallies": late_points[:30],
        "fabio_review_rallies": FABIO_REVIEW_RALLIES,
        "fabio_review": fabio_review,
        "rallies": rallies,
        "summary_flags": summary_flags,
        "acceptance_gates": acceptance_gates,
        "event_counts": df[ecol].value_counts().to_dict(),
    }


def compare_to_ground_truth(
    events_csv: Path,
    ground_truth_csv: Path,
    tolerance_sec: float = 0.5,
) -> Dict[str, Any]:
    """P3 gold scoring: event match ±tolerance, plus optional jersey exact match."""
    pred = _safe_read_csv(events_csv)
    gt = _safe_read_csv(ground_truth_csv)
    if pred.empty or gt.empty:
        return {"error": "pred or ground truth empty"}

    pe_col = _event_col(pred)
    ge_col = _event_col(gt)
    pred_times = pred.copy()
    gt_times = gt.copy()
    pred_times["time_start_sec"] = pd.to_numeric(pred_times["time_start_sec"], errors="coerce")
    gt_times["time_start_sec"] = pd.to_numeric(gt_times["time_start_sec"], errors="coerce")

    used_gt: set = set()
    matched = 0
    jersey_checked = 0
    jersey_correct = 0
    by_event: Dict[str, Dict[str, int]] = {
        ev: {"tp": 0, "fp": 0, "fn": 0, "ground_truth": 0, "predicted": 0}
        for ev in EVENT_NAMES
    }

    for ev in EVENT_NAMES:
        by_event[ev]["ground_truth"] = int((gt_times[ge_col] == ev).sum()) if ge_col in gt_times else 0
        by_event[ev]["predicted"] = int((pred_times[pe_col] == ev).sum()) if pe_col in pred_times else 0

    for _, prow in pred_times.iterrows():
        pe, pt = prow.get(pe_col), prow["time_start_sec"]
        if pd.isna(pt):
            continue
        match = None
        best_dt = None
        for gi, grow in gt_times.iterrows():
            if gi in used_gt:
                continue
            if grow.get(ge_col) != pe:
                continue
            dt = abs(float(grow["time_start_sec"]) - float(pt))
            if dt <= tolerance_sec and (best_dt is None or dt < best_dt):
                match = gi
                best_dt = dt
        if match is not None:
            used_gt.add(match)
            matched += 1
            if pe in by_event:
                by_event[pe]["tp"] += 1
            grow = gt_times.loc[match]
            pj = str(prow.get("jersey_number", "") or "").strip()
            gj = str(grow.get("jersey_number", "") or "").strip()
            if pj.lower() in ("nan", "none"):
                pj = ""
            if gj.lower() in ("nan", "none"):
                gj = ""
            if gj:
                jersey_checked += 1
                if pj and pj == gj:
                    jersey_correct += 1
        else:
            if pe in by_event:
                by_event[pe]["fp"] += 1

    missed = len(gt_times) - len(used_gt)
    for ev in EVENT_NAMES:
        by_event[ev]["fn"] = max(0, by_event[ev]["ground_truth"] - by_event[ev]["tp"])
        tp, fp, fn = by_event[ev]["tp"], by_event[ev]["fp"], by_event[ev]["fn"]
        prec = tp / max(tp + fp, 1)
        rec = tp / max(tp + fn, 1)
        f1 = (2 * prec * rec / max(prec + rec, 1e-9)) if (tp + fp + fn) else 0.0
        by_event[ev]["precision"] = round(prec, 3)
        by_event[ev]["recall"] = round(rec, 3)
        by_event[ev]["f1"] = round(f1, 3)

    precision = matched / max(len(pred_times), 1)
    recall = matched / max(len(gt_times), 1)
    f1 = (2 * precision * recall / max(precision + recall, 1e-9)) if matched else 0.0

    return {
        "tolerance_sec": tolerance_sec,
        "matched": matched,
        "missed": missed,
        "extra": max(0, len(pred_times) - matched),
        "precision": round(precision, 3),
        "recall": round(recall, 3),
        "f1": round(f1, 3),
        "jersey_exact_checked": jersey_checked,
        "jersey_exact_correct": jersey_correct,
        "jersey_exact_accuracy": (
            round(jersey_correct / jersey_checked, 3) if jersey_checked else None
        ),
        "by_event": by_event,
    }


def write_evaluation_report(
    events_csv: Path,
    out_json: Path,
    duration_sec: float = 90.0,
    ground_truth_csv: Optional[Path] = None,
    tolerance_sec: float = 0.5,
) -> Dict[str, Any]:
    report: Dict[str, Any] = {
        "events_analysis": analyze_events_csv(events_csv, duration_sec=duration_sec),
        "rally_analysis": analyze_rallies(events_csv),
    }
    if ground_truth_csv and ground_truth_csv.exists():
        report["ground_truth"] = compare_to_ground_truth(
            events_csv, ground_truth_csv, tolerance_sec=tolerance_sec
        )

    out_json.parent.mkdir(parents=True, exist_ok=True)
    out_json.write_text(json.dumps(report, indent=2), encoding="utf-8")
    return report


def summarize_holdout_runs(events_dir: Path, out_csv: Path) -> pd.DataFrame:
    rows = []
    for p in sorted(events_dir.glob("*_events.csv")):
        stem = p.stem.replace("_events", "")
        analysis = analyze_events_csv(p, duration_sec=90.0)
        rally = analyze_rallies(p)
        rows.append({
            "stem": stem,
            **{k: v for k, v in analysis.items() if k != "path"},
            "serve_only_pct": rally.get("serve_only_pct"),
            "reception_serve_ratio": rally.get("reception_serve_ratio"),
            "median_touches_per_rally": rally.get("median_touches_per_rally"),
            "median_serve_duration_sec": rally.get("median_serve_duration_sec"),
            "same_player_serve_reception": rally.get("same_player_serve_reception"),
            "jersey_fill_rate_touches": rally.get("jersey_fill_rate_touches"),
        })
    df = pd.DataFrame(rows)
    if not df.empty:
        df.to_csv(out_csv, index=False)
    return df


def _print_rally_summary(report: Dict[str, Any]) -> None:
    print(f"path                  : {report.get('path')}")
    print(f"n_rallies             : {report.get('n_rallies')}")
    print(f"serve_only            : {report.get('serve_only_rallies')} "
          f"({report.get('serve_only_pct')}%)")
    print(f"reception/serve ratio : {report.get('reception_serve_ratio')}")
    print(f"median touches/rally  : {report.get('median_touches_per_rally')}")
    print(f"median serve duration : {report.get('median_serve_duration_sec')} s")
    print(f"Set+Block share       : {report.get('set_block_share')}")
    print(f"mean touch rate       : {report.get('mean_touch_rate_per_sec')} /s")
    print(f"inferred contact rate : {report.get('inferred_contact_rate')}")
    print(f"jersey fill (touches) : {report.get('jersey_fill_rate_touches')}")
    print(f"jersey dominance max  : {report.get('jersey_dominance_max_frac')}")
    print(f"same-player Serve→Rec : {report.get('same_player_serve_reception')}")
    gaps = report.get("point_to_next_rally_gaps") or {}
    print(f"Point→Rally gap med   : {gaps.get('median_sec')} s "
          f"(<{1}s: {gaps.get('count_lt_1s')}/{gaps.get('n')}; "
          f">30s: {gaps.get('count_gt_30s', 0)})")
    print(f"flags                 : {report.get('summary_flags')}")
    gates = report.get("acceptance_gates") or {}
    if gates:
        passed = sum(1 for v in gates.values() if v)
        print(f"acceptance gates      : {passed}/{len(gates)} pass")
        for k, v in gates.items():
            print(f"  [{'PASS' if v else 'FAIL'}] {k}")
    print(f"event_counts          : {report.get('event_counts')}")
    long_gaps = report.get("long_silence_gaps") or []
    if long_gaps:
        print(f"\nGap report (>30s silence, {len(long_gaps)}):")
        for g in long_gaps[:20]:
            tag = "timeout" if g.get("likely_timeout") else "check_missed_play"
            print(
                f"  after R{g['after_rally_id']} → before R{g['before_rally_id']}: "
                f"{g['gap_sec']}s  [{g['silence_start_sec']}–{g['silence_end_sec']}]  ({tag})"
            )
    print("\nFabio priority rallies:")
    for r in report.get("fabio_review") or []:
        print(f"  rally {r['rally_id']:3d}  touches={r['n_touches']}  "
              f"serve_dur={r['serve_duration_sec']}  flags={r['flags']}")
        print(f"           {r['sequence']}")
    late = report.get("late_point_rallies") or []
    if late:
        print(f"\nLate points (first {min(10, len(late))}):")
        for lp in late[:10]:
            print(f"  rally {lp['rally_id']}: +{lp['gap_sec']}s after last touch")


def main(argv: Optional[List[str]] = None) -> None:
    parser = argparse.ArgumentParser(
        description="VolleyScout rally-level evaluation report (optional gold CSV)"
    )
    parser.add_argument("--csv", type=str, required=True, help="Path to *_events.csv")
    parser.add_argument(
        "--out-json", type=str, default="",
        help="Optional JSON output path (default: <csv_stem>_rally_report.json)",
    )
    parser.add_argument("--duration-sec", type=float, default=0.0,
                        help="Optional duration for events_per_10s in summary")
    parser.add_argument(
        "--ground-truth", type=str, default="",
        help="Optional gold-label CSV for P3 precision/recall/jersey scoring",
    )
    parser.add_argument(
        "--tolerance-sec", type=float, default=0.5,
        help="Match window for gold scoring (default 0.5s)",
    )
    args = parser.parse_args(argv)

    csv_path = Path(args.csv).expanduser().resolve()
    if not csv_path.exists():
        raise SystemExit(f"CSV not found: {csv_path}")

    rally = analyze_rallies(csv_path)
    _print_rally_summary(rally)

    out_json = Path(args.out_json) if args.out_json else csv_path.with_name(
        csv_path.stem.replace("_events", "") + "_rally_report.json"
    )
    duration = args.duration_sec if args.duration_sec > 0 else 90.0
    gt = Path(args.ground_truth).expanduser().resolve() if args.ground_truth else None
    report = write_evaluation_report(
        csv_path, out_json, duration_sec=duration,
        ground_truth_csv=gt, tolerance_sec=args.tolerance_sec,
    )
    if report.get("ground_truth") and "error" not in report["ground_truth"]:
        gt_rep = report["ground_truth"]
        print("\nGold scoring:")
        print(f"  precision={gt_rep['precision']} recall={gt_rep['recall']} f1={gt_rep['f1']}")
        print(f"  jersey exact accuracy={gt_rep.get('jersey_exact_accuracy')}")
    print(f"\nWrote report -> {out_json}")


if __name__ == "__main__":
    main()
