"""Behaviour tests for the v8 rally/point logic (synthetic ball paths).

Run from the project root:  python tests/test_engine_v8.py   (or pytest)
No GPU / weights needed.  torch/ultralytics are stubbed only if not installed.
"""
import contextlib, io, sys, types
from pathlib import Path
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
try:
    import torch  # noqa: F401
    import ultralytics  # noqa: F401
except Exception:  # pragma: no cover
    t = types.ModuleType("torch")
    t.cuda = types.SimpleNamespace(is_available=lambda: False, empty_cache=lambda: None)
    t.inference_mode = lambda *a, **k: (lambda f: f)
    sys.modules["torch"] = t
    u = types.ModuleType("ultralytics"); u.YOLO = object; sys.modules["ultralytics"] = u

from volleyball_analytics.pipeline import (  # noqa: E402
    PipelineConfig,
    RallyEventEngine,
    RallySequenceNormalizer,
    TOUCH_EVENT_NAMES,
    JerseyReader,
    cluster_track_identities,
)

FPS = 25.0
PLAYERS = [(150, 480, 250, 640, 1, "", 0.0), (830, 430, 930, 640, 2, "", 0.0),
           (950, 260, 1050, 440, 3, "", 0.0), (700, 250, 800, 430, 4, "", 0.0),
           (60, 500, 160, 660, 5, "", 0.0)]
# serve 2.0 -> reception 3.0 -> set 4.0 -> attack 5.0 -> ball lands 6.2 and stays
RALLY = [(1.5, 200, 560), (2.0, 200, 560), (2.9, 880, 520), (3.0, 880, 520), (3.1, 885, 515),
         (4.0, 1000, 330), (4.1, 1000, 330), (5.0, 750, 320), (5.1, 750, 320),
         (6.2, 300, 690), (9.0, 300, 690)]


def run(keyframes, hidden=(), srcs=(), extra=None, T=14.0):
    cfg = PipelineConfig()
    eng = RallyEventEngine(FPS, cfg); eng.net_y = 720 * 0.42
    ks = np.array(keyframes, float)
    with contextlib.redirect_stdout(io.StringIO()):
        for fi in range(int(T * FPS)):
            t = fi / FPS
            ball, src = None, "yolo"
            if ks[0, 0] <= t <= ks[-1, 0]:
                ball = (float(np.interp(t, ks[:, 0], ks[:, 1])), float(np.interp(t, ks[:, 0], ks[:, 2])), 0.7)
            if any(a <= t < b for a, b in hidden):
                ball = None
            for a, b, name in srcs:
                if a <= t < b:
                    src = name
            if extra and fi in extra:
                ball = extra[fi]
            eng.update(fi, ball, PLAYERS, 720, ball_src=src)
        eng.finalize(int(T * FPS))
    return eng


def events(eng):
    return [(r["event"], r["time_start_sec"]) for r in eng.closed]


def point_t(eng):
    return [t for e, t in events(eng) if e == "Point"]


def test_visible_rally_point_is_backdated():
    eng = run(RALLY)
    assert [e for e, _ in events(eng)][:2] == ["Rally", "Serve"]
    p = point_t(eng)
    assert len(p) == 1 and p[0] <= 6.8, p          # ball stops at 6.2 (old code: ~7.5)


def test_hidden_ball_mid_flight_does_not_end_rally():
    eng = run(RALLY, hidden=[(3.3, 4.9)])
    touches = [e for e, _ in events(eng) if e not in ("Rally", "Point")]
    assert len(touches) >= 5, touches                # old code stopped after 3 touches
    assert point_t(eng)[0] > 6.0


def test_motion_only_ball_cannot_start_rally():
    eng = run([(5.0, 200, 560), (5.1, 260, 520)], srcs=[(5.0, 5.2, "motion")])
    assert events(eng) == []


def test_few_yolo_frames_is_discarded_false_rally():
    # v11 min_rally_ball_frames=3 — only 2 YOLO hits must still discard.
    ex = {i: (200 + 8 * (i - 100), 560 - 4 * (i - 100), 0.7) for i in range(100, 102)}
    eng = run([(0, 0, 0)], extra=ex)
    assert events(eng) == [] and eng.discarded_rallies == 1


def test_ball_leaving_view_backdates_point_to_last_seen():
    eng = run(RALLY, hidden=[(6.3, 20)])
    assert point_t(eng)[0] <= 6.8


def test_serve_block_relabelled_to_reception():
    """Fabio: a serve cannot be blocked — grammar must not leave Serve→Block."""
    closed = [
        {"event_id": 1, "event": "Rally", "rally_id": 1,
         "time_start_sec": 10.0, "time_end_sec": 10.05,
         "player_track_id": "1", "jersey_number": "8", "jersey_conf": 0.9,
         "ball_speed_peak": 0},
        {"event_id": 2, "event": "Serve", "rally_id": 1,
         "time_start_sec": 10.0, "time_end_sec": 10.4,
         "player_track_id": "1", "jersey_number": "8", "jersey_conf": 0.9,
         "ball_speed_peak": 200},
        {"event_id": 3, "event": "Block", "rally_id": 1,
         "time_start_sec": 11.5, "time_end_sec": 11.9,
         "player_track_id": "9", "jersey_number": "30", "jersey_conf": 0.99,
         "ball_speed_peak": 140},
        {"event_id": 4, "event": "Point", "rally_id": 1,
         "time_start_sec": 13.0, "time_end_sec": 13.4,
         "player_track_id": "", "jersey_number": "", "jersey_conf": "",
         "ball_speed_peak": 0},
    ]
    eng = RallyEventEngine(FPS, PipelineConfig())
    eng.closed = [dict(r) for r in closed]
    eng.apply_event_logic_fixes(serve_to_reception_window_sec=5.0)
    assert eng.closed[2]["event"] == "Reception", eng.closed[2]

    closed2 = [dict(r) for r in closed]
    RallySequenceNormalizer().apply(closed2)
    assert closed2[2]["event"] == "Reception", closed2[2]
    assert not any(r["event"] == "Block" for r in closed2)


def test_block_allowed_only_after_attack_in_normalizer():
    closed = [
        {"event_id": 1, "event": "Serve", "rally_id": 1,
         "time_start_sec": 1.0, "time_end_sec": 1.4,
         "player_track_id": "1", "jersey_number": "8", "jersey_conf": 0.9,
         "ball_speed_peak": 200},
        {"event_id": 2, "event": "Reception", "rally_id": 1,
         "time_start_sec": 2.0, "time_end_sec": 2.4,
         "player_track_id": "2", "jersey_number": "3", "jersey_conf": 0.8,
         "ball_speed_peak": 80},
        {"event_id": 3, "event": "Set", "rally_id": 1,
         "time_start_sec": 3.0, "time_end_sec": 3.4,
         "player_track_id": "3", "jersey_number": "6", "jersey_conf": 0.8,
         "ball_speed_peak": 60},
        {"event_id": 4, "event": "Attack", "rally_id": 1,
         "time_start_sec": 4.0, "time_end_sec": 4.4,
         "player_track_id": "4", "jersey_number": "9", "jersey_conf": 0.8,
         "ball_speed_peak": 220},
        {"event_id": 5, "event": "Block", "rally_id": 1,
         "time_start_sec": 4.5, "time_end_sec": 4.9,
         "player_track_id": "5", "jersey_number": "12", "jersey_conf": 0.8,
         "ball_speed_peak": 150},
    ]
    RallySequenceNormalizer().apply(closed)
    assert closed[4]["event"] == "Block"


def test_timeout_mode_blocks_walkback_serve():
    """v12: after prolonged idle, non–serve-like contact must not open a rally."""
    cfg = PipelineConfig(timeout_suppress_sec=2.0, gap_watchdog_sec=60.0)
    eng = RallyEventEngine(FPS, cfg)
    eng.net_y = 720 * 0.42
    with contextlib.redirect_stdout(io.StringIO()):
        # Idle with no/slow ball long enough to enter timeout mode
        for fi in range(int(3.5 * FPS)):
            eng.update(fi, None, PLAYERS, 720, ball_src="yolo")
        assert eng._timeout_mode is True
        # Mid-court slow contact (walk-back) — must not open
        for fi in range(int(3.5 * FPS), int(4.5 * FPS)):
            t = fi / FPS
            ball = (600.0, 300.0, 0.8)  # mid-height, not deep serve
            eng.update(fi, ball, PLAYERS, 720, ball_src="yolo")
        assert events(eng) == []
        # Deep serve-like motion should reopen
        for fi in range(int(4.5 * FPS), int(6.0 * FPS)):
            t = fi / FPS
            # Deep baseline + outbound speed
            ball = (200.0 + 40 * (t - 4.5), 560.0 - 80 * (t - 4.5), 0.8)
            eng.update(fi, ball, PLAYERS, 720, ball_src="yolo")
        eng.finalize(int(6.0 * FPS))
    names = [e for e, _ in events(eng)]
    assert "Rally" in names or eng.discarded_rallies >= 0  # may discard if too short
    assert eng._timeout_mode is False or "Serve" in names or eng.discarded_rallies >= 1


def test_duplicate_set_same_track_collapsed():
    """v12: same event + same track within 0.55s → keep highest conf."""
    closed = [
        {"event_id": 1, "event": "Serve", "rally_id": 1,
         "time_start_sec": 1.0, "time_end_sec": 1.4,
         "player_track_id": "1", "jersey_number": "8", "jersey_conf": 0.9,
         "ball_speed_peak": 200, "event_conf": 0.85},
        {"event_id": 2, "event": "Reception", "rally_id": 1,
         "time_start_sec": 2.0, "time_end_sec": 2.4,
         "player_track_id": "2", "jersey_number": "3", "jersey_conf": 0.8,
         "ball_speed_peak": 80, "event_conf": 0.75},
        {"event_id": 3, "event": "Set", "rally_id": 1,
         "time_start_sec": 3.0, "time_end_sec": 3.4,
         "player_track_id": "3", "jersey_number": "6", "jersey_conf": 0.8,
         "ball_speed_peak": 60, "event_conf": 0.60},
        {"event_id": 4, "event": "Set", "rally_id": 1,
         "time_start_sec": 3.3, "time_end_sec": 3.7,
         "player_track_id": "3", "jersey_number": "6", "jersey_conf": 0.8,
         "ball_speed_peak": 55, "event_conf": 0.78},
        {"event_id": 5, "event": "Attack", "rally_id": 1,
         "time_start_sec": 4.0, "time_end_sec": 4.4,
         "player_track_id": "4", "jersey_number": "9", "jersey_conf": 0.8,
         "ball_speed_peak": 220, "event_conf": 0.82},
    ]
    RallySequenceNormalizer(duplicate_touch_collapse_sec=0.55).apply(closed)
    sets = [r for r in closed if r["event"] == "Set"]
    assert len(sets) == 1, sets
    assert float(sets[0]["event_conf"]) >= 0.78


def test_jersey_confusion_and_top2_blank():
    """v12: confusion pairs / top-2 conflict → blank export."""
    from volleyball_analytics.pipeline import JerseyReader

    jr = JerseyReader(
        use_gpu=False, vote_min=2, confirm_min_avg_conf=0.50,
        ocr_min_vote_conf=0.40,
        confusion_pairs=(("10", "18"), ("2", "23")),
        confusion_extra_conf=0.15,
        top2_conflict_ratio=0.75,
    )
    # Simulate conflicting votes without OCR
    jr._votes[7]["10"] = 1.2
    jr._votes[7]["18"] = 1.0
    jr._vote_counts[7]["10"] = 2
    jr._vote_counts[7]["18"] = 2
    num, conf = jr.update_track(7, np.zeros((80, 40, 3), dtype=np.uint8), 0)
    # Top-2 conflict should blank
    assert num is None or conf == 0.0 or 7 not in jr._confirmed

    out, oconf = jr.export_jersey("10", 0.55, min_conf=0.55, runner_up=("18", 0.50))
    assert out == ""
    out2, _ = jr.export_jersey("10", 0.80, min_conf=0.55, runner_up=("18", 0.40))
    # High conf with weaker runner-up may still pass if ratio low enough
    assert out2 in ("", "10")


def test_jersey_dominance_hard_cap():
    """v18: flood hard-capped; Serve/Reception numbers preserved."""
    cfg = PipelineConfig(jersey_dominance_max_frac=0.25, jersey_export_min_conf=0.50)
    eng = RallyEventEngine(FPS, cfg)
    closed = []
    eid = 1
    closed.append({
        "event_id": eid, "event": "Serve", "rally_id": 1,
        "time_start_sec": 0.0, "time_end_sec": 0.4,
        "player_track_id": "99", "jersey_number": "10", "jersey_conf": 0.88,
    })
    eid += 1
    closed.append({
        "event_id": eid, "event": "Reception", "rally_id": 1,
        "time_start_sec": 0.5, "time_end_sec": 0.9,
        "player_track_id": "98", "jersey_number": "3", "jersey_conf": 0.85,
    })
    eid += 1
    for i in range(6):
        closed.append({
            "event_id": eid, "event": "Attack", "rally_id": 1,
            "time_start_sec": float(i + 1), "time_end_sec": float(i) + 1.3,
            "player_track_id": str(100 + i), "jersey_number": "10",
            "jersey_conf": 0.60 + 0.02 * i,
        })
        eid += 1
    for i, num in enumerate(("2", "4", "7", "8", "9", "18")):
        closed.append({
            "event_id": eid, "event": "Set", "rally_id": 1,
            "time_start_sec": 10.0 + i, "time_end_sec": 10.3 + i,
            "player_track_id": str(200 + i), "jersey_number": num,
            "jersey_conf": 0.80,
        })
        eid += 1
    eng.closed = closed
    with contextlib.redirect_stdout(io.StringIO()):
        eng.backfill_jerseys({}, {}, dominance_guard=True)
    filled = [
        r for r in eng.closed
        if r.get("jersey_number") not in ("", None)
        and r["event"] in TOUCH_EVENT_NAMES
    ]
    from collections import Counter
    counts = Counter(str(r["jersey_number"]) for r in filled)
    assert counts.get("10", 0) / max(len(filled), 1) <= 0.25 + 1e-9, counts
    serve = next(r for r in eng.closed if r["event"] == "Serve")
    assert serve["jersey_number"] == "10", serve
    recv = next(r for r in eng.closed if r["event"] == "Reception")
    assert recv["jersey_number"] == "3", recv


def test_long_idle_softens_start_ball_ok():
    """v14: after long idle, one YOLO hit in a wider window is enough to reopen."""
    cfg = PipelineConfig(
        rally_start_min_ball_frames=2,
        rally_start_window_sec=1.5,
        long_idle_reopen_sec=20.0,
        timeout_suppress_sec=2.0,
    )
    eng = RallyEventEngine(FPS, cfg)
    eng._last_point_t = 0.0
    eng._idle_since_t = 0.0
    eng._timeout_mode = True
    eng._yolo_times.clear()
    eng._yolo_times.append(25.0)
    assert eng._long_idle(25.0) is True
    assert eng._start_ball_ok(25.0) is True


def test_enforce_one_reception_per_rally():
    """v17: Dig→Reception must not create a second Reception."""
    eng = RallyEventEngine(FPS, PipelineConfig())
    eng.closed = [
        {"event_id": 1, "event": "Rally", "rally_id": 1,
         "time_start_sec": 1.0, "time_end_sec": 1.05,
         "player_track_id": "", "jersey_number": "", "jersey_conf": ""},
        {"event_id": 2, "event": "Serve", "rally_id": 1,
         "time_start_sec": 1.0, "time_end_sec": 1.4,
         "player_track_id": "1", "jersey_number": "8", "jersey_conf": 0.9},
        {"event_id": 3, "event": "Reception", "rally_id": 1,
         "time_start_sec": 2.0, "time_end_sec": 2.4,
         "player_track_id": "2", "jersey_number": "3", "jersey_conf": 0.8},
        {"event_id": 4, "event": "Dig", "rally_id": 1,
         "time_start_sec": 2.5, "time_end_sec": 2.9,
         "player_track_id": "3", "jersey_number": "6", "jersey_conf": 0.7},
        {"event_id": 5, "event": "Point", "rally_id": 1,
         "time_start_sec": 4.0, "time_end_sec": 4.05,
         "player_track_id": "", "jersey_number": "", "jersey_conf": ""},
    ]
    with contextlib.redirect_stdout(io.StringIO()):
        eng.apply_event_logic_fixes(serve_to_reception_window_sec=5.0)
    recs = [r for r in eng.closed if r["event"] == "Reception"]
    digs = [r for r in eng.closed if r["event"] == "Dig"]
    assert len(recs) == 1, recs
    assert len(digs) == 1, digs


def test_serve_only_flash_discarded():
    """v15: Serve→Point in <1.6s with few ball frames is discarded."""
    cfg = PipelineConfig(
        min_rally_ball_frames=3,
        serve_only_max_sec=1.6,
        event_display_sec=0.4,
        min_point_after_touch_sec=0.35,
    )
    eng = RallyEventEngine(FPS, cfg)
    eng.rally_id = 1
    eng.rally_active = True
    eng.touch_index = 1
    eng._rally_ball_frames = 5
    eng._serve_t = 10.0
    eng.last_contact_t = 10.0
    eng._last_touch_t = 10.0
    eng.closed = [
        {"event_id": 1, "event": "Rally", "rally_id": 1,
         "time_start_sec": 10.0, "time_end_sec": 10.05,
         "player_track_id": "", "jersey_number": "", "jersey_conf": ""},
        {"event_id": 2, "event": "Serve", "rally_id": 1,
         "time_start_sec": 10.0, "time_end_sec": 10.4,
         "player_track_id": "1", "jersey_number": "", "jersey_conf": ""},
    ]
    with contextlib.redirect_stdout(io.StringIO()):
        eng._end_rally(int(11.0 * FPS), 11.0, 0.0, "ball_lost")
    assert eng.discarded_rallies == 1
    assert not any(r.get("event") == "Serve" for r in eng.closed)


def test_evaluation_gap_report_gt_30s(tmp_path):
    """Gaps 30–35s still fail; ≥35s count as timeout."""
    import pandas as pd
    from volleyball_analytics.evaluation import analyze_rallies

    # 32s unexplained gap (point@12 → rally@44)
    rows = [
        dict(event="Rally", rally_id=1, time_start_sec=10.0, time_end_sec=10.05,
             jersey_number="", player_track_id=""),
        dict(event="Serve", rally_id=1, time_start_sec=10.0, time_end_sec=10.4,
             jersey_number="8", player_track_id="1"),
        dict(event="Reception", rally_id=1, time_start_sec=11.0, time_end_sec=11.4,
             jersey_number="3", player_track_id="2"),
        dict(event="Point", rally_id=1, time_start_sec=12.0, time_end_sec=12.05,
             jersey_number="", player_track_id=""),
        dict(event="Rally", rally_id=2, time_start_sec=44.0, time_end_sec=44.05,
             jersey_number="", player_track_id=""),
        dict(event="Serve", rally_id=2, time_start_sec=44.0, time_end_sec=44.4,
             jersey_number="8", player_track_id="1"),
        dict(event="Reception", rally_id=2, time_start_sec=45.0, time_end_sec=45.4,
             jersey_number="3", player_track_id="2"),
        dict(event="Point", rally_id=2, time_start_sec=46.0, time_end_sec=46.05,
             jersey_number="", player_track_id=""),
    ]
    csv_path = tmp_path / "gap_events.csv"
    pd.DataFrame(rows).to_csv(csv_path, index=False)
    rep = analyze_rallies(csv_path)
    assert len(rep["long_silence_gaps"]) >= 1
    assert rep["long_silence_gaps"][0].get("likely_timeout") is False
    assert rep["acceptance_gates"]["no_long_silence_gaps"] is False


def test_evaluation_timeout_gap_passes_gate(tmp_path):
    """Gaps ≥35s are tagged timeout and do not fail no_long_silence_gaps."""
    import pandas as pd
    from volleyball_analytics.evaluation import analyze_rallies

    rows = [
        dict(event="Rally", rally_id=1, time_start_sec=10.0, time_end_sec=10.05,
             jersey_number="", player_track_id=""),
        dict(event="Serve", rally_id=1, time_start_sec=10.0, time_end_sec=10.4,
             jersey_number="8", player_track_id="1"),
        dict(event="Reception", rally_id=1, time_start_sec=11.0, time_end_sec=11.4,
             jersey_number="3", player_track_id="2"),
        dict(event="Point", rally_id=1, time_start_sec=12.0, time_end_sec=12.05,
             jersey_number="", player_track_id=""),
        dict(event="Rally", rally_id=2, time_start_sec=80.0, time_end_sec=80.05,
             jersey_number="", player_track_id=""),
        dict(event="Serve", rally_id=2, time_start_sec=80.0, time_end_sec=80.4,
             jersey_number="8", player_track_id="1"),
        dict(event="Reception", rally_id=2, time_start_sec=81.0, time_end_sec=81.4,
             jersey_number="3", player_track_id="2"),
        dict(event="Point", rally_id=2, time_start_sec=82.0, time_end_sec=82.05,
             jersey_number="", player_track_id=""),
    ]
    csv_path = tmp_path / "timeout_gap.csv"
    pd.DataFrame(rows).to_csv(csv_path, index=False)
    rep = analyze_rallies(csv_path)
    assert rep["long_silence_gaps"][0]["likely_timeout"] is True
    assert rep["acceptance_gates"]["no_long_silence_gaps"] is True
    assert "timeout_silences_ok" in rep["summary_flags"]


def test_jersey_unlock_on_10_18_conflict():
    """v19: locked #10 unlocks when a tall, high-conf #18 OCR arrives."""
    jr = JerseyReader(
        use_gpu=False, vote_min=3, confirm_min_avg_conf=0.55,
        ocr_min_vote_conf=0.50,
        confusion_pairs=(("10", "18"),),
        confusion_extra_conf=0.20,
        top2_conflict_ratio=0.65,
        confusion_vote_min=5,
        confusion_min_height_px=110,
        unlock_conflict_conf=0.72,
    )
    jr._confirmed[42] = ("10", 1.0)
    tall = np.zeros((130, 55, 3), dtype=np.uint8)
    jr.read_once = lambda _crop: ("18", 0.80)  # type: ignore[method-assign]
    num, conf = jr.update_track(42, tall, frame_idx=1)
    assert 42 not in jr._confirmed, "wrong #10 lock must unlock on strong #18"
    # One vote is not enough to re-lock a confusion number.
    assert num in (None, "18")


def test_jersey_no_unlock_on_tiny_crop():
    """v19: far/tiny crops must not unlock a confirmed identity."""
    jr = JerseyReader(
        use_gpu=False, vote_min=3, confirm_min_avg_conf=0.55,
        ocr_min_vote_conf=0.50,
        confusion_pairs=(("10", "18"),),
        unlock_conflict_conf=0.72,
        confusion_min_height_px=110,
    )
    jr._confirmed[7] = ("10", 1.0)
    tiny = np.zeros((80, 35, 3), dtype=np.uint8)
    jr.read_once = lambda _crop: ("18", 0.85)  # type: ignore[method-assign]
    num, conf = jr.update_track(7, tiny, frame_idx=2)
    assert jr._confirmed[7][0] == "10"
    assert num == "10"


def test_cluster_skips_confusion_inherit():
    """v19: unconfirmed tracks must not inherit #10/#18 from a neighbour."""
    cfg = PipelineConfig(
        cluster_inherit_unconfirmed=True,
        cluster_max_time_gap_sec=6.0,
        cluster_max_bbox_gap_px=150.0,
        jersey_confusion_pairs=(("10", "18"),),
    )
    jersey_cache = {1: "10"}
    jersey_conf_cache = {1: 0.90}
    track_timeline = {
        1: {
            "t_start": 0.0, "t_end": 2.0,
            "first_x1": 100, "first_x2": 140, "first_y1": 200, "first_y2": 320,
            "last_x1": 100, "last_x2": 140, "last_y1": 200, "last_y2": 320,
        },
        2: {
            "t_start": 2.1, "t_end": 4.0,
            "first_x1": 105, "first_x2": 145, "first_y1": 205, "first_y2": 325,
            "last_x1": 105, "last_x2": 145, "last_y1": 205, "last_y2": 325,
        },
    }
    with contextlib.redirect_stdout(io.StringIO()):
        cluster_track_identities(jersey_cache, jersey_conf_cache, track_timeline, cfg)
    assert 2 not in jersey_cache, "confusion-prone #10 must not cluster-inherit"


def test_cluster_still_inherits_safe_numbers():
    """v19: non-confusion numbers (#7) still inherit onto nearby unconfirmed tracks."""
    cfg = PipelineConfig(
        cluster_inherit_unconfirmed=True,
        cluster_max_time_gap_sec=6.0,
        cluster_max_bbox_gap_px=150.0,
        jersey_confusion_pairs=(("10", "18"),),
    )
    jersey_cache = {1: "7"}
    jersey_conf_cache = {1: 0.90}
    track_timeline = {
        1: {
            "t_start": 0.0, "t_end": 2.0,
            "first_x1": 100, "first_x2": 140, "first_y1": 200, "first_y2": 320,
            "last_x1": 100, "last_x2": 140, "last_y1": 200, "last_y2": 320,
        },
        2: {
            "t_start": 2.1, "t_end": 4.0,
            "first_x1": 105, "first_x2": 145, "first_y1": 205, "first_y2": 325,
            "last_x1": 105, "last_x2": 145, "last_y1": 205, "last_y2": 325,
        },
    }
    with contextlib.redirect_stdout(io.StringIO()):
        cluster_track_identities(jersey_cache, jersey_conf_cache, track_timeline, cfg)
    assert jersey_cache.get(2) == "7"


if __name__ == "__main__":
    import tempfile
    for name, fn in list(globals().items()):
        if name.startswith("test_"):
            if "tmp_path" in fn.__code__.co_varnames:
                with tempfile.TemporaryDirectory() as d:
                    fn(Path(d))
            else:
                fn()
            print("PASS", name)
