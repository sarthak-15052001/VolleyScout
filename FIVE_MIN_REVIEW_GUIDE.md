# 5-minute Villadoro review guide (CSV ↔ video)

**Next accuracy work (full match):** see [FULL_MATCH_ACCURACY_IMPROVEMENT_PLAN.md](FULL_MATCH_ACCURACY_IMPROVEMENT_PLAN.md).

Use this when checking the latest 5-minute export with your team lead / Fabio.

## Files to open together

| File | Purpose |
|------|---------|
| `output/..._analytics.mp4` | Annotated video (seek by time) |
| `output/..._events.csv` | Event list with timestamps |
| `output/..._rally_report.json` | Auto summary (serve-only %, sequences) |

Video stem (this run):

`2025#09#11#KERAKOLL SASSUOLO MO#NTRANSPORTS VILLADORO MO#Amichevole#01_Villadoro_bmb_25`

---

## How to match CSV rows to the video

### 1. Use the `timestamp` / `time_start_sec` column

Each CSV row has:

- `time_start_sec` — seconds from video start (best for exact seek)
- `time_end_sec` — short display window (~0.4 s for touches)
- `timestamp` — human form `HH:MM:SS` (same moment, rounded)

**Rule:** seek the MP4 to `time_start_sec` (or `timestamp`). That is when the model claims the event happened.

Example — Rally 2:

| event_type | time_start_sec | timestamp | What to look for on video |
|------------|----------------|-----------|---------------------------|
| Rally | 16.68 | 00:00:17 | New rally starts |
| Serve | 16.68 | 00:00:17 | Server contact / ball leave hand |
| Reception | 20.24 | 00:00:20 | First pass after serve |
| Attack | 20.72 | 00:00:21 | Spike / attack swing |
| Reception | 21.44 | 00:00:21 | Defensive touch / dig-like contact |
| Point | 22.92 | 00:00:23 | Rally ends (ball dead / out) |

In VLC / Windows player: seek to **00:00:17**, watch through **00:00:23**, and confirm each action lines up with the table.

### 2. Group by `rally_id`

Do **not** review the CSV as a flat list. Review **one rally at a time**:

1. Filter CSV where `rally_id = N`
2. Sort by `time_start_sec`
3. Read the sequence: `Rally → Serve → Reception → Set → Attack → … → Point`
4. Seek video to that rally’s Serve time and watch until Point

### 3. What “correct” looks like (client / Fabio checklist)

For each rally, ask:

1. **Serve** — Is there a real serve near that timestamp?
2. **In-rally events** — After Serve, do Reception / Set / Attack (kill) appear when they happen on court?
3. **Order** — Does the sequence make volleyball sense?  
   Typical: `Serve → Reception → Set → Attack`  
   Defense: `Attack → Block/Dig → Set → Attack`
4. **Point timing** — Does Point fire when the rally actually ends (not during walk-back)?
5. **Jersey** — If `jersey_number` is filled, does that player’s shirt match? Blank is OK when unsure.
6. **No fake continuation** — After Point, the next Rally should start near the next real serve, not during ball retrieval.

### 4. Fast review workflow (for team lead)

1. Open MP4 + CSV side by side.
2. Pick Fabio priority rallies first: **1, 2, 4, 6, 7, 10**.
3. For each rally, seek to Serve `timestamp`, watch until Point.
4. Mark: OK / Missed event / Wrong label / Wrong time / Wrong jersey.
5. Optionally run:

```bash
python -m volleyball_analytics.evaluation --csv "output/<stem>_events.csv" --duration-sec 300
```

That prints serve-only %, Reception/Serve ratio, same-player Serve→Reception count,
jersey fill rate (blanks correctly excluded), and per-rally sequences.

Optional gold scoring (P3) after labeling rallies:

```bash
python -m volleyball_analytics.evaluation --csv "output/<stem>_events.csv" \
  --ground-truth "labels/<stem>_gold.csv" --tolerance-sec 0.5 --duration-sec 300
```

Prefer `time_start_sec` or `timestamp_ms` for exact seek; `timestamp` is rounded HH:MM:SS.

### 5. Worked examples from this 5-minute export

**Rally 4** (good multi-touch example) — seek **00:00:35**:

```text
Rally → Serve → Reception → Attack → Block → Set → Set → Point
(34.8s … 43.2s)
```

**Rally 7** — seek **00:01:34**:

```text
Rally → Serve → Reception → Attack → Block → Point
(94.0s … 98.2s)
```

**Rally 10** — seek **00:01:58**:

```text
Rally → Serve → Reception → Block → Attack → Set → Set → Dig → Point
(117.8s … 126.9s)
```

**Rally 1** (still weak / serve-only) — seek **00:00:01**:

```text
Rally → Serve → Point
```

If the video shows receive/set after that serve, this is a **missed in-rally** case to note for the lead.

---

## What the black bar at the top of the video means

This export was run with `--debug-overlay`. The top HUD is for engineers / review — not for the final client product.

### Top status line (always on)

```text
Event: Point | Players: 12 | Jerseys: 3 | 1234/7500
```

| Field | Meaning |
|-------|---------|
| `Event:` | Last event type the engine emitted |
| `Players:` | How many player boxes are tracked this frame |
| `Jerseys:` | How many track IDs currently have a jersey number |
| `1234/7500` | Current frame / total frames in this export |

Player boxes also show `ID123` and optionally `#6(0.82)` = jersey + OCR confidence.

### Debug block (with `--debug-overlay`)

Your screenshot example:

```text
R5 active=False touch=0
ball_conf=0.225 speed=0.0 miss=91 since_ball=6.2
contacts=0 dead=0.0 lockout=0.0
last=Point
measured=(234,448)/0.22
predicted=(720,511)/0.16
```

| Line / field | Meaning |
|--------------|---------|
| `R5` | Current **rally_id** (rally number 5) |
| `active=False` | No live rally right now (between points) |
| `touch=0` | Touch index inside the rally (0 = no active rally touches) |
| `ball_conf` | Confidence of the ball used for contact logic |
| `speed` | Estimated ball speed (px/s); `0` = stationary / no track |
| `miss` | Consecutive YOLO frames without a ball detection |
| `since_ball` | Seconds since a trustworthy ball was seen |
| `contacts` | Contact candidates near a player this frame |
| `dead` | Dead-ball timer (seconds) toward ending the Point |
| `lockout` | Seconds left before a new Rally is allowed after Point |
| `last=` | Last event name (e.g. `Point`, `Serve`, `Reception`) |
| `measured=(x,y)/conf` | Last YOLO/measured ball position + conf |
| `predicted=(x,y)/conf` | Kalman predicted ball when YOLO misses |

### How to read that screenshot for the lead

- `last=Point` + `active=False` → point already closed; waiting for next serve.
- `miss=91` + `since_ball=6.2` → ball was lost for a long stretch (common between rallies).
- `measured` vs `predicted` far apart → tracker is guessing; do not trust contacts in that moment.
- `lockout=0.0` → system is allowed to start a new Rally if a serve-like contact appears.

---

## CSV columns (quick reference)

| Column | Use when reviewing |
|--------|--------------------|
| `event_id` | Unique row id |
| `video_name` | Source file name |
| `rally_id` | Group events into one point |
| `event_type` | Rally / Serve / Reception / Set / Attack / Block / Dig / Point |
| `time_start_sec` | **Seek here in the video** |
| `time_end_sec` | Short display end |
| `timestamp` | HH:MM:SS for humans |
| `player_track_id` | Tracker ID of touching player |
| `jersey_number` | Shirt number (blank if uncertain — preferred over wrong) |
| `jersey_confidence` | OCR confidence |
| `event_confidence` | Heuristic label confidence |

---

## Match start (skip training / warm-up)

Fabio: pre-match training (multi-ball, no real jerseys) must **not** be analysed.

**Cross-video contract:** the operator sets the match start time. Do **not** rely on auto warm-up detection.

| Layer | Parameter |
|-------|-----------|
| CLI | `--start-sec 330` (e.g. match begins at 5:30) |
| API upload | Form field `start_seconds` |
| Pipeline | `process_video(..., start_seconds=)` |

Rules:
- `max_seconds` is the **duration from** `start_seconds` (not from t=0).
- CSV / overlay timestamps stay in **original video time**.
- Outputs get a `_from<N>s` suffix when `start_seconds > 0`.

Always note which `start_seconds` was used in the client demo.

Example — Cremonese (training in first ~5 min):

```bash
python export_analytics.py \
  --video "holdout/2025#10#11#ASD PALLAVOLO CREMONESE CR#NTRANSPORTS VILLADORO MO#Serie BM B#01_Cremonese_bmb_25.MP4" \
  --start-sec 330 \
  --max-seconds 600 \
  --max-jersey-number 30 \
  --debug-overlay
```

Optional roster (blank-over-wrong): `--roster 1,3,6,8,9,10,12,18,23,30`

---

## What to tell the team lead (short script)

1. Confirm **match start** (`--start-sec` / `start_seconds`) so training frames are excluded.
2. Open the analytics MP4 and the events CSV together.
3. For each rally, jump to `time_start_sec` of **Serve**, watch until Point.
4. Check Fabio’s asks: **in-rally events present**, **Point ends on time**, **jersey blank or correct**, **no Block after Serve**.
5. Use the top debug HUD only to explain *why* something failed (ball miss, lockout, dead-ball) — not as the client deliverable.

---

## Re-run the same 5-minute check

```bash
python export_analytics.py \
  --video "holdout/2025#09#11#KERAKOLL SASSUOLO MO#NTRANSPORTS VILLADORO MO#Amichevole#01_Villadoro_bmb_25.MP4" \
  --start-sec 0 \
  --max-seconds 300 \
  --debug-overlay

python -m volleyball_analytics.evaluation \
  --csv "output/2025#09#11#KERAKOLL SASSUOLO MO#NTRANSPORTS VILLADORO MO#Amichevole#01_Villadoro_bmb_25_events.csv" \
  --duration-sec 300
```
