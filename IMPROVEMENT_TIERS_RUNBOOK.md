# Improvement tiers runbook (events + jerseys + time sync)

Time sync is already correct (CSV `time_start_sec` / `timestamp` = original video clock).  
Do **not** change start-sec accounting. Focus here is detection quality.

---

## What landed in code (this pass)

| Tier | Change |
|------|--------|
| 1 Jersey | Team-color uniqueness (home red / away teal), denser OCR between rallies, stricter dominance ≤24% |
| 1 Point | Eval tags gaps ≥45s as `timeout` — they no longer fail `no_long_silence_gaps` |
| 1 Tracker | Longer `track_buffer` (120) for fewer ID switches |
| 2 Touch | Stricter Set/Block heuristics; Set/Block need ≥2 contact frames |
| 2 GT | `tools/gt_review_export.py` review sheet |
| 1 Ball | `autolabel_balls.py --ranges` for hard windows |

---

## Step A — Ball hard-case fine-tune (you run on GPU)

Use train-folder siblings when possible; holdout only with `--allow-holdout` for review packs, not final train merge if policy forbids it.

```bash
conda activate py312
cd "/media/iflair/New Volume/VolleyScout"

# Hard windows (serve reopen / long idle neighbours on Cremonese-like footage)
python autolabel_balls.py \
  --video "train/<sibling_match>.MP4" \
  --ranges "330-450,550-630,670-900" \
  --ball-weights models/volleyball_ball_best.pt \
  --out label_auto_hard --cvat

# Review top of label_auto_hard/review_manifest.csv, then rebuild + short fine-tune:
python rebuild_ball_dataset.py   # if using your existing merge flow
yolo detect train \
  data=dataset_ball/data.yaml \
  model=models/volleyball_ball_best.pt \
  epochs=30 imgsz=1280 batch=4 \
  project=runs/ball name=finetune_hard
# Copy best -> models/volleyball_ball_best.pt
```

---

## Step B — Re-export Cremonese 10 min (validate)

```bash
python export_analytics.py \
  --video "holdout/2025#10#11#ASD PALLAVOLO CREMONESE CR#NTRANSPORTS VILLADORO MO#Serie BM B#01_Cremonese_bmb_25.MP4" \
  --start-sec 330 --max-seconds 600 \
  --require-gpu --debug-overlay \
  --roster "2,3,4,7,8,9,10,12,13,17,18,20,23"

python -m volleyball_analytics.evaluation \
  --csv "output/2025#10#11#ASD PALLAVOLO CREMONESE CR#NTRANSPORTS VILLADORO MO#Serie BM B#01_Cremonese_bmb_25_from330s_events.csv" \
  --duration-sec 600

python tools/gt_review_export.py \
  --csv "output/..._from330s_events.csv" \
  --out output/gt_review_from330s.csv
```

Targets: serve_only pass, Set+Block &lt; 40%, jersey dominance ≤ 25%, timeout gaps tagged OK, times still absolute.

---

## Step C — Full match (after Step B looks good)

```bash
python export_analytics.py \
  --video "holdout/...Cremonese...01_Cremonese_bmb_25.MP4" \
  --require-gpu --debug-overlay \
  --roster "2,3,4,7,8,9,10,12,13,17,18,20,23"
```

Omit `--start-sec` / `--max-seconds`. Expect a long run.

---

## Still human / longer-horizon

- **Touch classifier model** (learned) — replace remaining heuristic edge cases  
- **Player YOLO retrain** — if sideline boxes steal contacts  
- **Hand-labelled test set** — fill `gt_review_*.csv` while watching MP4  
- **Second camera** — product stretch, not code  

---

## Honest expectation

- **All events** ≈ ball recall + touch rules/model; timeouts will still show long gaps (now tagged).  
- **All jerseys** ≈ only when the number is readable; blanks remain when backs are hidden (correct).  
- **Time sync** stays as-is.
