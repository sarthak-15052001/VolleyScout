# Ball label review checklist

Auto-labels are a **first pass**. Clean them before expecting strong ball mAP.

## Goal

- Train positives: **≥ 1000** good ball boxes (ideally 2k–3k)
- Test set: **300–500** hand-checked frames never used in training
- Target after clean retrain: **mAP50 ≥ 0.55** on held-out test (stretch: 0.65+)

## Pipeline

```bash
conda activate py312
cd "/media/iflair/New Volume/VolleyScout"

# 1) Rebuild denser trajectory dataset (train/ videos only)
python rebuild_ball_dataset.py --max-seconds 900 --label-stride 1 --keep-staging

# 2) Export top uncertain frames for review
python prepare_ball_review.py --top 200

# 3) Fix labels (see below), then retrain
```

Optional: also feed current ball weights as candidates:

```bash
python rebuild_ball_dataset.py --max-seconds 900 --label-stride 1 \
  --use-coco-candidates --pretrained models/volleyball_ball_best.pt --keep-staging
```

## Review each overlay (`output/ball_review_pack/overlays/`)

For every frame:

| Check | Action if wrong |
|-------|-----------------|
| Box on the **volleyball** | Keep / tighten YOLO line in matching `dataset_ball/labels/.../*.txt` |
| Box on head / shoe / scoreboard / reflection | **Delete** box → empty `.txt` (negative) or delete image+label pair |
| Ball visible but **no** box | Add YOLO line: `0 cx cy w h` (normalised 0–1) |
| Interpolated blur / wrong side of net | Prefer blank over a wrong box |

Mark rows in `output/ball_review_pack/CHECKLIST.md` as you go.

## YOLO label format

```text
0 <cx> <cy> <w> <h>
```

All values normalised by image width/height. One ball per frame is enough.

## After review — retrain

```bash
python -c "
from ultralytics import YOLO
from pathlib import Path
import shutil
YOLO('yolo26n.pt').train(
    data='data_ball.yaml', epochs=50, imgsz=1280, batch=2, device=0,
    project='runs', name='volleyball_ball_detection', exist_ok=True,
    patience=20, workers=0, cos_lr=True, close_mosaic=8)
Path('models').mkdir(exist_ok=True)
shutil.copy2('runs/volleyball_ball_detection/weights/best.pt', 'models/volleyball_ball_best.pt')
print('OK -> models/volleyball_ball_best.pt')
"
```

Then re-export Cremonese and compare inferred-contact rate / gaps.

## Do not

- Label from `holdout/` into train
- Train on unreviwed COCO-only pseudo-labels as the only signal
- Keep huge empty/background sets that hide missed balls
