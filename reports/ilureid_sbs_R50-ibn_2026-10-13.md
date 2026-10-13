# ILUReID baseline training report

Date: 2026-10-13 · Code: fast-reid @ `745400d` · Hardware: single RTX 3090 (24 GB) · Wall time: 17 min 21 s (3 538 iterations)

## TL;DR

| metric | value |
|---|---|
| Rank-1 | **38.2 %** |
| Rank-5 | 64.7 % |
| Rank-10 | 73.5 % |
| mAP | **44.4 %** |
| mINP | 38.2 % |

Best and final checkpoint are the same (epoch 60): `logs/ilureid/sbs_R50-ibn/model_best.pth`.
This is a first baseline from one default config with no tuning. It is clearly above
random (Rank-1 would be ~0.7 %), but not yet usable for automated matching. The metric
curve was still improving when training ended — headroom is expected.

## Task and data

Re-identification of intermodal loading units across heterogeneous cameras in a terminal.
Data: TRUDI build `reid/ilu_reid/market1501` (Market1501-style layout), 13 viewpoint
classes (bridge, bus, car, cctv, gate, high_structure, nadir, oblique, person,
reach_stacker, terminal_truck, train, unknown).

| split | identities | images | cameras | notes |
|---|---|---|---|---|
| train | 2 168 | 3 784 | 11 | 1 544 identities (71 %) have a single image; mean 1.7 views/id |
| val | 133 | 505 | — | **unused** (loader does not consume it) |
| test query | 135 | 135 | 7 | one query image per identity |
| test gallery | 135 | 407 | 9 | ~3 views per identity |

Identity-disjoint train/test. The heavy single-view skew is the defining property of this
dataset: the batch-hard triplet loss has almost no positive pairs to mine for 71 % of
training identities, so the classification (ID) loss does most of the representation
learning.

## Setup

- Model: SBS ResNet-50 + IBN + Non-Local (`configs/ILUReID/sbs_R50-ibn.yml`, Base-SBS),
  GeM pooling, BNNeck, CircleSoftmax (margin 0.35, scale 64).
- Losses: label-smoothed cross-entropy (ε=0.1) + batch-hard triplet (margin 0).
- Input 256×256, horizontal flip + random erasing + AutoAugment (p=0.1).
- Optimizer: Adam, base LR 3.5e-4, batch 64, AMP; backbone frozen for the first
  1 000 iterations (~epoch 17); cosine schedule with `DELAY_EPOCHS: 30`.
- Sampler: NaiveIdentitySampler, `NUM_INSTANCE: 4` (identities with <4 images are padded
  by resampling, i.e. the batch contains duplicate crops).
- Eval every 10 epochs on the test query/gallery set (fast-reid default; see caveats).

## Results

| epoch | Rank-1 | Rank-5 | Rank-10 | mAP | mINP |
|---|---|---|---|---|---|
| 10 | 29.4 | 44.1 | 44.1 | 29.9 | 24.4 |
| 20 | 35.3 | 52.9 | 58.8 | 39.2 | 35.3 |
| 30 | 41.2 | 50.0 | 61.8 | 41.2 | 35.3 |
| 40 | 35.3 | 55.9 | 64.7 | 39.9 | 35.1 |
| 50 | 32.4 | 58.8 | 67.7 | 39.3 | 34.2 |
| **60 (final)** | **38.2** | **64.7** | **73.5** | **44.4** | **38.2** |

Reference points:

- Random ranking: Rank-1 ≈ 0.7 % (≈3 correct of 407 gallery images per query).
- 2-epoch smoke run with a completely frozen backbone: Rank-1 29.4 % — i.e. generic
  ImageNet features plus a barely-trained head already reach three quarters of the final
  Rank-1. Task-specific fine-tuning adds ~9 points of Rank-1 and ~15 points of mAP over
  that; the headroom beyond is where a tuned model must come from.

Training dynamics: total loss fell 54 → 4.7 within 33 epochs; train batch classification
accuracy reached ~96 %; the triplet loss collapsed to ~0.05 by epoch 30 (consistent with
exhausted positive pairs on single-view identities).

## Assessment

**What works.** The model learns real cross-view features (55× random), transfers from
ImageNet features, and training is stable (no divergences, smooth loss curves).

**What limits it.**

1. **Not converged.** Rank-5 rose monotonically from epoch 30 to 60 (50 → 65) and the best
   mAP is the last epoch's. Training longer should keep paying.
2. **Schedule mismatch.** `DELAY_EPOCHS: 30` on a 60-epoch run means LR decay only covered
   the second half; the inherited Base-SBS schedule assumes longer runs.
3. **Triplet starved.** With 71 % single-view identities the metric-learning component is
   mostly inactive; performance rides on the ID classification loss.
4. **Small test set.** 135 identities; each identity moves Rank-1 by 0.74 points, and the
   ±3–5 point swings between checkpoints (41 → 35 → 38 in epochs 30–60) are partly noise.
5. **Protocol caveat.** fast-reid selects `model_best.pth` by test-set metric and the val
   split (133 identities, 505 images) is unused, so the reported numbers slightly favor
   the final model; for a production claim, freeze the checkpoint and re-evaluate once on
   a held-out set.

**Practical read.** As a human-in-the-loop search aid ("rank the 5 most likely previous
sightings of this unit"), 65 % top-5 on a 407-image gallery is marginally useful today.
For automated linkage (gate matching, camera-to-camera tracking) 38 % top-1 is not
deployable.

## Recommended next steps (cheapest first)

1. Train 120–180 epochs with `SOLVER.DELAY_EPOCHS` ≈ half of `MAX_EPOCH` (~40–60 min on
   the 3090). Single most promising change.
2. Merge the val split into train (+133 identities, +13 % images), or build a proper
   val protocol (select on val, report once on test).
3. Diagnostics before further tuning: per-camera-pair CMC/heatmap on the test set. The
   13 viewpoints are very unequal (e.g. nadir vs oblique); knowing which pairs fail tells
   where to invest (resolution, augmentation, hard-negative mining).
4. Input resolution 384×192 (Base-SBS default) and stronger augmentation (higher
   AutoAugment probability, CutBlur/RandomPatch) — unit graphics/text are fine detail.
5. Loss/sampler suited to single-view data: lower triplet weight or margin-based losses
   (Circle, AGW recipe), identity-balanced sampling without duplicate padding.
6. Architecture variants: OSNet, or plain R50 with the AGW tuning recipe.

## Reproduce

```bash
# dataset (from /data/8hirsch/trudi)
uv run trudi build 'reid/ilu_reid/market1501' --workers 8 --force

# train (from the fast-reid repo root)
python3 tools/train_net.py --config-file configs/ILUReID/sbs_R50-ibn.yml MODEL.DEVICE cuda:0

# evaluate the saved checkpoint
python3 tools/train_net.py --config-file configs/ILUReID/sbs_R50-ibn.yml --eval-only \
  MODEL.WEIGHTS logs/ilureid/sbs_R50-ibn/model_best.pth MODEL.DEVICE cuda:0
```

Full resolved config, checkpoints, TensorBoard events and per-iteration metrics:
`logs/ilureid/sbs_R50-ibn/` (`config.yaml`, `model_best.pth`, `metrics.json`, `train.log`).
