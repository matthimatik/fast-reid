# AGENTS.md

FastReID: PyTorch re-identification research platform, detectron2-style (yacs configs + registries + `build_*` factories). No `setup.py`/`pip install -e .` — `fastreid` is importable only from the repo root.

## Always run from the repo root

Every entrypoint does `sys.path.append('.')`, so `fastreid` resolves only when cwd is the repo root (verified: fails from any other directory). Applies to `tools/*`, `demo/*`, and project scripts (`projects/*/train_net.py`, `projects/FastTune/tune_net.py`).

## Setup (this clone)

- Venv at `.venv/` (Python 3.11, torch 2.4.1+cu121, yacs, faiss-cpu, gdown, tensorboard, opencv-headless, Cython). Activate with `source .venv/bin/activate`; no conda, system pip is PEP-668 locked.
- Code is patched for torch ≥ 2 / Python 3.11: `collections.abc.Mapping` (`fastreid/data/build.py`, `fastreid/evaluation/testing.py`) and `optimizer.zero_grad(set_to_none=False)` in both trainers (`fastreid/engine/train_loop.py`) — the latter is required because `ContiguousParams` shares one grad buffer across param groups; `set_to_none=True` (torch≥2 default) severs it and the optimizer sees no grads (`AssertionError: No inf checks were recorded`). Do not revert.
- Cython rank eval compiled (`make all` in `fastreid/evaluation/rank_cylib`); generated `.c` files are gitignored.
- Verified end-to-end: 40-epoch synthetic Market1501 run trains on the RTX 3090, checkpoints save, `--eval-only` from a checkpoint evaluates, ImageNet backbone download works.

## Commands

```bash
# train — trailing opts are bare yacs "KEY VALUE" pairs (argparse REMAINDER)
python3 tools/train_net.py --config-file configs/Market1501/bagtricks_R50.yml MODEL.DEVICE cuda:0
python3 tools/train_net.py --config-file <cfg> --num-gpus 4

# eval
python3 tools/train_net.py --config-file <cfg> --eval-only \
  MODEL.WEIGHTS /path/to/model_final.pth MODEL.DEVICE cuda:0

# demo scripts differ: overrides must follow --opts
python3 demo/demo.py --config-file <cfg> --input img.jpg --opts MODEL.WEIGHTS /path/to.pth
```

- Multi-machine: `--num-machines/--machine-rank/--dist-url` plus `GLOO_SOCKET_IFNAME`/`NCCL_SOCKET_IFNAME` (see `GETTING_STARTED.md`). Shipped configs assume 1 GPU (`SOLVER.IMS_PER_BATCH`).
- Trainer is **epoch-based**: `SOLVER.MAX_EPOCH`, `SOLVER.CHECKPOINT_PERIOD`, `TEST.EVAL_PERIOD` are in epochs — there is no `SOLVER.MAX_ITER`. `DefaultTrainer.train()` asserts an eval ran during training, so `TEST.EVAL_PERIOD` must fire within `MAX_EPOCH` (0 disables eval and crashes at end of training).
- LR schedule quirks (verified with instrumented hooks, torch 2.4):
  - `WarmupLR` steps per **iteration** while `next_iter <= SOLVER.WARMUP_ITERS`; `MultiStepLR` steps per **epoch** after warmup. With AMP, `LRScheduler.after_step` skips the warmup step whenever the GradScaler scale changed (overflow); the scaler's initial backoff (65536→~256) eats the first ~8 iterations, so tiny `WARMUP_ITERS` (<10) can be fully swallowed and LR stays at `BASE_LR * WARMUP_FACTOR`. Defaults use `WARMUP_ITERS: 2000`, which is fine.
  - torch ≥2 `MultiStepLR.get_lr()` multiplies the **current** `param_group["lr"]` at milestones (not `base_lrs`), so milestone decay stacks on whatever LR warmup left behind.
- Optional but recommended: compile Cython rank eval, otherwise a slow pure-Python fallback with a warning is used:
  `cd fastreid/evaluation/rank_cylib && make all`

## Dependencies (no root requirements.txt)

`INSTALL.md` is the dependency list; CI references a `requirements.txt` that does not exist. Verified failure points:

- `yacs` — any `fastreid.config` import (config-only code needs just yacs + pyyaml).
- `faiss` (`pip install faiss-cpu`) — import of `fastreid.evaluation`/`fastreid.engine` via `fastreid/utils/compute_dist.py:10`, so **any** train/eval script.
- `gdown` — building a model with `MODEL.BACKBONE.PRETRAIN: True` (default): downloads ImageNet weights into `$TORCH_HOME/checkpoints` (`~/.cache/torch/checkpoints`). Offline: place files there, or set `MODEL.BACKBONE.PRETRAIN_PATH`.
- Optional: `apex` for `SOLVER.FP16_ENABLED`, `tensorboard` for TB logs, `ipdb` (unused by real workflows).

## Datasets and outputs

- Dataset root from `$FASTREID_DATASETS`, default `datasets/` relative to cwd (`fastreid/data/build.py:34`). Expected layouts in `datasets/README.md`.
- Each run writes `config.yaml`, `metrics.json`, and checkpoints into `cfg.OUTPUT_DIR` (e.g. `logs/market1501/bagtricks_R50`, gitignored). `config.yaml` there is the full resolved config — the source of truth for reproducing a run.

## Config system

- New keys **must** be declared in `fastreid/config/defaults.py` first; unknown keys raise `AssertionError` on merge (yacs is strict).
- CLI overrides are `KEY VALUE` pairs; tuples/lists need shell quoting: `'("CrossEntropyLoss",)'`.
- YAML inheritance via `_BASE_: ../Base-bagtricks.yml` — allowed only in files, not on the CLI (`fastreid/config/config.py`). Base files: `configs/Base-{bagtricks,AGW,SBS,MGN}.yml`; per-dataset files live in `configs/<Dataset>/`.
- `COMPUTED_*` keys are insertion-only even when frozen.

## Extending (registry pattern)

- Components self-register at import time via `@*_REGISTRY.register()`: meta-archs, backbones, heads, losses, datasets, samplers.
- A new module is invisible until it is imported in the package `__init__.py` (e.g. `fastreid/modeling/backbones/__init__.py`, `fastreid/data/datasets/__init__.py`); otherwise `build_*` raises a KeyError on the name.
- Builders look up `cfg.<...>.NAME`; classes use `@configurable` plus a `from_config` classmethod that reads values from cfg.

## Tests / verification

- **`pytest` is not a usable gate.** `tests/` is stale: pre-v1.0 imports (`from data import ...`, `from config import ...`), hardcoded `/export/home/lxy/...` paths, live `ipdb.set_trace()` calls, CUDA + personal checkpoint paths, fastai. Collection fails; CI runs it with `|| true`. Do not try to make `pytest tests/` pass unless explicitly asked.
- CI (`.github/workflows/lint_python.yml`) only enforces `flake8 . --count --select=E9,F63,F7,F82` (syntax errors / undefined names). `black`, `isort`, `codespell` run with `|| true` and there is no formatter config — match the surrounding style (4-space, detectron2-ish, mixed quote styles).
- Checks that actually work:
  ```bash
  python3 -m compileall -q fastreid tools demo projects tests
  python3 -c "from fastreid.config import get_cfg; c=get_cfg(); c.merge_from_file('configs/Market1501/bagtricks_R50.yml')"
  ```
- Data-free model smoke test: build from a config with `MODEL.BACKBONE.PRETRAIN False` to avoid the weight download. Full train/eval smoke test uses the synthetic Market1501-format set at `datasets/Market-1501-v15.09.15/` (10 train / 10 test ids, generated locally, gitignored): `python3 tools/train_net.py --config-file configs/Market1501/bagtricks_R50.yml SOLVER.MAX_EPOCH 40 TEST.EVAL_PERIOD 20 MODEL.BACKBONE.PRETRAIN False OUTPUT_DIR /tmp/smoke MODEL.DEVICE cuda:0`.

## projects/

Research projects layered on the library (`projects/README.md`). Run their entrypoints from the repo root too, e.g. `python3 projects/FastDistill/train_net.py --config-file projects/FastDistill/configs/...`; project-specific modules register themselves via top-level `import` in that script. Caveats:

- Entrypoints vary: `train_net.py` in most, `tune_net.py` in FastTune, a C++/pybind build in FastRT.
- `CrossDomainReID`, `DG-ReID`, `HAA` are README-only pointers to external repos — no code here.
- Support/stability vary per project; the per-project README is the only trustworthy source.

## Docs

Sphinx: `cd docs && make html` (deps in `docs/requirements.txt`); published at fast-reid.readthedocs.io.
