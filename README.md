# Laya × Multi3NLU++ (Turkish banking) with Ray

Fine-tune and evaluate [Laya](https://github.com/NandhaKishorM/laya) `laya-multilingual` on the
Turkish banking split of [Multi3NLU++](https://huggingface.co/datasets/uoe-nlp/multi3-nlu)
(CC-BY 4.0), using Ray Train for distributed training and Ray Data for distributed evaluation.

## Task framing

Multi3NLU++ is **multi-label**: each message carries 1–6 of 48 compositional intent tags
(`card`, `lost_stolen`, `request_info`, …). Each tag becomes one Laya `noul` (yes/no)
question, so one forward pass returns a probability per tag. Tag wordings are in `intents.py`.

| split | folds | messages |
|-------|-------|----------|
| train | 0–15  | 1,585    |
| test  | 16–19 | 398      |

Laya trains one item per labelled question: all 48 tags × 1,585 messages = **76,080 items**.
`--neg 8` keeps each message's positive tags plus 8 random negatives: **16,417 items** (~4.6× cheaper).

## Files

| file | what it does |
|------|--------------|
| `intents.py` | the 48 tags phrased as yes/no questions |
| `prepare_data.py` | Multi3NLU++ JSON folds → Laya JSONL (`state`, `questions`, `expected`) |
| `ray_train.py` | Ray Train `TorchTrainer` (DDP) fine-tune using Laya's own loss, calibration and checkpoint format |
| `evaluate.py` | Ray Data batch inference + micro/macro F1, top-1, exact match, per-tag F1 |

## Run

```bash
pip install laya "ray[train,data]"
# download turkish/banking/fold0..19.json from uoe-nlp/multi3-nlu into DATA_DIR
python prepare_data.py DATA_DIR data_full                 # test.jsonl labels all 48 tags
python prepare_data.py DATA_DIR data_neg8 --neg 8         # cheaper training set
laya-train --data data_neg8/train.jsonl --base multilingual --dry-run   # sanity check

# zero-shot baseline
python evaluate.py data_full/test.jsonl --checkpoint multilingual --workers 4 --gpus-per-worker 1

# distributed fine-tune: 4 GPUs (one node or several; use a shared --storage on multi-node)
python ray_train.py --data data_neg8/train.jsonl --base multilingual --out ckpt \
    --workers 4 --gpu --epochs 4 --loss soft-ce
python evaluate.py data_full/test.jsonl --checkpoint ./ckpt --workers 4 --gpus-per-worker 1
```

Useful flags: `--freeze-encoder` (head only, fits small CPUs/GPUs, gains much less),
`--max-messages N` (subsample for smoke tests), `--loss rlcd|soft-ce`.

## Why a custom Ray loop

`laya.train.finetune` / `laya-train` are single-device. `ray_train.py` reuses Laya's building
blocks (`items_from_rows`, `encode_item`, `soft_ce_loss` / `rlcd_loss`, `calibration_records`,
`fit_temperature_map`, `save_checkpoint`) and adds: rank sharding, DDP via
`ray.train.torch.prepare_model`, `no_sync` on gradient-accumulation steps, per-epoch checkpoints
reported to Ray, and temperature calibration on a slice held out before training (rank 0).
The output loads with `laya.load(path)`. It does not reproduce `laya-train`'s before/after report
or abstention-threshold fitting — use `evaluate.py` for held-out metrics.

## Results so far (CPU sandbox, 2 vCPU / 7 GB)

Zero-shot `laya-multilingual`, 100 test messages (fixed seed): see `results/`.

| run (100 test messages) | micro-F1 @0.5 | macro-F1 @0.5 | top-1 hit | tags predicted / gold |
|---|---|---|---|---|
| always predict the most common tag | — | — | 0.26 | — |
| zero-shot `laya-multilingual` | 0.186 | 0.187 | 0.21 | 4.7 / 2.29 |
| Ray smoke fine-tune (120 msgs, head only, 2 epochs, CPU) | 0.194 | 0.168 | 0.24 | 3.6 / 2.29 |

The smoke run proves the pipeline (Ray Train → calibrated Laya checkpoint → Ray Data eval), not
accuracy: it saw 8% of the training data with the encoder frozen. A real result needs a full
fine-tune (`data_neg8` or `data_full`, encoder unfrozen) on GPUs. Two Ray workers ran one DDP epoch
here before the 7 GB sandbox ran out of memory (each worker holds ~2.6 GB); use ≥16 GB per worker.
