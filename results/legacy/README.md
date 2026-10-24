# Legacy results (legacy protocol, not comparable)

These two reports predate the frozen splits in `data/splits.json`. They follow the legacy protocol:

- they scored only 100 messages from the old test split, and that split had no no-intent rows;
- the majority row used a tag derived from the test labels themselves, and top-1 used the old denominator;
- metrics were computed at the fixed thresholds 0.3/0.5/0.7/0.9, and values were rounded during aggregation,
  not only for display.

They are kept for provenance only. They are **not comparable** to the full-test reports under `results/reports/` (from
the baselines onward), which score every message of the frozen test split (no-intent rows included), take the majority
tag and label prior from the matching training split, and are recomputed from committed logit dumps under
`results/dumps/`.

| file | what it is |
|---|---|
| `zeroshot_multilingual.json` | zero-shot `laya-multilingual` (pinned hub weights, no fine-tuning) on the 100 legacy test messages |
| `finetuned_smoke.json` | Ray smoke fine-tune: 120 training messages, head only (encoder frozen), 2 epochs, CPU; a pipeline check, not an accuracy result |

The bytes of both files are unchanged since they were first committed; they only moved here from `results/`.
