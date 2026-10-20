# Verified execution status

Prepared October 20, 2026 for the CSI 436/536 midterm.

**Group 8 — Yuhan Sun, Zeyu Pan, Jiayu Liao**

## Executed

- Downloaded official BioScope abstracts XML, SHA-256
  `ea275d3faf6659b888cf0c04b259e2088fe1e46062aa4d19247ce8cf822e423d`.
- Parsed 11,871 sentences, 1,273 source documents. Removed 295 both-cue
  sentences and 17 redundant duplicates; retained 11,559 sentences.
- Built a fixed group-separated pilot: 1,500 train, 400 validation, 600 test.
- Ran TF-IDF + LR, TF-IDF + linear SVM, BERT + LR, BioBERT + LR,
  and PubMedBERT + LR at 10/25/50/100 examples per class plus full pilot pool.
- Three training-sampling seeds per setting, yielding 75 evaluations and
  225 hyperparameter candidate fits. No convergence warnings were recorded.
- Extracted 768-dimensional frozen embeddings for all 2,500 pilot sentences
  from all three encoders on CPU. BERT and BioBERT each truncated 2 sentences
  at 192 tokens; PubMedBERT truncated 1. Model commits and cache fingerprints
  are saved with results.
- Ran ten synthetic demonstration pairs using all three encoders and TF-IDF.
  They are separate from benchmark data and are not scored as clinical accuracy.
- Four unit tests passed. Result audit verified all 75 runs, including common
  training examples, test IDs, validation C selection, and recomputed metrics.
- Generated raw metrics, summary CSV, learning curves and confusion matrices.
- Compiled the LaTeX report to exactly one page; also compiled successfully
  in the desktop LaTeX editor. Generated a six-slide editable PPTX and a
  450-word English script with a 4:40 timing plan.

## Selected measured results

Test macro-F1 in percent. First column: mean ± sample SD across the three
training subsets. Full-pool repetitions are deterministic.

| Method | 100/class (300 train sentences) | Full pilot pool (1,500 train) |
|---|---:|---:|
| TF-IDF + LR | 59.1 ± 0.8 | 81.7 |
| TF-IDF + SVM | 59.0 ± 0.8 | 84.3 |
| BERT + LR | 64.4 ± 1.3 | 77.7 |
| BioBERT + LR | 67.8 ± 1.1 | 80.4 |
| PubMedBERT + LR | 65.8 ± 1.7 | 82.6 |

The first comparison uses balanced training subsets. The full pilot pool has
natural imbalance, so the differences between these columns are not caused by
sample size alone. Both use the same 400 labeled validation examples and fixed
600-sentence test. These are training-label budgets, not total annotation costs.
The experiment supports a preliminary statement that relative performance
depends on the training setting, not a universal biomedical-model advantage.

## Implemented but not benchmarked

- Optional target-span pooling (unit-tested), because BioScope lacks disease
  target labels suitable for the proposed assertion task.
- Optional linear SVM on embedding features.
- Configuration for uncapped corpus experiments.

## Remaining limitations

- No disease-level Affirmed/Negated/Uncertain benchmark or held-out-disease
  result. The measured task is sentence-level annotated cue presence.
- One fixed group split. Sampling SD is not a confidence interval over test
  populations. No formal significance claim is made.
- Article and exact-duplicate leakage checked; near-duplicate templates need
  additional review. Possible overlap with model pretraining text is unresolved.
- Mixed-cue sentences excluded, limiting coverage. Biomedical abstracts and
  patient clinical records are different domains.
- Synthetic pairs are authored diagnostics, not clinician-validated labels.
- Deck checked by rendering and package/layout/import checks, not by opening
  Microsoft PowerPoint. Report visually checked after final compilation.

All reported results can be regenerated using the commands in `README.md`.
Raw article text and model weights are local caches and are excluded from
the portable handoff archive.
