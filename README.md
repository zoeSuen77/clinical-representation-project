# Clinical representation project

**Group 8 — Yuhan Sun, Zeyu Pan, Jiayu Liao**

CSI 436/536, Project P. Broad research title: **Probing Clinical Meaning in
Pretrained Language Representations: Negation, Uncertainty, and Terminology
Generalization**.

The implemented midterm task is **sentence-level BioScope cue-presence
classification**. The model predicts `no_cue`, `negation_only`, or
`speculation_only`. These labels describe annotated linguistic cues; `no_cue`
does not mean a patient has a disease. Disease-level assertion and terminology
generalization remain research extensions requiring additional labels.

## Reproduce

### Ready for AutoDL validation

Local preparation before full training is complete. See the
[stage report](docs/STAGE_REPORT.md) and [AutoDL runbook](docs/AUTODL_RUNBOOK.md).
A fresh dependency environment passed 22 tests and a 30-sentence synthetic
end-to-end check across all five methods. Final BioScope evaluations remain 0/65;
Linux/CUDA validation is pending the rented instance.

`python -m experiments.launch_final` only prints the plan. After remote checks,
`--device cuda --execute` runs the experiment; `--resume` audits completed feature
groups before skipping them and refits incomplete groups. Cache checksums,
device-specific keys, per-group logs, and a launcher file lock protect the run.
The runner rechecks the prepared-data fingerprints before final fitting.

For a local technical check after dependency setup:

```bash
python -m experiments.check_environment --device cpu
python -m experiments.smoke_models --device cpu --workdir build/new-smoke
```

The smoke workspace must be new. It uses authored sentences and never trains on
the final BioScope data. Add `--offline` when all pinned model files are cached.

### Day 1 final-protocol preparation

The final protocol and data preparation are ready; final model evaluations have
not been run. See [the fixed protocol](docs/FINAL_PROTOCOL.md) and
[the Day 1 completion record](docs/DAY1_COMPLETION.md).
Set up Python and install dependencies using the instructions below first.

```bash
python -m experiments.prepare_data --config configs/final_main.json
python -m experiments.preflight --config configs/final_main.json
python -m pytest -q
```

This split has 6,939 training, 400 validation, and 2,308 test sentences.
Its 65 planned evaluations use conservative near-duplicate document grouping,
four balanced budgets with three seeds, and one full-pool fit per method.
Preflight success does not mean training or test evaluation has completed.

For subsequent training, use `--config configs/final_main.json` with the
experiment runner, then run `python -m experiments.audit_results --config configs/final_main.json`
and `python -m src.visualization --name final_main`.
The audit rejects an incomplete matrix. Existing method results are protected;
use `--overwrite` only for an intentional rerun of the same protocol. Changes
to the dataset or protocol require a new experiment name. Optional methods
must also be listed in the config's `methods` field for a complete audit.

### Reproduce the original pilot

Use Python 3.12. All commands run from this directory. Standard Python:

```bash
python3.12 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements-lock.txt
python -m experiments.prepare_data --config configs/pilot.json
python -m pytest -q
# The repository includes pilot results; these flags explicitly replace them
# when intentionally reproducing the pilot in your own checkout.
python -m experiments.run_experiments --config configs/pilot.json --features tfidf --overwrite
python -m experiments.run_experiments --config configs/pilot.json --features bert biobert pubmedbert --overwrite
python -m src.visualization --name pilot
python -m experiments.minimal_pairs --features tfidf bert biobert pubmedbert
python -m experiments.audit_results
python -m experiments.build_midterm_text
```

With `uv`, replace environment setup by `uv venv --python 3.12 .venv` and
`uv pip install --python .venv/bin/python -r requirements-lock.txt`.
On macOS, PyTorch can use MPS. Pass `--device cpu` for CPU, or `--device cuda`
on a compatible GPU. Device-specific floating-point differences may produce
small deviations. Checkpoints total roughly 1.3 GB; cache stays under `data/cache`.
First-time dataset/model installation requires internet access.

Extract features without fitting a classifier:

```bash
python -m experiments.run_experiments --features bert --extract-only
```

The first download resolves a model to a commit SHA in `configs/models.json`.
All later runs use that revision. Cache keys include the ordered text and IDs,
model revision, library versions, length limit, and pooling method.
The current cache format also records device, GPU/runtime details and batch size,
and verifies file checksums. Earlier pilot feature caches are left intact and
will be recomputed once if the pilot is intentionally rerun with the new code.

For the uncapped corpus run, execute `experiments.prepare_data` and
`experiments.run_experiments` with `--config configs/full.json`, then
`python -m src.visualization --name full`. Add `--embedding-svm` to compare
Linear SVM on pretrained features. These are additional runs, not claimed in
midterm material unless they have actually completed.

## Protocol

1. Parse the official XML with `ElementTree`. Strip annotation markup from
   input, retain the original text, character offsets, document IDs, cue types
   and scopes. This includes titles and abstract-body sentences.
2. Count mixed negation/speculation sentences separately and exclude them from
   the three-class experiment. A later multilabel task could retain them.
3. Normalize case/whitespace to find duplicate sentences. Link documents
   sharing duplicates, remove redundant copies, and exclude inconsistent
   duplicate labels. Keep connected document groups in one partition.
4. Use five fixed stratified group folds: fold 0 test, fold 1 validation, folds
   2–4 train. Split seed is 1729. For the pilot, stratified sampling within
   each partition caps counts at 1500/400/600 using seed 2026. This preserves
   approximate natural class proportions while keeping article groups disjoint.
   Exact duplicates are checked; near-duplicate templates are not fully audited.
5. Draw nested **10/25/50/100 labeled sentences per class** using seeds
   13, 37, 73. The exact same example IDs go to every method. `full` means
   the full 1500-sentence pilot training pool and uses its natural imbalance.
   Do not interpret the jump to full as a pure sample-size change: class
   composition changes too. This full point is excluded from the balanced-budget
   learning-curve line. All settings additionally use the same 400 labeled
   validation sentences. Thus these are **training budgets**, not total
   annotation budgets. Full-pool repetitions are deterministic, so zero
   standard deviation there does not express uncertainty.
6. TF-IDF uses word unigrams/bigrams and fits vocabulary/IDF on the selected
   labeled training subset only. Encoders run in eval/inference mode with frozen
   weights, last-layer mean pooling excluding padding and special tokens,
   maximum length 192. Record truncation per model. Optional span pooling
   rejects missing/truncated targets and is tested, but not used on BioScope.
7. Standardize embedding dimensions using only the selected training subset.
   Use balanced class weights for LR and SVM. Tune C in `[0.1, 1, 10]` by
   validation macro-F1, with the first C winning ties. Never tune on test.
8. Save each run's predictions, metrics, training IDs, C search, and convergence
   warnings. Report mean and sample standard deviation across training subsets.
   One fixed test split is reused; these SDs are not confidence intervals and
   do not quantify uncertainty across independent test populations.

## Experiment coverage

- **A:** Implemented cue-presence classification supported by actual labels.
  Accuracy, macro precision/recall/F1, per-class metrics and confusion matrices.
- **B:** Learning curves for multiple training budgets and three seeds.
- **C:** Ten authored sentence pairs, explicitly marked synthetic demonstrations.
  Cosine similarity and predicted cue labels only. No benchmark accuracy or
  clinician-validated semantic labels. Never mixed into train/validation/test.
- **D:** Disease-held-out testing is **not run**. BioScope abstracts do not supply
  disease identities or target-specific assertion labels. A robust extension
  requires annotated entity spans/concepts, an entity-group split, and audit of
  document/template overlap. Sentence-level success cannot prove scope resolution.

## Outputs

- `results/metrics/pilot_data_audit.json`: raw source hash and verified counts.
- `data/processed/pilot_manifest.json`: source IDs and split assignments.
- `results/metrics/pilot/`: individual runs, predictions, feature metadata,
  `raw_metrics.csv`, `summary.csv`, and demonstration diagnostics.
- `results/figures/pilot/`: learning curves and confusion matrices.
- `slides/midterm_presentation.pptx`: six-slide English presentation.
- `slides/presentation_script.md`: timed English script with source notes.
- `report/midterm_one_pager.tex` and `.pdf`: one-page midterm report.
- `docs/project_guide_zh.html`: Chinese project explanation and presentation guide.
- `results/EXECUTION_STATUS.md`: exact execution status and limitations.

Team information is stored in `configs/team.json` and included in the report,
title slide, and speaking script. Slides are due before the October 20 class; the one-pager
is due on October 20. Course instructions require LaTeX PDF and Python code.

Rebuild the one-page PDF with `tectonic --outdir report report/midterm_one_pager.tex`
or an ordinary LaTeX distribution. The report source and speaking script are
generated from the verified summary by `experiments.build_midterm_text`.
The editable deck builder is `experiments/build_slides.mjs`; it requires
`@oai/artifact-tool` and the Codex presentation runtime. Provide
`PRESENTATION_SKILL_DIR`, `ARTIFACT_RUNTIME_PYTHON`, and `RUNTIME_NODE_MODULES`
for a different host. Use a fresh output path through `MIDTERM_PPTX`
when rebuilding an existing deck. This authoring runtime is not needed to
reproduce any ML result.

## Data and authorship

See `references.md` for primary sources, model identities, dataset access
limitations, pretraining overlap, and software attribution. Raw BioScope text
and model weights are not intended for Git distribution. Re-download from the
official source using the provided preparation command. The code was authored
with Codex assistance; each member should review it and comply with the
course's disclosure and collaboration rules.
