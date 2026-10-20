# Minimal-pair diagnostics

SYNTHETIC DEMONSTRATION ONLY. Authored examples, not clinician-validated benchmark labels. Never included in training or hyperparameter selection.

All classifiers were fitted on the full pilot pool (seed 13). The examples were never used to train or tune models. Their clinical wording differs from the biomedical abstract training domain.

| Method | Pair | Cosine | Prediction A | Prediction B |
|---|---|---:|---|---|
| tfidf | negation_1 | 0.8466 | no_cue | negation_only |
| tfidf | uncertainty_1 | 0.4613 | no_cue | speculation_only |
| tfidf | scope_1 | 1.0000 | negation_only | negation_only |
| tfidf | scope_2 | 0.4747 | no_cue | negation_only |
| bert | negation_1 | 0.9742 | no_cue | no_cue |
| bert | uncertainty_1 | 0.9502 | no_cue | speculation_only |
| bert | scope_1 | 0.9983 | negation_only | negation_only |
| bert | scope_2 | 0.9472 | negation_only | negation_only |
| biobert | negation_1 | 0.9779 | no_cue | negation_only |
| biobert | uncertainty_1 | 0.9821 | no_cue | speculation_only |
| biobert | scope_1 | 0.9981 | negation_only | negation_only |
| biobert | scope_2 | 0.9801 | no_cue | negation_only |
| pubmedbert | negation_1 | 0.9921 | no_cue | negation_only |
| pubmedbert | uncertainty_1 | 0.9887 | no_cue | speculation_only |
| pubmedbert | scope_1 | 0.9987 | negation_only | negation_only |
| pubmedbert | scope_2 | 0.9915 | no_cue | negation_only |

The pneumonia/no-pneumonia pair demonstrates that a high cosine can coexist with different predicted cue labels. Geometry alone is not an assertion test.

In `scope_1`, pneumonia and diabetes exchange roles while both sentences retain a negation cue. Equal sentence-level predictions are compatible with the trained task and cannot tell us whether the model resolved the target's status. In `scope_2`, an extra negation changes the construction; a cue-presence classifier cannot reliably adjudicate the resulting target meaning.

Abbreviation and paraphrase pairs are included in the CSV/JSON. Agreement on a few authored pairs is a descriptive observation, not evidence of generalization to an independent medical benchmark. No synthetic benchmark accuracy is reported.
