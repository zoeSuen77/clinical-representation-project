# English presentation script

**Group 8 — Yuhan Sun, Zeyu Pan, Jiayu Liao**

Six slides. Planned delivery: 4 minutes 40 seconds, with 20 seconds of buffer. Spoken text: approximately 450 words. Rehearse at about 120–130 words/minute.

## Slide 1: Background and Motivation (45 seconds)

Our project asks whether pretrained text representations preserve useful information about negation and uncertainty. Consider three sentences: the patient has pneumonia, has no pneumonia, and may have pneumonia. Their vocabulary is almost identical, but their meanings differ. We use frozen language models to produce vectors, then train simple classifiers on those vectors. This connects the application to Project P and the standard machine learning methods covered in class.

## Slide 2: Problem Setup and Research Questions (45 seconds)

Our broad questions concern semantic information, sample efficiency, and terminology generalization. We checked the actual BioScope annotations before defining the pilot. They mark negation and speculation cues and their scopes, rather than disease-specific assertion labels. Our current classes are no cue, negation only, and speculation only. We exclude sentences containing both types. Therefore, this experiment tests cue presence in biomedical abstracts. Target-specific clinical assertion remains an extension.

## Slide 3: Proposed Method (55 seconds)

We parsed eleven thousand eight hundred seventy-one sentences from the official XML. After removing mixed-cue sentences and duplicates, eleven thousand five hundred fifty-nine remain. We separate document groups before selecting a pilot with fifteen hundred training sentences, four hundred validation sentences, and six hundred test sentences. We compare TF-IDF with BERT, BioBERT, and PubMedBERT. All encoders stay frozen. We fit logistic regression, and also a TF-IDF linear SVM. Hyperparameters use validation data only. The different methods receive identical labeled training examples.

## Slide 4: Current Progress (55 seconds)

We have completed all five methods at four balanced training budgets and the full pilot pool, using three sampling seeds. The code saves model revisions, predictions, metrics, and learning curves. At one hundred training examples per class, BioBERT + LR reaches 67.8 percent mean macro-F1. With the full training pool, TF-IDF + SVM reaches 84.3 percent. The table reports the complete comparison. These are preliminary results on one fixed test set. The full pool also has different class proportions, so the change is not solely a sample-size effect.

## Slide 5: Challenges (45 seconds)

There are several limits to the interpretation. A model may identify cue words without resolving their scope. The majority class has no annotated cue, so we report macro-F1 and per-class recall. Our low-data budgets count training labels, while every run also uses four hundred validation labels. The standard deviations measure training-subset variation, not uncertainty across independent test sets. Finally, biomedical pretraining may already include some source abstracts, and disease-level labels are still missing.

## Slide 6: Future Plans (35 seconds)

Next, we will expand to the uncapped corpus and analyze errors by cue family. We will consider two binary labels so that mixed-cue sentences can remain in the task. Our small authored sentence pairs provide qualitative diagnostics, not benchmark results. With reliable target annotations, we can then test span pooling and unseen disease generalization. Our aim is to establish when pretrained representations help, and to describe the conditions where simpler features remain competitive.

## Questions to prepare for

- Why not classify Affirmed/Negated/Uncertain? BioScope lacks disease-level assertion labels. The implemented labels follow the actual annotation schema.
- Does higher cosine similarity show misunderstanding? No. Similarity measures overall vector geometry; a classifier may still read out a useful distinction.
- Is this a 10-label-per-class experiment overall? It uses 10 training labels per class plus a fixed 400-label validation set. This caveat is explicit.
- Are the full-pool SDs evidence of certainty? No. Those fits reuse the same examples with deterministic solvers. Independent group splits remain future work.
- Was terminology generalization tested? Only authored qualitative examples. There is no validated disease-held-out benchmark result yet.

Sources: see `references.md`; all numeric results come from `results/metrics/pilot/summary.csv` and the data/integrity audit JSON files.
