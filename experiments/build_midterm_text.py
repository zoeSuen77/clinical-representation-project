"""Write report source and speaking script from completed, verified pilot results."""
import json
from pathlib import Path
import pandas as pd

METHODS = [("tfidf","lr","TF-IDF + LR"),("tfidf","svm","TF-IDF + SVM"),
           ("bert","lr","BERT + LR"),("biobert","lr","BioBERT + LR"),
           ("pubmedbert","lr","PubMedBERT + LR")]


def main():
    team = json.loads(Path("configs/team.json").read_text())
    members = ", ".join(team["members"])
    base = Path("results/metrics/pilot")
    audit = json.loads((base / "integrity_audit.json").read_text())
    if not audit["passed"] or audit["evaluations_checked"] != 75:
        raise ValueError("The requested five-method pilot is not fully verified")
    df = pd.read_csv(base / "summary.csv", dtype={"budget_per_class":str})

    def metric(f, c, budget):
        row = df[(df.features == f) & (df.classifier == c) &
                 (df.budget_per_class == str(budget))].iloc[0]
        return float(row.macro_f1_mean), float(row.macro_f1_std)

    cells = []
    for f,c,label in METHODS:
        m,s = metric(f,c,100)
        full,_ = metric(f,c,"full")
        cells.append(f"{label} & ${100*m:.1f}\\pm{100*s:.1f}$ & {100*full:.1f} \\\\")
    table_rows = "\n".join(cells)
    best100 = max(METHODS, key=lambda t:metric(t[0],t[1],100)[0])
    bestfull = max(METHODS, key=lambda t:metric(t[0],t[1],"full")[0])
    b100 = metric(best100[0],best100[1],100)[0]*100
    bfull = metric(bestfull[0],bestfull[1],"full")[0]*100
    report = r"""\documentclass[10pt,letterpaper]{article}
\usepackage[margin=0.67in]{geometry}
\usepackage{booktabs,amsmath,microtype}
\usepackage[hidelinks]{hyperref}
\usepackage{titlesec}
\titleformat{\section}{\normalsize\bfseries}{}{0pt}{}
\titlespacing*{\section}{0pt}{6pt}{3pt}
\setlength{\parindent}{0pt}
\setlength{\parskip}{3pt}
\pagestyle{empty}
\begin{document}
\begin{center}
{\large\bfseries Probing Clinical Meaning in Pretrained Language Representations:\\
Negation, Uncertainty, and Terminology Generalization}\\[3pt]
{\small CSI 436/536 Project P \quad Midterm Report \quad October 20, 2026}\\
{\small Group ID: \textbf{[GROUP ID]} \quad Members: \textbf{[MEMBER NAMES]}}
\end{center}
\vspace{-6pt}
\section{Problem Statement}
We investigate whether frozen pretrained representations preserve useful information
about negation and uncertainty, improve learning with limited labeled training data,
and eventually support terminology generalization. The implemented pilot predicts
\emph{sentence-level annotated cue presence} in BioScope abstracts [1]:
\texttt{no\_cue}, \texttt{negation\_only}, or \texttt{speculation\_only}.
These labels do not establish a particular patient's disease status. Target-specific
assertion and unseen-disease evaluation require additional entity-level labels.
\section{Data and Proposed Method}
We downloaded the official BioScope XML and parsed 11,871 sentences from 1,273 abstracts,
including titles. Removing 295 mixed-cue sentences and 17 redundant duplicates leaves
11,559 sentences. Documents connected by duplicate sentences stay in one partition.
Five fixed stratified group folds allocate three folds to training, one to validation,
and one to test. The pilot samples 1,500/400/600 sentences within those partitions,
approximately preserving class proportions. The test contains 439 no-cue, 68 negation,
and 93 speculation sentences.

We compare word unigram/bigram TF-IDF with frozen BERT-base-cased [2], BioBERT [3],
and PubMedBERT [4] features. The encoders use final-layer mean pooling,
excluding padding and special tokens, with a 192-token limit. We fit feature
preprocessing only on each labeled training subset. Class-balanced logistic regression
(LR), plus a TF-IDF linear SVM, select $C\in\{0.1,1,10\}$ on validation macro-F1.
\section{Current Progress and Verified Pilot Results}
The Python pipeline, embedding caches, pinned model revisions, split manifests,
prediction files, and automatic figures are implemented. All five methods completed
10/25/50/100 examples per class and the full pilot training pool with three sampling
seeds: 75 evaluations in total. Integrity checks confirm shared training IDs and
recompute metrics from saved predictions.
\begin{center}
\small
\begin{tabular}{lcc}
\toprule
Method & 100/class: macro-F1 (\%) & Full pool: macro-F1 (\%)\\
\midrule
TABLE_ROWS
\bottomrule
\end{tabular}
\end{center}
\vspace{-6pt}
{\small Values at 100/class are mean $\pm$ sample SD over three training subsets on
one fixed test set. Full pool means 1,500 training sentences with natural imbalance;
its repetitions are deterministic. All runs also use 400 labeled validation sentences,
so the budgets count training labels only.}

The pilot suggests that the relative benefit depends on the training setting.
The full-pool comparison also changes class composition, so its difference from
balanced-budget runs is not a pure sample-size effect. These preliminary results
measure cue detection and do not demonstrate clinical understanding.
\section{Challenges and Next Steps}
Explicit cue words may permit lexical shortcuts. Mixed cues, class imbalance,
scope ambiguity, and possible overlap with biomedical pretraining text limit interpretation.
Exact duplicate removal does not guarantee removal of all near-duplicate templates.
Next we will run the uncapped corpus, retain mixed labels in a multilabel task,
and analyze errors by cue family. Ten authored minimal pairs support qualitative
diagnostics only. Disease-held-out tests and target pooling need suitable annotated
targets. We will preserve document and template separation for those extensions.
\vspace{3pt}
\hrule
\vspace{3pt}
{\fontsize{8}{9}\selectfont
[1] Vincze et al., BioScope, \emph{BMC Bioinformatics}, 2008,
\href{https://doi.org/10.1186/1471-2105-9-S11-S9}{doi:10.1186/1471-2105-9-S11-S9}.
[2] Devlin et al., BERT, NAACL 2019.
[3] Lee et al., BioBERT, \emph{Bioinformatics}, 2020.
[4] Gu et al., Domain-Specific Language Model Pretraining, \emph{ACM TCH}, 2021.
Code authored with Codex assistance; group review and course-policy compliance remain
the authors' responsibility.}
\end{document}
"""
    report = (report.replace("TABLE_ROWS", table_rows)
              .replace("[GROUP ID]", str(team["group_id"]))
              .replace("[MEMBER NAMES]", members))
    Path("report/midterm_one_pager.tex").write_text(report)
    scripts = [
        ("Background and Motivation",45,
         "Our project asks whether pretrained text representations preserve useful information "
         "about negation and uncertainty. Consider three sentences: the patient has pneumonia, "
         "has no pneumonia, and may have pneumonia. Their vocabulary is almost identical, but "
         "their meanings differ. We use frozen language models to produce vectors, then train "
         "simple classifiers on those vectors. This connects the application to Project P and "
         "the standard machine learning methods covered in class."),
        ("Problem Setup and Research Questions",45,
         "Our broad questions concern semantic information, sample efficiency, and terminology "
         "generalization. We checked the actual BioScope annotations before defining the pilot. "
         "They mark negation and speculation cues and their scopes, rather than disease-specific "
         "assertion labels. Our current classes are no cue, negation only, and speculation only. "
         "We exclude sentences containing both types. Therefore, this experiment tests cue "
         "presence in biomedical abstracts. Target-specific clinical assertion remains an extension."),
        ("Proposed Method",55,
         "We parsed eleven thousand eight hundred seventy-one sentences from the official XML. "
         "After removing mixed-cue sentences and duplicates, eleven thousand five hundred "
         "fifty-nine remain. We separate document groups before selecting a pilot with fifteen "
         "hundred training sentences, four hundred validation sentences, and six hundred test "
         "sentences. We compare TF-IDF with BERT, BioBERT, and PubMedBERT. All encoders stay frozen. "
         "We fit logistic regression, and also a TF-IDF linear SVM. Hyperparameters use validation "
         "data only. The different methods receive identical labeled training examples."),
        ("Current Progress",55,
         f"We have completed all five methods at four balanced training budgets and the full "
         f"pilot pool, using three sampling seeds. The code saves model revisions, predictions, "
         f"metrics, and learning curves. At one hundred training examples per class, "
         f"{best100[2]} reaches {b100:.1f} percent mean macro-F1. With the full training pool, "
         f"{bestfull[2]} reaches {bfull:.1f} percent. The table reports the complete comparison. "
         "These are preliminary results on one fixed test set. The full pool also has different "
         "class proportions, so the change is not solely a sample-size effect."),
        ("Challenges",45,
         "There are several limits to the interpretation. A model may identify cue words without "
         "resolving their scope. The majority class has no annotated cue, so we report macro-F1 "
         "and per-class recall. Our low-data budgets count training labels, while every run also "
         "uses four hundred validation labels. The standard deviations measure training-subset "
         "variation, not uncertainty across independent test sets. Finally, biomedical pretraining "
         "may already include some source abstracts, and disease-level labels are still missing."),
        ("Future Plans",35,
         "Next, we will expand to the uncapped corpus and analyze errors by cue family. We will "
         "consider two binary labels so that mixed-cue sentences can remain in the task. Our "
         "small authored sentence pairs provide qualitative diagnostics, not benchmark results. "
         "With reliable target annotations, we can then test span pooling and unseen disease "
         "generalization. Our aim is to establish when pretrained representations help, and "
         "to describe the conditions where simpler features remain competitive.")
    ]
    words = sum(len(t.split()) for _,_,t in scripts)
    script = "# English presentation script\n\n"
    script += f"**Group {team['group_id']} — {members}**\n\n"
    script += (f"Six slides. Planned delivery: 4 minutes 40 seconds, with 20 seconds of buffer. "
               f"Spoken text: approximately {words} words. Rehearse at about 120–130 words/minute."
               "\n\n")
    for i,(title,seconds,text) in enumerate(scripts,1):
        script += f"## Slide {i}: {title} ({seconds} seconds)\n\n{text}\n\n"
    script += ("## Questions to prepare for\n\n"
               "- Why not classify Affirmed/Negated/Uncertain? BioScope lacks disease-level "
               "assertion labels. The implemented labels follow the actual annotation schema.\n"
               "- Does higher cosine similarity show misunderstanding? No. Similarity measures "
               "overall vector geometry; a classifier may still read out a useful distinction.\n"
               "- Is this a 10-label-per-class experiment overall? It uses 10 training labels "
               "per class plus a fixed 400-label validation set. This caveat is explicit.\n"
               "- Are the full-pool SDs evidence of certainty? No. Those fits reuse the same "
               "examples with deterministic solvers. Independent group splits remain future work.\n"
               "- Was terminology generalization tested? Only authored qualitative examples. "
               "There is no validated disease-held-out benchmark result yet.\n\n"
               "Sources: see `references.md`; all numeric results come from "
               "`results/metrics/pilot/summary.csv` and the data/integrity audit JSON files.\n")
    Path("slides/presentation_script.md").write_text(script)
    payload = {"scripts":[{"title":t,"seconds":s,"text":x} for t,s,x in scripts],
               "methods":[{"features":f,"classifier":c,"label":l,
                           "ten":metric(f,c,10),"hundred":metric(f,c,100),
                           "full":metric(f,c,"full")} for f,c,l in METHODS],
               "word_count":words}
    Path("build").mkdir(exist_ok=True)
    Path("build/midterm_content.json").write_text(json.dumps(payload,indent=2))
    print(f"Wrote report source and {words}-word speaking script.")


if __name__ == "__main__":
    main()
