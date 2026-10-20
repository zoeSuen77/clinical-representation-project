# Sources and attribution

Access checked October 20, 2026. Implementation code in `src/`, `experiments/`,
and `tests/` was authored for this project with Codex assistance. It calls the
public APIs of the libraries listed in `requirements-lock.txt`; it does not copy
an existing project's experiment implementation. Group members must review and
understand the code and follow their course's AI-use policy.

## Dataset feasibility

| Resource | Actual annotations and access | Decision |
|---|---|---|
| [BioScope official page](https://rgai.inf.u-szeged.hu/node/105) | Negation/speculation cues and linguistic scopes. Abstract XML publicly downloadable for research. Clinical portion has separate licensing/source requirements. | Use abstracts, including annotated title sentences. Source XML checksum and exact parsing counts saved in data audit. |
| [BioScope full articles](https://rgai.inf.u-szeged.hu/node/105) | Same cue/scope schema, very small number of full articles. | Potential later genre-transfer test, not used in reported pilot. |
| [SemEval 2015 Task 14](https://physionet.org/content/semeval2015/2.0/) | Disorder spans and separate negation/uncertainty attributes. Credentialing, training and DUA required. | Not downloaded. Potential target-level extension after appropriate access. |
| [GENIA Event / BioScope comparison](https://pmc.ncbi.nlm.nih.gov/articles/PMC3239308/) | Biological events and linguistic scopes represent different targets. Event negation cannot be treated as patient disease assertion. | Requires a distinct event task and format investigation; not used. |
| [SFU Review negation/speculation corpus](https://www.sfu.ca/~mtaboada/download/downloadNegation.html) | Negation/speculation in reviews rather than biomedical documents. | Possible domain-shift resource, not a biomedical assertion benchmark. Not downloaded. |

The official BioScope page permits research use but does not establish a general
redistribution license for all underlying article text. Raw text and processed
sentence files are ignored by Git. Share code, manifests, hashes, and aggregate
metrics; collaborators should retrieve the source via the official download.
This is a course research pipeline, not a clinical decision tool.

## Papers and checkpoints

1. Vincze et al. (2008). *The BioScope corpus: biomedical texts annotated for
   uncertainty, negation and their scopes*. BMC Bioinformatics 9(Suppl 11):S9.
   https://doi.org/10.1186/1471-2105-9-S11-S9
2. Devlin et al. (2019). *BERT: Pre-training of Deep Bidirectional Transformers
   for Language Understanding*. https://aclanthology.org/N19-1423/
   Checkpoint: https://huggingface.co/google-bert/bert-base-cased
3. Lee et al. (2020). *BioBERT: a pre-trained biomedical language representation
   model for biomedical text mining*. https://doi.org/10.1093/bioinformatics/btz682
   Checkpoint: https://huggingface.co/dmis-lab/biobert-base-cased-v1.2
4. Gu et al. (2021). *Domain-Specific Language Model Pretraining for Biomedical
   Natural Language Processing*. https://doi.org/10.1145/3458754
   Checkpoint: https://huggingface.co/microsoft/BiomedNLP-BiomedBERT-base-uncased-abstract-fulltext
   This checkpoint was formerly named PubMedBERT (abstracts + full text).
5. Elhadad et al. (2015). *SemEval-2015 Task 14: Analysis of Clinical Text*.
   https://aclanthology.org/S15-2051/

Each downloaded model's immutable revision is recorded in `configs/models.json`
and in the feature metadata. BERT and BioBERT are both cased. PubMedBERT is
uncased and has a different tokenizer; model comparisons do not isolate the
causal effect of medical pretraining from all tokenizer/training differences.

BioScope abstracts may overlap biomedical pretraining corpora. Our document
split prevents supervised downstream leakage; it cannot establish that a
pretrained encoder never saw a test abstract during unsupervised pretraining.
