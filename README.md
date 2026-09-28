# Proactive Creation of Chinese Neologisms — Code Release

Release package for the paper **"You Can Create Words Rather Than Just Detect Them: Towards A New Paradigm of Chinese Neologisms"** (Hansi Wang, Yue Wang, Qiliang Liang, Yang Liu).

The pipeline constructs disyllabic Chinese word candidates from a morpheme–word-formation space, discards morphologically invalid combinations, verifies the survivors semantically, and enriches them with a definition and five context sentences. This repository contains the **code and documentation**; the **candidate sets** are hosted separately on Hugging Face.

## Contents

```
.
├── code/
│   ├── stage1_kg_retrieval/         # SimKGC link prediction over the morpheme/formation graph
│   ├── stage2_morphological/        # rule-based morphological discarding
│   ├── stage3_filtering/            # discriminative (Qwen3-Embedding + MLP) filtering
│   ├── stage3_llm_verification/     # closed-source LLM verification (majority vote >= 2)
│   └── enrichment/                  # definition and example-sentence generation
├── evaluation/
│   ├── perplexity/                  # PPL computation for MLM and causal models
│   ├── human_agreement/             # Fleiss' kappa and bootstrap analysis
│   └── corpus_attestation/          # corpus retrieval + LLM-based semantic verification
├── prompts/prompts.md               # prompt templates used at each LLM stage
├── docs/
│   ├── DATA.md                      # data card: files, fields, provenance
│   └── REPRODUCIBILITY.md           # what is included, what is not, what can be re-checked
├── LICENSE                          # MIT (code)
└── requirements.txt
```

## Data

The data release (CC-BY-4.0) is hosted on Hugging Face:

```
https://huggingface.co/datasets/COOLPKU/Chinese_neologism_creation
```

| File | Records | Content |
|---|---|---|
| `candidates_pos.jsonl.gz` | 18,662 | accepted candidates: morphemes, senses, word-formation type, retrieval score, generated definition, five context sentences |
| `candidates_neg.jsonl.gz` | 590,011 | rejected candidates (same schema) |
| `exist_words.jsonl.gz` | 42,844 | established disyllabic words used as the upper reference set |

See `docs/DATA.md` for field-level documentation.

## Installation

```bash
pip install -r requirements.txt
```

The closed-source stages call the official OpenAI / Anthropic / Google APIs; supply your own credentials through the environment variables expected by `api_based_process/api_check.py`.

## Reproducibility status

Included in the release:

- the three candidate sets above, with definitions, generated contexts and formation metadata;
- the full pipeline code (KG retrieval, morphological discarding, discriminative filtering, LLM verification, enrichment) and the training configurations.

Runnable or checkable from the release alone:

- **Model perplexity** — the four evaluation models are public; the sentences they score are part of the released sets, so the PPL table can be recomputed with `evaluation/perplexity/`.
- **Stage 2 (morphological discarding)** — deterministic rules; re-running it requires the KG-retrieval pool, which can be regenerated with `code/stage1_kg_retrieval/` once COOL is available.

Not included:

- **COOL / MiCLS**, the morpheme and word-formation resources that seed the graph (not redistributed here);
- **model checkpoints** (SimKGC and the discriminative filter); training code and configurations are included so both can be retrained;
- the intermediate decisions (per-stage labels, LLM verification verdicts) and the raw human annotations and corpus-attestation records used for the paper's evaluation.

Consequently, the full pipeline cannot be re-executed end-to-end from this package alone, and the evaluation tables are documented rather than re-derivable here. `docs/REPRODUCIBILITY.md` states exactly what each release item does and does not support.

## Links

- Code repository: https://github.com/COOLPKU/Chinese_neologism_creation
- Data release: https://huggingface.co/datasets/COOLPKU/Chinese_neologism_creation

## License

- Code (this repository): **MIT** — see `LICENSE`.
- Data (Hugging Face release): **CC-BY-4.0**.

## Citation

```bibtex
@inproceedings{wang-etal-2026-proactive,
  title     = {You Can Create Words Rather Than Just Detect Them: Towards A New Paradigm of Chinese Neologisms},
  author    = {Wang, Hansi and Wang, Yue and Liang, Qiliang and Liu, Yang},
  booktitle = {Proceedings of the Asia-Pacific Chapter of the Association for Computational Linguistics},
  year      = {2026},
  note      = {To appear}
}
```
