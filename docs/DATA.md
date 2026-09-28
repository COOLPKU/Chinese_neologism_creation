# Data card

Hosted on Hugging Face: `https://huggingface.co/datasets/COOLPKU/Chinese_neologism_creation` — license **CC-BY-4.0**.

All files are gzip-compressed JSON Lines (one record per line, UTF-8).
Record counts are exact and match the paper.

## 1. `data/candidates_pos.jsonl.gz` — 18,662 records

The accepted candidates, i.e. the pipeline's output resource.

| Field | Meaning |
|---|---|
| `head_id` | first morpheme identifier (sense-indexed, e.g. `心02795_1_05_02`; the first character is the morpheme form) |
| `head_pos` | morpheme part of speech of the first morpheme |
| `head_sense` | sense gloss of the first morpheme |
| `tail_id`, `tail_pos`, `tail_sense` | same for the second morpheme |
| `relation` | word-formation type (one of the 16 types, e.g. 主谓, 定中, 状中) |
| `confidence` | relation-conditioned SimKGC retrieval score |
| `definition` | LLM-generated dictionary-style definition |
| `sentences` | five generated context sentences; `～` marks the position of the target word |

## 2. `data/candidates_neg.jsonl.gz` — 590,011 records

Candidates that passed KG retrieval but were rejected by a later stage; used as the
negative comparison set. Identical schema to `candidates_pos`.

## 3. `data/exist_words.jsonl.gz` — 42,844 records

Established disyllabic words used as the upper reference set.
Fields: `head_id`, `tail_id`, `relation`, `definition`, `sentences`.

## 4. Provenance

| Stage | Script |
|---|---|
| KG retrieval (SimKGC) | `code/stage1_kg_retrieval/` |
| Morphological discarding | `code/stage2_morphological/` |
| Discriminative filtering | `code/stage3_filtering/` |
| LLM verification (majority vote ≥ 2) | `code/stage3_llm_verification/api_check.py` |
| Definition / sentence generation | `code/enrichment/` |

Candidate funnel reported in the paper: `1.44 × 10^8` enumerated triplets → 802,108
retrieved by KG link prediction → 73,381 survivors of the morphological whitelist →
46,998 after surface-form de-duplication → 18,662 accepted.

## 5. Not included

- **COOL / MiCLS**, the morpheme and word-formation resources that seed the graph.
- Model checkpoints (SimKGC; discriminative filter); training code is provided instead.
- Intermediate per-stage labels, LLM verification verdicts, raw human annotations and the
  corpus-attestation records used for the evaluation tables.
- The large retrieval-context dumps (word-in-context spans) produced during corpus
  attestation.

## 6. Notes on interpretation

- Definitions and context sentences are **generated**, not attested; the five sentences per
  candidate use `～` for the target slot.
- "Novel" follows the paper's operational definition — absent from four standard Chinese
  dictionaries — and does not mean "never used"; some candidates occur in corpora.

## 7. Citation

See the repository README.
