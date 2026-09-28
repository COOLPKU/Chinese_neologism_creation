# Reproducibility notes

## 1. What the release contains

| Component | Status |
|---|---|
| Accepted candidate set (18,662) with morphemes, senses, word-formation type, retrieval score, definition, five generated sentences | included |
| Rejected candidate set (590,011) | included |
| Established-word reference set (42,844) | included |
| Pipeline code: KG retrieval, morphological discarding, discriminative filtering, LLM verification, enrichment | included |
| Training configurations (DeepSpeed) and prompt templates | included |
| Evaluation code: PPL, Fleiss' κ and bootstrap, corpus-attestation analysis | included |

## 2. What the release does not contain

- **COOL / MiCLS** — the morpheme inventory, sense categories and word-formation
  annotations that seed the knowledge graph. They are project resources and are not
  redistributed, so the graph cannot be rebuilt from scratch here.
- **Model checkpoints** — neither SimKGC (Stage 1) nor the discriminative filter
  (Stage 3). Training code and configurations are included, so both can be retrained once
  the underlying resource is available.
- **Intermediate decisions and evaluation records** — per-stage labels, LLM verification
  verdicts, the raw annotations of the three experts, and the corpus-attestation records.

## 3. What can and cannot be re-derived from the release

| Paper item | Re-derivable? | Notes |
|---|---|---|
| Model perplexity table (POS 7.20 / NEG 8.32 / EXIST 5.62 average) | **Yes** | the four evaluation models (BERT-Base-Chinese, ModernBERT-large, Llama-3.2-3B, Qwen3-1.7B) are public and the scored sentences are in the released sets; see `evaluation/perplexity/` |
| Morphological-discarding behaviour | partly | deterministic rules in `code/stage2_morphological/`; the KG-retrieval pool they act on must first be regenerated |
| Candidate funnel (802,108 → 73,381 → 46,998 → 18,662) | No | intermediate pools are not released; the final stage count (18,662) is |
| Human judgment rates and Fleiss' κ | No | raw annotations are not released |
| Bootstrap confidence intervals | No | computed from the unreleased annotations |
| Corpus-attestation validity ratios | No | sampled contexts and their verification verdicts are not released |
| Zero-shot vs. attested analysis | No | requires the unreleased attestation records |

## 4. Known points of non-determinism

1. **Proprietary models.** Definitions, context sentences and verification verdicts were
   produced by hosted models (Claude, GPT, Gemini families) whose behaviour can change
   between versions and snapshots. Prompts and the single-call setting are fixed, but
   sampling seeds were not recorded, so re-running the API stages will not reproduce the
   released strings exactly.
2. **Model lists.** The model sets used in the paper are: Stage 1 definitions —
   Claude-3.7-Sonnet; Stage 1 sentences — Gemini-2.0-flash, Gemini-2.5-flash, GPT-4o-mini,
   GPT-4.1-mini, GPT-5-mini; Stage 3 verification — GPT-4o, GPT-5-mini, Gemini-2.5-Flash.
   The released scripts take model names as command-line arguments, and some of their
   docstring examples show different sets; the paper's lists are authoritative.
3. **Sampling.** The human-evaluation sample (200 POS + 200 NEG) and the 10,000 contexts
   sampled per corpus-attestation set were drawn from the full pools; those pools and the
   sample indices are not part of this release.
4. **Environment paths.** The scripts retain the absolute paths and directory layout of the original working environment; adapt the input/output paths to your own setup before running them.
5. **Hardware / library versions.** `requirements.txt` pins the versions used for training
   and inference of the two learned components; results may drift marginally under other
   versions.

## 5. Licensing

Code: MIT. Data: CC-BY-4.0. COOL/MiCLS-derived resources are **not** redistributed; any
reuse of those resources is subject to their own terms.
