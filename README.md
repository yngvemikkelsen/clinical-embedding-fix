# clinical-embedding-fix

[![License: CC BY 4.0](https://img.shields.io/badge/License-CC%20BY%204.0-lightgrey.svg)](https://creativecommons.org/licenses/by/4.0/)
[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)

Post-hoc corrections for degenerate clinical text embeddings. No retraining required.

Many transformer models produce near-degenerate embedding geometry on clinical text — high pairwise cosine similarity, low effective dimensionality — which degrades retrieval quality in RAG pipelines. This toolkit diagnoses the problem and fixes it with a single function call.

## Installation

```bash
pip install clinical-embedding-fix
```

Or from source:

```bash
git clone https://github.com/yngvemikkelsen/clinical-embedding-fix.git
cd clinical-embedding-fix
pip install -e .
```

## Quick Start

### Diagnose

```python
from clinical_embedding_fix import embedding_report

report = embedding_report(doc_embeddings, name="BioBERT clinical docs")
# === Embedding Report: BioBERT clinical docs ===
#   Samples: 500, Dim: 768
#   Participation ratio: 8.3
#   Avg cosine similarity: 0.9580
#   Anisotropy (SVD): 0.4721
#   Whitening recommended: YES
```

### Fix

```python
from clinical_embedding_fix import corpus_only_whitening

fixed_docs, fixed_queries = corpus_only_whitening(doc_embeddings, query_embeddings)

# Use fixed embeddings for retrieval
scores = fixed_queries @ fixed_docs.T
```

### Evaluate layers

```python
from clinical_embedding_fix import find_best_layer

best_layer, best_mrr, profile = find_best_layer(query_layers, doc_layers)
print(f"Best: layer {best_layer} (MRR={best_mrr:.3f})")
print(f"Final: layer {max(profile)} (MRR={profile[max(profile)]:.3f})")
```

## What's included

### Whitening (`whitening.py`)

| Function | Description | When to use |
|----------|-------------|-------------|
| `corpus_only_whitening()` | ZCA fitted on documents only | **Deployment** — no access to queries at index time |
| `transductive_whitening()` | ZCA fitted on docs + queries | Diagnostic upper bound |
| `zca_whitening_matrix()` | Returns the raw transform matrix | Custom pipelines |

### Diagnostics (`diagnostics.py`)

| Function | Description | Healthy range |
|----------|-------------|---------------|
| `participation_ratio()` | Effective dimensionality | >20 |
| `avg_cosine_similarity()` | Mean pairwise cosine | <0.7 |
| `anisotropy_svd()` | Variance in top singular direction | <0.1 |
| `embedding_report()` | All metrics + whitening recommendation | — |

### Layer Selection (`layer_select.py`)

| Function | Description |
|----------|-------------|
| `find_best_layer()` | Identify optimal layer by MRR@k |
| `layer_mrr_profile()` | MRR@k at every layer |

## Empirical basis

Based on experiments across 11 model configurations, 3 clinical corpora, and 2 query formats (~1,400 conditions):

- **Corpus-only whitening** improved MRR@10 on all 16 heterogeneous clinical text conditions (MTSamples, PMC-Patients) and was negative on all 8 structurally uniform conditions (Synthetic)
- **Transductive whitening** improved MRR@10 on all 24 conditions, with gains of +0.16 to +0.27 on degraded models
- **Participation ratio** correlates with MRR@10 at ρ = 0.736 (p = 0.010); excluding EOS-pooled models: ρ = 0.964 (p < 0.001)
- **All models** exhibit a U-shaped MRR curve across layers with mid-layer collapse

For full details see:

> Mikkelsen Y. Context or Tuning? Layer-Level Analysis of Embedding Degradation in Clinical Document Retrieval. Submitted to JAMIA. 2026.

> Mikkelsen Y. Context Matters More Than Model Choice: A Multi-Corpus Benchmark of Embedding Models for Clinical RAG. JMIR Preprints. 2026. doi: [10.2196/preprints.94241](https://doi.org/10.2196/preprints.94241)

## When to use (and when not to)

**Use whitening when:**
- Your embedding model shows high avg cosine (>0.7) or low participation ratio (<20)
- You're using a domain model not fine-tuned for retrieval (BioBERT, ClinicalBERT)
- You're using an LLM encoder (E5-Mistral, Phi-3) on clinical text
- Your corpus is heterogeneous clinical text (mixed specialties, note types)

**Don't use whitening when:**
- Your model already has healthy geometry (BGE, GTE, Nomic — check with `embedding_report`)
- Your corpus is structurally uniform (e.g., synthetically generated from templates)
- You have very few documents (<50) — covariance estimate will be unstable

## Running tests

```bash
pip install -e ".[dev]"
pytest tests/ -v
```

## License

CC BY 4.0. See [LICENSE](LICENSE) for details.
