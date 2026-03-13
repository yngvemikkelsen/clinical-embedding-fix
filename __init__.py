"""
clinical-embedding-fix: Post-hoc corrections for degenerate clinical text embeddings.

A lightweight toolkit for diagnosing and fixing embedding geometry problems
in clinical retrieval-augmented generation (RAG) pipelines. No retraining required.

Based on empirical findings from:
    Mikkelsen Y. Context or Tuning? Layer-Level Analysis of Embedding Degradation
    in Clinical Document Retrieval. Submitted to JAMIA. 2026.
"""

__version__ = "0.1.0"

from .whitening import corpus_only_whitening, transductive_whitening, zca_whitening_matrix
from .diagnostics import participation_ratio, avg_cosine_similarity, anisotropy_svd, embedding_report
from .layer_select import find_best_layer, layer_mrr_profile
