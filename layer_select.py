"""
Optimal layer selection for transformer-based clinical retrieval.

All tested models exhibit a U-shaped MRR@10 curve across layers, with
mid-layer collapse and variable final-layer recovery. For models with
poor final-layer recovery (BioBERT, ClinicalBERT, Phi-3-mini), selecting
an alternative layer can improve retrieval without retraining.

Key finding (Mikkelsen, 2026):
    Final-layer embeddings are consistently preferable for well-trained
    retrieval models (BGE, GTE, Nomic, BioLORD). Layer selection primarily
    benefits domain encoders and LLMs not fine-tuned for retrieval.
"""

import numpy as np
from typing import Dict, List, Optional, Tuple


def compute_mrr_at_k(
    query_embeddings: np.ndarray,
    doc_embeddings: np.ndarray,
    k: int = 10,
) -> float:
    """
    Compute MRR@k with 1-to-1 query-document mapping.

    Assumes query[i] should retrieve document[i] (identity ground truth).

    Parameters
    ----------
    query_embeddings : np.ndarray, shape (n, dim)
        Query embeddings (L2-normalized).
    doc_embeddings : np.ndarray, shape (n, dim)
        Document embeddings (L2-normalized).
    k : int, default 10
        Cutoff for reciprocal rank.

    Returns
    -------
    mrr : float
        Mean Reciprocal Rank @ k.
    """
    sim = query_embeddings @ doc_embeddings.T
    n = sim.shape[0]
    ranks = np.array([(sim[i] > sim[i, i]).sum() + 1 for i in range(n)], dtype=float)
    rr = np.where(ranks <= k, 1.0 / ranks, 0.0)
    return float(rr.mean())


def layer_mrr_profile(
    layer_query_embeddings: Dict[int, np.ndarray],
    layer_doc_embeddings: Dict[int, np.ndarray],
    k: int = 10,
) -> Dict[int, float]:
    """
    Compute MRR@k at every available layer.

    Parameters
    ----------
    layer_query_embeddings : dict of {layer_idx: np.ndarray}
        Query embeddings per layer (from model with output_hidden_states=True).
    layer_doc_embeddings : dict of {layer_idx: np.ndarray}
        Document embeddings per layer.
    k : int, default 10
        MRR cutoff.

    Returns
    -------
    profile : dict of {layer_idx: mrr_score}

    Examples
    --------
    >>> from clinical_embedding_fix import layer_mrr_profile
    >>> profile = layer_mrr_profile(q_layers, d_layers)
    >>> for layer, mrr in sorted(profile.items()):
    ...     print(f"Layer {layer}: MRR@10 = {mrr:.3f}")
    """
    common_layers = sorted(set(layer_query_embeddings.keys()) & set(layer_doc_embeddings.keys()))
    profile = {}
    for layer in common_layers:
        q = layer_query_embeddings[layer]
        d = layer_doc_embeddings[layer]
        profile[layer] = compute_mrr_at_k(q, d, k)
    return profile


def find_best_layer(
    layer_query_embeddings: Dict[int, np.ndarray],
    layer_doc_embeddings: Dict[int, np.ndarray],
    k: int = 10,
) -> Tuple[int, float, Dict[int, float]]:
    """
    Identify the layer with highest MRR@k.

    Parameters
    ----------
    layer_query_embeddings : dict of {layer_idx: np.ndarray}
        Query embeddings per layer.
    layer_doc_embeddings : dict of {layer_idx: np.ndarray}
        Document embeddings per layer.
    k : int, default 10
        MRR cutoff.

    Returns
    -------
    best_layer : int
        Layer index with highest MRR@k.
    best_mrr : float
        MRR@k at the best layer.
    profile : dict of {layer_idx: mrr_score}
        Full layer-wise MRR profile.

    Notes
    -----
    This requires access to ground-truth query-document pairs for evaluation.
    In deployment, use participation_ratio as a proxy: layers with higher PR
    tend to produce better retrieval (rho=0.736 across models).

    Examples
    --------
    >>> best_layer, best_mrr, profile = find_best_layer(q_layers, d_layers)
    >>> final_layer = max(profile.keys())
    >>> print(f"Best: layer {best_layer} (MRR={best_mrr:.3f})")
    >>> print(f"Final: layer {final_layer} (MRR={profile[final_layer]:.3f})")
    >>> print(f"Gain from layer selection: {best_mrr - profile[final_layer]:+.3f}")
    """
    profile = layer_mrr_profile(layer_query_embeddings, layer_doc_embeddings, k)
    if not profile:
        raise ValueError("No common layers found between query and doc embeddings")
    best_layer = max(profile, key=profile.get)
    return best_layer, profile[best_layer], profile
