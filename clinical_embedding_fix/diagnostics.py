"""
Embedding geometry diagnostics for clinical text.

Three complementary metrics for assessing embedding quality:
    - participation_ratio: effective dimensionality (higher = better, range 1..dim)
    - avg_cosine_similarity: mean pairwise cosine (lower = better for retrieval)
    - anisotropy_svd: concentration in top singular value (lower = better)

Key finding (Mikkelsen, 2026):
    Final-layer participation ratio correlates with MRR@10 at rho=0.736 (p=0.010),
    outperforming average cosine similarity as a retrieval quality predictor.
    Excluding EOS-pooled models: rho=0.964 (p<0.001).
"""

import numpy as np
from typing import Optional, Dict


def participation_ratio(
    embeddings: np.ndarray,
    max_samples: int = 1000,
    seed: int = 42,
) -> float:
    """
    Compute participation ratio (effective dimensionality) of embeddings.

    PR = (sum(sigma_i^2))^2 / sum(sigma_i^4)

    where sigma_i are the singular values of the centered embedding matrix.
    A PR of 1.0 means all variance is in one dimension (maximally degenerate).
    A PR equal to the embedding dimension means perfectly isotropic.

    Parameters
    ----------
    embeddings : np.ndarray, shape (n_samples, embedding_dim)
        Input embeddings (L2-normalized recommended).
    max_samples : int, default 1000
        Subsample to this many embeddings for efficiency.
    seed : int, default 42
        Random seed for subsampling.

    Returns
    -------
    pr : float
        Participation ratio. Higher values indicate healthier geometry.

    Examples
    --------
    >>> from clinical_embedding_fix import participation_ratio
    >>> pr = participation_ratio(doc_embeddings)
    >>> if pr < 20:
    ...     print("Degenerate geometry detected — consider whitening")
    """
    centered = embeddings - embeddings.mean(axis=0)
    if len(centered) > max_samples:
        rng = np.random.RandomState(seed)
        idx = rng.choice(len(centered), max_samples, replace=False)
        centered = centered[idx]
    try:
        _, S, _ = np.linalg.svd(centered, full_matrices=False)
        S_sq = S ** 2
        return float((S_sq.sum()) ** 2 / (S_sq ** 2).sum())
    except np.linalg.LinAlgError:
        return float("nan")


def avg_cosine_similarity(
    embeddings: np.ndarray,
    n_pairs: int = 10000,
    seed: int = 42,
) -> float:
    """
    Compute average pairwise cosine similarity (sampled).

    High values (>0.8) indicate embedding collapse / anisotropy.
    For L2-normalized embeddings, cosine similarity = dot product.

    Parameters
    ----------
    embeddings : np.ndarray, shape (n_samples, embedding_dim)
        Input embeddings (L2-normalized recommended).
    n_pairs : int, default 10000
        Number of random pairs to sample.
    seed : int, default 42
        Random seed for reproducibility.

    Returns
    -------
    avg_cos : float
        Average pairwise cosine similarity. Lower is generally better
        for retrieval (more discriminative).
    """
    n = embeddings.shape[0]
    if n < 2:
        return float("nan")
    rng = np.random.RandomState(seed)
    idx_a = rng.randint(0, n, size=n_pairs)
    idx_b = rng.randint(0, n, size=n_pairs)
    mask = idx_a != idx_b
    idx_a, idx_b = idx_a[mask], idx_b[mask]
    if len(idx_a) == 0:
        return float("nan")
    cos_sims = np.sum(embeddings[idx_a] * embeddings[idx_b], axis=1)
    return float(cos_sims.mean())


def anisotropy_svd(
    embeddings: np.ndarray,
    max_samples: int = 1000,
    seed: int = 42,
) -> float:
    """
    Compute SVD-based anisotropy: sigma_1^2 / sum(sigma_i^2).

    Measures how much variance is concentrated in the top singular direction.
    Value of 1.0 = maximally anisotropic (all embeddings on a line).
    Value of 1/dim = perfectly isotropic.

    Parameters
    ----------
    embeddings : np.ndarray, shape (n_samples, embedding_dim)
        Input embeddings.
    max_samples : int, default 1000
        Subsample for efficiency.
    seed : int, default 42
        Random seed.

    Returns
    -------
    aniso : float
        Anisotropy score (lower = healthier).
    """
    centered = embeddings - embeddings.mean(axis=0)
    if len(centered) > max_samples:
        rng = np.random.RandomState(seed)
        idx = rng.choice(len(centered), max_samples, replace=False)
        centered = centered[idx]
    try:
        _, S, _ = np.linalg.svd(centered, full_matrices=False)
        S_sq = S ** 2
        return float(S_sq[0] / S_sq.sum())
    except np.linalg.LinAlgError:
        return float("nan")


def embedding_report(
    embeddings: np.ndarray,
    name: str = "embeddings",
    max_samples: int = 1000,
    seed: int = 42,
) -> Dict[str, float]:
    """
    Generate a diagnostic report for a set of embeddings.

    Parameters
    ----------
    embeddings : np.ndarray, shape (n_samples, embedding_dim)
        Input embeddings.
    name : str, default "embeddings"
        Label for display.
    max_samples : int, default 1000
        Subsample for efficiency.
    seed : int, default 42
        Random seed.

    Returns
    -------
    report : dict
        Dictionary with keys: n_samples, embedding_dim, participation_ratio,
        avg_cosine, anisotropy, needs_whitening.

    Examples
    --------
    >>> from clinical_embedding_fix import embedding_report
    >>> report = embedding_report(doc_embeddings, name="BioBERT clinical docs")
    >>> if report["needs_whitening"]:
    ...     print("Whitening recommended")
    """
    pr = participation_ratio(embeddings, max_samples, seed)
    avg_cos = avg_cosine_similarity(embeddings, seed=seed)
    aniso = anisotropy_svd(embeddings, max_samples, seed)

    # Heuristic thresholds based on empirical findings
    # PR < 20 and/or avg_cosine > 0.7 typically indicates degenerate geometry
    needs_whitening = (pr < 20) or (avg_cos > 0.7)

    report = {
        "name": name,
        "n_samples": embeddings.shape[0],
        "embedding_dim": embeddings.shape[1],
        "participation_ratio": round(pr, 2),
        "avg_cosine": round(avg_cos, 4),
        "anisotropy": round(aniso, 4),
        "needs_whitening": needs_whitening,
    }

    print(f"=== Embedding Report: {name} ===")
    print(f"  Samples: {report['n_samples']}, Dim: {report['embedding_dim']}")
    print(f"  Participation ratio: {report['participation_ratio']:.1f}")
    print(f"  Avg cosine similarity: {report['avg_cosine']:.4f}")
    print(f"  Anisotropy (SVD): {report['anisotropy']:.4f}")
    print(f"  Whitening recommended: {'YES' if needs_whitening else 'No'}")

    return report
