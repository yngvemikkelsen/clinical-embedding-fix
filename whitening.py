"""
ZCA whitening for clinical text embeddings.

Provides two deployment modes:
    - corpus_only_whitening: fits on document embeddings only (deployment-realistic)
    - transductive_whitening: fits on documents + queries jointly (oracle/upper bound)

Empirical evidence (Mikkelsen, 2026):
    - Corpus-only whitening: positive on 16/16 heterogeneous clinical text conditions
      (MTSamples, PMC-Patients), negative on 8/8 structurally uniform conditions (Synthetic)
    - Transductive whitening: positive on all 24 conditions tested
    - Largest gains on models with degenerate geometry (BioBERT +0.187, E5-Mistral +0.265)
"""

import numpy as np
from typing import Tuple, Optional


def zca_whitening_matrix(
    embeddings: np.ndarray,
    regularization: float = 1e-5,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Compute ZCA whitening transform from a set of embeddings.

    Parameters
    ----------
    embeddings : np.ndarray, shape (n_samples, embedding_dim)
        Input embeddings to fit the whitening transform on.
    regularization : float, default 1e-5
        Regularization added to singular values to prevent division by zero.
        Larger values produce more conservative whitening.

    Returns
    -------
    W : np.ndarray, shape (embedding_dim, embedding_dim)
        The ZCA whitening matrix. Apply as: whitened = (X - mean) @ W.T
    mean : np.ndarray, shape (embedding_dim,)
        The mean vector used for centering.

    Notes
    -----
    ZCA (Zero-phase Component Analysis) whitening decorrelates dimensions while
    preserving the original coordinate alignment, unlike PCA whitening which
    rotates into principal component space. This makes ZCA more suitable for
    embeddings where individual dimensions may carry interpretable information.

    The transform is: W = U @ diag(1/sqrt(S + reg)) @ U^T
    where U, S are from the SVD of the covariance matrix.
    """
    if embeddings.ndim != 2:
        raise ValueError(f"Expected 2D array, got shape {embeddings.shape}")
    if embeddings.shape[0] < embeddings.shape[1]:
        import warnings
        warnings.warn(
            f"Fewer samples ({embeddings.shape[0]}) than dimensions "
            f"({embeddings.shape[1]}). Covariance estimate will be rank-deficient. "
            f"Consider using more documents or increasing regularization.",
            stacklevel=2,
        )

    mean = embeddings.mean(axis=0)
    centered = embeddings - mean
    cov = centered.T @ centered / (len(centered) - 1)

    U, S, Vt = np.linalg.svd(cov)
    S_inv_sqrt = np.diag(1.0 / np.sqrt(S + regularization))
    W = U @ S_inv_sqrt @ U.T

    return W, mean


def _apply_whitening(
    embeddings: np.ndarray,
    W: np.ndarray,
    mean: np.ndarray,
    normalize: bool = True,
) -> np.ndarray:
    """Apply a precomputed whitening transform and optionally L2-normalize."""
    whitened = (embeddings - mean) @ W.T
    if normalize:
        norms = np.linalg.norm(whitened, axis=1, keepdims=True)
        whitened = whitened / np.clip(norms, 1e-9, None)
    return whitened


def corpus_only_whitening(
    doc_embeddings: np.ndarray,
    query_embeddings: np.ndarray,
    regularization: float = 1e-5,
    normalize: bool = True,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Apply ZCA whitening fitted on document embeddings only.

    This is the deployment-realistic variant: the whitening transform is
    estimated from the document corpus (available at index time) and applied
    to both documents and queries. No access to queries at fit time.

    Parameters
    ----------
    doc_embeddings : np.ndarray, shape (n_docs, embedding_dim)
        Document embeddings (L2-normalized recommended).
    query_embeddings : np.ndarray, shape (n_queries, embedding_dim)
        Query embeddings (L2-normalized recommended).
    regularization : float, default 1e-5
        Regularization for the whitening matrix.
    normalize : bool, default True
        Whether to L2-normalize after whitening.

    Returns
    -------
    fixed_docs : np.ndarray, shape (n_docs, embedding_dim)
    fixed_queries : np.ndarray, shape (n_queries, embedding_dim)

    Notes
    -----
    Empirically effective on heterogeneous clinical text (MTSamples, PMC-Patients).
    May degrade performance on structurally uniform corpora where the covariance
    is already low-rank, as whitening amplifies noise in sparse directions.

    Examples
    --------
    >>> from clinical_embedding_fix import corpus_only_whitening
    >>> fixed_docs, fixed_queries = corpus_only_whitening(doc_emb, query_emb)
    >>> # Use fixed embeddings for retrieval
    >>> scores = fixed_queries @ fixed_docs.T
    """
    W, mean = zca_whitening_matrix(doc_embeddings, regularization)
    fixed_docs = _apply_whitening(doc_embeddings, W, mean, normalize)
    fixed_queries = _apply_whitening(query_embeddings, W, mean, normalize)
    return fixed_docs, fixed_queries


def transductive_whitening(
    doc_embeddings: np.ndarray,
    query_embeddings: np.ndarray,
    regularization: float = 1e-5,
    normalize: bool = True,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Apply ZCA whitening fitted on both documents and queries jointly.

    This is the oracle/upper-bound variant: uses all available embeddings
    to estimate the whitening transform. Not realistic for deployment
    (requires access to queries at index time) but useful as a diagnostic
    upper bound on whitening effectiveness.

    Parameters
    ----------
    doc_embeddings : np.ndarray, shape (n_docs, embedding_dim)
        Document embeddings.
    query_embeddings : np.ndarray, shape (n_queries, embedding_dim)
        Query embeddings.
    regularization : float, default 1e-5
        Regularization for the whitening matrix.
    normalize : bool, default True
        Whether to L2-normalize after whitening.

    Returns
    -------
    fixed_docs : np.ndarray, shape (n_docs, embedding_dim)
    fixed_queries : np.ndarray, shape (n_queries, embedding_dim)
    """
    all_emb = np.vstack([doc_embeddings, query_embeddings])
    W, mean = zca_whitening_matrix(all_emb, regularization)
    fixed_docs = _apply_whitening(doc_embeddings, W, mean, normalize)
    fixed_queries = _apply_whitening(query_embeddings, W, mean, normalize)
    return fixed_docs, fixed_queries
