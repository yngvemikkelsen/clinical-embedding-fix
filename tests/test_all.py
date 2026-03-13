"""Tests for clinical-embedding-fix."""

import numpy as np
import pytest
from clinical_embedding_fix import (
    corpus_only_whitening,
    transductive_whitening,
    zca_whitening_matrix,
    participation_ratio,
    avg_cosine_similarity,
    anisotropy_svd,
    embedding_report,
    find_best_layer,
    layer_mrr_profile,
)


@pytest.fixture
def random_embeddings():
    """Generate random L2-normalized embeddings."""
    rng = np.random.RandomState(42)
    emb = rng.randn(100, 768).astype(np.float32)
    emb = emb / np.linalg.norm(emb, axis=1, keepdims=True)
    return emb


@pytest.fixture
def degenerate_embeddings():
    """Generate embeddings with high anisotropy (near-degenerate)."""
    rng = np.random.RandomState(42)
    # Most variance in first 2 dimensions
    emb = rng.randn(100, 768).astype(np.float32) * 0.01
    emb[:, 0] += rng.randn(100) * 10
    emb[:, 1] += rng.randn(100) * 5
    emb = emb / np.linalg.norm(emb, axis=1, keepdims=True)
    return emb


@pytest.fixture
def query_doc_pairs():
    """Generate matched query-document pairs for retrieval testing."""
    rng = np.random.RandomState(42)
    n, dim = 50, 128
    docs = rng.randn(n, dim).astype(np.float32)
    docs = docs / np.linalg.norm(docs, axis=1, keepdims=True)
    # Queries are noisy versions of their matched document
    queries = docs + rng.randn(n, dim).astype(np.float32) * 0.3
    queries = queries / np.linalg.norm(queries, axis=1, keepdims=True)
    return queries, docs


# ── Whitening tests ──


class TestZCAWhiteningMatrix:
    def test_returns_correct_shapes(self, random_embeddings):
        W, mean = zca_whitening_matrix(random_embeddings)
        assert W.shape == (768, 768)
        assert mean.shape == (768,)

    def test_whitened_covariance_is_identity(self):
        # Need more samples than dimensions for full-rank covariance
        rng = np.random.RandomState(42)
        emb = rng.randn(200, 64).astype(np.float32)
        emb = emb / np.linalg.norm(emb, axis=1, keepdims=True)
        W, mean = zca_whitening_matrix(emb, regularization=1e-5)
        whitened = (emb - mean) @ W.T
        cov = whitened.T @ whitened / (len(whitened) - 1)
        diag = np.diag(cov)
        assert np.allclose(diag[:10], 1.0, atol=0.5)

    def test_warns_on_few_samples(self):
        small = np.random.randn(5, 768).astype(np.float32)
        with pytest.warns(UserWarning, match="Fewer samples"):
            zca_whitening_matrix(small)

    def test_rejects_1d_input(self):
        with pytest.raises(ValueError, match="2D"):
            zca_whitening_matrix(np.array([1.0, 2.0, 3.0]))


class TestCorpusOnlyWhitening:
    def test_returns_correct_shapes(self, query_doc_pairs):
        q, d = query_doc_pairs
        fd, fq = corpus_only_whitening(d, q)
        assert fd.shape == d.shape
        assert fq.shape == q.shape

    def test_output_is_normalized(self, query_doc_pairs):
        q, d = query_doc_pairs
        fd, fq = corpus_only_whitening(d, q)
        norms = np.linalg.norm(fd, axis=1)
        assert np.allclose(norms, 1.0, atol=1e-6)

    def test_reduces_anisotropy(self, degenerate_embeddings):
        rng = np.random.RandomState(99)
        queries = degenerate_embeddings + rng.randn(*degenerate_embeddings.shape) * 0.1
        queries = queries / np.linalg.norm(queries, axis=1, keepdims=True)

        aniso_before = anisotropy_svd(degenerate_embeddings)
        fixed_d, _ = corpus_only_whitening(degenerate_embeddings, queries)
        aniso_after = anisotropy_svd(fixed_d)
        assert aniso_after < aniso_before


class TestTransductiveWhitening:
    def test_returns_correct_shapes(self, query_doc_pairs):
        q, d = query_doc_pairs
        fd, fq = transductive_whitening(d, q)
        assert fd.shape == d.shape
        assert fq.shape == q.shape

    def test_output_is_normalized(self, query_doc_pairs):
        q, d = query_doc_pairs
        fd, fq = transductive_whitening(d, q)
        norms = np.linalg.norm(fq, axis=1)
        assert np.allclose(norms, 1.0, atol=1e-6)


# ── Diagnostics tests ──


class TestParticipationRatio:
    def test_isotropic_embeddings_high_pr(self):
        rng = np.random.RandomState(42)
        emb = rng.randn(200, 64).astype(np.float32)
        emb = emb / np.linalg.norm(emb, axis=1, keepdims=True)
        pr = participation_ratio(emb)
        # Isotropic random should have high PR (close to min(n, dim))
        assert pr > 20

    def test_degenerate_embeddings_low_pr(self, degenerate_embeddings):
        pr = participation_ratio(degenerate_embeddings)
        assert pr < 5

    def test_returns_float(self, random_embeddings):
        assert isinstance(participation_ratio(random_embeddings), float)


class TestAvgCosineSimilarity:
    def test_orthogonal_embeddings_low_cosine(self):
        # Identity-like embeddings should have low pairwise cosine
        emb = np.eye(100, dtype=np.float32)
        cos = avg_cosine_similarity(emb, n_pairs=5000)
        assert cos < 0.1

    def test_identical_embeddings_high_cosine(self):
        emb = np.ones((100, 10), dtype=np.float32)
        emb = emb / np.linalg.norm(emb, axis=1, keepdims=True)
        cos = avg_cosine_similarity(emb)
        assert cos > 0.99

    def test_single_sample_returns_nan(self):
        emb = np.array([[1.0, 0.0, 0.0]])
        assert np.isnan(avg_cosine_similarity(emb))


class TestAnisotropySVD:
    def test_degenerate_high_anisotropy(self, degenerate_embeddings):
        aniso = anisotropy_svd(degenerate_embeddings)
        assert aniso > 0.5

    def test_isotropic_low_anisotropy(self):
        rng = np.random.RandomState(42)
        emb = rng.randn(200, 64).astype(np.float32)
        aniso = anisotropy_svd(emb)
        assert aniso < 0.1


class TestEmbeddingReport:
    def test_returns_dict_with_required_keys(self, random_embeddings):
        report = embedding_report(random_embeddings, name="test")
        assert "participation_ratio" in report
        assert "avg_cosine" in report
        assert "anisotropy" in report
        assert "needs_whitening" in report

    def test_degenerate_flags_whitening(self, degenerate_embeddings):
        report = embedding_report(degenerate_embeddings, name="degenerate")
        assert report["needs_whitening"] is True


# ── Layer selection tests ──


class TestLayerMRRProfile:
    def test_returns_profile_for_all_layers(self, query_doc_pairs):
        q, d = query_doc_pairs
        # Simulate 3 layers
        q_layers = {0: q * 0.5, 1: q * 0.1, 2: q}
        d_layers = {0: d * 0.5, 1: d * 0.1, 2: d}
        profile = layer_mrr_profile(q_layers, d_layers)
        assert len(profile) == 3
        assert all(isinstance(v, float) for v in profile.values())


class TestFindBestLayer:
    def test_finds_correct_best_layer(self, query_doc_pairs):
        q, d = query_doc_pairs
        # Layer 2 should be best (unscaled = highest similarity structure)
        q_layers = {0: q + np.random.randn(*q.shape) * 2, 1: q + np.random.randn(*q.shape) * 5, 2: q}
        d_layers = {0: d + np.random.randn(*d.shape) * 2, 1: d + np.random.randn(*d.shape) * 5, 2: d}
        # Normalize
        for k in q_layers:
            q_layers[k] = q_layers[k] / np.linalg.norm(q_layers[k], axis=1, keepdims=True)
            d_layers[k] = d_layers[k] / np.linalg.norm(d_layers[k], axis=1, keepdims=True)

        best_layer, best_mrr, profile = find_best_layer(q_layers, d_layers)
        assert best_layer == 2
        assert best_mrr == profile[2]

    def test_raises_on_empty_input(self):
        with pytest.raises(ValueError):
            find_best_layer({}, {})
