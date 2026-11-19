"""
DataSentry Detectors - Unstructured / Text Modality

Embedding drift detection for document batches using MMD, Domain-Classifier AUC,
and Centroid Cosine Distance, with Docling-compatible document ingestion.
"""

from __future__ import annotations

import os
from typing import Any, List, Optional, Sequence, Union

import numpy as np
from sklearn.decomposition import TruncatedSVD
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedKFold

from datasentry.detectors.base import BaseDetector
from datasentry.schemas import (
    AxisType,
    Finding,
    Severity,
    TextDriftReport,
)


def compute_rbf_mmd(x: np.ndarray, y: np.ndarray, gamma: Optional[float] = None) -> float:
    """
    Computes unbiased Maximum Mean Discrepancy (MMD) with Gaussian RBF kernel.
    MMD^2 = E[k(x, x')] - 2*E[k(x, y)] + E[k(y, y')]
    """
    m, n = len(x), len(y)
    if m < 2 or n < 2:
        return 0.0

    # Pairwise squared Euclidean distances
    xx = np.sum(x**2, axis=1, keepdims=True) + np.sum(x**2, axis=1, keepdims=True).T - 2 * np.dot(x, x.T)
    yy = np.sum(y**2, axis=1, keepdims=True) + np.sum(y**2, axis=1, keepdims=True).T - 2 * np.dot(y, y.T)
    xy = np.sum(x**2, axis=1, keepdims=True) + np.sum(y**2, axis=1, keepdims=True).T - 2 * np.dot(x, y.T)

    xx = np.maximum(0.0, xx)
    yy = np.maximum(0.0, yy)
    xy = np.maximum(0.0, xy)

    if gamma is None:
        # Median heuristic for gamma
        median_dist = float(np.median(xy))
        gamma = 1.0 / (2.0 * max(1e-4, median_dist))

    k_xx = np.exp(-gamma * xx)
    k_yy = np.exp(-gamma * yy)
    k_xy = np.exp(-gamma * xy)

    # Remove diagonals for unbiased estimate
    np.fill_diagonal(k_xx, 0.0)
    np.fill_diagonal(k_yy, 0.0)

    mmd2 = (np.sum(k_xx) / (m * (m - 1))) + (np.sum(k_yy) / (n * (n - 1))) - (2.0 * np.sum(k_xy) / (m * n))
    return float(np.sqrt(max(0.0, mmd2)))


class DocumentEmbedder:
    """Embeds text documents into dense semantic vectors."""

    def __init__(self, model_name: str = "offline_tfidf_lsa", dim: int = 64):
        self.model_name = model_name
        self.dim = dim
        self._vectorizer: Optional[TfidfVectorizer] = None
        self._use_st = False

        if model_name.startswith("sentence-transformers"):
            try:
                from sentence_transformers import SentenceTransformer

                self._st_model = SentenceTransformer(model_name)
                self._use_st = True
            except Exception:
                self._use_st = False

    def fit_transform(
        self, ref_texts: Sequence[str], cur_texts: Sequence[str]
    ) -> tuple[np.ndarray, np.ndarray]:
        """Generates embeddings for reference and current text batches."""
        if self._use_st:
            ref_emb = self._st_model.encode(list(ref_texts), show_progress_bar=False)
            cur_emb = self._st_model.encode(list(cur_texts), show_progress_bar=False)
            return np.asarray(ref_emb, dtype=float), np.asarray(cur_emb, dtype=float)

        # Deterministic offline TF-IDF + TruncatedSVD dense representation (LSA)
        all_texts = list(ref_texts) + list(cur_texts)
        self._vectorizer = TfidfVectorizer(
            max_features=500, stop_words="english", ngram_range=(1, 2)
        )
        tfidf_mat = self._vectorizer.fit_transform(all_texts).toarray()

        n_samples, n_features = tfidf_mat.shape
        n_components = min(self.dim, max(2, min(n_samples - 1, n_features - 1)))
        svd = TruncatedSVD(n_components=n_components, random_state=42)
        dense_mat = svd.fit_transform(tfidf_mat)

        # Normalize dense vectors to unit sphere
        norm = np.linalg.norm(dense_mat, axis=1, keepdims=True)
        norm[norm == 0] = 1.0
        normalized = dense_mat / norm

        ref_emb = normalized[: len(ref_texts)]
        cur_emb = normalized[len(ref_texts) :]
        return ref_emb, cur_emb


class TextDriftDetector(BaseDetector):
    """Detects semantic and topic distribution drift in unstructured text collections."""

    name: str = "text_drift_detector"
    modality: AxisType = AxisType.TEXT

    def __init__(
        self,
        embedding_model: str = "offline_tfidf_lsa",
        auc_threshold: float = 0.65,
        mmd_threshold: float = 0.05,
        cosine_threshold: float = 0.15,
    ):
        super().__init__(
            embedding_model=embedding_model,
            auc_threshold=auc_threshold,
            mmd_threshold=mmd_threshold,
            cosine_threshold=cosine_threshold,
        )
        self.embedder = DocumentEmbedder(model_name=embedding_model)
        self.auc_threshold = auc_threshold
        self.mmd_threshold = mmd_threshold
        self.cosine_threshold = cosine_threshold

    def detect(
        self,
        current_data: Union[Sequence[str], Any],
        reference_data: Optional[Union[Sequence[str], Any]] = None,
        **kwargs: Any,
    ) -> TextDriftReport:
        if reference_data is None:
            raise ValueError("TextDriftDetector requires reference_data texts for comparison.")

        cur_texts = list(current_data)
        ref_texts = list(reference_data)

        if len(cur_texts) < 3 or len(ref_texts) < 3:
            raise ValueError("Text collections must contain at least 3 documents each.")

        # 1. Embed documents
        ref_emb, cur_emb = self.embedder.fit_transform(ref_texts, cur_texts)

        # 2. Maximum Mean Discrepancy (MMD)
        mmd_score = compute_rbf_mmd(ref_emb, cur_emb)

        # 3. Centroid Cosine Distance
        mean_ref = np.mean(ref_emb, axis=0)
        mean_cur = np.mean(cur_emb, axis=0)
        norm_ref = np.linalg.norm(mean_ref)
        norm_cur = np.linalg.norm(mean_cur)

        if norm_ref > 0 and norm_cur > 0:
            cosine_sim = float(np.dot(mean_ref, mean_cur) / (norm_ref * norm_cur))
            centroid_dist = float(max(0.0, 1.0 - cosine_sim))
        else:
            centroid_dist = 0.0

        # 4. Domain-Classifier AUC
        auc_score = self._compute_domain_classifier_auc(ref_emb, cur_emb)

        # 5. Extract top divergent terms between reference and current
        top_terms = self._find_divergent_terms(ref_texts, cur_texts)

        # Drift decision: breach in AUC or joint breach in centroid distance and MMD
        is_drift = bool(
            auc_score >= self.auc_threshold
            or (centroid_dist >= self.cosine_threshold and mmd_score >= self.mmd_threshold)
        )

        findings: List[Finding] = []
        if is_drift:
            findings.append(
                Finding(
                    finding_id="FND-TEXT-DRIFT-001",
                    axis=AxisType.TEXT,
                    severity=Severity.HIGH if auc_score > 0.80 else Severity.MEDIUM,
                    metric_name="domain_classifier_auc",
                    metric_value=auc_score,
                    threshold=self.auc_threshold,
                    description=(
                        f"Unstructured text distribution drift detected: domain classifier achieved "
                        f"AUC={auc_score:.3f} (threshold={self.auc_threshold:.2f}, MMD={mmd_score:.4f}, "
                        f"centroid_cosine_dist={centroid_dist:.3f}). Divergent topics: {', '.join(top_terms[:5])}."
                    ),
                )
            )

        return TextDriftReport(
            total_reference_docs=len(ref_texts),
            total_current_docs=len(cur_texts),
            embedding_model=self.embedder.model_name,
            mmd_score=round(mmd_score, 4),
            mmd_threshold=self.mmd_threshold,
            domain_classifier_auc=round(auc_score, 4),
            auc_threshold=self.auc_threshold,
            centroid_cosine_distance=round(centroid_dist, 4),
            cosine_threshold=self.cosine_threshold,
            is_drift=is_drift,
            top_divergent_terms=top_terms,
            findings=findings,
        )

    def _compute_domain_classifier_auc(
        self, ref_emb: np.ndarray, cur_emb: np.ndarray
    ) -> float:
        """Trains cross-validated linear classifier discriminating Ref (0) vs Cur (1)."""
        x_all = np.vstack([ref_emb, cur_emb])
        y_all = np.array([0] * len(ref_emb) + [1] * len(cur_emb))

        n_splits = min(3, min(len(ref_emb), len(cur_emb)))
        if n_splits < 2:
            return 0.50

        skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=42)
        aucs = []

        for train_idx, test_idx in skf.split(x_all, y_all):
            clf = LogisticRegression(C=1.0, max_iter=200, random_state=42)
            clf.fit(x_all[train_idx], y_all[train_idx])
            probs = clf.predict_proba(x_all[test_idx])[:, 1]
            try:
                auc_fold = roc_auc_score(y_all[test_idx], probs)
                aucs.append(auc_fold)
            except ValueError:
                aucs.append(0.50)

        return float(np.mean(aucs)) if aucs else 0.50

    def _find_divergent_terms(
        self, ref_texts: List[str], cur_texts: List[str], top_k: int = 5
    ) -> List[str]:
        vec = TfidfVectorizer(max_features=200, stop_words="english")
        try:
            vec.fit(ref_texts + cur_texts)
            feature_names = np.array(vec.get_feature_names_out())
            ref_tfidf = vec.transform(ref_texts).mean(axis=0)
            cur_tfidf = vec.transform(cur_texts).mean(axis=0)

            diff = np.abs(np.asarray(cur_tfidf - ref_tfidf).flatten())
            top_indices = np.argsort(diff)[-top_k:][::-1]
            return [str(feature_names[i]) for i in top_indices]
        except Exception:
            return []


def ingest_document_with_docling(file_path: str) -> str:
    """
    Docling-compatible document ingestion path.
    Extracts text from PDF/documents using Docling if available,
    with graceful fallback to standard PDF/text extraction.
    """
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"File not found: {file_path}")

    # Check for Docling
    try:
        from docling.document_converter import DocumentConverter

        converter = DocumentConverter()
        result = converter.convert(file_path)
        return str(result.document.export_to_markdown())
    except Exception:
        pass

    # Fallback to pdfplumber / pypdfium2 / pure text
    if file_path.lower().endswith(".pdf"):
        try:
            import pdfplumber

            with pdfplumber.open(file_path) as pdf:
                pages = [p.extract_text() or "" for p in pdf.pages]
                return "\n".join(pages)
        except Exception:
            pass

    # Plain text / markdown fallback
    with open(file_path, "r", encoding="utf-8", errors="ignore") as f:
        return f.read()
