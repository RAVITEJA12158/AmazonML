"""Semantic retrieval — the most important architectural constraint here is
WHAT POOL this runs against. Per the restructured flow:

    S1 -> geographic blocking -> cheap ranking -> semantic retrieval (THIS) -> Top-K -> pair model

This module is never called against the full S2/S3 table. Callers (see
src/blocking/union.py) must pass an already-reduced candidate pool.

embedding_backend is pluggable:
  - "tfidf" (default): character n-gram TF-IDF + cosine similarity, runs
    fully locally, no external calls. This is what actually runs in this
    environment.
  - "qwen": the real target backend (Qwen3-Embedding-8B). Not runnable in
    this sandbox (no internet access to model-hosting sites, no GPU). Swap
    this in once you have the model weights available locally — the
    interface (`SemanticRetriever.fit_corpus` / `.query`) doesn't change.
"""
from abc import ABC, abstractmethod
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity


class EmbeddingBackend(ABC):
    @abstractmethod
    def fit_corpus(self, texts: list) -> None:
        ...

    @abstractmethod
    def embed(self, texts: list) -> np.ndarray:
        ...


class TfidfBackend(EmbeddingBackend):
    """Local fallback. Character n-gram TF-IDF is a reasonable stand-in for
    a real multilingual embedding on typo/transliteration-heavy business
    names — it isn't semantic in the neural sense, but it's cheap, runs
    anywhere, and still catches a meaningful chunk of the near-duplicate
    pattern this task cares about."""

    def __init__(self):
        self.vectorizer = TfidfVectorizer(analyzer="char_wb", ngram_range=(3, 5), min_df=1)
        self._fitted = False

    def fit_corpus(self, texts: list) -> None:
        texts = [t if t else "" for t in texts]
        if not texts:
            return
        self.vectorizer.fit(texts)
        self._fitted = True

    def embed(self, texts: list) -> np.ndarray:
        texts = [t if t else "" for t in texts]
        if not self._fitted:
            self.fit_corpus(texts)
        return self.vectorizer.transform(texts)


class QwenBackend(EmbeddingBackend):
    """Placeholder for the real backend. Raises clearly rather than silently
    doing something else, so a misconfiguration is never mistaken for a
    working Qwen retrieval run."""

    def __init__(self, model_name: str = "Qwen/Qwen3-Embedding-8B"):
        self.model_name = model_name

    def fit_corpus(self, texts: list) -> None:
        pass

    def embed(self, texts: list) -> np.ndarray:
        raise NotImplementedError(
            "QwenBackend requires model weights + a GPU-capable environment "
            "that this sandbox does not have. Install sentence-transformers, "
            "download the model weights in your own environment, and "
            "implement `embed()` to call the model — the rest of the "
            "pipeline (union.py, prune.py, features) does not need to change."
        )


def get_backend(name: str) -> EmbeddingBackend:
    if name == "tfidf":
        return TfidfBackend()
    if name == "qwen":
        return QwenBackend()
    raise ValueError(f"Unknown embedding_backend: {name}")


class SemanticRetriever:
    """Runs three parallel embeddings — name, address, and concatenated
    name+address — never collapsing to a single combined vector, so
    downstream features can tell "name matches, address doesn't" apart from
    "both genuinely match" (see the flow-diagram addendum on this)."""

    def __init__(self, backend_name: str = "tfidf"):
        self.backend_name = backend_name
        self.backend_name_fields = {}  # field -> EmbeddingBackend instance

    def _backend_for(self, field: str) -> EmbeddingBackend:
        if field not in self.backend_name_fields:
            self.backend_name_fields[field] = get_backend(self.backend_name)
        return self.backend_name_fields[field]

    def retrieve(self, s1_norm: dict, pool_norm: list, top_k_name: int,
                 top_k_address: int, top_k_combined: int) -> dict:
        """Returns {record_id: {"semantic_name": .., "semantic_address": ..,
        "semantic_combined": .., "retrieved_semantic_name": bool, ...}}
        for the candidates that made it into at least one of the three
        top-k lists. `pool_norm` must already be the reduced pool — this
        method does not itself do any size control.
        """
        if not pool_norm:
            return {}

        ids = [c["record_id"] for c in pool_norm]
        names = [c["normalized_name"] for c in pool_norm]
        addresses = [c["normalized_address"] for c in pool_norm]
        combined = [f"{c['normalized_name']} {c['normalized_address']}" for c in pool_norm]

        s1_name = s1_norm["normalized_name"]
        s1_address = s1_norm["normalized_address"]
        s1_combined = f"{s1_name} {s1_address}"

        results = {}

        for field, corpus, query, top_k, flag in (
            ("semantic_name", names, s1_name, top_k_name, "retrieved_semantic_name"),
            ("semantic_address", addresses, s1_address, top_k_address, "retrieved_semantic_address"),
            ("semantic_combined", combined, s1_combined, top_k_combined, "retrieved_semantic_combined"),
        ):
            backend = self._backend_for(field)
            backend.fit_corpus(corpus)
            corpus_vec = backend.embed(corpus)
            query_vec = backend.embed([query])
            sims = cosine_similarity(query_vec, corpus_vec).ravel()

            top_idx = np.argsort(-sims)[:top_k]
            for idx in top_idx:
                rid = ids[idx]
                results.setdefault(rid, {"semantic_name": 0.0, "semantic_address": 0.0,
                                          "semantic_combined": 0.0,
                                          "retrieved_semantic_name": False,
                                          "retrieved_semantic_address": False,
                                          "retrieved_semantic_combined": False})
                results[rid][field] = float(sims[idx])
                results[rid][flag] = True

            # also record the raw similarity for every candidate (not just
            # top-k) so downstream features have a real number, not just a
            # retrieval flag, for whichever candidates DID make the pool.
            for i, rid in enumerate(ids):
                results.setdefault(rid, {"semantic_name": 0.0, "semantic_address": 0.0,
                                          "semantic_combined": 0.0,
                                          "retrieved_semantic_name": False,
                                          "retrieved_semantic_address": False,
                                          "retrieved_semantic_combined": False})
                results[rid][field] = float(sims[i])

        return results
