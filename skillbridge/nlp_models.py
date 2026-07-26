"""
SkillBridge — NLP Skill Extractors
===================================
Owner: Nalluraj Babu.  CSC 503 Data Mining, Summer 2026.

Every extractor implements one interface:
    fit(texts, Y, train_idx)   learn from TRAIN rows only
    score(texts)               -> dense (n_docs, n_skills) score matrix
    threshold_grid             the cutoff range to search on VALIDATION

A score matrix rather than hard predictions, because the cutoff is a separate
decision tuned per skill in tune_thresholds(). One global cutoff cannot serve
both "communication" (26k postings) and a skill at the frequency floor.

Usage:
    from skillbridge.nlp_models import (GazetteerExtractor, TfidfOvRExtractor,
                                        HybridExtractor, EmbeddingExtractor,
                                        tune_thresholds, map_skills_to_oasis)
"""

from __future__ import annotations

import re
import time
from pathlib import Path

import numpy as np
from scipy import sparse


# ══════════════════════════════════════════════════════════════
# THRESHOLD SELECTION
# ══════════════════════════════════════════════════════════════

def tune_thresholds(score_val: np.ndarray, Y_val: np.ndarray,
                    grid: np.ndarray, candidate_val: np.ndarray | None = None
                    ) -> np.ndarray:
    """
    Per-skill cutoff maximising F1 on VALIDATION. Never sees test.

    candidate_val restricts the search to a candidate set (used by the hybrid,
    where only gazetteer-proposed skills are eligible).
    """
    n_labels = Y_val.shape[1]
    pos = Y_val.sum(axis=0)
    best_f1 = np.zeros(n_labels, dtype=np.float32)
    best_t = np.full(n_labels, grid[len(grid) // 2], dtype=np.float32)
    for t in grid:
        P = score_val > t
        if candidate_val is not None:
            P = P & candidate_val
        tp = (P & Y_val).sum(axis=0)
        pc = P.sum(axis=0)
        prec = np.divide(tp, pc, out=np.zeros(n_labels, dtype=np.float32), where=pc > 0)
        rec = np.divide(tp, pos, out=np.zeros(n_labels, dtype=np.float32), where=pos > 0)
        f1 = np.divide(2 * prec * rec, prec + rec,
                       out=np.zeros(n_labels, dtype=np.float32), where=(prec + rec) > 0)
        upd = f1 > best_f1
        best_f1[upd], best_t[upd] = f1[upd], t
    return best_t


class BaseExtractor:
    """Common interface. Baseline or learned, every extractor implements this."""
    name: str = "base"
    threshold_grid: np.ndarray = np.arange(-1.5, 1.51, 0.125, dtype=np.float32)

    def fit(self, texts, Y, train_idx):
        raise NotImplementedError

    def score(self, texts) -> np.ndarray:
        raise NotImplementedError


# ══════════════════════════════════════════════════════════════
# BASELINE: GAZETTEER
# ══════════════════════════════════════════════════════════════

class GazetteerExtractor(BaseExtractor):
    """
    Predict a skill when its name occurs verbatim, matched on whole words.

    Word boundaries matter: naive substring matching fires "go" inside
    "category" and "r" inside every word in the document. Implemented by
    sliding word n-grams and looking them up in a set, so boundaries hold by
    construction and no extra dependency is needed.

    No learning at all. This is the bar the trained models must clear.
    """
    name = "gazetteer"
    threshold_grid = np.array([0.5], dtype=np.float32)   # binary already

    WORD = re.compile(r"[a-z0-9][a-z0-9+#.\-']*")

    def __init__(self, vocab: list[str]):
        self.vocab = vocab
        self.skill_id = {s: j for j, s in enumerate(vocab)}
        self.by_len: dict[int, set] = {}
        for s in vocab:
            self.by_len.setdefault(len(s.split()), set()).add(s)

    def fit(self, texts, Y, train_idx):
        return self                      # nothing to learn

    def score(self, texts) -> np.ndarray:
        n = len(texts)
        S = np.zeros((n, len(self.vocab)), dtype=np.float16)
        t0 = time.time()
        for i, text in enumerate(texts):
            toks = self.WORD.findall(text)
            for k, pool in self.by_len.items():
                if k > len(toks):
                    continue
                for a in range(len(toks) - k + 1):
                    phrase = " ".join(toks[a:a + k])
                    if phrase in pool:
                        S[i, self.skill_id[phrase]] = 1.0
            if i and i % 20000 == 0:
                print(f"    scanned {i:,}/{n:,}")
        print(f"  gazetteer matched in {time.time() - t0:.0f}s")
        return S


# ══════════════════════════════════════════════════════════════
# TF-IDF + ONE LINEAR CLASSIFIER PER SKILL
# ══════════════════════════════════════════════════════════════

class TfidfOvRExtractor(BaseExtractor):
    """
    One binary classifier per skill, one-vs-rest over TF-IDF features.

    Negative subsampling: a skill tagged on 200 of 38k training postings has a
    0.5% positive rate. Training on every negative is slow and lets them
    dominate the boundary, so we keep all positives and a capped random sample
    of negatives.
    """
    name = "tfidf"

    def __init__(self, vocab: list[str], seed: int = 42,
                 max_features: int = 200_000):
        self.vocab = vocab
        self.seed = seed
        self.max_features = max_features
        self.vec = None
        self.X = None
        self.S = None

    def fit(self, texts, Y, train_idx):
        from sklearn.feature_extraction.text import TfidfVectorizer
        from sklearn.linear_model import SGDClassifier

        self.vec = TfidfVectorizer(lowercase=True, ngram_range=(1, 2), min_df=5,
                                   max_features=self.max_features,
                                   sublinear_tf=True, strip_accents="unicode")
        self.vec.fit([texts[i] for i in train_idx])
        self.X = self.vec.transform(texts)
        print(f"  TF-IDF matrix: {self.X.shape[0]:,} x {self.X.shape[1]:,}")

        Xtr = self.X[train_idx]
        Ytr = Y[train_idx].tocsc()
        n_labels = len(self.vocab)
        self.S = np.full((self.X.shape[0], n_labels), -10.0, dtype=np.float16)

        rng = np.random.default_rng(self.seed)
        t0 = time.time()
        for j in range(n_labels):
            y = np.asarray(Ytr[:, j].todense()).ravel()
            n_pos = int(y.sum())
            if n_pos < 2:
                continue
            pos = np.flatnonzero(y)
            neg = np.flatnonzero(y == 0)
            n_neg = min(len(neg), max(1500, 10 * n_pos))
            take = np.concatenate([pos, rng.choice(neg, n_neg, replace=False)])
            clf = SGDClassifier(loss="hinge", alpha=1e-5, max_iter=15, tol=1e-3,
                                class_weight="balanced", random_state=self.seed + j)
            clf.fit(Xtr[take], y[take])
            self.S[:, j] = self.X @ clf.coef_.ravel() + clf.intercept_[0]
            if j and j % 250 == 0:
                print(f"    trained {j:,}/{n_labels:,}  ({time.time() - t0:.0f}s)")
        print(f"  trained {n_labels:,} classifiers in {time.time() - t0:.0f}s")
        return self

    def score(self, texts) -> np.ndarray:
        return self.S


# ══════════════════════════════════════════════════════════════
# HYBRID: MENTIONED **AND** SALIENT
# ══════════════════════════════════════════════════════════════

class HybridExtractor(BaseExtractor):
    """
    Predict a skill only when it is written in the text (gazetteer) AND the
    classifier scores it salient for this posting.

    This falls straight out of the error analysis of the other two. The
    gazetteer has coverage and no judgement: it fires on every skill an ad
    happens to name, including benefits and boilerplate. The classifier has
    judgement but poor coverage of rare skills. Requiring both keeps the
    coverage and adds the discrimination.
    """
    name = "hybrid"
    threshold_grid = np.arange(-1.0, 1.51, 0.25, dtype=np.float32)

    def __init__(self, gazetteer: GazetteerExtractor, classifier: TfidfOvRExtractor):
        self.gaz = gazetteer
        self.clf = classifier
        self.candidates = None

    def fit(self, texts, Y, train_idx):
        return self                      # both parts are already fitted

    def score(self, texts) -> np.ndarray:
        self.candidates = self.gaz.score(texts).astype(bool)
        return self.clf.score(texts)

    def predict(self, S: np.ndarray, thr: np.ndarray) -> sparse.csr_matrix:
        return sparse.csr_matrix(
            (self.candidates & (S.astype(np.float32) > thr[None, :])).astype(np.int8))


# ══════════════════════════════════════════════════════════════
# SENTENCE EMBEDDINGS
# ══════════════════════════════════════════════════════════════

_SENT = re.compile(r"[.!?\n•·;]+")


def sentences(text: str, lo: int = 8, hi: int = 300, cap: int = 60) -> list[str]:
    """Split a posting into sentences, trimmed and capped."""
    return [p.strip()[:hi] for p in _SENT.split(text) if len(p.strip()) >= lo][:cap]


class SbertBackend:
    """Pretrained Sentence-BERT. The approach the proposal names."""
    tag = "sbert"

    def __init__(self, model_name: str = "all-MiniLM-L6-v2"):
        from sentence_transformers import SentenceTransformer
        self.model = SentenceTransformer(model_name)

    def encode(self, items):
        return self.model.encode(list(items), batch_size=256,
                                 normalize_embeddings=True, show_progress_bar=False)


class SvdBackend:
    """
    TruncatedSVD over our own TF-IDF space: an embedding learned from this
    corpus alone. Serves as the no-pretraining ablation, isolating what
    pretrained semantics actually buy.
    """
    tag = "tfidf_svd"

    def __init__(self, vectorizer, X_train, n_components: int = 192, seed: int = 42):
        from sklearn.decomposition import TruncatedSVD
        svd = TruncatedSVD(n_components=n_components, random_state=seed, n_iter=4)
        svd.fit(X_train)
        self.vec = vectorizer
        self.comp = svd.components_.astype(np.float32)

    def encode(self, items):
        E = np.asarray(self.vec.transform(list(items)) @ self.comp.T, dtype=np.float32)
        n = np.linalg.norm(E, axis=1, keepdims=True)
        return E / np.where(n == 0, 1.0, n)


class EmbeddingExtractor(BaseExtractor):
    """
    Embed every sentence of a posting and every skill name into one space;
    a skill scores as the maximum cosine similarity to any sentence.

    Scores are cached to disk: embedding ~1.5M sentences is the slowest step
    in the pipeline and does not change between runs.
    """
    threshold_grid = np.arange(0.20, 0.91, 0.05, dtype=np.float32)

    def __init__(self, vocab: list[str], backend, cache_dir: Path | None = None,
                 chunk: int = 2000):
        self.vocab = vocab
        self.backend = backend
        self.cache_dir = cache_dir
        self.chunk = chunk
        self.name = f"emb_{backend.tag}"

    def fit(self, texts, Y, train_idx):
        return self                      # nothing supervised here

    def score(self, texts) -> np.ndarray:
        cache = (self.cache_dir / f"nlp_embscore_{self.backend.tag}.npy"
                 if self.cache_dir else None)
        if cache is not None and cache.exists():
            print(f"  {self.backend.tag}: loaded cached scores")
            return np.load(cache)

        sk = self.backend.encode(self.vocab)
        out = np.zeros((len(texts), len(self.vocab)), dtype=np.float16)
        t0 = time.time()
        for start in range(0, len(texts), self.chunk):
            idx = range(start, min(start + self.chunk, len(texts)))
            sents, owner = [], []
            for k, i in enumerate(idx):
                ss = sentences(texts[i])
                sents += ss
                owner += [k] * len(ss)
            if not sents:
                continue
            sims = self.backend.encode(sents) @ sk.T
            owner = np.array(owner)
            uniq, starts = np.unique(owner, return_index=True)
            out[np.array(idx)[uniq]] = np.maximum.reduceat(
                sims, starts, axis=0).astype(np.float16)
            if start and start % 20000 == 0:
                print(f"    embedded {start:,}/{len(texts):,}  ({time.time() - t0:.0f}s)")
        if cache is not None:
            np.save(cache, out)
        print(f"  {self.backend.tag}: embedded in {time.time() - t0:.0f}s")
        return out


# ══════════════════════════════════════════════════════════════
# OaSIS MAPPING  (the integration bridge)
# ══════════════════════════════════════════════════════════════

def _toks(t: str) -> set:
    return {w.rstrip("s") for w in re.split(r"[^a-z0-9]+", t.lower()) if len(w) > 2}


def map_skills_to_oasis(vocab: list[str], descriptor_names: list[str],
                        descriptor_texts: list[str], encode,
                        match_threshold: float = 0.55,
                        lexical_threshold: float = 0.8):
    """
    Map the extracted market vocabulary onto the OaSIS descriptors by two
    routes, because neither alone is enough:

      lexical    token containment against the descriptor NAME, which catches
                 near-exact matches an embedding space can underrate
      semantic   cosine between the skill name and the descriptor's own
                 DEFINITION text, which carries far more signal than the name

    Returns (descriptor_index, cosine, match_type, mapped) as numpy arrays.
    Skills matching neither route are the enrichment finding: in demand in the
    labour market, absent from the taxonomy.
    """
    sims = encode(vocab) @ encode(descriptor_texts).T
    best = sims.argmax(axis=1)
    best_cos = sims.max(axis=1)

    dtoks = [_toks(n) for n in descriptor_names]
    lex_best = np.full(len(vocab), -1)
    lex_score = np.zeros(len(vocab))
    for i, s in enumerate(vocab):
        st = _toks(s)
        if not st:
            continue
        sc = [len(st & dt) / min(len(st), len(dt)) if dt else 0.0 for dt in dtoks]
        j = int(np.argmax(sc))
        lex_best[i], lex_score[i] = j, sc[j]

    use_lex = lex_score >= lexical_threshold
    final = np.where(use_lex, lex_best, best)
    mapped = use_lex | (best_cos >= match_threshold)
    match_type = np.where(use_lex, "lexical", np.where(mapped, "semantic", "unmapped"))
    return final, best_cos, match_type, mapped
