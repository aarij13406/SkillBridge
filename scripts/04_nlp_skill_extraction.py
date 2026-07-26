"""
SkillBridge — 04: NLP Skill Extraction
=======================================
Owner: Nalluraj Babu.  CSC 503 Data Mining, Summer 2026.

Free-text job postings in.  Structured competencies out.


WHAT THIS COMPONENT IS FOR
--------------------------
scripts/12_query.py states the limitation plainly: the recommender knows
exactly 181 OaSIS competencies and does not know "Python", "Excel" or
"patient care". A real person types those. This script is the bridge.


THE LABEL SOURCE, AND WHY IT IS THE STORY
------------------------------------------
Supervision comes from LinkedIn's own skill tags: the SILVER standard,
automatically produced and never checked by a human. Every score against them
carries a hidden question -- are we measuring the model, or the tags?

Section 6 answers it with a 100-posting GOLD set, annotated by reading the
postings without looking at the tags. Scoring the tags themselves against that
set gives micro-F1 ~0.33: they agree with a careful reader about a third of
the time, which caps what any model can score against them. That measurement,
not the headline F1, is this component's main result.


MODELS COMPARED  (skillbridge/nlp_models.py; same split, one shared scorer)
---------------------------------------------------------------------------
    gazetteer      verbatim word-boundary match, no learning
    tfidf          TF-IDF 1-2 grams -> one linear classifier per skill
    hybrid         mentioned AND scored salient          <-- best
    emb_sbert      Sentence-BERT, sentence-vs-skill cosine
    emb_tfidf_svd  corpus-trained embedding, no-pretraining ablation

DATA
----
    datasets/clean/linkedin_postings_canada.csv   metadata + province
    datasets/clean/linkedin_skills_canada.csv     silver labels
    datasets/raw/linkedin/job_summary.csv         the text (5 GB, cached)
    datasets/clean/descriptor_lookup.csv          the 181 OaSIS descriptors
    datasets/clean/oasis_skill_descriptions.csv   descriptor definitions
    datasets/clean/nlp_gold_sample.csv            100 hand-annotated postings

RUN
---
    python scripts/04_nlp_skill_extraction.py

First run streams the 5 GB summary file once and caches it; later runs reuse
the cache. Full run ~25-40 min, mostly the per-skill classifiers and the
corpus embedding.
"""

from __future__ import annotations

import re
import sys
import time
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import sparse

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from skillbridge.config import (CLEAN_DIR, DATASETS_DIR, RESULTS_DIR, SEED,
                                set_all_seeds)
from skillbridge.metrics import multilabel_report, save_result
from skillbridge.nlp_models import (EmbeddingExtractor, GazetteerExtractor,
                                    HybridExtractor, SbertBackend, SvdBackend,
                                    TfidfOvRExtractor, map_skills_to_oasis,
                                    tune_thresholds)

set_all_seeds(SEED)

# A few hundred per-skill classifiers hit their iteration cap: rare skills have
# ~35 positive examples and more iterations will not find signal that is not
# there. Expected, and silenced so the results stay readable.
from sklearn.exceptions import ConvergenceWarning
warnings.filterwarnings("ignore", category=ConvergenceWarning)

COMPONENT = "nlp_extraction"

MIN_DF        = 50      # a skill must be tagged on >= this many postings
TEST_FRAC     = 0.15
VAL_FRAC      = 0.15
MAX_SKILL_LEN = 60      # longer "skills" are sentence fragments
SBERT_MODEL   = "all-MiniLM-L6-v2"
SVD_DIMS      = 192

RAW_SUMMARY = DATASETS_DIR / "raw" / "linkedin" / "job_summary.csv"
CACHE       = CLEAN_DIR / "nlp_ca_postings.parquet"
GOLD_FILE   = CLEAN_DIR / "nlp_gold_sample.csv"


def hr(text: str) -> None:
    print(f"\n{'=' * 70}\n  {text}\n{'=' * 70}")


# ══════════════════════════════════════════════════════════════
# SECTION 1: WORKING DATASET
# ══════════════════════════════════════════════════════════════
# The clean extract carries metadata and province but not the posting text
# (17 MB for 56k postings; the text alone is ~200 MB). So the text comes from
# the raw file, streamed once and cached.

hr("SECTION 1: Working dataset")

if CACHE.exists():
    df = pd.read_parquet(CACHE)
    print(f"  loaded cache: {CACHE.name}  ({len(df):,} postings)")
else:
    import csv
    t0 = time.time()
    posts = pd.read_csv(CLEAN_DIR / "linkedin_postings_canada.csv")
    wanted = set(posts["job_link"])
    print(f"  clean postings      : {len(posts):,}")

    # csv.reader, not pandas: summaries contain embedded newlines and commas,
    # so a chunked read splits records across chunk boundaries.
    csv.field_size_limit(sys.maxsize)
    links, text = [], []
    with open(RAW_SUMMARY, newline="", encoding="utf-8", errors="replace") as fh:
        rdr = csv.reader(fh)
        next(rdr, None)
        for row in rdr:
            if len(row) >= 2 and row[0] in wanted:
                links.append(row[0])
                text.append(row[1])
    summ = pd.DataFrame({"job_link": links, "job_summary": text}).drop_duplicates("job_link")
    print(f"  summaries matched   : {len(summ):,}  ({time.time() - t0:.0f}s)")

    df = posts.merge(summ, on="job_link", how="inner")
    df["job_summary"] = (df["job_summary"].fillna("")
                         .str.replace(r"\s*show more\s*show less\s*$", "",
                                      regex=True, case=False)
                         .str.replace(r"[ \t]+", " ", regex=True).str.strip())
    df = df[df["job_summary"].str.len() > 0].copy()

    # ~16% of Canadian postings are French; a stopword-ratio test separates
    # them at no dependency cost.
    EN = frozenset("the and to of in for with you we our is are will as on be or at an".split())
    FR = frozenset("le la les de des et un une du en pour avec vous nous est sont au aux dans sur que qui".split())
    tok = re.compile(r"[a-zàâçéèêëîïôûùüÿœ]+")

    def detect_lang(t: str) -> str:
        w = tok.findall(t.lower()[:4000])
        if not w:
            return "other"
        en, fr = sum(x in EN for x in w), sum(x in FR for x in w)
        return "other" if en == fr == 0 else ("en" if en >= fr else "fr")

    df["language"] = df["job_summary"].map(detect_lang)
    df["province"] = df["province"].fillna("OTHER")
    df = df.reset_index(drop=True)
    df.to_parquet(CACHE, compression="zstd", index=False)
    print(f"  cached -> {CACHE.name}")

print(f"  postings with text  : {len(df):,}")
print(f"  language            : {df['language'].value_counts().to_dict()}")
print(f"  provinces           : {df['province'].nunique()}")


# ══════════════════════════════════════════════════════════════
# SECTION 2: LABEL SPACE
# ══════════════════════════════════════════════════════════════
# The ETL already collapsed spelling variants into `skill_clean`, so we
# lowercase and apply a frequency floor rather than re-normalising. 235k
# distinct strings for what should be a few thousand real skills is the first
# sign the tag source is noisy; most appear exactly once.

hr("SECTION 2: Label space")

skills = pd.read_csv(CLEAN_DIR / "linkedin_skills_canada.csv")
skills["skill"] = skills["skill_clean"].astype(str).str.lower().str.strip()
skills = skills[(skills["skill"].str.len() > 1) &
                (skills["skill"].str.len() <= MAX_SKILL_LEN) &
                (skills["job_link"].isin(set(df["job_link"])))]

doc_freq = skills.groupby("skill")["job_link"].nunique()
vocab = sorted(doc_freq[doc_freq >= MIN_DF].index)
skill_id = {s: j for j, s in enumerate(vocab)}
row_id = {l: i for i, l in enumerate(df["job_link"])}

kept = skills[skills["skill"].isin(skill_id)]
Y = sparse.csr_matrix(
    (np.ones(len(kept), dtype=np.int8),
     (kept["job_link"].map(row_id).to_numpy(), kept["skill"].map(skill_id).to_numpy())),
    shape=(len(df), len(vocab)))
Y.sum_duplicates()
Y.data[:] = 1
per_doc = np.asarray(Y.sum(axis=1)).ravel()

print(f"  distinct skill strings : {doc_freq.size:,}")
print(f"  appearing only once    : {int((doc_freq == 1).sum()):,}")
print(f"  label space (df >= {MIN_DF}) : {len(vocab):,}")
print(f"  tag coverage retained  : {len(kept) / len(skills):.1%}")
print(f"  labels per posting     : mean {per_doc.mean():.1f}, median {np.median(per_doc):.0f}")


# ══════════════════════════════════════════════════════════════
# SECTION 3: FROZEN SPLIT
# ══════════════════════════════════════════════════════════════
# Stratified by (language, province) so EN/FR and regional mixes match across
# splits -- otherwise the French gap in Section 5 could be an artefact of an
# unlucky partition rather than a property of the models.
#
# The gold-annotated postings are pinned to TEST first. They exist to audit
# held-out performance; if any landed in training, Section 6 would score the
# models on data they had already seen.

hr("SECTION 3: Split")

rng = np.random.default_rng(SEED)
gold_links = (set(pd.read_csv(GOLD_FILE, usecols=["job_link"])["job_link"])
              if GOLD_FILE.exists() else set())
is_gold = df["job_link"].isin(gold_links).to_numpy()
strat = (df["language"] + "|" + df["province"]).to_numpy()

train_idx, val_idx, test_idx = [], [], list(np.flatnonzero(is_gold))
for key in np.unique(strat):
    idx = np.flatnonzero((strat == key) & ~is_gold)
    rng.shuffle(idx)
    n_te = int(round(len(idx) * TEST_FRAC))
    n_va = int(round(len(idx) * VAL_FRAC))
    test_idx.extend(idx[:n_te])
    val_idx.extend(idx[n_te:n_te + n_va])
    train_idx.extend(idx[n_te + n_va:])
tr, va, te = (np.array(sorted(train_idx)), np.array(sorted(val_idx)),
              np.array(sorted(test_idx)))

if gold_links:
    print(f"  {int(is_gold.sum())} gold-annotated postings pinned to test")
print(f"  train {len(tr):,} | val {len(va):,} | test {len(te):,}  (seed {SEED})")
top_prov = df["province"].value_counts().index[0]
for nm, ix in (("train", tr), ("val", va), ("test", te)):
    sub = df.iloc[ix]
    print(f"    {nm:<5} en={(sub.language == 'en').mean():.1%}  "
          f"{top_prov}={(sub.province == top_prov).mean():.1%}")
assert not (set(tr) & set(te)) and not (set(va) & set(te)), "split leakage"
print("  no overlap between splits")

texts = df["job_summary"].str.lower().tolist()
Yva = Y[va].toarray().astype(bool)
dfreq = np.array([doc_freq[s] for s in vocab])
results: dict[str, dict] = {}
preds: dict[str, sparse.csr_matrix] = {}


def evaluate(name: str, pred: sparse.csr_matrix, extra: dict | None = None) -> dict:
    """Score on test with the shared harness, save the JSON, print one line."""
    rep = multilabel_report(Y[te].toarray(), pred[te].toarray())

    # language breakdown -- input for the fairness audit (Dharnesh)
    lang = df["language"].to_numpy()[te]
    for lg in ("en", "fr"):
        m = lang == lg
        if m.sum():
            rep[f"micro_f1_{lg}"] = multilabel_report(
                Y[te][m].toarray(), pred[te][m].toarray())["micro_f1"]

    # frequency bands -- where a model is weak (head vs tail skills)
    yt, yp = Y[te].toarray().astype(bool), pred[te].toarray().astype(bool)
    tp, pc, tc = (yt & yp).sum(axis=0), yp.sum(axis=0), yt.sum(axis=0)
    prec = np.divide(tp, np.maximum(pc, 1), where=pc > 0, out=np.zeros(len(vocab)))
    rec = np.divide(tp, np.maximum(tc, 1), where=tc > 0, out=np.zeros(len(vocab)))
    f1l = np.divide(2 * prec * rec, np.maximum(prec + rec, 1e-12),
                    where=(prec + rec) > 0, out=np.zeros(len(vocab)))
    for band, mask in (("head_df500+", dfreq >= 500),
                       ("mid_df100_499", (dfreq >= 100) & (dfreq < 500)),
                       ("tail_df50_99", dfreq < 100)):
        rep[f"macro_f1_{band}"] = float(f1l[mask].mean()) if mask.any() else None

    save_result(rep, COMPONENT, name, RESULTS_DIR, extra=extra)
    results[name], preds[name] = rep, pred
    print(f"  [{name:<14}] micro-F1={rep['micro_f1']:.4f}  macro-F1={rep['macro_f1']:.4f}  "
          f"P={rep['micro_precision']:.3f}  R={rep['micro_recall']:.3f}  "
          f"(en {rep.get('micro_f1_en', float('nan')):.3f} / "
          f"fr {rep.get('micro_f1_fr', float('nan')):.3f})")
    return rep


# ══════════════════════════════════════════════════════════════
# SECTION 4: RUN THE EXTRACTORS
# ══════════════════════════════════════════════════════════════
# Baseline first, then the two approaches the proposal names, then the hybrid
# that came out of their error analysis. Every cutoff is tuned on VALIDATION.

hr("SECTION 4: Extractors")

gaz = GazetteerExtractor(vocab).fit(texts, Y, tr)
S_gaz = gaz.score(texts)
evaluate("gazetteer", sparse.csr_matrix((S_gaz > 0.5).astype(np.int8)),
         extra={"note": "verbatim word-boundary match, no learning"})

clf = TfidfOvRExtractor(vocab, seed=SEED).fit(texts, Y, tr)
S_clf = clf.score(texts)
thr = tune_thresholds(S_clf[va].astype(np.float32), Yva, clf.threshold_grid)
evaluate("tfidf", sparse.csr_matrix((S_clf > thr[None, :].astype(np.float16)).astype(np.int8)),
         extra={"classifier": "SGD hinge, one-vs-rest", "n_classifiers": len(vocab)})

hyb = HybridExtractor(gaz, clf)
S_hyb = hyb.score(texts)
thr_h = tune_thresholds(S_hyb[va].astype(np.float32), Yva, hyb.threshold_grid,
                        candidate_val=hyb.candidates[va])
evaluate("hybrid", hyb.predict(S_hyb, thr_h),
         extra={"rule": "gazetteer candidate AND classifier score > per-skill threshold"})

# --- the two embedding backends ---------------------------------------
backends = []
try:
    backends.append(SbertBackend(SBERT_MODEL))
except Exception as exc:                                   # offline / not installed
    print(f"  ! Sentence-BERT unavailable ({exc}); ablation still runs")
backends.append(SvdBackend(clf.vec, clf.X[tr], SVD_DIMS, SEED))

emb_thr: dict[str, np.ndarray] = {}
embedders: dict[str, EmbeddingExtractor] = {}
for backend in backends:
    ex = EmbeddingExtractor(vocab, backend, cache_dir=CLEAN_DIR)
    S_emb = ex.score(texts)
    emb_thr[ex.name] = tune_thresholds(S_emb[va].astype(np.float32), Yva, ex.threshold_grid)
    embedders[ex.name] = ex
    evaluate(ex.name,
             sparse.csr_matrix((S_emb.astype(np.float32) > emb_thr[ex.name][None, :]).astype(np.int8)),
             extra={"backend": backend.tag, "match": "max cosine over sentences"})


# ══════════════════════════════════════════════════════════════
# SECTION 5: GOLD STANDARD -- HOW GOOD ARE THE LABELS?
# ══════════════════════════════════════════════════════════════
# 100 test-split postings annotated by reading each ad and recording the skills
# it genuinely asks for, without consulting LinkedIn's tags. Every model AND
# the tags themselves are scored against those annotations. The tags' own score
# is the ceiling: no model graded against them can do much better than they
# agree with reality.

hr("SECTION 5: Gold-standard evaluation")

if not GOLD_FILE.exists():
    print(f"  ! {GOLD_FILE.name} not found -- skipping")
else:
    gold = pd.read_csv(GOLD_FILE)
    by_key = {}
    for s, j in skill_id.items():
        by_key.setdefault(re.sub(r"[^a-z0-9+#]", "", s), j)

    def to_vocab(raw: str):
        g = re.sub(r"\s+", " ", raw.lower().strip())
        if g in skill_id:
            return skill_id[g]
        k = re.sub(r"[^a-z0-9+#]", "", g)
        if k in by_key:
            return by_key[k]
        if g.endswith(" skills") and g[:-7] in skill_id:
            return skill_id[g[:-7]]
        return skill_id.get(g + " skills")

    rows = np.array([row_id[l] for l in gold["job_link"] if l in row_id])
    Yg = np.zeros((len(rows), len(vocab)), dtype=bool)
    total = mapped_n = 0
    for r, (_, rec) in enumerate(gold[gold["job_link"].isin(row_id)].iterrows()):
        for raw in str(rec.get("gold_skills", "")).split(","):
            if not raw.strip():
                continue
            total += 1
            j = to_vocab(raw)
            if j is not None:
                mapped_n += 1
                Yg[r, j] = True

    def gold_score(pb):
        tp = int((pb & Yg).sum())
        p = tp / max(int(pb.sum()), 1)
        r = tp / max(int(Yg.sum()), 1)
        return {"micro_precision": round(p, 4), "micro_recall": round(r, 4),
                "micro_f1": round(2 * p * r / (p + r), 4) if p + r else 0.0}

    gold_out = {"n_docs": int(len(rows)), "gold_skill_mentions": total,
                "vocab_coverage_of_gold": round(mapped_n / max(total, 1), 4),
                "silver_tags": gold_score(Y[rows].toarray().astype(bool))}
    for name, P in preds.items():
        gold_out[name] = gold_score(P[rows].toarray().astype(bool))
    save_result(gold_out, COMPONENT, "gold_standard", RESULTS_DIR)

    print(f"  gold postings {gold_out['n_docs']} | {total} skill mentions | "
          f"{gold_out['vocab_coverage_of_gold']:.0%} expressible in the label space")
    print(f"  SILVER TAGS vs gold : micro-F1 {gold_out['silver_tags']['micro_f1']:.4f}"
          "   <-- the ceiling on any model graded against them")
    for k in preds:
        print(f"    {k:<14} {gold_out[k]['micro_f1']:.4f}")


# ══════════════════════════════════════════════════════════════
# SECTION 6: OaSIS BRIDGE  (the integration deliverable)
# ══════════════════════════════════════════════════════════════
# 12_query.py accepts only the 181 OaSIS descriptors. Skills matching none of
# them are the enrichment finding: in demand, absent from the taxonomy.

hr("SECTION 6: OaSIS mapping and handoff")

desc = pd.read_csv(CLEAN_DIR / "descriptor_lookup.csv").merge(
    pd.read_csv(CLEAN_DIR / "oasis_skill_descriptions.csv"),
    left_on="descriptor_name", right_on="skill_name", how="left")
desc["text"] = (desc["descriptor_name"] + ". " +
                desc["skill_description"].fillna("")).str.lower()
print(f"  OaSIS descriptors: {len(desc)}  "
      f"({desc['skill_description'].notna().sum()} with definition text)")

best_encoder = next((b for b in backends if b.tag == "sbert"), backends[-1])
final, best_cos, match_type, is_mapped = map_skills_to_oasis(
    vocab, desc["descriptor_name"].tolist(), desc["text"].tolist(),
    best_encoder.encode, match_threshold=0.55 if best_encoder.tag == "sbert" else 0.45)

mapping = pd.DataFrame({
    "skill": vocab, "doc_freq": dfreq,
    "descriptor_id": desc["descriptor_id"].to_numpy()[final],
    "descriptor_name": desc["descriptor_name"].to_numpy()[final],
    "cosine": np.round(best_cos, 4), "match_type": match_type, "mapped": is_mapped,
})
mapping.to_csv(RESULTS_DIR / "nlp_extraction__oasis_mapping.csv", index=False)

emerging = mapping[~mapping["mapped"]].sort_values("doc_freq", ascending=False)
emerging.to_csv(RESULTS_DIR / "nlp_extraction__emerging_skills.csv", index=False)

coo = preds["hybrid"].tocoo()
pd.DataFrame({
    "job_link": df["job_link"].to_numpy()[coo.row],
    "skill": np.array(vocab)[coo.col],
    "descriptor_name": np.where(is_mapped[coo.col],
                                mapping["descriptor_name"].to_numpy()[coo.col], ""),
}).to_csv(RESULTS_DIR / "nlp_extraction__extracted_skills.csv.gz",
          index=False, compression="gzip")   # ~30 MB raw, ~4 MB gzipped

print(f"  mapped to OaSIS : {int(is_mapped.sum()):,} / {len(vocab):,} ({is_mapped.mean():.0%})")
print(f"  emerging (no descriptor): {len(emerging):,}")
for _, r in emerging.head(10).iterrows():
    print(f"    {r['skill'][:34]:<36} df={r['doc_freq']:>6,}  "
          f"nearest: {r['descriptor_name'][:30]:<32} cos={r['cosine']:.2f}")
print(f"  handoff rows: {int(preds['hybrid'].sum()):,}")

save_result({"vocab_size": len(vocab), "mapped_to_oasis": int(is_mapped.sum()),
             "mapped_fraction": float(is_mapped.mean()),
             "emerging_skills": int(len(emerging)),
             "handoff_rows": int(preds["hybrid"].sum())},
            COMPONENT, "oasis_bridge", RESULTS_DIR,
            extra={"encoder": best_encoder.tag})


# ══════════════════════════════════════════════════════════════
# SUMMARY TABLE
# ══════════════════════════════════════════════════════════════

hr("SUMMARY (test split, silver standard)")
print(f"  {'model':<16}{'micro-P':>9}{'micro-R':>9}{'micro-F1':>10}{'macro-F1':>10}")
print("  " + "-" * 54)
for name, r in results.items():
    print(f"  {name:<16}{r['micro_precision']:>9.3f}{r['micro_recall']:>9.3f}"
          f"{r['micro_f1']:>10.4f}{r['macro_f1']:>10.4f}")
print(f"\n  results written to {RESULTS_DIR}")


# ============================================================
# SUMMARY -- NLP Skill Extraction (Nalluraj)
# for the final report. paste/rewrite from here.
# ============================================================
#
# TASK: extract structured competencies from the free-text job_summary field of
# Canadian LinkedIn postings and map them onto the 181-descriptor OaSIS
# taxonomy. Multi-label: ~2,300 possible skills, about 12 true per posting.
#
# DATA: 55,972 Canadian postings; ~53,900 carry both a summary and skill tags.
# 84% English, 16% French. Silver labels are LinkedIn's own tags: 227k distinct
# strings, most appearing once, reduced to a ~2,300-skill space at df >= 50
# (about 55% of tag occurrences).
#
# BASELINE: verbatim word-boundary matching, no learning. Recall ~0.53 shows
# most tagged skills really are written in the ad; precision ~0.13 shows that
# mention alone is nearly worthless, because ads name far more skills than they
# require.
#
# MODEL COMPARISON (test split, micro-F1):
#   gazetteer (baseline)     0.206     coverage, no judgement
#   tfidf + linear OvR       0.213     good on common skills, fails on rare
#   hybrid (mention+salient) 0.370     best; precision roughly triples
#   sentence-BERT            0.143     penalised by verbatim labels
#   tfidf-SVD ablation       0.150     isolates what pretraining buys
#
# The hybrid is the finding: neither component alone works. The gazetteer
# proposes candidates well but cannot judge them; the classifier judges well but
# proposes poorly, especially in the tail. Requiring both keeps the coverage and
# adds the discrimination.
#
# WHY SENTENCE-BERT LOSES TO A KEYWORD BASELINE: it matches meaning, and the
# silver tags are near-verbatim. Predicting "collaboration" where the ad writes
# "team player" is right by a human standard and wrong by this answer key. Its
# gains over the no-pretraining ablation land where lexical matching fails --
# rare skills and French postings -- consistent with that explanation rather
# than with the model being broken.
#
# THE TARGET (F1 > 0.6 on the silver standard) WAS NOT REACHED, and the gold set
# explains why rather than excusing it. Scoring LinkedIn's own tags against 100
# hand-annotated postings gives micro-F1 0.328: the labels agree with careful
# human reading about a third of the time. That is a ceiling on any model graded
# against them, and the best model (0.259 on gold) sits just under it. The 0.6
# target was set before label quality could be measured; per the team's stated
# policy we report the shortfall and its cause instead of revising the target.
# The model ranking is identical on gold and on silver, so the comparison
# between models remains valid even though the absolute values are depressed.
#
# A SECOND KIND OF UNLEARNABLE LABEL: some tags carry hundreds of postings and
# score exactly zero -- "in-person tutoring", "self-employed", "set own rates".
# These come from templated gig-platform ads where the tag describes the job
# ARRANGEMENT, which the summary text never states. No text model can predict a
# label the text does not contain.
#
# FAIRNESS SIGNAL (input for Dharnesh): French postings score roughly half of
# English across every model (hybrid 0.387 EN vs 0.208 FR), and the trained
# models widen the gap rather than narrowing it. The vocabulary is
# English-dominated.
#
# INTEGRATION (the dependency named in 12_query.py): only ~22% of the market
# skill vocabulary maps onto an OaSIS descriptor. The unmapped remainder is
# dominated by named tools, languages and credentials -- Excel, Microsoft
# Office, French, driver's licence -- because OaSIS describes ABILITIES while
# employers hire for TOOLS AND CREDENTIALS. That gap is the quantified
# open-vocabulary problem. Deliverables:
#     results/nlp_extraction__extracted_skills.csv.gz  posting -> skill -> descriptor
#     results/nlp_extraction__oasis_mapping.csv        skill -> descriptor + confidence
#     results/nlp_extraction__emerging_skills.csv      demand-ranked, no descriptor
#
# HONEST TAKEAWAY: micro-F1 0.370 against a label source that is itself only
# ~0.33 accurate. The bottleneck is label quality, not model capacity, and this
# component is the only one that built the instrument to prove it.
