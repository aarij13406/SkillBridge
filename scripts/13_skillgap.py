"""
scripts/13_skillgap.py  -  Skill-Gap Recommender  (Anand Anto)  [ALL-IN-ONE]
============================================================================
SkillBridge, Component 4.  CSC 503 Data Mining, Summer 2026.

Given a person's current job, a target job, and (optionally) their own
skills, this ranks what they should learn next, in TWO tracks:

  1. OaSIS competencies - matrix factorization (k=64, chosen by 5-fold CV)
     over the dense 900x181 occupation-competency matrix; personalized by
     cold-start fold-in; ranked by gap x leverage; each with a plain reason.
  2. Market tools       - the SAME gap logic over a second matrix built from
     real LinkedIn postings (occupation x market-skill demand), so it can
     recommend concrete tools (Python, SQL...) the OaSIS taxonomy omits.

Free-text skills the user types are mapped onto OaSIS descriptors through
Nalluraj's NLP bridge (the open-vocabulary integration).

MODES (extra command-line flags):
    python scripts/13_skillgap.py                default: evaluate + plots + demo
    python scripts/13_skillgap.py --ask          type your own job / skills
    python scripts/13_skillgap.py --cv-k         5-fold CV to choose k    (slow)
    python scripts/13_skillgap.py --cv-cut       5-fold CV of the @N cutoff (slow)
    python scripts/13_skillgap.py --market-eval  popularity vs MF vs item-item

REQUIRES:  numpy, pandas, scikit-learn, matplotlib
DATA (read from the shared data folder; tracks whose data is missing are
skipped gracefully, so the file is safe to run anywhere):
    oasis_descriptors_long.csv                             (required)
    nlp_oasis_mapping.csv  OR  results/nlp_extraction__oasis_mapping.csv
    linkedin_postings_canada.csv + linkedin_skills_canada.csv
        (market track; the market_edges/skills caches auto-build on first run)
"""

from __future__ import annotations
import sys
import re
from pathlib import Path
import numpy as np
import pandas as pd

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))

from skillbridge.config import SEED, set_all_seeds
from skillbridge import models
from skillbridge.baselines import MostCommonMissingRecommender
from skillbridge.splits import leave_one_skill_out
from skillbridge.metrics import recall_at_k, ndcg_at_k

def _find_data_dir():
    """Locate the shared data folder across common project layouts."""
    for c in (REPO.parent / "data" / "processed", REPO / "data" / "processed",
              REPO / "datasets" / "clean", REPO.parent / "datasets" / "clean"):
        if (c / "oasis_descriptors_long.csv").exists():
            return c
    return REPO.parent / "data" / "processed"        # sensible default

DATA_DIR = _find_data_dir()
FIG_DIR = REPO / "figures"; FIG_DIR.mkdir(exist_ok=True)

# ════════════════════════════════════════════════════════════════════
#  CONTROL PANEL  -  change these three lines for a live demo
# ════════════════════════════════════════════════════════════════════
CURRENT_JOB = "software developer"    # the job the person does NOW
TARGET_JOB  = "data scien"            # the job they WANT to move into
MY_SKILLS   = ""                      # optional extra skills, comma-separated, e.g.
#                                       "team player, 3d modeling, accounting, python"
# ════════════════════════════════════════════════════════════════════

# Tuned by cross-validation (see --cv-k):
MF_K, MF_REG, MF_EPOCHS = 64, 0.02, 300
LEV_WEIGHT = 0.5                       # leverage bonus ("opens many doors")


# ════════════════════════════════════════════════════════════════════
# 1. DATA
# ════════════════════════════════════════════════════════════════════

def load_and_prepare():
    df, n_occ, n_desc, occ_names, desc_names, noc = models.load_raw(DATA_DIR)
    edges = df[["occupation_id", "descriptor_id", "rating"]].copy()

    held = {}
    for _t, occ, hd in leave_one_skill_out(edges, min_rating=4, seed=SEED):
        held[occ] = hd
    held_pairs = set(zip(held.keys(), held.values()))
    keys = list(zip(edges["occupation_id"], edges["descriptor_id"]))
    mask = np.array([k in held_pairs for k in keys])
    train_edges = edges[~mask].reset_index(drop=True)

    core = train_edges[train_edges["rating"] >= 4]
    current_core = core.groupby("occupation_id")["descriptor_id"].apply(set).to_dict()
    return (edges, train_edges, held, current_core,
            n_occ, n_desc, occ_names, desc_names)


def load_nlp_map():
    """Nalluraj's NLP bridge: market skill string -> OaSIS descriptor id.
    Looks in the data folder and in the repo's results/ folder.
    Returns {} if not found (the free-text mapping is then simply skipped)."""
    for path in (DATA_DIR / "nlp_oasis_mapping.csv",
                 REPO / "results" / "nlp_extraction__oasis_mapping.csv"):
        if path.exists():
            m = pd.read_csv(path)
            m = m[m["mapped"] == True]                            # noqa: E712
            return {str(s).strip().lower(): int(d)
                    for s, d in zip(m["skill"], m["descriptor_id"])}
    return {}


# ════════════════════════════════════════════════════════════════════
# 2. EVALUATION
# ════════════════════════════════════════════════════════════════════

def evaluate(scorer, held, current_core, n_desc, ks=(10,)):
    per_k = {k: [] for k in ks}; ndcg = []
    for occ, hd in held.items():
        have = current_core.get(occ, set())
        cands = np.array([d for d in range(n_desc) if d not in have])
        yt = (cands == hd).astype(int)
        s = scorer(occ, cands)
        for k in ks:
            per_k[k].append(recall_at_k(yt, s, k))
        ndcg.append(ndcg_at_k(yt, s, 10))
    return {k: np.array(v) for k, v in per_k.items()}, np.array(ndcg)


def bootstrap_ci(values, n_boot=2000, seed=SEED):
    rng = np.random.default_rng(seed)
    v = np.asarray(values, float)
    means = [rng.choice(v, size=len(v), replace=True).mean() for _ in range(n_boot)]
    return float(v.mean()), float(np.percentile(means, 2.5)), float(np.percentile(means, 97.5))


# ════════════════════════════════════════════════════════════════════
# 3. MODEL BUILDERS
# ════════════════════════════════════════════════════════════════════

def fit_mf(train_edges, n_occ, n_desc, occ_names, desc_names,
           k=MF_K, reg=MF_REG, epochs=MF_EPOCHS):
    rng = np.random.default_rng(SEED)
    R = np.zeros((n_occ, n_desc)); M = np.zeros((n_occ, n_desc))
    R[train_edges["occupation_id"], train_edges["descriptor_id"]] = train_edges["rating"]
    M[train_edges["occupation_id"], train_edges["descriptor_id"]] = 1.0
    dummy = pd.DataFrame({"occupation_id": [0], "descriptor_id": [0], "rating": [5.0]})
    mx = models.Matrix(n_occ, n_desc, train_edges, dummy, R, M, occ_names, desc_names, {})
    return models.MFRecommender(k=k, reg=reg, epochs=epochs).fit(mx, rng)


def build_leverage(train_edges, n_desc):
    cnt = train_edges[train_edges["rating"] >= 4]["descriptor_id"].value_counts()
    lev = np.zeros(n_desc); lev[cnt.index.values] = cnt.values
    return lev / lev.max()


# ════════════════════════════════════════════════════════════════════
# 4. THE RECOMMENDER  (person + target -> ranked skills, with reasons)
# ════════════════════════════════════════════════════════════════════

def find_job(occ_names, query):
    q = query.strip().lower()
    return next((o for o, nm in occ_names.items()
                 if isinstance(nm, str) and q in nm.lower()), None)


def suggest_jobs(occ_names, query, n=8):
    """When a typed job isn't found, offer close official titles.
    Matches on the first 4 letters of each word, so 'banker' finds 'Banking...'."""
    stems = [w[:4] for w in query.lower().split() if len(w) >= 3]
    out = []
    for nm in occ_names.values():
        if isinstance(nm, str) and any(s in nm.lower() for s in stems) and nm not in out:
            out.append(nm)
    return out[:n]


def map_user_skills(text, desc_names, nlp_map):
    """Turn free-text skills into OaSIS descriptor ids using THREE routes:
       1) exact/substring OaSIS descriptor name  (e.g. 'Leadership')
       2) Nalluraj's NLP mapping                 (e.g. 'team player' -> Coordinating...)
       3) otherwise: unmapped (emerging skill the taxonomy doesn't hold, e.g. 'Python')."""
    name_to_id = {str(v).strip().lower(): k for k, v in desc_names.items()}
    matched, unmapped = [], []
    for raw in [x.strip() for x in text.split(",") if x.strip()]:
        low = raw.lower()
        did = name_to_id.get(low)
        how = "exact"
        if did is None:                                  # loose OaSIS name match
            did = next((k for k, v in desc_names.items()
                        if low in str(v).strip().lower()), None)
        if did is None and low in nlp_map:               # NLP bridge
            did, how = nlp_map[low], "nlp"
        if did is None:
            unmapped.append(raw)
        else:
            matched.append((int(did), str(desc_names[did]).strip(), how))
    return matched, unmapped


def recommend(mf, edges, occ_names, desc_names, leverage, n_desc, nlp_map,
              current_query, target_query, extra_skills_text="", topk=5):
    current_job = find_job(occ_names, current_query)
    target_job = find_job(occ_names, target_query)
    missing = ([("current job", current_query)] if current_job is None else []) \
        + ([("target job", target_query)] if target_job is None else [])
    if missing:
        lines = []
        for label, q in missing:
            sugg = suggest_jobs(occ_names, q)
            hint = ("did you mean: " + "; ".join(sugg)) if sugg else "no close matches found"
            lines.append(f"Could not find a {label} matching '{q}'.\n    {hint}")
        raise SystemExit("\n  ".join(lines))

    cur = edges[(edges["occupation_id"] == current_job) & (edges["rating"] >= 4)]
    desc_ids = list(cur["descriptor_id"].to_numpy())
    ratings = list(cur["rating"].to_numpy())
    have = set(desc_ids)

    matched, unmapped = map_user_skills(extra_skills_text, desc_names, nlp_map)
    for did, _nm, _how in matched:
        if did not in have:
            desc_ids.append(did); ratings.append(5.0); have.add(did)

    p_u, b_u = models.fold_in(mf, np.array(desc_ids), np.array(ratings))

    rows = []
    for d in range(n_desc):
        if d in have:
            continue
        need = mf.score(np.array([target_job]), np.array([d]))[0]
        if need < 4.0:
            continue
        level = mf.mu + b_u + mf.bi[d] + float(p_u @ mf.Q[d])
        gap = need - level
        rows.append(dict(skill=str(desc_names.get(d, "?")).strip(),
                         need=float(need), level=float(max(level, 0.0)),
                         gap=float(gap), lev=float(leverage[d]),
                         score=float(gap + LEV_WEIGHT * leverage[d])))
    rows.sort(key=lambda r: -r["score"])
    meta = dict(current=occ_names.get(current_job), target=occ_names.get(target_job),
                matched=matched, unmapped=unmapped, n_current=len(cur))
    return rows[:topk], meta


# ════════════════════════════════════════════════════════════════════
# 4b. MARKET MODEL  (our skill-gap logic applied to LinkedIn demand data)
# ════════════════════════════════════════════════════════════════════
# OaSIS lists 181 abstract competencies, NOT concrete tools like Python or
# SQL. So we build a SECOND matrix from real LinkedIn postings:
#     rows    = occupations (aligned to OaSIS ids by title matching)
#     columns = the ~400 most common market skills/tools
#     value   = prevalence (how often that job's postings mention the skill), x5
# We then rank with a HYBRID model: gap = target demand - your current level,
# where your current level and the target demand come from OBSERVED data
# (honest zeros, never predicted), and an implicit-feedback MF only BACKFILLS
# demand for target jobs whose postings are too thin to show a needed tool.
#
# NOTE (an honest finding, see --market-eval): plain DENSE matrix factorization
# - which wins big on the DENSE OaSIS matrix - does NOT beat a popularity
# baseline on this SPARSE demand matrix, and it hallucinates skills a person
# does not have (e.g. crediting a nurse with Machine Learning). So we read the
# person's current skills from data and use MF only as a thin-data safety-net.

def build_market_matrix():
    """One-time build of market_edges.csv + market_skills.csv from LinkedIn."""
    posts_p = DATA_DIR / "linkedin_postings_canada.csv"
    skills_p = DATA_DIR / "linkedin_skills_canada.csv"
    occ_p = DATA_DIR / "occupation_lookup.csv"
    if not (posts_p.exists() and skills_p.exists() and occ_p.exists()):
        return False
    MIN_POSTS, MIN_DF, TOP = 15, 30, 400
    occ = pd.read_csv(occ_p)[["occupation_id", "occupation_name"]].drop_duplicates()
    posts = pd.read_csv(posts_p, usecols=["job_link", "job_title"]).dropna()
    skills = pd.read_csv(skills_p).dropna()

    def sing(t):
        return " ".join(w[:-1] if len(w) > 3 and w.endswith("s") else w
                        for w in re.findall(r"[a-z0-9+#.]+", str(t).lower()))
    SEP = re.compile(r",| and | including | except |/| - ")
    posts["ts"] = posts["job_title"].map(sing)

    occ_links = {}
    for oid, name in zip(occ["occupation_id"], occ["occupation_name"]):
        ph = sing(SEP.split(name)[0]).strip()
        if len(ph) < 4:
            continue
        m = posts["ts"].str.contains(ph, regex=False, na=False)
        if m.sum() >= MIN_POSTS:
            occ_links[int(oid)] = set(posts.loc[m, "job_link"])
    if not occ_links:
        return False
    kept = set().union(*occ_links.values())
    sk = skills[skills["job_link"].isin(kept)]
    vocab = [s for s, c in sk["skill_clean"].value_counts().items() if c >= MIN_DF][:TOP]
    sid = {s: j for j, s in enumerate(vocab)}
    by_link = sk[sk["skill_clean"].isin(sid)].groupby("job_link")["skill_clean"].apply(list).to_dict()
    rows = []
    for oid, links in occ_links.items():
        cnt = np.zeros(len(vocab)); nl = len(links)
        for lk in links:
            for s in by_link.get(lk, []):
                cnt[sid[s]] += 1
        for j, p in enumerate(cnt / nl):
            if p > 0:
                rows.append((oid, j, round(float(p), 4)))
    pd.DataFrame(rows, columns=["occupation_id", "skill_id", "prevalence"]).to_csv(
        DATA_DIR / "market_edges.csv", index=False)
    pd.DataFrame({"skill_id": range(len(vocab)), "skill": vocab}).to_csv(
        DATA_DIR / "market_skills.csv", index=False)
    return True


def load_market():
    ep, sp = DATA_DIR / "market_edges.csv", DATA_DIR / "market_skills.csv"
    if not (ep.exists() and sp.exists()):
        print("  building market matrix from LinkedIn (one-time)...")
        if not build_market_matrix():
            return None
    return pd.read_csv(ep), pd.read_csv(sp)["skill"].tolist()


def _market_R(edges, vocab):
    occ_ids = sorted(edges["occupation_id"].unique())
    row_of = {o: i for i, o in enumerate(occ_ids)}
    R = np.zeros((len(occ_ids), len(vocab)))
    for o, s, p in zip(edges["occupation_id"], edges["skill_id"], edges["prevalence"]):
        R[row_of[o], s] = p * 5.0
    return R, row_of


def _fit_market_mf(R):
    """Implicit-feedback MF: train on the POSITIVE demand cells only (not the
    dense grid). This is the correct MF for sparse demand data; used ONLY to
    backfill demand the target job's thin postings may have missed."""
    n, n_skill = R.shape
    nz = np.argwhere(R > 0)
    imp = pd.DataFrame({"occupation_id": nz[:, 0], "descriptor_id": nz[:, 1],
                        "rating": R[nz[:, 0], nz[:, 1]]})
    mx = models.Matrix(n, n_skill, imp,
                       pd.DataFrame({"occupation_id": [0], "descriptor_id": [0], "rating": [5.0]}),
                       R, np.ones((n, n_skill)), {}, {}, {})
    mf = models.MFRecommender(k=32, reg=0.02, epochs=200).fit(mx, np.random.default_rng(SEED))
    return mf


def market_recommend(edges, vocab, current_occ, target_occ, exclude=(), topk=8):
    """HYBRID model: the person's CURRENT level comes from OBSERVED data (honest
    zeros, never predicted), the TARGET demand is observed too, and an implicit
    MF only BACKFILLS demand where the target's postings are too thin to show it.
    Returns (rows, missing_reason). Each row carries src='data' or 'model'."""
    R, row_of = _market_R(edges, vocab)
    n, n_skill = R.shape
    if target_occ not in row_of:
        return None, "target"
    lev = (R >= 1).sum(axis=0); lev = lev / max(lev.max(), 1.0)

    mf = _fit_market_mf(R)                                   # backfill safety-net
    tr = row_of[target_occ]
    obs = R[tr]                                              # observed target demand
    have = R[row_of[current_occ]] if current_occ in row_of else np.zeros(n_skill)
    mfn = mf.score(np.full(n_skill, tr), np.arange(n_skill))            # model demand
    glob = np.array([mf.score(np.arange(n), np.full(n, j)).mean()       # skill's global avg
                     for j in range(n_skill)])
    exl = {e.strip().lower() for e in exclude}

    rows = []
    for j in range(n_skill):
        src, need = "data", obs[j]
        # backfill ONLY where observed demand is missing AND the model says the
        # skill is genuinely occupation-specific (not just globally popular).
        if obs[j] < 1.0 and mfn[j] >= 2.0 and (mfn[j] - glob[j]) >= 0.5:
            src, need = "model", float(mfn[j])
        if need < 1.5 or vocab[j].lower() in exl or have[j] >= need:
            continue
        rows.append(dict(skill=vocab[j], need=float(need), level=float(have[j]),
                         gap=float(need - have[j]), lev=float(lev[j]), src=src))
    rows.sort(key=lambda r: -(r["gap"] + 0.3 * r["lev"]))
    return rows[:topk], None


def market_eval(edges, vocab):
    """Honest comparison on the sparse demand matrix: popularity vs MF vs
    item-item. Shows that MF does NOT win here (unlike on OaSIS)."""
    R, _ = _market_R(edges, vocab)
    n, n_skill = R.shape
    rng = np.random.default_rng(SEED)
    held = {}
    for i in range(n):
        core = np.where(R[i] >= 2.0)[0]
        if len(core) >= 2:
            held[i] = int(rng.choice(core))

    def ev(scorer):
        rec, nd = [], []
        for i, h in held.items():
            have = set(np.where(R[i] >= 2.0)[0]) - {h}
            cands = np.array([j for j in range(n_skill) if j not in have])
            yt = (cands == h).astype(int); s = scorer(i, cands)
            rec.append(recall_at_k(yt, s, 10)); nd.append(ndcg_at_k(yt, s, 10))
        return np.mean(rec), np.mean(nd)

    pop = (R >= 2).sum(axis=0)
    B = (R >= 2).astype(float); coln = B / (np.linalg.norm(B, axis=0, keepdims=True) + 1e-9)
    S = coln.T @ coln; np.fill_diagonal(S, 0.0)

    def item(i, c):
        known = list(np.where(R[i] >= 2.0)[0])
        return S[np.ix_(c, known)].mean(axis=1) if known else np.zeros(len(c))

    Rtr = R.copy()
    for i, h in held.items():
        Rtr[i, h] = 0.0
    ii, jj = np.meshgrid(np.arange(n), np.arange(n_skill), indexing="ij")
    trdf = pd.DataFrame({"occupation_id": ii.ravel(), "descriptor_id": jj.ravel(), "rating": Rtr.ravel()})
    mx = models.Matrix(n, n_skill, trdf,
                       pd.DataFrame({"occupation_id": [0], "descriptor_id": [0], "rating": [5.0]}),
                       Rtr, np.ones((n, n_skill)), {}, {}, {})
    mf = models.MFRecommender(k=32, reg=0.05, epochs=200).fit(mx, np.random.default_rng(SEED))

    print(f"  evaluated on {len(held)} occupations (hide one in-demand tool, recover it):")
    print("    popularity baseline      R@10=%.3f  NDCG@10=%.3f" % ev(lambda i, c: pop[c]))
    print("    matrix factorization     R@10=%.3f  NDCG@10=%.3f" % ev(lambda i, c: mf.score(np.full(len(c), i), c)))
    print("    item-item co-occurrence  R@10=%.3f  NDCG@10=%.3f" % ev(item))
    print("  Finding: MF does NOT beat popularity on this SPARSE demand matrix,")
    print("  the opposite of its win on the DENSE OaSIS matrix. Right model, right data.")


# ════════════════════════════════════════════════════════════════════
# 5. PLOTS
# ════════════════════════════════════════════════════════════════════

def plot_metrics(results, path):
    import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
    names = list(results.keys())
    rm, rerr, nm, nerr = [], [[], []], [], [[], []]
    for n in names:
        rec10, ndcg = results[n]
        m, lo, hi = bootstrap_ci(rec10); rm.append(m); rerr[0].append(m-lo); rerr[1].append(hi-m)
        m, lo, hi = bootstrap_ci(ndcg);  nm.append(m); nerr[0].append(m-lo); nerr[1].append(hi-m)
    x = np.arange(len(names)); w = 0.36
    fig, ax = plt.subplots(figsize=(8, 4.6))
    for bars in (ax.bar(x-w/2, rm, w, yerr=rerr, capsize=4, label="Recall@10", color="#2E7D7D"),
                 ax.bar(x+w/2, nm, w, yerr=nerr, capsize=4, label="NDCG@10", color="#1F3A5F")):
        for b in bars:
            ax.text(b.get_x()+b.get_width()/2, b.get_height()+0.02, f"{b.get_height():.2f}",
                    ha="center", va="bottom", fontsize=8)
    ax.set_xticks(x); ax.set_xticklabels(names, rotation=10, ha="right", fontsize=8)
    ax.set_ylim(0, 1); ax.set_ylabel("score (0-1)")
    ax.set_title("Skill-Gap Recommender: model comparison (95% CI)")
    ax.legend(); ax.grid(axis="y", ls=":", alpha=.5)
    fig.tight_layout(); fig.savefig(path, dpi=140); plt.close(fig)
    print(f"  saved team plot   -> {path}")


def plot_user(rows, meta, path):
    import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
    rows = rows[::-1]
    skills = [r["skill"] for r in rows]; need = [r["need"] for r in rows]; level = [r["level"] for r in rows]
    y = list(range(len(skills)))
    fig, ax = plt.subplots(figsize=(8.4, 4.6))
    for i in y:
        ax.plot([level[i], need[i]], [i, i], color="#d0d0d0", lw=4, zorder=1)
        ax.text(need[i]+0.08, i, f"gap {need[i]-level[i]:.1f}", va="center", fontsize=8, color="#555")
    ax.scatter(level, y, color="#2E7D7D", s=90, zorder=2, label="Where you are")
    ax.scatter(need, y, color="#C0392B", s=90, zorder=2, label="What the job needs")
    ax.set_yticks(y); ax.set_yticklabels(skills, fontsize=9)
    ax.set_xlim(0, 5.6); ax.set_xlabel("importance (0 = not needed, 5 = essential)")
    ax.set_title(f"Skills to learn next\n{meta['current']}  ->  {meta['target']}", fontsize=11)
    ax.legend(loc="lower right", fontsize=8); ax.grid(axis="x", ls=":", alpha=.5)
    fig.tight_layout(); fig.savefig(path, dpi=140); plt.close(fig)
    print(f"  saved seeker plot -> {path}")


def plot_market_gap(rows, meta, path):
    """Dumbbell: how much your current field uses each tool vs the target's demand."""
    import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
    rows = rows[::-1]
    names = [r["skill"] for r in rows]; need = [r["need"] for r in rows]; lvl = [r["level"] for r in rows]
    y = list(range(len(names)))
    fig, ax = plt.subplots(figsize=(8.4, 4.8))
    for i in y:
        ax.plot([lvl[i], need[i]], [i, i], color="#d0d0d0", lw=4, zorder=1)
        ax.text(need[i] + 0.06, i, f"gap {need[i]-lvl[i]:.1f}", va="center", fontsize=8, color="#555")
    ax.scatter(lvl, y, color="#2E7D7D", s=90, zorder=2, label="Your current field uses it")
    ax.scatter(need, y, color="#C0392B", s=90, zorder=2, label="Target job demands it")
    ax.set_yticks(y); ax.set_yticklabels(names, fontsize=9)
    ax.set_xlim(0, 5.4); ax.set_xlabel("demand (0 = not asked for, 5 = in nearly every posting)")
    ax.set_title(f"Tools & technologies to learn (from real job postings)\n"
                 f"{meta['current']}  ->  {meta['target']}", fontsize=10)
    ax.legend(loc="lower right", fontsize=8); ax.grid(axis="x", ls=":", alpha=.5)
    fig.tight_layout(); fig.savefig(path, dpi=140); plt.close(fig)
    print(f"  saved tools plot  -> {path}")


def safe_plot(fn, *a):
    try:
        fn(*a)
    except ImportError:
        print("  (install matplotlib to get the plots)")


# ════════════════════════════════════════════════════════════════════
# 6. CROSS-VALIDATION (optional, slow) -- both were used to tune the model
# ════════════════════════════════════════════════════════════════════

def cv_choose_k(edges, n_occ, n_desc, occ_names, desc_names, K=5, k_values=(32, 64, 96, 128)):
    """5-fold CV over each occupation's core skills, to choose k. Saves a plot."""
    key = list(zip(edges["occupation_id"].to_numpy(), edges["descriptor_id"].to_numpy()))
    occ_core = edges[edges["rating"] >= 4].groupby("occupation_id")["descriptor_id"].apply(list).to_dict()
    rng0 = np.random.default_rng(SEED)
    folds = {o: [p.tolist() for p in np.array_split(rng0.permutation(sk), K)]
             for o, sk in occ_core.items() if len(sk) >= K}
    table = {}
    for k in k_values:
        recs = []
        for f in range(K):
            held = {o: set(pt[f]) for o, pt in folds.items() if len(pt[f]) > 0}
            hp = {(o, d) for o, ds in held.items() for d in ds}
            tr = edges[~np.array([kp in hp for kp in key])].reset_index(drop=True)
            mf = fit_mf(tr, n_occ, n_desc, occ_names, desc_names, k=k, epochs=120)
            r = []
            for o, hidden in held.items():
                have = set(occ_core[o]) - hidden
                cands = np.array([d for d in range(n_desc) if d not in have])
                yt = np.array([1 if d in hidden else 0 for d in cands])
                r.append(recall_at_k(yt, mf.score(np.full(len(cands), o), cands), 10))
            recs.append(np.nanmean(r))
        table[k] = (float(np.mean(recs)), float(np.std(recs)))
        print(f"  k={k:<4} CV Recall@10 = {np.mean(recs):.3f} ± {np.std(recs):.3f}")
    best = max(table, key=lambda k: table[k][0])
    print(f"  best k = {best}")
    return table


def cv_cutoff(edges, n_occ, n_desc, occ_names, desc_names, seeds=(42, 7, 13, 21, 99), ns=(5, 10, 15, 20, 25)):
    """5-fold CV of the @N cutoff for the tuned model. Shows where Recall crosses 90%."""
    occ_core = edges[edges["rating"] >= 4].groupby("occupation_id")["descriptor_id"].apply(set).to_dict()
    key = list(zip(edges["occupation_id"].to_numpy(), edges["descriptor_id"].to_numpy()))
    agg = {N: [] for N in ns}
    for sd in seeds:
        held = {}
        for _t, o, hd in leave_one_skill_out(edges, min_rating=4, seed=sd):
            held[o] = hd
        hp = set(zip(held.keys(), held.values()))
        tr = edges[~np.array([k in hp for k in key])].reset_index(drop=True)
        cc = tr[tr["rating"] >= 4].groupby("occupation_id")["descriptor_id"].apply(set).to_dict()
        mf = fit_mf(tr, n_occ, n_desc, occ_names, desc_names)
        per = {N: [] for N in ns}
        for o, hd in held.items():
            have = cc.get(o, set()); cands = np.array([d for d in range(n_desc) if d not in have])
            yt = (cands == hd).astype(int); s = mf.score(np.full(len(cands), o), cands)
            for N in ns:
                per[N].append(recall_at_k(yt, s, N))
        for N in ns:
            agg[N].append(np.mean(per[N]))
        print(f"  seed {sd} done")
    for N in ns:
        print(f"  Recall@{N:<2} = {np.mean(agg[N]):.3f} ± {np.std(agg[N]):.3f}")
    return agg


# ════════════════════════════════════════════════════════════════════
# 7. MAIN
# ════════════════════════════════════════════════════════════════════

def ask(prompt, default):
    try:
        v = input(f"  {prompt} [{default or 'none'}]: ").strip()
    except EOFError:
        v = ""
    return v or default


def _run():
    set_all_seeds(SEED)

    # ------------------------------------------------------------------
    # COMMAND-LINE FLAGS  (extra system arguments you can pass after the
    # filename; they are read from sys.argv - the words typed after it):
    #
    #   (no flag)  normal run: evaluate the model ladder, save both plots,
    #              and print the personalized demo.  ~15 seconds.
    #              (this is what the VS Code "Run" button does)
    #
    #   --ask      interactive: prompts you for the current job, the target
    #              job, and extra skills (comma-separated) instead of using
    #              the CONTROL PANEL defaults at the top of this file.
    #
    #   --cv-k     5-fold cross-validation over k = 32/64/96/128 to choose
    #              the number of latent factors.  SLOW (a few minutes).
    #
    #   --cv-cut   5-fold cross-validation of the @N cutoff = 5/10/15/20/25,
    #              shows where Recall crosses 90%.  SLOW (a few minutes).
    #
    #   --market-eval  compare popularity vs matrix factorization vs item-item
    #              on the LinkedIn demand matrix (shows MF is NOT the right model
    #              for the sparse market data - an honest finding).
    #
    # Examples:  python scripts/13_skillgap.py
    #            python scripts/13_skillgap.py --ask
    #            python scripts/13_skillgap.py --cv-k
    #            python scripts/13_skillgap.py --market-eval
    # ------------------------------------------------------------------
    args = sys.argv[1:]

    (edges, train_edges, held, current_core,
     n_occ, n_desc, occ_names, desc_names) = load_and_prepare()
    nlp_map = load_nlp_map()
    print(f"  occupations tested: {len(held)}   training ratings: {len(train_edges):,}")
    print(f"  NLP skill mappings loaded (open-vocabulary bridge): {len(nlp_map)}")

    if "--cv-k" in args:
        print("\n  == 5-fold CV to choose k ==")
        cv_choose_k(edges, n_occ, n_desc, occ_names, desc_names); return
    if "--cv-cut" in args:
        print("\n  == 5-fold CV of the @N cutoff ==")
        cv_cutoff(edges, n_occ, n_desc, occ_names, desc_names); return
    if "--market-eval" in args:
        print("\n  == Market-track model comparison (LinkedIn demand matrix) ==")
        m = load_market()
        if m:
            market_eval(m[0], m[1])
        else:
            print("  (LinkedIn market data not found)")
        return

    # ---- models ----
    mcm = MostCommonMissingRecommender().fit(train_edges, n_occ, n_desc)
    base_scorer = lambda occ, c: mcm.score(occ, c)
    mf = fit_mf(train_edges, n_occ, n_desc, occ_names, desc_names)
    mf_scorer = lambda occ, c: mf.score(np.full(len(c), occ), c)
    leverage = build_leverage(train_edges, n_desc)
    lev_scorer = lambda occ, c: mf.score(np.full(len(c), occ), c) + LEV_WEIGHT * leverage[c]

    # ---- ladder + confidence intervals ----
    print("\n  ── Model ladder (Recall@10, @15, @20 / NDCG@10, with 95% CI) ──")
    results = {}
    for name, sc in [("Most-common-missing", base_scorer),
                     ("Matrix factorization", mf_scorer),
                     ("MF + leverage", lev_scorer)]:
        recs, ndcg = evaluate(sc, held, current_core, n_desc, ks=(10, 15, 20))
        results[name] = (recs[10], ndcg)
        r10 = bootstrap_ci(recs[10]); nd = bootstrap_ci(ndcg)
        print(f"    {name:<22} R@10={r10[0]:.3f} [{r10[1]:.3f},{r10[2]:.3f}]  "
              f"R@15={recs[15].mean():.3f}  R@20={recs[20].mean():.3f}  "
              f"NDCG@10={nd[0]:.3f}")
    safe_plot(plot_metrics, results, FIG_DIR / "skillgap_metrics.png")

    # ---- personalized demo (with NLP-mapped skills) ----
    print("\n  ── Personalized recommendation ──")
    if "--ask" in args:
        cj = ask("Your current job", CURRENT_JOB)
        tj = ask("The job you want", TARGET_JOB)
        sk = ask("Extra skills you have (comma-separated)", MY_SKILLS)
    else:
        cj, tj, sk = CURRENT_JOB, TARGET_JOB, MY_SKILLS

    rows, meta = recommend(mf, edges, occ_names, desc_names, leverage, n_desc, nlp_map, cj, tj, sk)
    print(f"    Current job : {meta['current']}  ({meta['n_current']} strong skills)")
    print(f"    Dream job   : {meta['target']}")
    print(f"    Career move : {meta['current']}  →  {meta['target']}")
    if meta["matched"]:
        for _did, nm, how in meta["matched"]:
            tag = "  (NLP-mapped)" if how == "nlp" else ""
            print(f"      + counted your skill: {nm}{tag}")
    if meta["unmapped"]:
        print(f"      ~ not in the taxonomy, ignored: {', '.join(meta['unmapped'])}")
    print(f"\n    Top {len(rows)} OaSIS competencies to develop, with reasons:")
    for r in rows:
        print(f"      • {r['skill']}")
        print(f"          the target job needs it {r['need']:.1f}/5; you're around "
              f"{r['level']:.1f}/5 (a gap of {r['gap']:.1f}); "
              f"it's core in ~{int(r['lev']*100)}% of jobs.")
    safe_plot(plot_user, rows, meta, FIG_DIR / "skillgap_recommendation.png")

    # ---- market track: HYBRID gap model on LinkedIn demand data ----
    market = load_market()
    if market is not None:
        cur_occ, tgt_occ = find_job(occ_names, cj), find_job(occ_names, tj)
        already = [x for x in sk.split(",")] + [nm for _d, nm, _h in meta["matched"]]
        mrows, miss = market_recommend(market[0], market[1], cur_occ, tgt_occ, exclude=already)
        if mrows:
            print("\n    Tools & technologies to learn (from real job postings):")
            for r in mrows:
                tag = "  [model-filled]" if r.get("src") == "model" else ""
                print(f"      • {r['skill']}: target demand {r['need']:.1f}/5, "
                      f"your field ~{r['level']:.1f}/5 (gap {r['gap']:.1f}){tag}")
            safe_plot(plot_market_gap, mrows,
                      {"current": meta["current"], "target": meta["target"]},
                      FIG_DIR / "skillgap_market_tools.png")
        else:
            print(f"\n    (no LinkedIn market data for the {miss} job - tools track skipped)")


class _Tee:
    """Write everything printed to BOTH the terminal and a .txt file."""
    def __init__(self, path):
        self.file = open(path, "w", encoding="utf-8")
        self.stdout = sys.stdout

    def write(self, s):
        self.stdout.write(s); self.file.write(s)

    def flush(self):
        self.stdout.flush(); self.file.flush()


def main():
    # Mirror the whole terminal output into outputs/skillgap_output.txt
    out_dir = REPO / "outputs"; out_dir.mkdir(exist_ok=True)
    log_path = out_dir / "skillgap_output.txt"
    old_stdout = sys.stdout
    sys.stdout = _Tee(log_path)
    try:
        _run()
    finally:
        sys.stdout.file.close()
        sys.stdout = old_stdout
        print(f"  (full terminal output also saved to {log_path})")


if __name__ == "__main__":
    main()
