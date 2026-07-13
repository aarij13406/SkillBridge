"""Smoke-test the harness on synthetic data shaped like enriched OaSIS."""
import sys; sys.path.insert(0, '.')
import numpy as np, pandas as pd
from skillbridge.splits import split_edges, build_query_sets, assert_no_edge_leakage
from skillbridge.metrics import ranking_report
from skillbridge.baselines import get_baselines

rng = np.random.default_rng(42)
N_OCC, N_DESC = 900, 210          # enriched OaSIS shape

# Simulate: 5 latent "occupation families", descriptors load on families.
K = 6
occ_f  = rng.random((N_OCC, K))
desc_f = rng.random((N_DESC, K))
latent = occ_f @ desc_f.T
latent = (latent - latent.min()) / (latent.max() - latent.min())

rows = []
for o in range(N_OCC):
    # each occupation rated on a random ~65% of descriptors (sparse!)
    ds = rng.choice(N_DESC, size=int(N_DESC*0.65), replace=False)
    for d in ds:
        r = int(np.clip(round(latent[o, d]*4 + 1 + rng.normal(0, 0.6)), 1, 5))
        rows.append((o, d, r))

edges = pd.DataFrame(rows, columns=["occupation_id","descriptor_id","rating"])
print(f"Synthetic graph: {len(edges):,} edges | density = {len(edges)/(N_OCC*N_DESC):.1%}")
print(f"Rating distribution: {edges['rating'].value_counts().sort_index().to_dict()}")

train, test = split_edges(edges, test_frac=0.10, seed=42)
assert_no_edge_leakage(train, test)

queries = build_query_sets(train, test, np.arange(N_DESC))
print(f"Eval queries (occupations with held-out edges): {len(queries)}")

print(f"\n{'Model':<18} {'AUC':>7} {'P@5':>7} {'P@10':>7} {'R@10':>7} {'MRR':>7} {'NDCG@10':>8}")
print("-"*70)
for m in get_baselines(seed=42):
    m.fit(train, N_OCC, N_DESC)
    yt = [q["y_true"] for q in queries]
    yg = [q["y_graded"] for q in queries]
    ys = [m.score(q["occupation_id"], q["candidates"]) for q in queries]
    r = ranking_report(yt, ys, yg, k_values=(5,10))
    print(f"{m.name:<18} {r['roc_auc']:>7.4f} {r['precision@5']:>7.4f} "
          f"{r['precision@10']:>7.4f} {r['recall@10']:>7.4f} {r['mrr']:>7.4f} {r['ndcg@10']:>8.4f}")
