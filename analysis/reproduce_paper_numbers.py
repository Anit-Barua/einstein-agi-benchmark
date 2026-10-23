"""Recompute every number quoted in the paper from the CSV/JSON files in ../results.
Run:  python analysis/reproduce_paper_numbers.py
"""
import json, pathlib, numpy as np, pandas as pd
from scipy import stats
R = pathlib.Path(__file__).resolve().parents[1] / "results"
core = pd.read_csv(R / "results_core_all_models.csv")
ebr = core[core.model.str.contains("EBR")].copy()
loc = core[~core.model.str.contains("EBR")]
fict = pd.read_csv(R / "results_contamination.csv")
st = json.load(open(R / "results_statistical_tests.json"))

print("== Oracle (EBR), n =", len(ebr))
print("mean %.2f  SD %.2f" % (ebr.total_score.mean(), ebr.total_score.std(ddof=1)))
print(ebr.groupby("domain").total_score.agg(["count", "mean", "std"]).round(2))
print(ebr[["logical_coherence", "novel_synthesis", "mathematical_accuracy", "extrapolation"]].mean().round(2).to_dict())
print("items scoring exactly 10:", int((ebr.total_score == 10).sum()))

print("\n== What drives the score (EBR)")
for col in ["depth_depth_score", "sai", "entropy", "depth_word_count"]:
    r, p = stats.spearmanr(ebr[col], ebr.total_score)
    print(f"Spearman({col}, total) = {r:+.2f}  p = {p:.3g}")
print("items with SAI >= 1/6 (sai5 saturated):", int((ebr.sai >= 1/6).sum()), "of", len(ebr))
rng = np.random.default_rng(20250106)
for col in ["depth_depth_score", "sai"]:
    bs = []
    for _ in range(10000):
        i = rng.integers(0, len(ebr), len(ebr))
        bs.append(stats.spearmanr(ebr[col].values[i], ebr.total_score.values[i])[0])
    print(f"bootstrap 95% CI for rho({col}) =", np.nanpercentile(bs, [2.5, 97.5]).round(2))
X = np.column_stack([np.ones(len(ebr)),
                     (ebr.sai - ebr.sai.mean()) / ebr.sai.std(),
                     (ebr.depth_depth_score - ebr.depth_depth_score.mean()) / ebr.depth_depth_score.std()])
b = np.linalg.lstsq(X, ebr.total_score.values, rcond=None)[0]
res = ebr.total_score.values - X @ b
print("standardised OLS: beta_SAI=%.2f beta_depth=%.2f R2=%.3f" % (b[1], b[2], 1 - (res**2).sum() / ((ebr.total_score - ebr.total_score.mean())**2).sum()))

print("\n== Equation check: recompute totals from logged signals (Eqs. 1-6)")
def total(r, sai_mode="real", dep_fixed=None):
    sai5 = min(5, 30 * r.sai) if sai_mode == "real" else sai_mode
    dep = min(5, .55 * r.depth_step_count + .35 * r.depth_formula_count) if dep_fixed is None else dep_fixed
    ent = min(5, max(0, .75 * (r.entropy - 1.5))); lam = min(1, r.depth_word_count / 120)
    cl = lambda x: min(5, max(0, x))
    return (round(cl(lam*(.7*dep+.3*ent))) + round(cl(lam*(.35*sai5+.3*ent+.35*dep)))
            + round(cl(lam*(.75*dep+.25*sai5))) + round(cl(lam*(.4*sai5+.4*dep+.2*ent))))
p = ebr.apply(total, axis=1)
print("exact matches: %d/%d  MAE %.3f  Spearman %.3f" % ((p == ebr.total_score).sum(), len(ebr), np.abs(p - ebr.total_score).mean(), stats.spearmanr(p, ebr.total_score)[0]))
for label, mode in [("SAI fixed at 5", 5), ("SAI set to 0", 0)]:
    q = ebr.apply(lambda r: total(r, mode), axis=1)
    print(f"{label}: Spearman vs logged {stats.spearmanr(q, ebr.total_score)[0]:.3f}, mean shift {np.mean(q - ebr.total_score):+.2f}")
med = (ebr.depth_step_count * .55 + ebr.depth_formula_count * .35).clip(upper=5).median()
q = ebr.apply(lambda r: total(r, dep_fixed=med), axis=1)
print(f"depth fixed at median: Spearman vs logged {stats.spearmanr(q, ebr.total_score)[0]:.3f}, totals range {q.min()}-{q.max()}")

print("\n== Contamination probes")
print("fictional mean %.2f SD %.2f; core EBR mean %.2f; relative drop %.1f%%" % (
    fict.total_score.mean(), fict.total_score.std(ddof=1), ebr.total_score.mean(),
    100 * (ebr.total_score.mean() - fict.total_score.mean()) / ebr.total_score.mean()))
print("one-sided Mann-Whitney:", stats.mannwhitneyu(ebr.total_score, fict.total_score, alternative="greater"))

print("\n== Stored statistical tests (results_statistical_tests.json)")
print(json.dumps(st, indent=1))
print("\nSAI >= 1/6 among simulated local-model responses:", int((loc.sai >= 1/6).sum()), "of", len(loc))
