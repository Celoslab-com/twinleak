# SPDX-License-Identifier: Apache-2.0
"""One command rebuilds every number and figure:  python -m twinleak.reproduce

For each target person, the twin sends a sequence of releases to a recipient. After every
release we record (a) the recipient's true posterior on the target in the real population and
(b) the person-side meter's estimate from a separate public reference population.
"""
import argparse, json, os
import numpy as np
from .population import generate
from .releases import make_releases
from .claims import extract
from .risk import KNOWS, claim_loglik, posterior_of, meter_estimate


def run(n_pop=20000, n_targets=300, n_rel=12, eps=1.0, seed=7, out="results"):
    os.makedirs(out, exist_ok=True)
    pop = generate(n_pop, seed)                 # the real (synthetic) population
    ref = generate(n_pop, seed + 1000)          # public reference: same generator, other draw
    rng = np.random.default_rng(seed + 1)
    targets = rng.choice(n_pop, n_targets, replace=False)
    rows = []
    for mode in ["raw", "generalise", "dp"]:
        for t in targets:
            texts = make_releases(pop, int(t), n_rel, mode, np.random.default_rng([seed, int(t)]), eps)
            claims = [c for s in texts for c in extract(s)]
            per_release = [extract(s) for s in texts]
            for K, known in KNOWS.items():
                ll_pop = np.zeros(pop.n)
                ll_ref = np.zeros(ref.n)
                for r, cl in enumerate(per_release, 1):
                    for attr, val in cl:
                        if attr in known:
                            ll_pop += claim_loglik(pop, attr, val, mode, eps)
                            ll_ref += claim_loglik(ref, attr, val, mode, eps)
                    p_true = posterior_of(ll_pop, int(t))
                    p_hat = meter_estimate(ll_pop[int(t)], ll_ref, pop.n)
                    rows.append((mode, K, int(t), r, p_true, p_hat))
    with open(os.path.join(out, "points.json"), "w") as f:
        json.dump({"columns": ["mode", "K", "target", "release", "p_true", "p_meter"], "rows": rows,
                   "config": dict(n_pop=n_pop, n_targets=n_targets, n_rel=n_rel, eps=eps, seed=seed)}, f)
    return rows


def summarise(rows, out="results", tol=0.10):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    arr = np.array([(r[3], r[4], r[5]) for r in rows])
    lines = ["| Mode | Recipient | Mean true risk after 12 releases | Share singled out (>0.5) | Meter mean abs. error | Meter within 0.10 |",
             "| --- | --- | --- | --- | --- | --- |"]
    summary = {}
    for mode in ["raw", "generalise", "dp"]:
        for K in KNOWS:
            sel = [r for r in rows if r[0] == mode and r[1] == K]
            fin = np.array([(r[4], r[5]) for r in sel if r[3] == max(x[3] for x in sel)])
            allp = np.array([(r[4], r[5]) for r in sel])
            err = np.abs(allp[:, 0] - allp[:, 1])
            s = dict(final_true=float(fin[:, 0].mean()), singled=float((fin[:, 0] > .5).mean()),
                     mae=float(err.mean()), within=float((err <= tol).mean()))
            summary[f"{mode}/{K}"] = s
            lines.append(f"| {mode} | {K} | {s['final_true']:.3f} | {s['singled']:.1%} | {s['mae']:.3f} | {s['within']:.1%} |")
    allerr = np.abs(arr[:, 1] - arr[:, 2])
    lines.append("")
    lines.append(f"All points: n={len(arr)}, meter within {tol:.2f} of true risk in {(allerr <= tol).mean():.1%}, MAE {allerr.mean():.3f}.")
    table = "\n".join(lines)
    with open(os.path.join(out, "summary.md"), "w") as f:
        f.write(table + "\n")
    with open(os.path.join(out, "summary.json"), "w") as f:
        json.dump(summary, f, indent=2)

    fig, axes = plt.subplots(1, 2, figsize=(10, 4))
    for mode, c in zip(["raw", "generalise", "dp"], ["#c0392b", "#2c7fb8", "#41ab5d"]):
        sel = [r for r in rows if r[0] == mode and r[1] == "K2"]
        R = sorted(set(r[3] for r in sel))
        axes[0].plot(R, [np.mean([r[4] for r in sel if r[3] == x]) for x in R], color=c, label=f"{mode}: true")
        axes[0].plot(R, [np.mean([r[5] for r in sel if r[3] == x]) for x in R], color=c, ls="--", label=f"{mode}: meter")
    axes[0].set(xlabel="Releases to recipient", ylabel="Mean P(recipient singles person out)",
                title="Risk accumulates across releases (recipient K2)")
    axes[0].legend(fontsize=7, frameon=False)
    axes[1].scatter(arr[:, 1], arr[:, 2], s=2, alpha=.15, color="#555")
    axes[1].plot([0, 1], [0, 1], color="k", lw=.8)
    axes[1].set(xlabel="True posterior (real population)", ylabel="Meter estimate (reference population)",
                title="Person-side meter vs. truth")
    fig.tight_layout()
    fig.savefig(os.path.join(out, "risk.png"), dpi=150)
    print(table)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--quick", action="store_true", help="small run for a smoke test")
    a = ap.parse_args()
    rows = run(n_targets=40, n_pop=5000) if a.quick else run()
    summarise(rows)


if __name__ == "__main__":
    main()
