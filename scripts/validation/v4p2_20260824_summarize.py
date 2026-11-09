"""
v4p2 reviewer-response summary table.

Collects the four held-out rollout JSONs produced by the run_chain.sh evals
and prints the comparison table for the response letter / manuscript 4.1:

  A  sim-split model      on 6 held-out sims (*_004)
  A' published frame-split model on the same 6 sims
  B  type-holdout model   on 5 unseen-type sims (sph_sphere_bounce)
  B' published model      on the same 5 sims (seen during its training)

Usage:  python scripts/validation/v4p2_20260824_summarize.py
"""
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
LOG = ROOT / "logs" / "v4p2_20260824"

ROWS = [
    ("A  sim-split (unseen sims)", "holdout_hybrid_simsplit_hybrid_*.json"),
    ("A' frame-split ref (same sims)", "holdout_ref_framesplit_on_004.json"),
    ("B  type-holdout (unseen type)", "holdout_hybrid_typeout_bounce_*.json"),
    ("B' frame-split ref (seen type)", "holdout_ref_framesplit_on_bounce.json"),
]
KEYS = ("vel_mse", "pos_mse", "ke_dev", "mom_drift", "domain_frac")


def find(pattern):
    hits = sorted(LOG.glob(pattern))
    return hits[-1] if hits else None


def main():
    print("%-34s %10s %10s %10s %10s %9s" % (
        "config", "vel_mse", "pos_mse", "|KE-1|", "mom", "domain%"))
    print("-" * 88)
    rows = {}
    for label, pat in ROWS:
        f = find(pat)
        if f is None:
            print("%-34s %s" % (label, "(missing: %s)" % pat))
            continue
        d = json.loads(f.read_text())
        s = d["stats"]
        rows[label] = s
        print("%-34s %10.3f %10.4f %10.4f %10.4f %8.2f%%" % (
            label, s["vel_mse"]["mean"], s["pos_mse"]["mean"],
            s["ke_dev"]["mean"], s["mom_drift"]["mean"],
            s["domain_frac"]["mean"] * 100))

    if len(rows) == 4:
        a, ar = rows[ROWS[0][0]], rows[ROWS[1][0]]
        b, br = rows[ROWS[2][0]], rows[ROWS[3][0]]
        print("\nrelative gaps (model vs published ref, same sims):")
        for k in KEYS:
            ga = (a[k]["mean"] / ar[k]["mean"] - 1) * 100 if ar[k]["mean"] else 0
            gb = (b[k]["mean"] / br[k]["mean"] - 1) * 100 if br[k]["mean"] else 0
            print("  %-12s  sim-split %+7.1f%%   type-holdout %+7.1f%%" % (k, ga, gb))


if __name__ == "__main__":
    main()
