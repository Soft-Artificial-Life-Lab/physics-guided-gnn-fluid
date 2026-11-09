"""
v5p2 result aggregation - seed statistics and the wall-offset sweep curve.

Collects every rollout JSON produced for the Water-3D study (seed-42 runs from
v5/v5p1 plus the seed-43/44 repeats and the offset sweep from v5p2) and prints
two tables ready for the paper:

  1. configuration comparison, mean +/- std over seeds, at 77 and 400 steps;
  2. wall-offset sweep (margin -> escape / KE deviation), the regime curve.

Usage:  python scripts/experiments/v5p2_20260922_summarize.py
"""
import json
import statistics as st
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
V5 = ROOT / "logs" / "v5_20260820_022909"
V5P1 = ROOT / "logs" / "v5p1_20260903"
V5P2 = ROOT / "logs" / "v5p2_20260922"

# configuration -> {seed: {steps: json path}}
CONFIGS = {
    "제안 (보정 벽 2겹 + 하이브리드)": {
        42: {77: V5 / "rollout_wall2_e100_77step_tol.json",
             400: V5 / "rollout_wall2_e100_400step_tol.json"},
        43: {77: V5P2 / "rollout_w3d_wall2_e100_s43_77step.json",
             400: V5P2 / "rollout_w3d_wall2_e100_s43_400step.json"},
        44: {77: V5P2 / "rollout_w3d_wall2_e100_s44_77step.json",
             400: V5P2 / "rollout_w3d_wall2_e100_s44_400step.json"},
    },
    "백본 + 보정 벽 2겹": {
        42: {77: V5 / "rollout_wall2_bb_e100_77step_tol.json",
             400: V5 / "rollout_wall2_bb_e100_400step_tol.json"},
        43: {77: V5P2 / "rollout_w3d_wall2_bb_e100_s43_77step.json",
             400: V5P2 / "rollout_w3d_wall2_bb_e100_s43_400step.json"},
        44: {77: V5P2 / "rollout_w3d_wall2_bb_e100_s44_77step.json",
             400: V5P2 / "rollout_w3d_wall2_bb_e100_s44_400step.json"},
    },
    "GNS (거리 특징, 재구현)": {
        42: {77: V5P1 / "rollout_gns_gns_e100_20260904_001444_v5p1_20260903_77step.json",
             400: V5P1 / "rollout_gns_gns_e100_20260904_001444_v5p1_20260903_400step.json"},
        43: {77: V5P2 / "rollout_gns_e100_s43_77step.json",
             400: V5P2 / "rollout_gns_e100_s43_400step.json"},
        44: {77: V5P2 / "rollout_gns_e100_s44_77step.json",
             400: V5P2 / "rollout_gns_e100_s44_400step.json"},
    },
    "메타데이터 벽 1겹 (오배치)": {
        42: {77: V5 / "rollout_e100_77step_tol.json",
             400: V5 / "rollout_e100_400step_tol.json"},
        43: {77: V5P2 / "rollout_w3d_wall_e100_s43_77step.json",
             400: V5P2 / "rollout_w3d_wall_e100_s43_400step.json"},
        44: {77: V5P2 / "rollout_w3d_wall_e100_s44_77step.json",
             400: V5P2 / "rollout_w3d_wall_e100_s44_400step.json"},
    },
    "벽 없음": {
        42: {77: V5 / "rollout_nowall_e100_77step_tol.json",
             400: V5 / "rollout_nowall_e100_400step_tol.json"},
        43: {77: V5P2 / "rollout_w3d_nowall_e100_s43_77step.json",
             400: V5P2 / "rollout_w3d_nowall_e100_s43_400step.json"},
        44: {77: V5P2 / "rollout_w3d_nowall_e100_s44_77step.json",
             400: V5P2 / "rollout_w3d_nowall_e100_s44_400step.json"},
    },
}

# wall margin (metadata box outward offset) -> rollout json stem, seed 42, 2 layers.
# measured true container is ~0.009 outward, so 0.000/0.005 sit inside the fluid.
SWEEP = {
    0.000: "rollout_w3d_sweep_m000_e100",
    0.005: "rollout_w3d_sweep_m005_e100",
    0.010: None,  # published run lives in the v5 log directory
    0.015: "rollout_w3d_sweep_m015_e100",
    0.025: "rollout_w3d_sweep_m025_e100",
    0.040: "rollout_w3d_sweep_m040_e100",
}
KEYS = ("domain_frac", "ke_dev", "vel_mse", "pos_mse", "mom_drift")


def load(path):
    if path is None or not Path(path).exists():
        return None
    return json.loads(Path(path).read_text(encoding="utf-8"))["summary"]


def fmt(vals, pct=False, sig=3):
    if not vals:
        return "     -     "
    m = st.mean(vals)
    s = st.stdev(vals) if len(vals) > 1 else 0.0
    if pct:
        return "%5.2f ± %4.2f" % (m * 100, s * 100)
    if m >= 1:
        return "%6.4g ± %-.3g" % (m, s)
    return "%8.3g ± %-.2g" % (m, s)


def seed_table(steps):
    print("\n## %d스텝 (평균 ± 표준편차, 시드 42/43/44)" % steps)
    head = "%-28s %14s %16s %16s %16s %16s %s" % (
        "구성", "이탈률(%)", "|KE-1|", "vel MSE", "pos MSE", "운동량", "n")
    print(head)
    print("-" * len(head))
    for name, seeds in CONFIGS.items():
        cols, n = [], 0
        for k in KEYS:
            vals = []
            for _, paths in sorted(seeds.items()):
                s = load(paths.get(steps))
                if s:
                    vals.append(s[k]["mean"])
            n = max(n, len(vals))
            cols.append(fmt(vals, pct=(k == "domain_frac")))
        print("%-28s %14s %16s %16s %16s %16s %d" % (name, *cols, n))


def sweep_table():
    print("\n## 벽 오프셋 스윕 (2겹, 시드 42) - 실측 용기 = margin 0.009")
    head = "%9s %10s %14s %14s %14s %14s" % (
        "margin", "위치", "이탈77(%)", "이탈400(%)", "|KE-1|@400", "pos@400")
    print(head)
    print("-" * len(head))
    for margin, stem in sorted(SWEEP.items()):
        if stem is None:
            p77 = V5 / "rollout_wall2_e100_77step_tol.json"
            p400 = V5 / "rollout_wall2_e100_400step_tol.json"
        else:
            p77 = V5P2 / (stem + "_77step.json")
            p400 = V5P2 / (stem + "_400step.json")
        s77, s400 = load(p77), load(p400)
        where = "안쪽" if margin < 0.009 else "바깥"
        if s400 is None:
            print("%9.3f %10s %14s %14s %14s %14s"
                  % (margin, where, "-" if s77 is None else "%.2f" % (s77["domain_frac"]["mean"] * 100),
                     "(실행 중)", "-", "-"))
            continue
        print("%9.3f %10s %14.2f %14.2f %14.4g %14.4g"
              % (margin, where,
                 s77["domain_frac"]["mean"] * 100 if s77 else float("nan"),
                 s400["domain_frac"]["mean"] * 100,
                 s400["ke_dev"]["mean"], s400["pos_mse"]["mean"]))


if __name__ == "__main__":
    seed_table(77)
    seed_table(400)
    sweep_table()
