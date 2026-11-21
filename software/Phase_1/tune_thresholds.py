"""
Choose detection thresholds from the trainer's labels - and check them honestly.

  python compare_labels.py --pred ...\\all_reps_predicted.csv --trainer ...\\trainer_labels.csv --out outputs\\matched_reps.csv
  python tune_thresholds.py outputs\\matched_reps.csv

For every error the trainer uses, and every per-rep metric that could detect it, it prints:
  AUC          how well the metric separates reps WITH the error from reps WITHOUT it  (0.5 = not at all, 1.0 = perfectly)
  current      precision / recall of the rule your analyzer uses now (approximated from the per-rep metric)
  best         the cut-off with the best F1 on ALL reps - optimistic, it is tuned and tested on the same reps
  held-out     leave-one-participant-out: the cut-off is chosen on the OTHER participants and applied to the held-out
               one; the pooled result is the number to QUOTE in a paper (the 'best' line is not).
"""
import argparse
import os
import re

import numpy as np
import pandas as pd

# trainer code -> [(metric column, direction, the analyzer's current limit)]   direction '>' = flag when metric is ABOVE the cut-off
RULES = {
    "FAST_LIFT":           [("m_peak_lift_vel", ">", 150)],
    "FAST_LOWER":          [("m_peak_lower_vel", ">", 150)],
    "LOWERED_TOO_QUICKLY": [("m_lower_s", "<", 1.0)],
    "TORSO_SWING":         [("m_peak_torso_vel", ">", 30), ("m_peak_back_vel", ">", 90)],
    "TORSO_LEAN":          [("m_torso_tilt_shift", ">", 8)],
    "TOO_HIGH":            [("peak_angle", ">", 100)],
    "NOT_HIGH_ENOUGH":     [("peak_angle", "<", 85)],
    "ELBOW_BENT":          [("min_elbow_angle", "<", 150), ("m_elbow_p10", "<", None)],
    "ELBOW_STRAIGHT":      [("m_elbow_median", ">", None), ("m_elbow_p90", ">", None), ("m_elbow_over170", ">", None)],
    "HEEL_LIFT":           [("m_heel_lift_max", ">", 0.10)],
    "FAST_DESCENT":        [("m_peak_descent_vel", ">", 200)],
    "BOUNCE":              [("m_bounce_out", ">", 60), ("m_bounce_in", "<", -60)],
}


def participant(v):
    v = re.sub(r"_[lr]$", "", str(v).lower())          # participant_101_ltr_l -> participant_101_ltr
    return re.sub(r"_[a-z]+$", "", v)                  #                       -> participant_101


def flag(x, thr, op):
    return x > thr if op == ">" else x < thr


def prf(tp, fp, fn):
    p = tp / (tp + fp) if tp + fp else np.nan
    r = tp / (tp + fn) if tp + fn else np.nan
    f = 2 * p * r / (p + r) if p == p and r == r and (p + r) else 0.0
    return p, r, f


def score(x, y, thr, op):
    f = flag(x, thr, op)
    return prf(int((f & y).sum()), int((f & ~y).sum()), int((~f & y).sum()))


def best_cut(x, y, op):
    u = np.unique(x)
    if len(u) > 120:
        u = np.unique(np.quantile(x, np.linspace(0.01, 0.99, 120)))
    if len(u) < 2:
        return None
    cands = (u[:-1] + u[1:]) / 2
    return max(cands, key=lambda t: score(x, y, t, op)[2])


def auc(x, y, op):
    s = x if op == ">" else -x
    r = pd.Series(s).rank().to_numpy()
    n1, n0 = int(y.sum()), int((~y).sum())
    return (r[y].sum() - n1 * (n1 + 1) / 2) / (n1 * n0)


def fmt(p, r, f):
    return f"precision {p:4.0%}  recall {r:4.0%}  F1 {f:.2f}" if p == p and r == r else "n/a"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("matched", help="matched_reps.csv written by compare_labels.py")
    ap.add_argument("--out", default=None, help="optional CSV of the suggestions")
    a = ap.parse_args()

    m = pd.read_csv(a.matched).fillna("")
    codes = m["trainer_errors"].astype(str).str.split(";")
    groups = m["video"].map(participant).to_numpy()
    print(f"{len(m)} reps, {len(set(groups))} participants\n")
    rows = []
    for code, rules in RULES.items():
        y = codes.map(lambda c: code in c).to_numpy()
        if y.sum() < 3 or (~y).sum() < 3:
            continue
        for col, op, current in rules:
            if col not in m.columns:
                continue
            x = pd.to_numeric(m[col], errors="coerce").to_numpy()
            ok = ~np.isnan(x)
            if ok.sum() < 10 or y[ok].sum() < 3:
                continue
            xv, yv, gv = x[ok], y[ok], groups[ok]
            n_part = len(set(gv[yv]))
            print(f"{code}   <- {col}  (flag when {op} cut-off)   trainer-positive: {int(yv.sum())} of {len(yv)} reps, "
                  f"{n_part} participant(s)")
            a_ = auc(xv, yv, op)
            print(f"   AUC {a_:.2f}" + ("   (barely separates the groups - don't expect a good rule from this metric)" if a_ < 0.6 else ""))
            if current is not None:
                print(f"   current ({op} {current:g}):  {fmt(*score(xv, yv, current, op))}")
            cut = best_cut(xv, yv, op)
            if cut is None:
                print()
                continue
            print(f"   best    ({op} {cut:.3g}):  {fmt(*score(xv, yv, cut, op))}   <- tuned on these same reps")
            tp = fp = fn = skipped = 0
            for g in sorted(set(gv)):                      # leave one participant out
                tr_, te_ = gv != g, gv == g
                if yv[tr_].sum() < 2 or (~yv[tr_]).sum() < 2:
                    skipped += 1
                    continue
                c_ = best_cut(xv[tr_], yv[tr_], op)
                f_ = flag(xv[te_], c_, op)
                tp += int((f_ & yv[te_]).sum()); fp += int((f_ & ~yv[te_]).sum()); fn += int((~f_ & yv[te_]).sum())
            hp, hr, hf = prf(tp, fp, fn)
            print(f"   held-out participants:  {fmt(hp, hr, hf)}   <- the honest number" + (f"  ({skipped} skipped)" if skipped else ""))
            print()
            rows.append({"code": code, "metric": col, "op": op, "current": current, "best_cut": round(float(cut), 4),
                         "auc": round(a_, 3), "heldout_precision": round(hp, 3), "heldout_recall": round(hr, 3),
                         "heldout_f1": round(hf, 3)})
    if a.out and rows:
        os.makedirs(os.path.dirname(a.out) or ".", exist_ok=True)
        pd.DataFrame(rows).to_csv(a.out, index=False)
        print("saved:", a.out)


if __name__ == "__main__":
    main()