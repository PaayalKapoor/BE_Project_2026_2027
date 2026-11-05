"""
Compare PowerPulse predictions with a certified trainer's annotations.

  python compare_labels.py --pred outputs/all_reps_predicted.csv --trainer trainer_labels.csv

trainer_labels.csv columns:  video,start_s,end_s,label,errors[,arm]
  label  : correct | incorrect
  errors : ';'-separated codes (blank if correct), e.g.
           squat : LEAN_FORWARD, TOO_DEEP, HIP_FLEXION, ANKLE_MOBILITY, KNEES_FORWARD,
                   HIPS_BACK, WEIGHT_BACK, SHALLOW
           lunge : LEAN_FORWARD, KNEES_FORWARD, BACK_KNEE_HIGH, BACK_KNEE_LOW, TOO_DEEP, SHALLOW
           raise : ELBOW_BENT, TOO_HIGH, NOT_HIGH_ENOUGH, UNEVEN_RAISE
           (any exercise: OTHER for faults the system does not check)
  arm    : only for lateral raises. left | right | both  (both = same label for both arms;
           split into two rows if the arms differ). Left/right = the PERSON's left/right.

Reps are matched by time overlap (IoU), so it still works if the system
misses a rep or invents one.
"""
import argparse
import os
import pandas as pd


def norm_video(s):
    return os.path.splitext(os.path.basename(str(s)))[0].lower()


def errset(s):
    return {e.strip().upper() for e in str(s).split(";") if e.strip() and e.strip().lower() != "nan"}


def iou(a0, a1, b0, b1):
    inter = max(0.0, min(a1, b1) - max(a0, b0))
    union = max(a1, b1) - min(a0, b0)
    return inter / union if union > 0 else 0.0


def match(pred, trn, min_iou):
    cands = []
    for i, p in pred.iterrows():
        for j, t in trn.iterrows():
            v = iou(p.start_s, p.end_s, t.start_s, t.end_s)
            if v >= min_iou:
                cands.append((v, i, j))
    cands.sort(reverse=True)
    used_p, used_t, pairs = set(), set(), []
    for v, i, j in cands:
        if i in used_p or j in used_t:
            continue
        used_p.add(i); used_t.add(j); pairs.append((i, j, v))
    return pairs, set(pred.index) - used_p, set(trn.index) - used_t


def prf(tp, fp, fn):
    p = tp / (tp + fp) if tp + fp else float("nan")
    r = tp / (tp + fn) if tp + fn else float("nan")
    f = 2 * p * r / (p + r) if p == p and r == r and (p + r) else float("nan")
    return p, r, f


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pred", required=True)
    ap.add_argument("--trainer", required=True)
    ap.add_argument("--min-iou", type=float, default=0.3)
    ap.add_argument("--out", default="outputs/matched_reps.csv")
    a = ap.parse_args()

    pred = pd.read_csv(a.pred).fillna("")
    trn = pd.read_csv(a.trainer).fillna("")
    pred["v"], trn["v"] = pred["video"].map(norm_video), trn["video"].map(norm_video)
    pred["label"] = pred["label"].str.strip().str.lower()
    trn["label"] = trn["label"].str.strip().str.lower()

    # Per-arm exercises: match reps only within the same video AND arm (both arms move at the same time)
    use_arm = "arm" in pred.columns and "arm" in trn.columns
    if use_arm:
        pred["arm"] = pred["arm"].astype(str).str.strip().str.lower()
        trn["arm"] = trn["arm"].astype(str).str.strip().str.lower()
        both = trn[trn.arm == "both"]
        trn = pd.concat([trn[trn.arm != "both"], both.assign(arm="left"), both.assign(arm="right")],
                        ignore_index=True)
        pred["key"] = pred["v"] + "|" + pred["arm"]
        trn["key"] = trn["v"] + "|" + trn["arm"]
    else:
        pred["key"], trn["key"] = pred["v"], trn["v"]

    matched, n_fp, n_fn = [], 0, 0
    for v in sorted(trn["key"].unique()):               # only videos (and arms) the trainer annotated
        p, t = pred[pred.key == v], trn[trn.key == v]
        pairs, up, ut = match(p, t, a.min_iou)
        n_fp += len(up); n_fn += len(ut)
        for i, j, s in pairs:
            matched.append({"video": v, "sys_start": p.loc[i, "start_s"], "sys_end": p.loc[i, "end_s"],
                            "trainer_start": t.loc[j, "start_s"], "trainer_end": t.loc[j, "end_s"],
                            "iou": round(s, 2),
                            "sys_label": p.loc[i, "label"], "trainer_label": t.loc[j, "label"],
                            "sys_errors": p.loc[i, "errors"], "trainer_errors": t.loc[j, "errors"]})
    m = pd.DataFrame(matched)
    if m.empty:
        raise SystemExit("No reps matched - check video names and timestamps (seconds).")

    # --- rep detection
    tp = len(m)
    p, r, f = prf(tp, n_fp, n_fn)
    print(f"\nREP DETECTION   matched={tp}  system-only={n_fp}  trainer-only(missed)={n_fn}")
    print(f"  precision={p:.3f}  recall={r:.3f}  F1={f:.3f}")

    # --- correct/incorrect (positive class = incorrect)
    sy, tr = m.sys_label == "incorrect", m.trainer_label == "incorrect"
    TP, FP, FN, TN = int((sy & tr).sum()), int((sy & ~tr).sum()), int((~sy & tr).sum()), int((~sy & ~tr).sum())
    n = len(m)
    acc = (TP + TN) / n
    pe = ((TP + FP) / n) * ((TP + FN) / n) + ((TN + FN) / n) * ((TN + FP) / n)
    kappa = (acc - pe) / (1 - pe) if pe < 1 else float("nan")
    p, r, f = prf(TP, FP, FN)
    print(f"\nFORM LABEL (correct vs incorrect) on {n} matched reps")
    print(f"  accuracy={acc:.3f}  Cohen's kappa={kappa:.3f}")
    print(f"  'incorrect' class: precision={p:.3f}  recall={r:.3f}  F1={f:.3f}")
    print("                       trainer:incorrect  trainer:correct")
    print(f"  system:incorrect     {TP:>10}        {FP:>10}")
    print(f"  system:correct       {FN:>10}        {TN:>10}")

    # --- per error type
    codes = sorted(set().union(*m.sys_errors.map(errset), *m.trainer_errors.map(errset)))
    if codes:
        print("\nPER ERROR TYPE")
        print(f"  {'code':<16}{'TP':>4}{'FP':>4}{'FN':>4}{'prec':>7}{'rec':>7}{'F1':>7}")
        for c in codes:
            s = m.sys_errors.map(lambda x: c in errset(x))
            t = m.trainer_errors.map(lambda x: c in errset(x))
            ctp, cfp, cfn = int((s & t).sum()), int((s & ~t).sum()), int((~s & t).sum())
            cp, cr, cf = prf(ctp, cfp, cfn)
            print(f"  {c:<16}{ctp:>4}{cfp:>4}{cfn:>4}{cp:>7.2f}{cr:>7.2f}{cf:>7.2f}")

    os.makedirs(os.path.dirname(a.out) or ".", exist_ok=True)
    m.to_csv(a.out, index=False)
    print(f"\nMatched pairs saved to {a.out} (inspect rows where sys_label != trainer_label).")


if __name__ == "__main__":
    main()