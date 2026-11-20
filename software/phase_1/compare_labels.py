"""
Compare PowerPulse predictions with a trainer's annotations.

  python compare_labels.py --pred outputs/all_reps_predicted.csv --trainer trainer_labels.csv

Trainer CSV columns:  video, start_s, end_s, label, errors [, arm]
  * header spaces, blank video cells (merged cells in a sheet) and capital letters are tolerated
  * label  : correct | incorrect
  * errors : the system's codes OR the trainer's own words, separated by , or ;   e.g. "Heel Lift, Forward Lean".
             '-' or blank = none.  Words are turned into codes by TERM_MAP below - edit it to match your trainer.
  * arm    : lateral raises only - the arm that did the lifting, left | right (the PERSON's left/right).
             ('both' is still accepted: the same label for both arms.)

What it prints:
  1. rep detection (skipped automatically if the trainer's windows are just the system's windows)
  2. binary correct/incorrect, RAW - with the always-guess-majority baseline next to it
  3. binary again SCOPED - errors the system cannot detect (--undetectable) removed from the trainer's labels
  4. per participant/video agreement, the disagreements, and a per-error table
  5. tuning table: the system's m_* metrics grouped by what the trainer said  (then run tune_thresholds.py)
"""
import argparse
import os
import re

import pandas as pd

# trainer's words -> code.  First match wins, matched on the lower-cased text.  Edit freely.
TERM_MAP = [
    ("neck", "NECK_DOWN"), ("destab", "UNSTABLE"), ("unstab", "UNSTABLE"),
    ("lock", "ELBOW_STRAIGHT"), ("straight", "ELBOW_STRAIGHT"), ("hyperext", "ELBOW_STRAIGHT"),   # lateral raise: BEFORE the generic "elbow" below
    ("sideways", "TORSO_LEAN"), ("torso", "TORSO_LEAN"),                                         # lateral raise: BEFORE the squat "lean" below
    ("lowered", "LOWERED_TOO_QUICKLY"), ("quick", "LOWERED_TOO_QUICKLY"),
    ("incomplete", "NOT_HIGH_ENOUGH"),                                                          # trainer's INCOMPLETE_ROM
    ("toe", "TOE_LIFT"), ("heel", "HEEL_LIFT"),
    ("lean", "LEAN_FORWARD"), ("swing", "TORSO_SWING"), ("momentum", "TORSO_SWING"),
    ("deeper", "SQUAT_DEEPER"), ("shallow", "SHALLOW"), ("bounce", "BOUNCE"),
    ("fast descent", "FAST_DESCENT"), ("too fast", "FAST_DESCENT"), ("too deep", "TOO_DEEP"),
    ("knee", "KNEES_FORWARD"), ("elbow", "ELBOW_BENT"),
    ("too high", "TOO_HIGH"), ("not high", "NOT_HIGH_ENOUGH"),
]
KNOWN_CODES = {c for _, c in TERM_MAP} | {
    "FAST_LIFT", "FAST_LOWER", "HIP_FLEXION", "ANKLE_MOBILITY", "HIPS_BACK", "WEIGHT_BACK", "BUTT_WINK",
    "BACK_KNEE_HIGH", "BACK_KNEE_LOW", "UNEVEN_RAISE", "ELBOW_STRAIGHT", "TORSO_LEAN", "LOWERED_TOO_QUICKLY", "OTHER"}


EXTRA_METRICS = ("peak_angle", "min_elbow_angle")      # non-m_ columns that are also useful for tuning


def norm_video(s):
    return os.path.splitext(os.path.basename(str(s).strip()))[0].lower()


def parse_errors(text, unmapped):
    """'Heel Lift, Forward Lean' or 'HEEL_LIFT;LEAN_FORWARD' -> {'HEEL_LIFT', 'LEAN_FORWARD'}."""
    out = set()
    for tok in re.split(r"[;,]", str(text)):
        t = tok.strip()
        if t.lower() in ("", "-", "nan", "none", "ok", "correct"):
            continue
        code = t.upper().replace(" ", "_")
        if code in KNOWN_CODES:
            out.add(code)
            continue
        for key, c in TERM_MAP:
            if key in t.lower():
                out.add(c)
                break
        else:
            out.add("OTHER")
            unmapped.add(t)
    return out


def errset(s):
    return {e for e in str(s).split(";") if e}


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


def binary_report(title, sys_inc, tr_inc):
    TP, FP = int((sys_inc & tr_inc).sum()), int((sys_inc & ~tr_inc).sum())
    FN, TN = int((~sys_inc & tr_inc).sum()), int((~sys_inc & ~tr_inc).sum())
    n = TP + FP + FN + TN
    acc = (TP + TN) / n
    pe = ((TP + FP) / n) * ((TP + FN) / n) + ((TN + FN) / n) * ((TN + FP) / n)
    kappa = (acc - pe) / (1 - pe) if pe < 1 else float("nan")
    prec, rec, _ = prf(TP, FP, FN)
    spec = TN / (TN + FP) if TN + FP else float("nan")
    base = max(tr_inc.mean(), 1 - tr_inc.mean())
    print(f"\n{title}   (n = {n} reps)")
    print(f"  trainer: {int(tr_inc.sum())} incorrect / {int((~tr_inc).sum())} correct"
          f"  ->  a system that always guessed the majority class would score {base:.1%}")
    print(f"  accuracy {acc:.1%}   Cohen's kappa {kappa:.2f}   balanced accuracy {(rec + spec) / 2:.1%}")
    print(f"  'incorrect' class: precision {prec:.1%}   recall {rec:.1%}      'correct' class: recall (specificity) {spec:.1%}")
    print("                         trainer: incorrect   trainer: correct")
    print(f"  system: incorrect      {TP:>10}           {FP:>10}")
    print(f"  system: correct        {FN:>10}           {TN:>10}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pred", required=True)
    ap.add_argument("--trainer", required=True)
    ap.add_argument("--min-iou", type=float, default=0.3)
    ap.add_argument("--undetectable", default="NECK_DOWN,UNSTABLE",
                    help="comma-separated codes the system cannot detect (used for the SCOPED comparison)")
    ap.add_argument("--ignore-codes", default="",
                    help="comma-separated SYSTEM codes that should not count against a rep - shows what the binary "
                         "result would be without them (no need to re-run the videos), e.g. ELBOW_BENT")
    ap.add_argument("--out", default="outputs/matched_reps.csv")
    a = ap.parse_args()
    UND = {c.strip().upper() for c in a.undetectable.split(",") if c.strip()}
    IGN = {c.strip().upper() for c in a.ignore_codes.split(",") if c.strip()}

    pred = pd.read_csv(a.pred)
    trn = pd.read_csv(a.trainer)
    pred.columns = [c.strip() for c in pred.columns]
    trn.columns = [c.strip() for c in trn.columns]
    trn["video"] = trn["video"].ffill()                      # merged cells leave blanks under the first row
    for df in (pred, trn):
        for c in ("start_s", "end_s"):
            df[c] = pd.to_numeric(df[c], errors="coerce")
    trn = trn.dropna(subset=["video", "start_s", "end_s"]).reset_index(drop=True)
    pred["v"], trn["v"] = pred["video"].map(norm_video), trn["video"].map(norm_video)
    pred["label"] = pred["label"].astype(str).str.strip().str.lower()
    trn["label"] = trn["label"].astype(str).str.strip().str.lower()
    bad = ~trn["label"].isin(["correct", "incorrect"])
    if bad.any():
        print(f"WARNING: {int(bad.sum())} trainer rows have a label that is not correct/incorrect - ignored.")
        trn = trn[~bad].reset_index(drop=True)

    unmapped = set()
    pred["sc"] = pred["errors"].fillna("").map(lambda x: parse_errors(x, set()))
    trn["tc"] = trn["errors"].fillna("").map(lambda x: parse_errors(x, unmapped))
    if unmapped:
        print("WARNING: trainer words that match no code (counted as OTHER) - add them to TERM_MAP:")
        for u in sorted(unmapped):
            print("   ", repr(u))

    use_arm = "arm" in pred.columns and "arm" in trn.columns
    if use_arm:
        pred["arm"] = pred["arm"].astype(str).str.strip().str.lower()
        trn["arm"] = trn["arm"].astype(str).str.strip().str.lower()
        both = trn[trn.arm == "both"]
        trn = pd.concat([trn[trn.arm != "both"], both.assign(arm="left"), both.assign(arm="right")], ignore_index=True)
        pred["key"], trn["key"] = pred["v"] + "|" + pred["arm"], trn["v"] + "|" + trn["arm"]
    else:
        pred["key"], trn["key"] = pred["v"], trn["v"]

    mcols = [c for c in pred.columns if c.startswith("m_") or c in EXTRA_METRICS]
    matched, n_fp, n_fn = [], 0, 0
    for k in sorted(trn["key"].unique()):                    # only videos (and arms) the trainer annotated
        p, t = pred[pred.key == k], trn[trn.key == k]
        pairs, up, ut = match(p, t, a.min_iou)
        n_fp += len(up); n_fn += len(ut)
        for i, j, s in pairs:
            matched.append({"video": p.loc[i, "v"], "arm": p.loc[i, "arm"] if use_arm else "",
                            "sys_start": p.loc[i, "start_s"], "sys_end": p.loc[i, "end_s"],
                            "trainer_start": t.loc[j, "start_s"], "trainer_end": t.loc[j, "end_s"], "iou": round(s, 2),
                            "sys_label": p.loc[i, "label"], "trainer_label": t.loc[j, "label"],
                            "sys_errors": ";".join(sorted(p.loc[i, "sc"])), "trainer_errors": ";".join(sorted(t.loc[j, "tc"])),
                            **{c: p.loc[i, c] for c in mcols}})
    m = pd.DataFrame(matched)
    if m.empty:
        raise SystemExit("No reps matched - check video names and timestamps (seconds).")

    not_annotated = sorted(set(pred.key) - set(trn.key))
    print(f"\n{len(pred)} system reps, {len(trn)} trainer rows -> {len(m)} matched")
    if not_annotated:
        n_skip = int(pred.key.isin(not_annotated).sum())
        print(f"NOT ANNOTATED by the trainer, left out of everything below ({n_skip} reps): {', '.join(not_annotated)}")

    # --- 1. rep detection
    if (m.iou > 0.999).all():
        print("\nREP DETECTION: not evaluated - every trainer window equals a system window exactly, so the trainer")
        print("  annotated the SYSTEM's reps. Rep counting would need the trainer to mark reps independently.")
    else:
        tp = len(m)
        p_, r_, f_ = prf(tp, n_fp, n_fn)
        print(f"\nREP DETECTION   matched={tp}  system-only={n_fp}  trainer-only(missed)={n_fn}")
        print(f"  precision={p_:.3f}  recall={r_:.3f}  F1={f_:.3f}")

    sys_inc = m.sys_label == "incorrect"
    tr_inc = m.trainer_label == "incorrect"
    tr_codes = m.trainer_errors.map(errset)

    # --- 2. + 3. binary comparison
    binary_report("BINARY (correct / incorrect) - RAW: trainer labels exactly as given", sys_inc, tr_inc)
    seen = sorted(set().union(*tr_codes) & UND)
    if seen:
        tr_scoped = tr_inc & tr_codes.map(lambda s: len(s - UND) > 0 or len(s) == 0)
        only = int((tr_inc & ~tr_scoped).sum())
        binary_report(f"BINARY - SCOPED: {', '.join(seen)} removed from the trainer's labels", sys_inc, tr_scoped)
        print(f"  ({only} reps had ONLY those errors, so they count as 'correct' here.  The system can't see these errors,")
        print("   so this separates 'cannot detect' from 'detects badly' - report it NEXT TO the raw numbers, never instead.)")

    if IGN:
        s_inc = m.sys_errors.map(lambda x: len(errset(x) - IGN) > 0)
        binary_report(f"BINARY - RAW again, but the system's {', '.join(sorted(IGN))} flags do NOT count against a rep",
                      s_inc, tr_inc)

    # --- 4. per video, disagreements, per error
    g = (m.assign(sys_inc=sys_inc, tr_inc=tr_inc, agree=sys_inc == tr_inc)
          .groupby(["video", "arm"] if use_arm else "video")
          .agg(reps=("agree", "size"), trainer_incorrect=("tr_inc", "sum"), system_incorrect=("sys_inc", "sum"),
               agree=("agree", "sum")))
    print("\nPER VIDEO (binary agreement)")
    print(g.to_string())

    dis = m[sys_inc != tr_inc]
    fn, fp = m[~sys_inc & tr_inc], m[sys_inc & ~tr_inc]
    print(f"\nDISAGREEMENTS: {len(dis)} reps")
    print(f"  system said correct, trainer said incorrect ({len(fn)}) - what the trainer saw:")
    for c, n in pd.Series([c for s in fn.trainer_errors.map(errset) for c in s]).value_counts().items():
        print(f"      {c:<16}{n:>4}" + ("   <- system cannot detect this" if c in UND else ""))
    print(f"  system said incorrect, trainer said correct ({len(fp)}) - what the system flagged:")
    for c, n in pd.Series([c for s in fp.sys_errors.map(errset) for c in s], dtype=object).value_counts().items():
        print(f"      {c:<16}{n:>4}")

    sys_codes = m.sys_errors.map(errset)
    codes = sorted(set().union(*sys_codes, *tr_codes))
    if codes:
        print("\nPER ERROR TYPE (on matched reps)")
        print(f"  {'code':<16}{'trainer':>8}{'system':>8}{'TP':>5}{'FP':>5}{'FN':>5}{'prec':>7}{'rec':>7}")
        for c in codes:
            s_, t_ = sys_codes.map(lambda x: c in x), tr_codes.map(lambda x: c in x)
            ctp, cfp, cfn = int((s_ & t_).sum()), int((s_ & ~t_).sum()), int((~s_ & t_).sum())
            cp, cr, _ = prf(ctp, cfp, cfn)
            note = "   cannot detect" if c in UND else ("   trainer never uses this code" if t_.sum() == 0 else "")
            print(f"  {c:<16}{int(t_.sum()):>8}{int(s_.sum()):>8}{ctp:>5}{cfp:>5}{cfn:>5}{cp:>7.2f}{cr:>7.2f}{note}")
        print("  (if the trainer lists only the MAIN errors of a rep, FP counts above are inflated - ask whether the lists are complete)")

    # --- 5. threshold tuning: system metrics grouped by what the trainer said
    mc = [c for c in m.columns if c.startswith("m_") or c in EXTRA_METRICS]
    if mc:
        num = m[mc].apply(pd.to_numeric, errors="coerce")
        masks = {"trainer: correct": ~tr_inc}
        counts = pd.Series([c for s in tr_codes for c in s]).value_counts()
        for c in counts[counts >= 3].index:
            masks[f"trainer: {c}"] = tr_codes.map(lambda s, c=c: c in s)
        masks = {k: v for k, v in masks.items() if v.sum()}
        summary = pd.DataFrame({k: num[v].median() for k, v in masks.items()}).T.round(3)
        summary.insert(0, "n", [int(v.sum()) for v in masks.values()])
        print("\nTUNING METRICS - median per trainer group (set each threshold between 'correct' and the error row)")
        print(summary.to_string())

    os.makedirs(os.path.dirname(a.out) or ".", exist_ok=True)
    m.assign(agree=sys_inc == tr_inc).to_csv(a.out, index=False)
    dis_path = os.path.splitext(a.out)[0] + "_disagreements.csv"
    dis.to_csv(dis_path, index=False)
    print(f"\nSaved: {a.out}  (every matched rep)   and   {dis_path}  (only the disagreements - review these with the trainer)")


if __name__ == "__main__":
    main()