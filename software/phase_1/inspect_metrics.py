"""
See how the tuning metrics (m_* columns) and error flags are spread across ALL your reps - useful for
choosing thresholds BEFORE you have trainer labels, and for spotting limits that flag far too many reps.

  python inspect_metrics.py outputs/all_reps_predicted.csv
  python inspect_metrics.py outputs                 # or a folder: reads every *_reps.csv in it

For each metric it prints the percentiles, and for the ones that drive an error it prints what share of
reps would be flagged at a range of candidate thresholds.
"""
import argparse
import glob
import os

import pandas as pd

# metric column -> (error code it drives, candidate thresholds to try)
CANDIDATES = {
    "m_peak_descent_vel": ("FAST_DESCENT", [100, 140, 170, 200, 250, 300]),   # squat, deg/s
    "m_peak_back_vel":    ("TORSO_SWING",  [40, 60, 90, 120, 160]),            # squat, deg/s
    "m_heel_lift_max":    ("HEEL_LIFT",    [0.03, 0.05, 0.08, 0.12, 0.20]),    # squat, fraction of shank
    "m_peak_lift_vel":    ("FAST_LIFT",    [100, 150, 200, 250, 300]),         # raise, deg/s
    "m_peak_lower_vel":   ("FAST_LOWER",   [100, 150, 200, 250, 300]),         # raise, deg/s
    "m_peak_torso_vel":   ("TORSO_SWING",  [15, 30, 45, 60, 90]),              # raise, deg/s
}


def load(path):
    if os.path.isdir(path):
        files = sorted(glob.glob(os.path.join(path, "*_reps.csv")))
    else:
        files = [path]
    if not files:
        raise SystemExit("No reps CSV found.")
    return pd.concat([pd.read_csv(f) for f in files], ignore_index=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("path", help="all_reps_predicted.csv, one <video>_reps.csv, or a folder of them")
    a = ap.parse_args()

    df = load(a.path).fillna("")
    n_videos = df["video"].nunique()
    print(f"\n{len(df)} reps from {n_videos} videos")

    # how often each error is flagged right now
    codes = df["errors"].astype(str).str.split(";").explode()
    codes = codes[codes != ""]
    print("\nERROR FLAG RATE (share of all reps)")
    if codes.empty:
        print("  no errors flagged")
    for code, n in codes.value_counts().items():
        print(f"  {code:<18}{n:>5}  {100 * n / len(df):5.1f}%")

    # metrics are only meaningful for full reps
    for col in ("reached_depth", "reached_top"):
        if col in df.columns:
            df = df[pd.to_numeric(df[col], errors="coerce") == 1]
            print(f"\n(metrics below use the {len(df)} reps that reached full depth/height)")

    mcols = [c for c in df.columns if c.startswith("m_")]
    num = df[mcols].apply(pd.to_numeric, errors="coerce")
    q = num.quantile([0, .10, .25, .50, .75, .90, .95, 1]).T
    q.columns = ["min", "10%", "25%", "median", "75%", "90%", "95%", "max"]
    print("\nMETRIC DISTRIBUTION")
    print(q.round(2).to_string())

    print("\nWHAT SHARE OF REPS WOULD BE FLAGGED AT EACH THRESHOLD")
    print("(peak-based, so it slightly over-counts: the real rule also needs several consecutive frames)")
    for col, (code, cands) in CANDIDATES.items():
        if col not in num.columns or num[col].dropna().empty:
            continue
        vals = num[col].dropna()
        row = "  ".join(f"{t:g}: {100 * (vals > t).mean():4.0f}%" for t in cands)
        print(f"  {code:<13}({col})\n      {row}")
    print("\nTip: until you have trainer labels, pick limits that flag only the fastest/most extreme ~5-15% of reps;")
    print("then check those reps against the video, and fine-tune with compare_labels.py once labelled.")


if __name__ == "__main__":
    main()