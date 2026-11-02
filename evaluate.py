#!/usr/bin/env python3
"""
Evaluate predicted valence/arousal against human annotations.

    python evaluate.py --scores out/scores.jsonl \
                       --truth data/liris/ACCEDEranking.txt \
                       --sep tab --id-col name \
                       --valence-col valenceRank --arousal-col arousalRank \
                       --report out/results.md

Deliberately format-agnostic: point it at any delimited file of human ratings and
name the columns. The LIRIS-ACCEDE column names above are a best guess -- run with
--inspect first to print the real header, then set the flags.

Reports Spearman (rank correlation, the right statistic when the ground truth is a
ranking) with a bootstrap CI, plus the baselines a reviewer will ask for. Nothing
here is a trained model; scipy is a statistics library, not an AI model.
"""
import argparse
import json
import os
import sys

import numpy as np
from scipy import stats


def load_scores(path):
    rows = {}
    with open(path) as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            r = json.loads(line)
            key = os.path.splitext(r["clip"])[0]
            rows[key] = r
    return rows


def load_truth(path, sep, id_col, v_col, a_col, inspect=False):
    delim = {"tab": "\t", "comma": ",", "semicolon": ";", "space": None}[sep]
    with open(path, encoding="utf-8", errors="replace") as fh:
        header = fh.readline().rstrip("\n").split(delim) if delim else \
                 fh.readline().split()
        header = [h.strip() for h in header]
        if inspect:
            print("columns:", header)
            for i, line in enumerate(fh):
                print(" ", line.rstrip()[:160])
                if i >= 2:
                    break
            sys.exit(0)
        for c in (id_col, v_col, a_col):
            if c not in header:
                print(f"column {c!r} not in header: {header}")
                sys.exit(1)
        ii, iv, ia = header.index(id_col), header.index(v_col), header.index(a_col)
        out = {}
        for line in fh:
            parts = line.rstrip("\n").split(delim) if delim else line.split()
            if len(parts) <= max(ii, iv, ia):
                continue
            key = os.path.splitext(parts[ii].strip())[0]
            try:
                out[key] = (float(parts[iv]), float(parts[ia]))
            except ValueError:
                continue
    return out


def boot_spearman(x, y, n=2000, seed=0):
    rng = np.random.default_rng(seed)
    vals = []
    idx = np.arange(len(x))
    for _ in range(n):
        s = rng.choice(idx, len(idx), replace=True)
        if len(set(y[s])) < 2 or len(set(x[s])) < 2:
            continue
        vals.append(stats.spearmanr(x[s], y[s]).statistic)
    if not vals:
        return (float("nan"), float("nan"))
    return tuple(np.percentile(vals, [2.5, 97.5]))


def block(name, pred, true):
    rho = stats.spearmanr(pred, true)
    r = stats.pearsonr(pred, true)
    lo, hi = boot_spearman(np.asarray(pred, float), np.asarray(true, float))
    return (f"### {name}\n\n"
            f"| statistic | value |\n|---|---|\n"
            f"| Spearman rho | **{rho.statistic:+.3f}** |\n"
            f"| Spearman p | {rho.pvalue:.2g} |\n"
            f"| 95% bootstrap CI | [{lo:+.3f}, {hi:+.3f}] |\n"
            f"| Pearson r | {r.statistic:+.3f} |\n\n")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--scores", required=True)
    ap.add_argument("--truth", required=True)
    ap.add_argument("--sep", default="tab",
                    choices=["tab", "comma", "semicolon", "space"])
    ap.add_argument("--id-col", default="name")
    ap.add_argument("--valence-col", default="valenceRank")
    ap.add_argument("--arousal-col", default="arousalRank")
    ap.add_argument("--report", default="out/results.md")
    ap.add_argument("--inspect", action="store_true",
                    help="print the truth file header and exit")
    args = ap.parse_args()

    truth = load_truth(args.truth, args.sep, args.id_col, args.valence_col,
                       args.arousal_col, args.inspect)
    scores = load_scores(args.scores)

    keys = sorted(set(scores) & set(truth))
    if len(keys) < 10:
        print(f"only {len(keys)} clips matched between the two files.")
        print("sample score ids:", list(scores)[:3])
        print("sample truth ids:", list(truth)[:3])
        print("The ids must match. Strip extensions or fix --id-col.")
        return 1

    pv = [scores[k]["valence"] for k in keys]
    pa = [scores[k]["arousal"] for k in keys]
    tv = [truth[k][0] for k in keys]
    ta = [truth[k][1] for k in keys]

    rng = np.random.default_rng(0)
    rand = rng.permutation(pa).tolist()

    md = [f"# Evaluation\n\n**n = {len(keys)} clips.** Predictions from "
          f"`{scores[keys[0]].get('model', '?')}`, ground truth from "
          f"`{os.path.basename(args.truth)}`.\n\n"]
    md.append(block("Arousal", pa, ta))
    md.append(block("Valence", pv, tv))
    md.append(block("Arousal, shuffled predictions (null baseline)", rand, ta))

    const = stats.spearmanr([50] * len(ta), ta)
    md.append(f"A constant prediction gives rho = {const.statistic:.3f} "
              f"(undefined / zero by construction).\n\n")
    md.append("## Honest reading\n\n"
              "These numbers describe how well a prediction made from film content "
              "alone ranks against aggregated human self-report. They are not a claim "
              "about any individual viewer's felt emotion. Rank correlation is "
              "reported because the reference annotations are rankings.\n")

    os.makedirs(os.path.dirname(args.report) or ".", exist_ok=True)
    with open(args.report, "w") as fh:
        fh.write("".join(md))

    print("".join(md))
    print(f"\nwritten to {args.report}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
