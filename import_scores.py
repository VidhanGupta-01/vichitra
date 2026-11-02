#!/usr/bin/env python3
"""
Bring scores made with score_clips.py into the app as a job.

    python import_scores.py --scores out/demo_v4.jsonl --clips out/demo_clips --name Sintel

Run once so the pre-scored demo shows up in the app without re-scoring anything.
Judges can then open the app with no API key and see a real result immediately.
"""
import argparse
import json
import os
import pathlib
import re
import shutil
import sys


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--scores", required=True)
    ap.add_argument("--clips", required=True)
    ap.add_argument("--name", default="demo")
    ap.add_argument("--seconds", type=int, default=10)
    ap.add_argument("--work", default="out/jobs")
    ap.add_argument("--job", default="demo", help="folder name inside --work")
    args = ap.parse_args()

    for p in (args.scores, args.clips):
        if not os.path.exists(p):
            print(f"not found: {p}")
            return 1

    dest = pathlib.Path(args.work) / args.job
    (dest / "clips").mkdir(parents=True, exist_ok=True)

    rows = []
    with open(args.scores) as fh:
        for line in fh:
            line = line.strip()
            if line:
                try:
                    rows.append(json.loads(line))
                except json.JSONDecodeError:
                    pass
    latest = {r["clip"]: r for r in rows}

    copied = 0
    for name in latest:
        src = pathlib.Path(args.clips) / name
        if src.exists():
            shutil.copy2(src, dest / "clips" / name)
            copied += 1

    with open(dest / "scores.jsonl", "w") as fh:
        for r in latest.values():
            fh.write(json.dumps(r) + "\n")

    starts = []
    for name in latest:
        m = re.search(r"_(\d+)_(\d+)\.mp4$", name)
        if m:
            starts.append(int(m.group(2)))

    with open(dest / "meta.json", "w") as fh:
        json.dump({"source": args.name, "seconds": args.seconds,
                   "clips": len(latest),
                   "start": min(starts) if starts else 0,
                   "end": (max(starts) + args.seconds) if starts else 0}, fh, indent=2)

    print(f"imported {len(latest)} scores and {copied} clips -> {dest}")
    if copied < len(latest):
        print(f"note: {len(latest)-copied} clips were missing; those excerpts "
              f"will show scores but no video.")
    print(f"\nStart the app and it will be listed under 'Earlier':\n  python app.py")
    return 0


if __name__ == "__main__":
    sys.exit(main())
