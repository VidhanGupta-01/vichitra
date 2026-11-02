#!/usr/bin/env python3
"""
Cut a film into fixed-length clips with ffmpeg. No ML model involved.

    python cut_clips.py --video sintel.mp4 --out out/demo_clips --seconds 10

Fixed windows (not shot detection) on purpose:
  - zero risk against the "every runtime AI model must be Meta's" rule
  - the window length matches LIRIS-ACCEDE excerpt length, so the demo pipeline
    and the evaluation pipeline score the same kind of unit

Clips are re-encoded small and quiet-ish: 480p, 24fps, CRF 30, 64k AAC mono.
That keeps upload size and input tokens down, which is most of your API bill.
Audio is kept -- the score and sound design carry much of the reaction.
"""
import argparse
import json
import os
import pathlib
import shutil
import subprocess
import sys


def duration(path):
    out = subprocess.run(
        ["ffprobe", "-v", "error", "-show_entries", "format=duration",
         "-of", "json", path],
        capture_output=True, text=True, check=True).stdout
    return float(json.loads(out)["format"]["duration"])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--video", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--seconds", type=int, default=10)
    ap.add_argument("--start", type=float, default=0.0)
    ap.add_argument("--end", type=float, default=0.0, help="0 = end of file")
    ap.add_argument("--height", type=int, default=480)
    ap.add_argument("--max-clips", type=int, default=0)
    args = ap.parse_args()

    for tool in ("ffmpeg", "ffprobe"):
        if not shutil.which(tool):
            print(f"{tool} not found. Install ffmpeg.")
            return 1

    total = duration(args.video)
    end = args.end if args.end > 0 else total
    os.makedirs(args.out, exist_ok=True)

    index = []
    t = args.start
    n = 0
    while t < end - 1.0:
        if args.max_clips and n >= args.max_clips:
            break
        length = min(args.seconds, end - t)
        name = f"clip_{n:04d}_{int(t):06d}.mp4"
        dest = os.path.join(args.out, name)
        subprocess.run(
            ["ffmpeg", "-nostdin", "-loglevel", "error", "-y",
             "-ss", f"{t:.3f}", "-i", args.video, "-t", f"{length:.3f}",
             "-vf", f"scale=-2:{args.height},fps=24",
             "-c:v", "libx264", "-preset", "veryfast", "-crf", "30",
             "-c:a", "aac", "-b:a", "64k", "-ac", "1",
             "-movflags", "+faststart", dest],
            check=True)
        index.append({"clip": name, "start_s": round(t, 2),
                      "end_s": round(t + length, 2),
                      "size_mb": round(os.path.getsize(dest) / 1e6, 2)})
        n += 1
        t += args.seconds

    with open(os.path.join(args.out, "index.json"), "w") as fh:
        json.dump({"source": os.path.basename(args.video),
                   "duration_s": round(total, 2),
                   "clip_seconds": args.seconds, "clips": index}, fh, indent=2)

    mb = sum(c["size_mb"] for c in index)
    print(f"{n} clips -> {args.out}  ({mb:.0f} MB total, "
          f"{mb/max(n,1):.2f} MB average)")
    print("index.json written. Next: python score_clips.py --clips "
          f"{args.out} --out out/demo.jsonl")
    return 0


if __name__ == "__main__":
    sys.exit(main())
