#!/usr/bin/env python3
"""
Batch-score a folder of clips. Resumable, parallel and cost-guarded.

    python score_clips.py --clips data\\liris\\clips --out out\\liris.jsonl --workers 8
    python score_clips.py --clips out\\demo_clips --out out\\demo.jsonl

Each clip is ~15s of wall clock (7s upload + 7s inference), almost all of it
waiting on the network, so running several at once is close to free. Results are
appended as soon as each clip finishes, so a crash or Ctrl-C never costs the same
clip twice and a re-run skips whatever is already in the output file.
"""
import argparse
import json
import os
import pathlib
import sys
import threading
import time
from concurrent.futures import ThreadPoolExecutor, as_completed

import museapi

# Standard-tier list price at time of writing: $1.25/1M input, $4.25/1M output.
# Check https://dev.meta.ai/docs/pricing-rate-limits before a large run.
USD_IN = 1.25 / 1_000_000
USD_OUT = 4.25 / 1_000_000


def already_done(path):
    done = set()
    if not os.path.exists(path):
        return done
    with open(path) as fh:
        for line in fh:
            try:
                done.add(json.loads(line)["clip"])
            except Exception:
                continue
    return done


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--clips", required=True, help="folder of .mp4 clips")
    ap.add_argument("--out", required=True, help="output .jsonl")
    ap.add_argument("--limit", type=int, default=0, help="stop after N new clips")
    ap.add_argument("--budget", type=float, default=10.0,
                    help="abort if estimated spend exceeds this many USD")
    ap.add_argument("--workers", type=int, default=6,
                    help="clips in flight at once; raise if you are not rate limited")
    ap.add_argument("--model", default=museapi.DEFAULT_MODEL)
    ap.add_argument("--seconds", type=int, default=10)
    ap.add_argument("--video-block", default="input_file",
                    choices=["input_file", "input_video"])
    ap.add_argument("--no-schema", action="store_true")
    args = ap.parse_args()

    clips = sorted(pathlib.Path(args.clips).glob("*.mp4"))
    if not clips:
        print(f"no .mp4 files in {args.clips}")
        return 1

    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    done = already_done(args.out)
    todo = [p for p in clips if p.name not in done]
    if args.limit:
        todo = todo[:args.limit]

    print(f"{len(clips)} clips found, {len(done)} already scored, {len(todo)} to do")
    if not todo:
        return 0
    print(f"{args.workers} at a time, budget ${args.budget:.2f}\n")

    client = museapi.get_client()
    lock = threading.Lock()
    state = {"spend": 0.0, "n": 0, "stop": False}
    failures = []
    t0 = time.time()
    out_fh = open(args.out, "a")

    def work(path):
        if state["stop"]:
            return None
        fid = museapi.upload_clip(client, str(path))
        data, usage = museapi.score_clip(
            client, fid, seconds=args.seconds, model=args.model,
            use_schema=not args.no_schema, video_block=args.video_block)
        return path, museapi.normalise(data), usage

    with ThreadPoolExecutor(max_workers=args.workers) as pool:
        futures = {pool.submit(work, p): p for p in todo}
        for fut in as_completed(futures):
            path = futures[fut]
            try:
                result = fut.result()
            except Exception as e:
                with lock:
                    failures.append(path.name)
                    print(f"  FAIL {path.name}: {type(e).__name__}: {e}")
                    if len(failures) >= 5 and state["n"] == 0:
                        print("\nFirst 5 clips all failed. Stopping before you "
                              "burn credits. Re-run timing_probe.py.")
                        state["stop"] = True
                continue
            if result is None:
                continue

            path, data, usage = result
            with lock:
                state["n"] += 1
                out_fh.write(json.dumps(
                    {"clip": path.name, "model": args.model, **data,
                     "usage": usage}) + "\n")
                out_fh.flush()

                if usage.get("input_tokens"):
                    state["spend"] += usage["input_tokens"] * USD_IN
                if usage.get("output_tokens"):
                    state["spend"] += usage["output_tokens"] * USD_OUT

                n, spend = state["n"], state["spend"]
                elapsed = time.time() - t0
                eta = (len(todo) - n) * elapsed / n
                flag = "  ?" if data.get("ambiguous") else ""
                print(f"  [{n}/{len(todo)}] {path.name}  v={data['valence']:>3} "
                      f"a={data['arousal']:>3}  {data['dominant']:<14}{flag:<4}"
                      f"~${spend:.2f}  eta {eta/60:.1f}m")

                if spend > args.budget:
                    print(f"\nBudget ${args.budget} reached. Stopping.")
                    state["stop"] = True

    out_fh.close()
    mins = (time.time() - t0) / 60
    per = (time.time() - t0) / max(state["n"], 1)
    print(f"\nwrote {args.out}  |  {state['n']} scored in {mins:.1f}m "
          f"({per:.1f}s/clip)  |  est. ${state['spend']:.2f}  "
          f"|  {len(failures)} failures")
    if failures:
        print("failed:", ", ".join(failures[:10]))
        print("re-run the same command to retry just those")
    print("\n'?' marks clips with no clear dominant reaction.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
