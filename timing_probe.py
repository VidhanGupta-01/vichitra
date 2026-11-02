#!/usr/bin/env python3
"""
Where do the 143 seconds go? Times upload, status-polling, and inference
separately on one clip.

    python timing_probe.py --clip samples\\test\\clip_0000_000300.mp4

Costs about one cent.
"""
import argparse
import os
import sys
import time

import museapi


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--clip", required=True)
    args = ap.parse_args()

    if not os.path.exists(args.clip):
        print(f"no such file: {args.clip}")
        return 1

    client = museapi.get_client()
    size_mb = os.path.getsize(args.clip) / 1e6
    print(f"clip: {args.clip}  ({size_mb:.2f} MB)\n")

    # --- upload, no polling ------------------------------------------------
    t = time.time()
    with open(args.clip, "rb") as fh:
        up = client.files.create(file=fh, purpose="user_data")
    t_upload = time.time() - t
    print(f"upload                 {t_upload:6.1f}s   -> {up.id}")

    # What does the API actually report? This is the suspected time sink.
    status = getattr(up, "status", "<no status attribute>")
    print(f"status right after upload: {status!r}")

    # --- polling loop, instrumented ----------------------------------------
    t = time.time()
    polls = 0
    for _ in range(40):
        try:
            f = client.files.retrieve(up.id)
        except Exception as e:
            print(f"retrieve failed: {type(e).__name__}: {e}")
            break
        polls += 1
        s = getattr(f, "status", None)
        print(f"  poll {polls:>2}  status={s!r}  t={time.time()-t:.1f}s")
        if s in (None, "processed", "active", "ready"):
            break
        if s in ("error", "failed"):
            print("  processing failed")
            return 1
        time.sleep(3)
    t_poll = time.time() - t
    print(f"polling                {t_poll:6.1f}s   ({polls} polls)")

    # --- inference ---------------------------------------------------------
    t = time.time()
    data, usage = museapi.score_clip(client, up.id, seconds=10)
    t_infer = time.time() - t
    print(f"inference              {t_infer:6.1f}s   "
          f"({usage.get('input_tokens')}/{usage.get('output_tokens')} tokens)")

    total = t_upload + t_poll + t_infer
    print(f"\ntotal                  {total:6.1f}s")
    print(f"  upload     {100*t_upload/total:5.1f}%")
    print(f"  polling    {100*t_poll/total:5.1f}%   <- if this is large, we delete it")
    print(f"  inference  {100*t_infer/total:5.1f}%   <- if this dominates, we parallelise")
    return 0


if __name__ == "__main__":
    sys.exit(main())
