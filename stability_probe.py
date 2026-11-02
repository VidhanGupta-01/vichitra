#!/usr/bin/env python3
"""
How reproducible is a score? Scores one clip N times and reports the spread.

    python stability_probe.py --clip samples\\test\\clip_0000_000300.mp4 --n 5
    python stability_probe.py --clip ... --n 5 --temperature 0

Reproducibility is worth knowing for its own sake -- a judge who re-runs the demo
and sees different answers concludes the tool is noise -- and the number belongs
in the README either way.
"""
import argparse
import statistics
import sys
from collections import Counter

import museapi


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--clip", required=True)
    ap.add_argument("--n", type=int, default=5)
    ap.add_argument("--temperature", type=float, default=None,
                    help="pass 0 to test whether the API honours it")
    args = ap.parse_args()

    client = museapi.get_client()
    fid = museapi.upload_clip(client, args.clip)
    print(f"uploaded once as {fid}, scoring {args.n}x\n")

    runs = []
    for i in range(args.n):
        kw = {}
        if args.temperature is not None:
            kw["temperature"] = args.temperature
        try:
            data, _ = museapi.score_clip(client, fid, seconds=10, **kw)
        except TypeError:
            print("score_clip does not accept temperature; run without the flag")
            return 1
        except Exception as e:
            if args.temperature is not None and "temperature" in str(e).lower():
                print(f"API rejected temperature: {e}")
                print("-> determinism is not available this way")
                return 1
            raise
        data = museapi.normalise(data)
        runs.append(data)
        print(f"  run {i+1}  v={data['valence']:>3}  a={data['arousal']:>3}  "
              f"{data['dominant']:<14} share={data['dominant_share']:>5}  "
              f"margin={data['margin']}")

    v = [r["valence"] for r in runs]
    a = [r["arousal"] for r in runs]
    labels = Counter(r["dominant"] for r in runs)

    print(f"\nvalence  mean {statistics.mean(v):5.1f}  sd {statistics.pstdev(v):4.1f}  "
          f"range {min(v)}-{max(v)}")
    print(f"arousal  mean {statistics.mean(a):5.1f}  sd {statistics.pstdev(a):4.1f}  "
          f"range {min(a)}-{max(a)}")
    print(f"dominant {dict(labels)}")
    top, count = labels.most_common(1)[0]
    print(f"\nmodal label {top!r} in {count}/{args.n} runs")

    if len(labels) > 1:
        print("\nThe label is not reproducible. Lead the UI with the "
              "valence/arousal curve and treat the label as secondary.")
    if statistics.pstdev(v) > 12 or statistics.pstdev(a) > 12:
        print("Valence/arousal are also unstable. Average several runs per clip.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
