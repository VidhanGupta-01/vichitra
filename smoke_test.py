#!/usr/bin/env python3
"""
Nov 1 smoke test. Run this BEFORE writing anything else.

It answers the only question that can kill the project:
  can Muse Spark watch an 8-second clip (with audio) and return clean JSON?

Usage:
    export MODEL_API_KEY=...
    python smoke_test.py                       # text call only
    python smoke_test.py --clip path/to.mp4    # full video path

Each step prints PASS / FAIL and the raw error. Nothing is cached; this is cheap.
"""
import argparse
import json
import os
import sys
import time

BASE_URL = "https://api.meta.ai/v1"

# Audio understanding in 1.3 is degraded per Meta's own docs, and our clips carry
# the audio track (score, sound design, dialogue), so the pipeline uses 1.2.
VIDEO_MODEL = "muse-spark-1.2"
TEXT_MODEL = "muse-spark-1.3"


def ok(msg):
    print(f"  PASS  {msg}")


def bad(msg, err=None):
    print(f"  FAIL  {msg}")
    if err:
        print(f"        {type(err).__name__}: {err}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--clip", help="path to a short .mp4 (8-10s) to test video input")
    args = ap.parse_args()

    key = os.environ.get("MODEL_API_KEY")
    if not key:
        print("MODEL_API_KEY is not set. export it and re-run.")
        return 1

    try:
        from openai import OpenAI
    except ImportError:
        print("pip install openai")
        return 1

    client = OpenAI(base_url=BASE_URL, api_key=key)

    # --- 1. auth + catalog -------------------------------------------------
    print("\n[1] auth / model list")
    try:
        models = client.models.list()
        ids = sorted(m.id for m in models.data)
        ok(f"{len(ids)} models visible")
        for wanted in (TEXT_MODEL, VIDEO_MODEL, "sam-3.1", "muse-voice-transcribe-1.0"):
            print(f"        {'yes' if wanted in ids else 'NO '}  {wanted}")
    except Exception as e:
        bad("could not list models - key, billing or region problem", e)
        return 1

    # --- 2. plain text call ------------------------------------------------
    print("\n[2] text call (Responses API)")
    try:
        r = client.responses.create(
            model=TEXT_MODEL,
            input="Reply with exactly: ok",
            max_output_tokens=4000,
        )
        ok(f"got: {r.output_text.strip()[:60]!r}")
    except Exception as e:
        bad("Responses API text call failed", e)
        print("        try the Chat Completions shape instead - see docs/protocols")
        return 1

    # --- 3. structured output ----------------------------------------------
    print("\n[3] structured JSON output")
    schema = {
        "type": "object",
        "properties": {"arousal": {"type": "integer"}, "note": {"type": "string"}},
        "required": ["arousal", "note"],
        "additionalProperties": False,
    }
    structured_works = False
    try:
        r = client.responses.create(
            model=TEXT_MODEL,
            input="Rate the arousal of a quiet empty room, 0-100. Answer as JSON.",
            max_output_tokens=4000,
            text={"format": {"type": "json_schema", "name": "probe",
                             "schema": schema, "strict": True}},
        )
        json.loads(r.output_text)
        structured_works = True
        ok("json_schema accepted -- use strict schemas in the scorer")
    except Exception as e:
        bad("json_schema rejected - fall back to prompt-enforced JSON", e)

    # --- 4. video ----------------------------------------------------------
    if not args.clip:
        print("\n[4] video - skipped (pass --clip path.mp4)")
        print("\nNext: cut one 8s clip and re-run with --clip. That is the real test.")
        return 0

    if not os.path.exists(args.clip):
        print(f"\n[4] FAIL  no such file: {args.clip}")
        return 1

    size_mb = os.path.getsize(args.clip) / 1e6
    print(f"\n[4] video upload ({size_mb:.1f} MB)")
    file_id = None
    try:
        with open(args.clip, "rb") as fh:
            up = client.files.create(file=fh, purpose="user_data")
        file_id = up.id
        ok(f"uploaded: {file_id}")
    except Exception as e:
        bad("Files API upload failed", e)
        return 1

    # Some APIs process async. Poll briefly if a status field exists.
    for _ in range(20):
        try:
            f = client.files.retrieve(file_id)
            status = getattr(f, "status", None)
            if status in (None, "processed", "active", "ready"):
                break
            if status in ("error", "failed"):
                bad(f"file processing failed: {status}")
                return 1
            time.sleep(3)
        except Exception:
            break

    print("\n[5] video understanding call")
    prompt = ("Watch this clip, including its audio. "
              "Reply with JSON only: {\"summary\": str, \"heard_audio\": bool, "
              "\"arousal_0_100\": int}")

    # input_file is what the organizers quoted. input_video is the other common
    # spelling. Try both before concluding video is unsupported.
    for block in ({"type": "input_file", "file_id": file_id},
                  {"type": "input_video", "file_id": file_id}):
        try:
            r = client.responses.create(
                model=VIDEO_MODEL,
                input=[{"role": "user", "content": [block, {"type": "input_text", "text": prompt}]}],
                max_output_tokens=4000,
            )
            txt = r.output_text.strip()
            ok(f"{block['type']} worked")
            print(f"        {txt[:400]}")
            try:
                parsed = json.loads(txt.strip("`").removeprefix("json").strip())
                if parsed.get("heard_audio"):
                    ok("model reports it heard the audio track")
                else:
                    print("        NOTE: model did not confirm audio. Check manually.")
            except Exception:
                print("        NOTE: output was not bare JSON. Tighten the prompt or use a schema.")
            u = getattr(r, "usage", None)
            if u:
                print(f"        tokens in/out: {getattr(u, 'input_tokens', '?')}/"
                      f"{getattr(u, 'output_tokens', '?')}  "
                      f"<- multiply by ~200 clips to size the bill")
            print(f"\nDONE. structured_output={structured_works} video_block={block['type']}")
            return 0
        except Exception as e:
            bad(f"{block['type']} rejected", e)

    print("\nVideo input did not work. Before changing the project, read:")
    print("  https://dev.meta.ai/docs/video-understanding")
    print("  https://dev.meta.ai/docs/file-handling")
    print("and ask in Discord #questions. Fallback: sample frames as images + "
          "Muse Voice Transcribe for the audio, all still Meta models.")
    return 1


if __name__ == "__main__":
    sys.exit(main())
