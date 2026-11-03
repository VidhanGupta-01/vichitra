"""
Shared Meta Model API client + the per-clip scoring call.

Every AI model called here is a Meta model (Muse Spark; optionally SAM 3.1).
No non-Meta model runs at runtime. ffmpeg / numpy / scipy are non-model libraries.
"""
import json
import os
import time

BASE_URL = "https://api.meta.ai/v1"

# 1.3 has degraded audio understanding per Meta's docs; our clips are scored with
# their audio track, so the default is 1.2.
DEFAULT_MODEL = os.environ.get("MUSE_MODEL", "muse-spark-1.2")

# The 8 reaction labels. Keep this list frozen once scoring starts -- changing it
# invalidates every cached score.
LABELS = [
    "amusement",      # laughter, lightness
    "tension",        # dread, suspense, fear
    "surprise",       # startle, reversal
    "sadness",        # grief, melancholy
    "disgust",
    "anger",
    "tenderness",     # warmth, awe, moved
    "disengagement",  # boredom, attention drop
]

SCORE_SCHEMA = {
    "type": "object",
    "properties": {
        "valence": {"type": "integer", "minimum": 0, "maximum": 100,
                    "description": "0 = maximally unpleasant, 50 = neutral, 100 = maximally pleasant"},
        "arousal": {"type": "integer", "minimum": 0, "maximum": 100,
                    "description": "0 = calm/still, 100 = maximally activating"},
        "expression": {
            "type": "object",
            "properties": {k: {"type": "integer", "minimum": 0, "maximum": 100} for k in LABELS},
            "required": LABELS,
            "additionalProperties": False,
            "description": "percentage of a typical audience showing each reaction; sums to ~100",
        },
        "dominant": {"type": "string", "enum": LABELS},
        "confidence": {"type": "integer", "minimum": 0, "maximum": 100},
        "cues": {"type": "array", "items": {"type": "string"}, "maxItems": 4,
                 "description": "concrete craft cues driving the prediction"},
        "rationale": {"type": "string"},
    },
    "required": ["valence", "arousal", "expression", "dominant", "confidence", "cues", "rationale"],
    "additionalProperties": False,
}

# PROMPT HISTORY -- read before editing this.
#
# This is the v2 prompt. A v3 was tried on 2026-11-02 that added worked valence
# anchors ("ordinary footage is 45-55", "grim but absorbing belongs at 50-70") to
# correct a measured -11.8 valence bias, then was reverted. Scored against the same
# 18 human-annotated excerpts:
#
#                     v2        v3        v4 (= v2 prompt, re-run)
#   valence rho      +0.30     +0.13     +0.27
#   valence bias     -11.8     +10.5      -9.8
#   valence sd        15.3       8.4      15.6     (annotator's own sd: 17.0)
#   arousal rho      +0.61     +0.38     +0.38
#
# Why it was reverted: the anchors moved the four targeted clips onto the annotator's
# numbers but dragged everything else up with them. Excerpts the annotator rated 20-30
# came back at 58-68, so the model lost its low end and its spread collapsed to roughly
# half the annotator's. Range compression, not discrimination.
#
# What the v4 column shows, and why it matters more: v4 re-runs the v2 prompt unchanged
# and gives arousal +0.38, not +0.61. So the arousal difference between v2 and v3 was
# NOT caused by the prompt -- it is run-to-run variance. At n=18 the correlation
# estimate itself swings by ~0.2, which is the real reason no accuracy claim is made
# anywhere in this repo.
#
# Lesson: anchoring one scale on 18 examples compresses range rather than teaching
# discrimination, and 18 examples cannot tell you whether a prompt change helped.
# Do not re-add anchors without a much larger validation set.
SYSTEM = """You are a reaction-prediction model for filmmakers. You watch a short film \
excerpt, including its audio track, and predict how a general cinema audience would \
visibly react while watching it.

Rules:
- Predict the AUDIENCE's reaction, not the emotion of the characters on screen. A \
character sobbing in a comedy can read as amusement; a character smiling in a thriller \
can read as tension.
- Ground every judgement in craft cues you can actually observe: shot scale, cut rhythm, \
camera movement, lighting, score, sound design, silence, dialogue delivery, framing.
- Do not assume genre from a single frame. Use the whole excerpt.
- Be calibrated. Most excerpts are unremarkable: arousal near 30-50, confidence below 70. \
Reserve extremes for excerpts that genuinely earn them.
- Output JSON only. No prose outside the JSON."""

USER = """Predict the audience reaction to this {secs}-second excerpt.

valence: 0 very unpleasant, 50 neutral, 100 very pleasant.
arousal: 0 calm and still, 100 maximally activating.
expression: percentage of a typical audience visibly showing each reaction. Must total \
about 100. Use "disengagement" for the share whose attention drifts.
cues: up to 4 short concrete observations (e.g. "hard cut on the score sting", \
"locked-off wide, no movement for 6s").
rationale: one sentence.

Return JSON matching the schema exactly."""


def get_client():
    from openai import OpenAI
    key = os.environ.get("MODEL_API_KEY")
    if not key:
        raise RuntimeError("MODEL_API_KEY is not set")
    return OpenAI(base_url=BASE_URL, api_key=key, timeout=240.0)


# Statuses that mean "still working, come back later". Anything else -- including
# 'uploaded', which is what this API actually returns and holds -- means ready.
# Waiting for a state the API never enters cost ~3 minutes per clip before this
# was measured. Only ever block on states known to be transient.
PROCESSING = {"processing", "pending", "in_progress", "queued", "running"}
FAILED = {"error", "failed", "cancelled", "expired"}


def upload_clip(client, path, wait=True, max_wait=30):
    """Upload an mp4 and return its file id."""
    with open(path, "rb") as fh:
        up = client.files.create(file=fh, purpose="user_data")

    status = getattr(up, "status", None)
    if status in FAILED:
        raise RuntimeError(f"upload failed for {path}: {status}")
    if not wait or status not in PROCESSING:
        return up.id

    deadline = time.time() + max_wait
    while time.time() < deadline:
        time.sleep(2)
        try:
            status = getattr(client.files.retrieve(up.id), "status", None)
        except Exception:
            return up.id
        if status in FAILED:
            raise RuntimeError(f"file processing failed for {path}: {status}")
        if status not in PROCESSING:
            return up.id
    return up.id


def _extract_json(text):
    text = text.strip()
    if text.startswith("```"):
        text = text.strip("`")
        if text.lstrip().lower().startswith("json"):
            text = text.lstrip()[4:]
    start, end = text.find("{"), text.rfind("}")
    if start == -1 or end == -1:
        raise ValueError(f"no JSON object in response: {text[:200]}")
    return json.loads(text[start:end + 1])


def score_clip(client, file_id, seconds=10, model=DEFAULT_MODEL,
               use_schema=True, video_block="input_file", temperature=None):
    """Score one uploaded clip. Returns (dict, usage)."""
    content = [
        {"type": video_block, "file_id": file_id},
        {"type": "input_text", "text": USER.format(secs=seconds)},
    ]
    # Muse Spark spends output tokens on internal reasoning before it emits any JSON,
    # so the ceiling has to cover reasoning + answer. At 4000 one clip in eighteen came
    # back cut off mid-array with no closing brace. 12000 leaves headroom; unused
    # tokens are not billed, so raising it costs nothing.
    kwargs = dict(
        model=model,
        instructions=SYSTEM,
        input=[{"role": "user", "content": content}],
        max_output_tokens=12000,
    )
    if temperature is not None:
        kwargs["temperature"] = temperature
    if use_schema:
        kwargs["text"] = {"format": {"type": "json_schema", "name": "reaction",
                                     "schema": SCORE_SCHEMA, "strict": True}}

    def call(kw):
        try:
            return client.responses.create(**kw)
        except Exception:
            if not kw.get("text"):
                raise
            kw = dict(kw)
            kw.pop("text")  # schema rejected -> rely on the prompt
            return client.responses.create(**kw)

    r = call(kwargs)

    # A response can come back truncated rather than failing. The API flags that as
    # status "incomplete"; a missing closing brace catches the rest.
    truncated = getattr(r, "status", None) == "incomplete"
    try:
        data = _extract_json(r.output_text)
    except ValueError:
        truncated = True
        data = None

    if truncated:
        retry = dict(kwargs, max_output_tokens=24000)
        r = call(retry)
        try:
            data = _extract_json(r.output_text)
        except ValueError as e:
            raise ValueError(
                f"response still truncated at 24000 output tokens "
                f"(status={getattr(r, 'status', None)}). Raw start: "
                f"{r.output_text[:160]!r}"
            ) from e

    usage = getattr(r, "usage", None)
    return data, {
        "input_tokens": getattr(usage, "input_tokens", None) if usage else None,
        "output_tokens": getattr(usage, "output_tokens", None) if usage else None,
    }


def normalise(data):
    """Clamp and renormalise so downstream code can trust the numbers.

    The model's own `dominant` is kept when it is valid, because the model reasoned
    about the excerpt as a whole. Overwriting it with the argmax of a flat
    distribution produced incoherent calls like arousal=74 with dominant=
    "disengagement", where one label led by two points over seven others.
    """
    data["valence"] = max(0, min(100, int(data.get("valence", 50))))
    data["arousal"] = max(0, min(100, int(data.get("arousal", 50))))

    exp = {k: max(0, int(data.get("expression", {}).get(k, 0))) for k in LABELS}
    total = sum(exp.values()) or 1
    exp = {k: round(v * 100 / total, 1) for k, v in exp.items()}
    data["expression"] = exp

    ranked = sorted(exp.items(), key=lambda kv: -kv[1])
    top, second = ranked[0], ranked[1]

    # The model's own `dominant` field is not trustworthy: measured over repeated
    # runs on one clip it named labels holding as little as 3% of the distribution
    # it generated in the same response. The argmax of `expression` is internally
    # consistent, so that is what we use. The model's claim is kept for inspection.
    data["model_said_dominant"] = data.get("dominant")
    data["dominant"] = top[0]
    data["dominant_share"] = top[1]
    data["margin"] = round(top[1] - second[1], 1)
    data["self_consistent"] = data["model_said_dominant"] == top[0]

    # Surfaced so the UI can say "no clear reaction" instead of inventing one.
    data["ambiguous"] = bool(top[1] < 30 or data["margin"] < 8)
    return data
