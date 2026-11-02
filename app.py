#!/usr/bin/env python3
"""
ViChitra -- predicted audience reaction, scene by scene.

    python app.py              # open http://localhost:8000

Upload a cut in the browser, pick the stretch to analyse, and get a reaction
curve you can mark your intent against and send to the cutting room.

Runs offline against already-scored files, so a judge can clone the repo and see
the whole thing work without an API key. Scoring new footage needs MODEL_API_KEY.
"""
import argparse
import json
import os
import pathlib
import re
import shutil
import subprocess
import threading
import time
import uuid

from fastapi import FastAPI, UploadFile, File, Form
from fastapi.responses import HTMLResponse, JSONResponse, FileResponse
from pydantic import BaseModel

import museapi

app = FastAPI()
CFG = {}
JOBS = {}
JOBS_LOCK = threading.Lock()

# Same word scale used to collect the human baseline, so intent is marked in the
# vocabulary a director would actually use rather than guessed as a number.
AROUSAL_WORDS = [
    ("nothing happening", 15), ("calm", 30), ("mildly engaged", 45),
    ("attentive", 58), ("intense", 72), ("gripped", 85), ("heart racing", 95),
]
GAP_THRESHOLD = 20
USD_PER_CLIP = 0.0067          # measured over 54 clips
SECS_PER_CLIP = 3.0            # measured at 6 workers
MAX_CLIPS = 400                # a guard, not a limit of the method


# ---------------------------------------------------------------- helpers

def tc(s):
    s = int(s)
    return f"{s//60:02d}:{s%60:02d}"


def reading(v, a):
    """Plain-English headline from valence and arousal only.

    These two repeat within a few points across runs; the single-word expression
    label does not, and its label set has no word for joy or awe. This is the
    circumplex model of affect -- the frame valence and arousal come from, and the
    one LIRIS-ACCEDE is built on.
    """
    hi_v, lo_v = v >= 60, v <= 40
    hi_a, lo_a = a >= 60, a <= 40
    if hi_v and hi_a:  return "pleasant, energising"
    if hi_v and lo_a:  return "pleasant, calm"
    if lo_v and hi_a:  return "unpleasant, intense"
    if lo_v and lo_a:  return "bleak and flat"
    if hi_a:           return "activating, mixed tone"
    if lo_a:           return "low engagement"
    return "moderate throughout"


def probe_duration(path):
    out = subprocess.run(
        ["ffprobe", "-v", "error", "-show_entries", "format=duration",
         "-of", "json", str(path)],
        capture_output=True, text=True, check=True).stdout
    return float(json.loads(out)["format"]["duration"])


def job_dir(job_id):
    d = pathlib.Path(CFG["work"]) / job_id
    d.mkdir(parents=True, exist_ok=True)
    return d


def load_rows(path):
    rows = {}
    if not os.path.exists(path):
        return rows
    with open(path) as fh:
        for line in fh:
            line = line.strip()
            if line:
                try:
                    r = json.loads(line)
                    rows[r["clip"]] = r
                except json.JSONDecodeError:
                    pass
    return rows


def read_json(path, default):
    if os.path.exists(path):
        try:
            with open(path) as fh:
                return json.load(fh)
        except Exception:
            pass
    return default


def write_json(path, obj):
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    with open(path, "w") as fh:
        json.dump(obj, fh, indent=2)


def timeline_for(job_id):
    d = job_dir(job_id)
    scores = load_rows(d / "scores.jsonl")
    marks = read_json(d / "marks.json", {})
    meta = read_json(d / "meta.json", {})
    out = []
    for name, r in scores.items():
        m = re.search(r"_(\d+)_(\d+)\.mp4$", name)
        r["index"] = int(m.group(1)) if m else 0
        r["start_s"] = int(m.group(2)) if m else 0
        r["seconds"] = meta.get("seconds", 10)
        r["reading"] = reading(r["valence"], r["arousal"])
        mk = marks.get(name, {})
        r["intent"] = mk.get("arousal")
        r["note"] = mk.get("note", "")
        out.append(r)
    return sorted(out, key=lambda r: r["start_s"]), meta


# ---------------------------------------------------------------- pipeline

def run_job(job_id, start, end, seconds):
    """Cut the chosen stretch and score every excerpt. Runs on a worker thread."""
    d = job_dir(job_id)
    clips = d / "clips"
    clips.mkdir(exist_ok=True)
    src = next(d.glob("source.*"))

    def set_state(**kw):
        with JOBS_LOCK:
            JOBS.setdefault(job_id, {}).update(kw)

    try:
        total = max(1, int((end - start) // seconds))
        set_state(stage="cutting", done=0, total=total,
                  msg=f"Cutting {total} excerpts")

        made = []
        t = float(start)
        n = 0
        while t < end - 1.0 and n < total:
            length = min(seconds, end - t)
            name = f"clip_{n:04d}_{int(t):06d}.mp4"
            subprocess.run(
                ["ffmpeg", "-nostdin", "-loglevel", "error", "-y",
                 "-ss", f"{t:.3f}", "-i", str(src), "-t", f"{length:.3f}",
                 "-vf", "scale=-2:480,fps=24",
                 "-c:v", "libx264", "-preset", "veryfast", "-crf", "30",
                 "-c:a", "aac", "-b:a", "64k", "-ac", "1",
                 "-movflags", "+faststart", str(clips / name)],
                check=True)
            made.append(name)
            n += 1
            t += seconds
            set_state(done=n, msg=f"Cutting excerpt {n} of {total}")

        set_state(stage="scoring", done=0, total=len(made),
                  msg=f"Watching {len(made)} excerpts")

        client = museapi.get_client()
        out_path = d / "scores.jsonl"
        done_already = set(load_rows(out_path))
        todo = [c for c in made if c not in done_already]
        lock = threading.Lock()
        counter = {"n": len(done_already), "fail": 0}

        from concurrent.futures import ThreadPoolExecutor, as_completed

        def one(name):
            fid = museapi.upload_clip(client, str(clips / name))
            data, usage = museapi.score_clip(client, fid, seconds=seconds)
            return name, museapi.normalise(data), usage

        with ThreadPoolExecutor(max_workers=6) as pool:
            futs = {pool.submit(one, c): c for c in todo}
            with open(out_path, "a") as fh:
                for fut in as_completed(futs):
                    try:
                        name, data, usage = fut.result()
                    except Exception:
                        with lock:
                            counter["fail"] += 1
                        continue
                    with lock:
                        fh.write(json.dumps({"clip": name, **data,
                                             "usage": usage}) + "\n")
                        fh.flush()
                        counter["n"] += 1
                        set_state(done=counter["n"],
                                  msg=f"Watched {counter['n']} of {len(made)}")

        write_json(d / "meta.json", {
            "source": read_json(d / "meta.json", {}).get("source", "your cut"),
            "start": start, "end": end, "seconds": seconds,
            "clips": len(made), "failed": counter["fail"],
        })
        set_state(stage="done", msg="Ready", failed=counter["fail"])

    except StopIteration:
        set_state(stage="error", msg="The uploaded file is missing.")
    except FileNotFoundError:
        set_state(stage="error",
                  msg="ffmpeg was not found. Install it and try again.")
    except Exception as e:
        set_state(stage="error", msg=f"{type(e).__name__}: {e}")


# ---------------------------------------------------------------- api

@app.post("/api/upload")
async def upload(file: UploadFile = File(...)):
    if not shutil.which("ffmpeg") or not shutil.which("ffprobe"):
        return JSONResponse({"error": "ffmpeg is not installed on this machine."},
                            status_code=400)
    job_id = uuid.uuid4().hex[:12]
    d = job_dir(job_id)
    ext = pathlib.Path(file.filename or "cut.mp4").suffix or ".mp4"
    dest = d / f"source{ext}"
    with open(dest, "wb") as fh:
        while chunk := await file.read(1 << 20):
            fh.write(chunk)
    try:
        dur = probe_duration(dest)
    except Exception:
        shutil.rmtree(d, ignore_errors=True)
        return JSONResponse({"error": "That file could not be read as video."},
                            status_code=400)
    write_json(d / "meta.json",
               {"source": pathlib.Path(file.filename or "your cut").stem,
                "duration": dur})
    return {"job": job_id, "duration": dur,
            "name": pathlib.Path(file.filename or "your cut").stem}


class Analyze(BaseModel):
    job: str
    start: float
    end: float
    seconds: int = 10


@app.post("/api/analyze")
def analyze(body: Analyze):
    if not os.environ.get("MODEL_API_KEY"):
        return JSONResponse(
            {"error": "MODEL_API_KEY is not set, so new footage cannot be scored. "
                      "The demo below works without it."}, status_code=400)
    n = int((body.end - body.start) // max(body.seconds, 1))
    if n < 1:
        return JSONResponse({"error": "That range is shorter than one excerpt."},
                            status_code=400)
    if n > MAX_CLIPS:
        return JSONResponse(
            {"error": f"That range makes {n} excerpts. The cap is {MAX_CLIPS} — "
                      f"shorten the range or use longer excerpts."}, status_code=400)
    with JOBS_LOCK:
        JOBS[body.job] = {"stage": "starting", "done": 0, "total": n,
                          "msg": "Starting"}
    threading.Thread(target=run_job,
                     args=(body.job, body.start, body.end, body.seconds),
                     daemon=True).start()
    return {"ok": True, "clips": n}


@app.get("/api/progress/{job}")
def progress(job: str):
    with JOBS_LOCK:
        return JOBS.get(job, {"stage": "unknown", "msg": ""})


@app.get("/api/timeline/{job}")
def timeline(job: str):
    clips, meta = timeline_for(job)
    return {"clips": clips, "meta": meta, "words": AROUSAL_WORDS,
            "threshold": GAP_THRESHOLD,
            "source": meta.get("source", "your cut"), "job": job}


class Mark(BaseModel):
    job: str
    clip: str
    arousal: int | None = None
    note: str | None = None


@app.post("/api/mark")
def mark(body: Mark):
    p = job_dir(body.job) / "marks.json"
    marks = read_json(p, {})
    entry = marks.get(body.clip, {})
    if body.arousal is not None or "arousal" not in entry:
        entry["arousal"] = body.arousal
    if body.note is not None:
        entry["note"] = body.note[:2000]
    if entry.get("arousal") is None and not entry.get("note"):
        marks.pop(body.clip, None)
    else:
        marks[body.clip] = entry
    write_json(p, marks)
    return {"ok": True}


@app.get("/api/jobs")
def jobs():
    out = []
    for d in sorted(pathlib.Path(CFG["work"]).glob("*"),
                    key=lambda p: p.stat().st_mtime, reverse=True):
        if not (d / "scores.jsonl").exists():
            continue
        meta = read_json(d / "meta.json", {})
        out.append({"job": d.name, "source": meta.get("source", d.name),
                    "clips": meta.get("clips", 0)})
    return {"jobs": out[:12]}


@app.get("/clip/{job}/{name}")
def clip(job: str, name: str):
    p = job_dir(job) / "clips" / os.path.basename(name)
    if not p.exists():
        return JSONResponse({"error": "not found"}, status_code=404)
    return FileResponse(p, media_type="video/mp4")


# ---------------------------------------------------------------- report

@app.get("/report/{job}", response_class=HTMLResponse)
def report(job: str, download: int = 0):
    clips, meta = timeline_for(job)
    source = meta.get("source", "your cut")
    secs = meta.get("seconds", 10)
    marked = [r for r in clips if r["intent"] is not None or r["note"]]
    scored = [r for r in marked if r["intent"] is not None]
    flagged = sorted([r for r in scored
                      if abs(r["intent"] - r["arousal"]) >= GAP_THRESHOLD],
                     key=lambda r: -abs(r["intent"] - r["arousal"]))

    W, H, PL, PB = 920, 240, 40, 26
    def X(i): return PL + (W - PL - 12) * (i / max(len(clips) - 1, 1))
    def Y(v): return 14 + (H - 14 - PB) * (1 - v / 100)

    svg = []
    for g in (0, 50, 100):
        svg.append(f'<line x1="{PL}" y1="{Y(g)}" x2="{W-12}" y2="{Y(g)}" stroke="#2C323E"/>'
                   f'<text x="{PL-8}" y="{Y(g)+4}" fill="#5C6473" font-size="11" '
                   f'text-anchor="end">{g}</text>')
    for i, r in enumerate(clips[:-1]):
        n = clips[i + 1]
        if r["intent"] is None or n["intent"] is None:
            continue
        big = (abs(r["intent"] - r["arousal"]) >= GAP_THRESHOLD or
               abs(n["intent"] - n["arousal"]) >= GAP_THRESHOLD)
        svg.append(f'<path d="M{X(i)} {Y(r["arousal"])} L{X(i+1)} {Y(n["arousal"])} '
                   f'L{X(i+1)} {Y(n["intent"])} L{X(i)} {Y(r["intent"])} Z" '
                   f'fill="#D9485F" opacity="{0.28 if big else 0.08}"/>')
    if clips:
        svg.append('<path fill="none" stroke="#E8B33C" stroke-width="2.5" d="' +
                   " ".join(("M" if i == 0 else "L") + f"{X(i):.1f} {Y(r['arousal']):.1f}"
                            for i, r in enumerate(clips)) + '"/>')
    pts = [(i, r) for i, r in enumerate(clips) if r["intent"] is not None]
    if len(pts) > 1:
        svg.append('<path fill="none" stroke="#4FC3D9" stroke-width="2" '
                   'stroke-dasharray="5 4" d="' +
                   " ".join(("M" if j == 0 else "L") + f"{X(i):.1f} {Y(r['intent']):.1f}"
                            for j, (i, r) in enumerate(pts)) + '"/>')
    for i, r in enumerate(clips):
        if i % max(len(clips) // 6, 1) == 0:
            svg.append(f'<text x="{X(i)}" y="{H-8}" fill="#5C6473" font-size="11" '
                       f'text-anchor="middle">{tc(r["start_s"])}</text>')

    def esc(s):
        return (s or "").replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")

    rows = "".join(
        f'<tr><td class="tc">{tc(r["start_s"])}–{tc(r["start_s"]+secs)}</td>'
        f'<td>{r["intent"] if r["intent"] is not None else "—"}</td>'
        f'<td>{r["arousal"]}</td>'
        f'<td class="{"miss" if r["intent"] is not None and abs(r["intent"]-r["arousal"])>=GAP_THRESHOLD else ""}">'
        f'{f"{r['arousal']-r['intent']:+d}" if r["intent"] is not None else "—"}</td>'
        f'<td>{r["reading"]}</td>'
        f'<td class="note">{esc(r["note"]) or "—"}</td>'
        f'<td class="cues">{esc("; ".join(r.get("cues") or []))}</td></tr>'
        for r in marked)

    if flagged:
        notes = f"Scenes to look at again ({source}):\n\n" + "\n".join(
            f'{tc(r["start_s"])}–{tc(r["start_s"]+secs)}  '
            + ("falls short by " + str(r["intent"] - r["arousal"])
               if r["arousal"] < r["intent"]
               else "overshoots by " + str(r["arousal"] - r["intent"]))
            + f' — wanted {r["intent"]}, reads {r["arousal"]} ({r["reading"]})'
            + (f'\n           note: {r["note"]}' if r["note"] else "")
            for r in flagged)
    elif scored:
        notes = ("The marked scene lands" if len(scored) == 1 else
                 f"All {len(scored)} marked scenes land") + \
                f" within {GAP_THRESHOLD} points of what was wanted. Nothing to re-cut."
    else:
        notes = "No scenes marked yet."

    if not scored:
        headline = "No scenes marked yet — open the app and mark what you wanted."
    elif not flagged:
        headline = ("The marked scene is doing what you wanted." if len(scored) == 1
                    else f"All {len(scored)} marked scenes are doing what you wanted.")
    elif len(flagged) == len(scored):
        headline = ("<b>The marked scene</b> is not landing." if len(scored) == 1
                    else f"<b>All {len(scored)} marked scenes</b> are not landing.")
    else:
        headline = (f"<b>{len(flagged)}</b> of {len(scored)} marked scenes "
                    f"{'is' if len(flagged) == 1 else 'are'} not landing.")

    bar = "" if download else f"""
<div class="actions">
  <a class="btn primary" href="/report/{job}?download=1" download="vichitra-notes.html">Download</a>
  <button class="btn" onclick="navigator.clipboard.writeText(
    document.querySelector('pre').textContent).then(()=>{{this.textContent='Copied'}})">
    Copy notes</button>
  <button class="btn" onclick="window.print()">Print or save as PDF</button>
</div>"""

    html = f"""<!doctype html><html lang="en"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<title>ViChitra notes — {esc(source)}</title><style>
body{{margin:0;background:#171A21;color:#E6E8EC;font:15px/1.5 -apple-system,
BlinkMacSystemFont,"Segoe UI",Roboto,sans-serif;padding:32px}}
.wrap{{max-width:1040px;margin:0 auto}}
h1{{font-size:20px;margin:0 0 4px}} .sub{{color:#8A91A0;font-size:13px;margin:0 0 20px}}
.card{{background:#1E222B;border:1px solid #2C323E;border-radius:6px;padding:20px;
margin-bottom:20px}}
.head{{font-size:16px;margin:0 0 16px}} .head b{{color:#D9485F}}
table{{width:100%;border-collapse:collapse;font-size:14px}}
th{{text-align:left;color:#8A91A0;font-weight:500;font-size:12px;padding:0 10px 8px 0;
border-bottom:1px solid #2C323E}}
td{{padding:9px 10px 9px 0;border-bottom:1px solid #22262F;vertical-align:top}}
.tc{{font-family:ui-monospace,Menlo,Consolas,monospace;font-size:13px;white-space:nowrap}}
.miss{{color:#D9485F;font-weight:600}}
.cues{{color:#8A91A0;font-size:13px}} .note{{color:#E6E8EC;font-size:13.5px}}
pre{{background:#171A21;border:1px solid #2C323E;border-radius:4px;padding:14px;
overflow-x:auto;white-space:pre-wrap;font-size:13px;color:#E6E8EC}}
.legend{{color:#8A91A0;font-size:12.5px;margin:10px 0 0}}
.legend i{{display:inline-block;width:16px;height:2px;vertical-align:middle;margin:0 6px 0 14px}}
footer{{color:#5C6473;font-size:12.5px;margin-top:28px;line-height:1.6}}
.actions{{display:flex;gap:10px;margin:0 0 22px;flex-wrap:wrap}}
.btn{{font:inherit;font-size:13.5px;padding:8px 14px;border-radius:4px;cursor:pointer;
border:1px solid #2C323E;background:transparent;color:#E6E8EC;text-decoration:none}}
.btn:hover{{border-color:#E8B33C}}
.btn.primary{{background:#E8B33C;border-color:#E8B33C;color:#171A21;font-weight:600}}
@media print{{body{{background:#fff;color:#000;padding:0}}
.card{{background:#fff;border-color:#ccc}} .actions{{display:none}}
.cues,.sub,footer,th{{color:#555}}}}
</style></head><body><div class="wrap">
<h1>ViChitra notes</h1>
<p class="sub">{esc(source)} · {len(clips)} excerpts of {secs}s · predicted reaction vs marked intent</p>
{bar}
<div class="card">
  <p class="head">{headline}</p>
  <svg viewBox="0 0 {W} {H}" style="width:100%;height:auto">{''.join(svg)}</svg>
  <p class="legend"><i style="background:#E8B33C"></i>what the cut delivers
     <i style="background:#4FC3D9"></i>what was intended</p>
</div>
<div class="card">
  <table><thead><tr><th>timecode</th><th>wanted</th><th>reads</th><th>gap</th>
  <th>reads as</th><th>your note</th><th>what the model saw</th></tr></thead>
  <tbody>{rows or '<tr><td colspan="7" style="color:#8A91A0">Nothing marked.</td></tr>'}</tbody></table>
</div>
<div class="card">
  <p class="head" style="font-size:13px;color:#8A91A0">Paste into an email or a shot list</p>
  <pre>{esc(notes)}</pre>
</div>
<footer>Reaction is predicted from the content of each excerpt by Meta's Muse Spark,
not measured from any viewer. It estimates how a general audience would be likely to
react; it is not a measurement of anyone's feelings, and it is a prompt to look at a
scene again, not a verdict on it.</footer>
</div></body></html>"""

    if download:
        return HTMLResponse(html, headers={
            "Content-Disposition": 'attachment; filename="vichitra-notes.html"'})
    return html


@app.get("/", response_class=HTMLResponse)
def index():
    return PAGE


PAGE = r"""<!doctype html>
<html lang="en"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<title>ViChitra</title>
<style>
 :root{
   --bg:#171A21; --panel:#1E222B; --line:#2C323E;
   --ink:#E6E8EC; --mute:#8A91A0; --dim:#5C6473;
   --predicted:#E8B33C; --intended:#4FC3D9; --gap:#D9485F;
   --mono:ui-monospace,"SF Mono",Menlo,Consolas,monospace;
 }
 *{box-sizing:border-box}
 body{margin:0;background:var(--bg);color:var(--ink);font:15px/1.5 -apple-system,
      BlinkMacSystemFont,"Segoe UI",Roboto,sans-serif;-webkit-font-smoothing:antialiased}
 header{display:flex;align-items:baseline;gap:18px;padding:16px 24px;
        border-bottom:1px solid var(--line)}
 header h1{margin:0;font-size:19px;font-weight:600;letter-spacing:-.01em;cursor:pointer}
 header .src{color:var(--mute);font-size:13px}
 header .right{margin-left:auto;display:flex;gap:16px;align-items:center;font-size:13px;
               color:var(--mute)}
 header i{display:inline-block;width:18px;height:2px;vertical-align:middle;
          margin-right:7px;border-radius:1px}
 .btn{font:inherit;font-size:13.5px;padding:7px 14px;border-radius:4px;cursor:pointer;
      border:1px solid var(--line);background:transparent;color:var(--ink);
      text-decoration:none;display:inline-block}
 .btn:hover{border-color:var(--predicted)}
 .btn:disabled{opacity:.45;cursor:default}
 .btn.primary{background:var(--predicted);border-color:var(--predicted);
              color:#171A21;font-weight:600}
 main{padding:24px;max-width:1240px;margin:0 auto}

 /* ---- upload ---- */
 .drop{border:1.5px dashed var(--line);border-radius:8px;padding:56px 24px;
       text-align:center;cursor:pointer;background:var(--panel)}
 .drop.over{border-color:var(--predicted);background:#22262F}
 .drop h2{margin:0 0 6px;font-size:18px;font-weight:600}
 .drop p{margin:0;color:var(--mute);font-size:14px}
 .setup{background:var(--panel);border:1px solid var(--line);border-radius:8px;
        padding:22px;margin-top:20px}
 .row{display:flex;gap:26px;flex-wrap:wrap;align-items:flex-end}
 label{display:block;font-size:12.5px;color:var(--mute);margin-bottom:5px}
 input[type=text],select{font:inherit;font-family:var(--mono);font-size:14px;
   background:var(--bg);border:1px solid var(--line);color:var(--ink);
   border-radius:4px;padding:8px 10px;width:92px}
 select{font-family:inherit;width:auto}
 input:focus,select:focus{outline:2px solid var(--predicted);outline-offset:1px}
 .estimate{color:var(--mute);font-size:13.5px;margin:16px 0 0}
 .estimate b{color:var(--ink);font-weight:600}
 .err{color:var(--gap);font-size:13.5px;margin:12px 0 0}
 .bar{height:3px;background:var(--line);border-radius:2px;overflow:hidden;margin-top:18px}
 .bar div{height:100%;background:var(--predicted);width:0;transition:width .3s}
 .recent{margin-top:28px;color:var(--mute);font-size:13.5px}
 .recent a{color:var(--ink);text-decoration:none;border-bottom:1px solid var(--line)}
 .recent a:hover{border-color:var(--predicted)}

 /* ---- timeline ---- */
 .chartwrap{background:var(--panel);border:1px solid var(--line);border-radius:6px;
            padding:20px 20px 8px}
 svg{display:block;width:100%;height:auto;overflow:visible}
 .axis{stroke:var(--line)} .axislabel{fill:var(--dim);font-size:11px;font-family:var(--mono)}
 .hit{fill:transparent;cursor:pointer} .node{cursor:pointer}
 .gapnote{margin:14px 0 0;color:var(--mute);font-size:13.5px}
 .gapnote b{color:var(--gap);font-weight:600}
 .cols{display:grid;grid-template-columns:minmax(0,1.1fr) minmax(0,1fr);gap:20px;
       margin-top:20px}
 .panel{background:var(--panel);border:1px solid var(--line);border-radius:6px;padding:18px}
 video{width:100%;border-radius:4px;background:#000;display:block}
 .tc{font-family:var(--mono);font-size:12.5px;color:var(--mute);display:flex;
     gap:16px;margin-top:12px;flex-wrap:wrap}
 .tc b{color:var(--ink);font-weight:500}
 h2{margin:0 0 12px;font-size:13px;font-weight:600;color:var(--mute)}
 .cues{margin:0;padding-left:18px;font-size:14px} .cues li{margin-bottom:5px}
 .rationale{margin:14px 0 0;color:var(--mute);font-size:14px;font-style:italic}
 .words{display:flex;flex-wrap:wrap;gap:7px;margin-top:10px}
 .words button{font:inherit;font-size:13px;padding:6px 11px;border-radius:4px;
   border:1px solid var(--line);background:transparent;color:var(--mute);cursor:pointer}
 .words button:hover{border-color:var(--intended);color:var(--ink)}
 .words button[aria-pressed="true"]{background:var(--intended);
   border-color:var(--intended);color:#0E1419;font-weight:600}
 textarea{width:100%;margin-top:12px;font:inherit;font-size:14px;background:var(--bg);
   border:1px solid var(--line);color:var(--ink);border-radius:4px;padding:10px;
   resize:vertical;min-height:62px}
 textarea:focus{outline:2px solid var(--intended);outline-offset:1px}
 .saved{font-size:12px;color:var(--dim);margin-top:5px;height:14px}
 .verdict{margin-top:14px;padding:11px 13px;border-radius:4px;font-size:14px;
          border:1px solid var(--line)}
 .verdict.miss{border-color:var(--gap);color:var(--gap)}
 .verdict.ok{color:var(--mute)}
 @media (max-width:860px){.cols{grid-template-columns:1fr}}
</style></head>
<body>
<header>
  <h1 onclick="go(null)">ViChitra</h1>
  <span class="src" id="src">predicted audience reaction, scene by scene</span>
  <div class="right" id="headright"></div>
</header>
<main id="main"></main>

<script>
let JOB=null, DATA=null, SEL=0, DUR=0, saveTimer=null;
const $ = s => document.querySelector(s);
const fmt = s => String(Math.floor(s/60)).padStart(2,'0')+':'+String(Math.floor(s)%60).padStart(2,'0');
const parseTc = t => { const p=String(t).trim().split(':').map(Number);
  return p.length===2 ? p[0]*60+p[1] : (Number(t)||0); };

function go(job){
  JOB = job;
  history.replaceState({}, '', job ? '#'+job : '#');
  job ? loadTimeline() : showUpload();
}

/* ------------------------------------------------ upload */
async function showUpload(){
  $('#src').textContent = 'predicted audience reaction, scene by scene';
  $('#headright').innerHTML = '';
  $('#main').innerHTML = `
    <div class="drop" id="drop" tabindex="0" role="button">
      <h2>Drop a cut here</h2>
      <p>or click to choose a file — mp4, mov or mkv</p>
      <input type="file" id="file" accept="video/*" hidden>
    </div>
    <div id="setup"></div>
    <div class="recent" id="recent"></div>`;
  const drop=$('#drop'), file=$('#file');
  drop.onclick = ()=>file.click();
  drop.onkeydown = e=>{ if(e.key==='Enter'||e.key===' '){e.preventDefault();file.click();} };
  drop.ondragover = e=>{e.preventDefault();drop.classList.add('over');};
  drop.ondragleave = ()=>drop.classList.remove('over');
  drop.ondrop = e=>{e.preventDefault();drop.classList.remove('over');
                    if(e.dataTransfer.files[0]) send(e.dataTransfer.files[0]);};
  file.onchange = ()=>file.files[0] && send(file.files[0]);

  const r = await (await fetch('/api/jobs')).json();
  if(r.jobs.length) $('#recent').innerHTML = 'Earlier: ' + r.jobs.map(j=>
    `<a href="#${j.job}" onclick="go('${j.job}');return false">${j.source} (${j.clips})</a>`
  ).join(' · ');
}

async function send(f){
  $('#drop').innerHTML = `<h2>Uploading ${f.name}</h2><p>${(f.size/1e6).toFixed(0)} MB</p>`;
  const fd = new FormData(); fd.append('file', f);
  const res = await fetch('/api/upload', {method:'POST', body:fd});
  const j = await res.json();
  if(!res.ok){ $('#drop').innerHTML =
      `<h2>That didn't work</h2><p class="err">${j.error}</p>`; return; }
  DUR = j.duration; JOB = j.job;
  $('#drop').innerHTML = `<h2>${j.name}</h2><p>${fmt(DUR)} long</p>`;
  $('#setup').innerHTML = `
    <div class="row">
      <div><label for="s">From</label><input type="text" id="s" value="00:00"></div>
      <div><label for="e">To</label><input type="text" id="e" value="${fmt(Math.min(DUR,180))}"></div>
      <div><label for="len">Each excerpt</label>
        <select id="len">
          <option value="5">5 seconds — fine detail</option>
          <option value="10" selected>10 seconds — balanced</option>
          <option value="20">20 seconds — cheap and broad</option>
        </select></div>
      <button class="btn primary" id="run">Analyse</button>
    </div>
    <p class="estimate" id="est"></p>
    <div class="bar" id="barwrap" hidden><div id="bar"></div></div>`;
  ['s','e','len'].forEach(id=>$('#'+id).oninput = estimate);
  $('#run').onclick = start;
  estimate();
}

function estimate(){
  const a=parseTc($('#s').value), b=parseTc($('#e').value), L=+$('#len').value;
  const n=Math.floor((b-a)/L);
  const est=$('#est');
  if(n<1){ est.innerHTML = '<span class="err">That range is shorter than one excerpt.</span>';
           $('#run').disabled=true; return; }
  $('#run').disabled=false;
  est.innerHTML = `<b>${n}</b> excerpts · about <b>${(n*3/60).toFixed(n*3<60?1:0)} min</b>`
    + ` · roughly <b>$${(n*0.0067).toFixed(2)}</b> of Meta API credit`
    + (b>DUR ? ' <span class="err">— past the end of the file</span>' : '');
}

async function start(){
  const body = {job:JOB, start:parseTc($('#s').value),
                end:Math.min(parseTc($('#e').value),DUR), seconds:+$('#len').value};
  const res = await fetch('/api/analyze',{method:'POST',
    headers:{'Content-Type':'application/json'},body:JSON.stringify(body)});
  const j = await res.json();
  if(!res.ok){ $('#est').innerHTML = `<span class="err">${j.error}</span>`; return; }
  $('#run').disabled = true; $('#barwrap').hidden = false;
  poll();
}

async function poll(){
  const p = await (await fetch('/api/progress/'+JOB)).json();
  const pct = p.total ? Math.round(100*p.done/p.total) : 0;
  $('#bar').style.width = pct+'%';
  $('#est').innerHTML = `${p.msg||''}${p.total?` — ${pct}%`:''}`;
  if(p.stage==='done'){ go(JOB); return; }
  if(p.stage==='error'){ $('#est').innerHTML = `<span class="err">${p.msg}</span>`;
                         $('#run').disabled=false; return; }
  setTimeout(poll, 1000);
}

/* ------------------------------------------------ timeline */
const W=1000,H=340,PAD={l:42,r:16,t:16,b:30};
const x = i => PAD.l + (W-PAD.l-PAD.r)*(DATA.clips.length<2?.5:i/(DATA.clips.length-1));
const y = v => PAD.t + (H-PAD.t-PAD.b)*(1-v/100);

async function loadTimeline(){
  DATA = await (await fetch('/api/timeline/'+JOB)).json();
  if(!DATA.clips.length){ showUpload(); return; }
  $('#src').textContent = DATA.source+' · '+DATA.clips.length+' excerpts';
  $('#headright').innerHTML =
    `<span><i style="background:var(--predicted)"></i>predicted</span>
     <span id="intlegend" hidden><i style="background:var(--intended)"></i>intended</span>
     <a class="btn" href="/report/${JOB}" target="_blank" rel="noopener">Open notes</a>
     <a class="btn" href="#" onclick="go(null);return false">New cut</a>`;
  $('#main').innerHTML = `
    <div class="chartwrap"><svg id="chart" viewBox="0 0 ${W} ${H}" role="img"
      aria-label="Predicted audience arousal across the cut"></svg></div>
    <p class="gapnote" id="gapnote"></p>
    <div class="cols">
      <div class="panel"><video id="player" controls preload="metadata"></video>
        <div class="tc" id="tc"></div></div>
      <div class="panel">
        <h2>What the model saw</h2>
        <ul class="cues" id="cuelist"></ul>
        <p class="rationale" id="rationale"></p>
        <h2 style="margin-top:22px">What did you want here?</h2>
        <div class="words" id="words"></div>
        <textarea id="note" placeholder="Optional note for your editor — what you were going for, what to try."></textarea>
        <div class="saved" id="saved"></div>
        <div class="verdict" id="verdict"></div>
      </div></div>`;
  SEL = 0; select(0);
}

function draw(){
  const c = DATA.clips; let s='';
  for(const v of [0,25,50,75,100]){
    s+=`<line class="axis" x1="${PAD.l}" y1="${y(v)}" x2="${W-PAD.r}" y2="${y(v)}"/>`;
    s+=`<text class="axislabel" x="${PAD.l-8}" y="${y(v)+4}" text-anchor="end">${v}</text>`;
  }
  for(let i=0;i<c.length-1;i++){
    const a=c[i], b=c[i+1];
    if(a.intent==null||b.intent==null) continue;
    const big = Math.abs(a.intent-a.arousal)>=DATA.threshold ||
                Math.abs(b.intent-b.arousal)>=DATA.threshold;
    s+=`<path d="M${x(i)} ${y(a.arousal)} L${x(i+1)} ${y(b.arousal)} `+
       `L${x(i+1)} ${y(b.intent)} L${x(i)} ${y(a.intent)} Z" fill="var(--gap)" `+
       `opacity="${big?0.26:0.08}"/>`;
  }
  const pts=[]; c.forEach((d,i)=>{ if(d.intent!=null) pts.push([i,d]); });
  if(pts.length>1) s+=`<path fill="none" stroke="var(--intended)" stroke-width="2"
    stroke-dasharray="5 4" d="${pts.map(([i,d],j)=>(j?'L':'M')+x(i).toFixed(1)+' '+y(d.intent).toFixed(1)).join(' ')}"/>`;
  s+=`<path fill="none" stroke="var(--predicted)" stroke-width="2.5" stroke-linejoin="round"
       d="${c.map((d,i)=>(i?'L':'M')+x(i).toFixed(1)+' '+y(d.arousal).toFixed(1)).join(' ')}"/>`;
  const step = Math.max(Math.floor(c.length/7),1);
  c.forEach((d,i)=>{
    const miss = d.intent!=null && Math.abs(d.intent-d.arousal)>=DATA.threshold;
    s+=`<circle class="node" cx="${x(i)}" cy="${y(d.arousal)}" r="${i===SEL?5.5:3.5}"
          fill="${miss?'var(--gap)':'var(--predicted)'}"/>`;
    if(d.intent!=null) s+=`<circle cx="${x(i)}" cy="${y(d.intent)}" r="3" fill="var(--intended)"/>`;
    s+=`<rect class="hit" x="${x(i)-14}" y="0" width="28" height="${H-PAD.b}"
          onclick="select(${i})"><title>${fmt(d.start_s)}</title></rect>`;
    if(i%step===0) s+=`<text class="axislabel" x="${x(i)}" y="${H-PAD.b+18}"
          text-anchor="middle">${fmt(d.start_s)}</text>`;
  });
  $('#chart').innerHTML = s;

  const missed = c.filter(d=>d.intent!=null && Math.abs(d.intent-d.arousal)>=DATA.threshold);
  const marked = c.filter(d=>d.intent!=null).length;
  const il = $('#intlegend'); if(il) il.hidden = !marked;
  const note = $('#gapnote');
  if(!marked) note.textContent = 'Pick an excerpt, then mark what you wanted it to do. '
    + 'Anything off by ' + DATA.threshold + ' points or more gets flagged.';
  else if(!missed.length) note.textContent =
    marked + ' of ' + c.length + ' marked. Nothing is off by ' + DATA.threshold + ' points.';
  else note.innerHTML = marked + ' marked · <b>' + missed.length + ' not landing</b>: '
    + missed.map(d=>fmt(d.start_s)).join(', ');
}

function select(i){
  SEL = i; const d = DATA.clips[i];
  $('#player').src = '/clip/'+JOB+'/'+d.clip;
  $('#tc').innerHTML = `<span>${fmt(d.start_s)}–${fmt(d.start_s+d.seconds)}</span>`
    + `<span><b>${d.reading}</b></span>`
    + `<span>arousal <b>${d.arousal}</b> · valence <b>${d.valence}</b></span>`;
  $('#cuelist').innerHTML = (d.cues||[]).map(c=>`<li>${c.replace(/[<>]/g,'')}</li>`).join('')
    || '<li style="color:var(--mute)">No cues recorded.</li>';
  $('#rationale').textContent = d.rationale||'';
  $('#words').innerHTML = DATA.words.map(([w,v])=>
    `<button type="button" aria-pressed="${d.intent===v}" onclick="mark(${v})">${w}</button>`
  ).join('') + `<button type="button" onclick="mark(null)">clear</button>`;
  const ta = $('#note'); ta.value = d.note||''; $('#saved').textContent='';
  ta.oninput = ()=>{ clearTimeout(saveTimer);
    saveTimer = setTimeout(()=>saveNote(ta.value), 600); };

  const vd = $('#verdict');
  if(d.intent==null){ vd.className='verdict ok'; vd.textContent='Not marked yet.'; }
  else{
    const gap = d.arousal - d.intent;
    if(Math.abs(gap) < DATA.threshold){ vd.className='verdict ok';
      vd.textContent = `Close enough — you wanted ${d.intent}, it reads ${d.arousal}.`; }
    else{ vd.className='verdict miss';
      vd.textContent = gap<0
        ? `Falling short by ${-gap}. You wanted ${d.intent}, the cut reads ${d.arousal}.`
        : `Overshooting by ${gap}. You wanted ${d.intent}, the cut reads ${d.arousal}.`; }
  }
  draw();
}

async function post(body){
  await fetch('/api/mark',{method:'POST',headers:{'Content-Type':'application/json'},
    body:JSON.stringify(Object.assign({job:JOB, clip:DATA.clips[SEL].clip}, body))});
}
async function mark(v){ DATA.clips[SEL].intent=v; await post({arousal:v}); select(SEL); }
async function saveNote(t){ DATA.clips[SEL].note=t; await post({note:t});
  $('#saved').textContent='Saved'; setTimeout(()=>{$('#saved').textContent='';},1400); }

addEventListener('keydown', e=>{
  if(document.activeElement && document.activeElement.tagName==='TEXTAREA') return;
  if(!DATA) return;
  if(e.key==='ArrowRight'&&SEL<DATA.clips.length-1) select(SEL+1);
  if(e.key==='ArrowLeft'&&SEL>0) select(SEL-1);
});

go(location.hash.slice(1) || null);
</script></body></html>"""


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--work", default="out/jobs", help="where uploads and scores live")
    ap.add_argument("--port", type=int, default=8000)
    args = ap.parse_args()
    CFG.update(work=args.work)
    os.makedirs(args.work, exist_ok=True)
    import uvicorn
    print(f"\n  ViChitra  ->  http://localhost:{args.port}\n")
    uvicorn.run(app, host="127.0.0.1", port=args.port, log_level="warning")
