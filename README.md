# ViChitra

**Predicted audience reaction, scene by scene — before anyone has watched it.**

*chitra* — picture. *vichitra* — strange, wonderful.

Built solo by Vidhan Gupta for the Meta Global AI Developer Hackathon
(Indie Developer track), 27 October – 4 November 2026.

---

## The problem

Filmmakers go blind to their own cut. You were there at 3am when you shot it, you
know what the scene cost and what it was supposed to mean — so when you watch it
back, you see your intention rather than the result. Your friends say it's good.
You find out it isn't at the premiere, in a room, in public, with the film locked.

Studios buy their way out of this with test screenings. India produces more feature
films than any country on earth — [1,724 in 2013](https://guinnessworldrecords.com/world-records/largest-annual-film-output)
by the Central Board of Film Certification's count — and the overwhelming majority
are regional, independent, student and creator films that will never get one.

**ViChitra doesn't compete with a test screening. It competes with not having one.**

## What it does

Upload a cut. It slices the stretch you choose into short excerpts, sends each one
to Meta's Muse Spark *with its audio*, and returns a reaction curve across the whole
sequence: valence (how pleasant it is to watch) and arousal (how activating).

Then the part that matters. Mark what you **intended** each stretch to do, in plain
words — *calm*, *attentive*, *gripped*. The tool draws your intent against what the
cut actually delivers and flags every scene where the two diverge by 20 points or
more. Download that as a single self-contained HTML file and send it to your editor;
they open it in any browser, no install and no account, and see timecodes with
reasons.

A test audience can tell you how they felt. It cannot tell you where the cut diverges
from what you meant, because they don't know what you meant. **Intent is an input
here — that is the thing this tool has that a screening room doesn't.**

## Meta models used, and where each one runs

| Model | Where it runs | What it does here |
|---|---|---|
| `muse-spark-1.2` | Meta Model API (`https://api.meta.ai/v1`) | Watches each excerpt with its audio track and returns the structured reaction prediction — valence, arousal, an expression distribution, and the craft cues behind the call. |

**Why 1.2 and not 1.3:** Meta's own model documentation states audio understanding in
Muse Spark 1.3 is not fully supported and may be degraded, and recommends 1.2 for
audio work. The audio track carries much of the reaction here — score, sound design,
silence, delivery — so the pipeline uses 1.2 deliberately.

Each excerpt is uploaded through the Files API with `purpose: "user_data"` and
referenced as an `input_file` block on the Responses API, with a strict
`json_schema` so the output is structured rather than parsed out of prose.

**No non-Meta AI model runs at any point.** The other dependencies are not models:
`ffmpeg` cuts clips, `numpy` and `scipy` compute correlation statistics, FastAPI
serves the page, and the OpenAI Python SDK is used purely as an HTTP client against
Meta Model API's compatible endpoint.

## Run it

Needs Python 3.10+, `ffmpeg` on PATH, and — only for scoring new footage — a Meta
Model API key.

```bash
git clone https://github.com/VidhanGupta-01/vichitra && cd vichitra
pip install -r requirements.txt
python app.py          # then open http://localhost:8000
```

### For judges — the fastest honest path

**The pre-scored demo needs no API key.** Run the two commands above, open
`localhost:8000`, and click the Sintel job listed at the bottom of the upload page.
That is 18 real excerpts of a real film, scored by Muse Spark, with the curve, the
intent marks, the gaps and the exportable notes all working offline.

To watch live inference, set `MODEL_API_KEY` and upload any video. Scoring is about
**3 seconds and 0.7 of a US cent per excerpt**, so a three-minute stretch costs
roughly **12 cents** and finishes in **under a minute**.

```bash
export MODEL_API_KEY=...        # Windows: $env:MODEL_API_KEY="..."
python smoke_test.py            # confirms auth, video upload, audio, structured output
```

## What works

**Fast and cheap enough to run on every version of a cut.** 2.2 seconds and
0.7 of a US cent per excerpt. A three-minute sequence costs about 12 cents and
finishes in under a minute. That economics is the whole product: it makes this a
check you run *during* the edit, not a verdict you get once at the end.

**Reproducible.** Same cut in, same curve out — run-to-run Spearman of **+0.88**
on valence and **+0.86** on arousal, and a single excerpt scored five times varies
by about 3 points. The intent comparison depends entirely on this, because a curve
that moved every run would make the gap meaningless.

**It explains itself.** Every excerpt comes back with the craft cues behind the
number — *"locked-off wide, no movement for 6s"*, *"hard cut on the score sting"*,
*"driving percussive score accelerating under frantic chase"*. A critic gives you a
verdict. This gives your editor something to change.

**Intent is an input.** No test audience can tell you where a cut diverges from what
you meant, because they don't know what you meant. Only the director does. This is
the one tool in this space where that goes in the comparison.

**It leaves the laptop as a single file.** Mark intent at home, hit Download, email
one self-contained HTML page. The cutting room opens it in any browser — no install,
no account, no server — and sees timecodes with reasons.

**Judges can run it with no API key.** The pre-scored Sintel demo ships in the repo.

**Three real bugs found by measuring rather than guessing** — a 184-second-per-clip
polling stall, a model label that contradicted its own output, and silent response
truncation. All three are documented below with the measurements that caught them.

## What was measured

Everything below is reproducible from this repo. No number here is an estimate.

### Reproducibility: strong

Scoring the same 18 excerpts twice with an unchanged prompt:

| | run-to-run rho | mean absolute difference |
|---|---|---|
| valence | **+0.88** | 3.7 points |
| arousal | **+0.86** | 6.2 points |

Scoring a single excerpt five times (`stability_probe.py`): valence sd **3.0**,
arousal sd **3.5**, dominant label identical 5/5.

Same cut in, materially the same curve out. The intent-gap feature depends on this,
because a comparison against a curve that moved every run would be meaningless.

### Agreement with a human: promising, not yet established

18 excerpts of *Sintel*, hand-annotated by the author on a fixed word scale before
seeing any model output (`my_ratings.csv`):

| | Spearman rho | p | 95% CI | bias | MAE |
|---|---|---|---|---|---|
| arousal | +0.38 | 0.12 | −0.10 to +0.72 | +0.2 | 13.2 |
| valence | +0.27 | 0.27 | −0.24 to +0.72 | −9.8 | 17.2 |

Both correlations point the right way, and the model's spread closely matches the
annotator's — valence sd 15.8 against 17.0, arousal 16.3 against 14.7 — so it is
discriminating between excerpts rather than hedging toward the middle. The arousal
bias is +0.2 points, essentially zero.

What 18 excerpts and one annotator cannot do is make those correlations
*statistically* established: both intervals still cross zero. So this is reported as
a promising signal and not as validation. A larger run against
[LIRIS-ACCEDE](https://liris-accede.ec-lyon.fr) is the obvious next step and
`evaluate.py` is already built for it.

### Why no accuracy claim is made

An earlier run of the identical prompt gave arousal rho **+0.61** with an interval
clear of zero. A later run of the same prompt gave **+0.38**. The correlation
estimate itself swings by around 0.2 between runs at this sample size. Reporting the
+0.61 would have been reporting a lucky draw.

### A prompt change that was tried and reverted

Measuring a −11.8 valence bias, worked anchors were added to the valence scale
(*"ordinary footage is 45-55"*, *"grim but absorbing belongs at 50-70"*). They moved
the four targeted excerpts onto the annotator's numbers — and dragged everything else
up with them. Excerpts the annotator rated 20–30 came back at 58–68, and the model's
spread collapsed to sd 8.4 against the annotator's 17.0. Range compression, not
discrimination. Reverted; the full numbers are in the comment at the top of
`museapi.py`.

### Engineering findings

- **Upload status polling was costing 184 seconds per excerpt.** The Files API returns
  status `uploaded` and holds it; the code was waiting for a state it never enters.
  With that removed and six requests in flight, scoring went from 143s to **2.2s per
  excerpt** — a 65× speedup, measured with `timing_probe.py`.
- **The model's own `dominant` label contradicts its own numbers.** Over repeated runs
  it named reactions holding as little as 3% of the distribution it generated in the
  same response. The label is now derived from the distribution's argmax, and it is
  not shown in the interface at all — the plain-English reading comes from valence and
  arousal, which reproduce. The model's claim is kept in the JSON for inspection.
- **Responses can truncate silently.** Muse Spark spends output tokens on reasoning
  before emitting JSON; at a 4000-token ceiling one excerpt in eighteen came back cut
  off mid-array. The ceiling is now 12000 with a retry at 24000 on an `incomplete`
  status.

## Design choices

**It reads the film, not faces.** The obvious version of this product watches an
audience through a camera. That approach rests on a premise the science contests:
Barrett, Adolphs, Marsella, Martinez and Pollak reviewed
[over 1,000 published findings](https://www.psychologicalscience.org/news/releases/emotion-expressions-pspi.html)
in *Psychological Science in the Public Interest* (2019) and concluded that *"it is
not possible to confidently infer happiness from a smile, anger from a scowl or
sadness from a frown"* — people scowl when angry roughly 25% of the time, and also
scowl when concentrating. They note that emotion-detection products apply *"what are
mistakenly believed to be the scientific facts."* The debate is live rather than
settled, but it is reason enough to analyse content instead of people. **No faces are
recorded, uploaded or stored. This tool never sees a viewer.**

**Fixed-length excerpts, not shot detection.** Window length is the user's choice
(5, 10 or 20 seconds). Fixed windows keep the cutting step free of any model, which
keeps the "every runtime AI model is Meta's" rule unambiguous, and 10 seconds matches
the excerpt length used by LIRIS-ACCEDE so the demo and any future evaluation score
the same kind of unit.

**The headline reading comes from valence and arousal only.** Those two reproduce;
the single-word label does not, and the label set has no word for joy or awe, so a
joyful excerpt produced a flat distribution with no winner. The plain reading
("pleasant, energising", "low engagement") is derived from the two stable numbers
using the circumplex model of affect — the frame valence and arousal come from.

## What it is not

A test screening and ViChitra answer different questions, and the honest version of
this section is what makes the rest of the page worth believing.

**It reads the film, not the viewer.** It estimates how a general audience would be
likely to react. It is not a measurement of anyone's feelings, and it is a prompt to
look at a scene again rather than a verdict on it.

**It cannot detect confusion.** A real audience tells you *"I didn't understand why
she went back."* This cannot, and that is the largest single gap against a screening
room — along with individual qualitative reactions of the *"I found the protagonist
unlikeable"* kind. Comedy is the weakest case of all, since whether a joke lands is
enormously audience-dependent.

**One audience, not yours.** "A typical cinema audience" is a convenient fiction:
reaction varies by culture, age, and whether someone is watching alone at 2am or in a
full theatre. The model has one undifferentiated prior.

**It sees each excerpt alone.** Fixed windows cut across shots and beats, and context
from earlier in the film — most of what makes a payoff land — sits outside the window.

**Marking intent is open to anchoring.** The interface shows the predicted curve
before you mark, so a director can drift toward it. Mark intent first, or without
studying the curve.

## Data and attribution

- **Demo film** — *Sintel* (2010), Blender Foundation, licensed CC BY 3.0,
  <https://durian.blender.org>. Eighteen 10-second excerpts from 05:00–08:00 are
  redistributed in `out/jobs/demo/` under that licence so the app runs without an API
  key. Licence screenshot in `licence_sintel.png`.
- **Human annotations** — `my_ratings.csv`, made by the author for this project.
- **LIRIS-ACCEDE** — a larger public evaluation against
  <https://liris-accede.ec-lyon.fr> (CC BY-NC-SA) is supported by `evaluate.py` but
  was not completed within the hackathon window. No dataset files are redistributed
  here; `evaluate.py` reads a local copy you download yourself.

## Built during the hackathon

Every file in this repository was written between 27 October and 4 November 2026.
There is no pre-existing code, no prior project and no reused private assets. The
commit history is the record.

Built with AI coding assistance (Claude), which the hackathon rules permit. All
design decisions, measurements, annotations and the decision to revert the v3 prompt
are the author's. Not written by the author and not claimed as such: the *Sintel*
film, and the open-source libraries in `requirements.txt`.

## What's in here

```
app.py              upload, timeline, intent marking, exportable notes
museapi.py          Meta Model API client, prompt, schema, normalisation
score_clips.py      batch scoring — parallel, resumable, budget-capped
cut_clips.py        ffmpeg slicing (CLI path)
import_scores.py    load an existing scoring run into the app as a job
smoke_test.py       end-to-end API check: auth, video, audio, structured output
timing_probe.py     where the per-excerpt seconds actually go
stability_probe.py  run-to-run reproducibility of a single excerpt
evaluate.py         Spearman with bootstrap CI against any annotation file
my_ratings.csv      the 18 human annotations used above
```

## Licence

MIT. *Sintel* and any dataset remain under their own licences as listed above.
