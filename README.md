# ViChitra

Predicts, scene by scene, how a cinema audience is likely to react to a cut —
before anyone has watched it. No cameras, no test screening, no audience.

*chitra* — picture. *vichitra* — strange, wonderful.

Built solo by Vidhan for the Meta Global AI Developer Hackathon
(Indie Developer track), 27 Oct – 4 Nov 2026.

---

## What this is

Upload a cut. The tool splits it into short excerpts, sends each one to Muse Spark
with its audio track, and returns a predicted audience-reaction curve: valence,
arousal, and a breakdown of which reactions the excerpt is likely to produce.

A director can also mark the reaction they *intended* for each stretch. The tool
then shows where intent and prediction diverge — the scenes that are supposed to
be funny and aren't, the stretch that is supposed to build tension and flatlines.

**This estimates the expected reaction to the content. It is not a measurement of
any individual's feelings.** See [Limitations](#limitations).

## Meta models used, and where each one runs

| Model | Where it runs | What it does here |
|---|---|---|
| `muse-spark-1.2` | Meta Model API (`https://api.meta.ai/v1`) | Watches each excerpt with its audio and returns the structured reaction prediction. 1.2 rather than 1.3 because Meta's docs state audio understanding is degraded in 1.3, and the audio track carries much of the reaction. |

<!-- Add rows as you integrate them, and delete the ones you don't ship: -->
<!-- | `sam-3.1` | Meta Model API | Segments on-screen subjects per excerpt to derive shot scale and subject count as explainable features alongside the prediction. | -->
<!-- | `muse-spark-1.3` | Meta Model API | Agent pass over the scored timeline that proposes concrete edit changes for the scenes with the largest intent/prediction gap. | -->

**No non-Meta AI model runs at any point.** The only other dependencies are
non-model libraries: `ffmpeg` for cutting clips, `numpy` and `scipy` for the
correlation statistics, and the OpenAI Python SDK used purely as an HTTP client
against Meta Model API's compatible endpoint.

## Built during the hackathon

Everything in this repository was written between 27 Oct and 4 Nov 2026. There is
no pre-existing code, no prior project, and no reused private assets. The commit
history is the record.

Not written by me, and not claimed as mine:
- The LIRIS-ACCEDE dataset and its human annotations (see [Data](#data)).
- The demo film (see [Data](#data)).
- Open-source libraries listed in `requirements.txt`.

## Run it

Needs Python 3.10+, `ffmpeg` on PATH, and a Meta Model API key.

```bash
git clone <this repo> && cd vichitra
pip install -r requirements.txt
export MODEL_API_KEY=your_key_here

# 1. confirm the API works end to end (cheap, a few cents)
python smoke_test.py --clip samples/one_clip.mp4

# 2. cut a film into 10-second excerpts
python cut_clips.py --video samples/demo_film.mp4 --out out/demo_clips --seconds 10

# 3. score them
python score_clips.py --clips out/demo_clips --out out/demo.jsonl --budget 5

# 4. open the timeline
python app.py          # then http://localhost:8000
```

`score_clips.py` is resumable and budget-capped: it appends one JSON line per
excerpt as soon as that excerpt is scored, skips anything already scored on a
re-run, and stops when estimated spend passes `--budget`.

### For judges

<!-- Fill this in on Nov 3-4. Judges need the fastest honest path to seeing it work. -->
- A hosted instance is at: `<URL>`. It serves pre-scored results for the demo film
  so it works without an API key and without waiting on inference.
- To see live inference, run steps 1–4 above with your own key. Scoring 30
  excerpts takes about `<N>` minutes and costs roughly `$<N>`.
- `out/demo.jsonl` and `out/results.md` in this repo are the exact outputs the
  demo video shows.

## Evaluation

Predictions are checked against **LIRIS-ACCEDE**, a public set of film excerpts
with human valence and arousal annotations collected from real viewers.

```bash
python evaluate.py --scores out/liris_scores.jsonl \
                   --truth data/liris/ACCEDEranking.txt --inspect   # see columns
python evaluate.py --scores out/liris_scores.jsonl \
                   --truth data/liris/ACCEDEranking.txt \
                   --id-col name --valence-col valenceRank --arousal-col arousalRank \
                   --report out/results.md
```

Results, including the shuffled-prediction null baseline, are in
[`out/results.md`](out/results.md). They are reported whatever they say.

## Data

- **LIRIS-ACCEDE** — film excerpts with human affective annotations,
  <https://liris-accede.ec-lyon.fr>. Licensed CC BY-NC-SA. Used here for
  non-commercial evaluation only. Clips are **not** redistributed in this
  repository; `evaluate.py` reads a local copy you download yourself.
  <!-- Cite the dataset paper here in full once confirmed. -->
- **Demo film** — `<title>` by `<author>`, licensed `<CC BY x.x>`, from `<URL>`.
  Used and attributed under that licence.

## Limitations

Stated plainly, because the honest version is the useful version.

1. **This predicts reaction from content. It does not measure emotion.** Visible
   expression is not a reliable readout of internal feeling, and this tool does not
   even see a viewer — it sees the film.
2. **"A typical cinema audience" is a fiction.** Reaction varies by culture, age,
   context, and whether someone is watching alone at 2am or in a full theatre. The
   model has one undifferentiated prior.
3. **Validated on a narrow slice.** LIRIS-ACCEDE is a specific corpus of short
   excerpts. Correlation there does not guarantee correlation on your cut.
4. **Excerpt boundaries are arbitrary.** Fixed-length windows cut across shots and
   beats. A reaction that builds over 40 seconds is scored in four pieces.
5. **It cannot see what it has not been given.** Context from earlier in the film,
   which is most of what makes a payoff land, is outside each excerpt's window.
6. **Not a decision-maker.** It is a prompt for a director to go look at a scene
   again, nothing more.

No faces are recorded, uploaded, or stored. The tool analyses film content, never
people.

## Licence

`<choose one — MIT is the usual default>`. Dataset and demo film are under their own
licences as listed above.
