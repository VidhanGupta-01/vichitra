# To post in #questions (one message, ask the dataset one first)

Post these tonight so an answer can land during Monday's office hours
(10–11 AM ET = 8:30–9:30 PM IST, Mon Nov 2).

---

**1. Research dataset licensing — blocking**

> Following up on the dataset question. I want to evaluate my predictions against
> LIRIS-ACCEDE, a public research dataset of film excerpts with human emotion
> annotations, licensed CC BY-NC-SA. I would not redistribute any clips — I
> download them locally, compute correlation statistics, and publish only the
> numbers, with full attribution in the README. Is that acceptable for a hackathon
> with cash prizes, or does the non-commercial clause rule it out? If it does, is
> there a dataset policy you'd point me to?

*If no answer by Monday evening IST, assume no and fall back:* self-annotate 30–40
excerpts from a CC BY film yourself, report it as a small hand-labelled validation
set, and say exactly that in the README. Weaker, but shippable and clean.

---

**2. Contributor-tier models**

> Does using the Contributor-mode model IDs (e.g. `muse-spark-1.2-contributor`)
> count the same as Standard mode for the "Use of Meta AI Tooling" criterion? The
> discount matters a lot for a solo build doing bulk inference.

---

**3. Demo film rights**

> For the demo I plan to use a Blender Foundation open movie (Sintel / Big Buck
> Bunny / Tears of Steel), which are CC BY. Attributing the film in the README and
> on screen in the video — is that sufficient under the "assets you have rights to"
> rule?

---

## Already answered — do not re-ask

- Every runtime AI model must be Meta's. Non-model libraries are fine. → settled.
- "Live product" = runs from the repo's setup instructions + 3-min video; hosting
  optional. → settled.
- Video goes to Muse Spark via Files API (`purpose: "user_data"`) referenced with
  `input_file`. → settled.
