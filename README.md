# EduBuddy Dynamic

Offline multilingual AI voice tutor — **English / Hindi / Tamil**.

A student speaks or types a question; the system transcribes it, detects the
language, retrieves relevant passages from indexed textbook PDFs, answers with a
local language model, and speaks the reply back in the same language. After a
one-time setup download **no internet is required** — nothing leaves the device.

Runs on both a **laptop (x86-64 Linux / WSL2)** and a **Raspberry Pi 4 or 5
(ARM64)** from the same codebase.

### What "Dynamic" means

This is the current build of the project. Against the earlier `edubuddy deploy`
it adds two things, both aimed at the same problem — a thirty-second wait on
CPU-only hardware.

**Progressive display.** The interface shows what was found about a second in,
while the model is still writing, instead of holding a blank screen until the
whole answer exists. Retrieval and generation have very different costs and the
interface now reflects that rather than hiding it.

**Question packs.** A book is turned into questions once, offline, by the model.
At query time a student's question is matched against those questions with
IDF-weighted character n-grams — no model runs — and the passage the matched
question was written from is returned **verbatim**, with its page citation, in
milliseconds. The interface labels these "From the textbook", because a
prewritten passage presented as EduBuddy's own sentence would be the dishonest
way to be fast.

```bash
python qa_pack.py --build --limit 20     # trial run first
python qa_pack.py --build                # the real thing (slow, one time)
python qa_pack.py --test "your question" # what would match, and by how much
```

**Library and Study.** Two more tabs beside Ask, both driven by
`library_questions.json` — question and answer pairs **you write by hand**.
Nothing is generated or retrieved, so both tabs respond instantly.

- **Library** browses the pairs with language and book filters, a citation on
  each, and a *Read all aloud* control that speaks every question and answer in
  turn.
- **Study** asks one question at a time. The student answers by typing or
  speaking, and is marked immediately on keyword overlap — no model, so the
  verdict arrives in milliseconds and can always be explained by which terms
  were found. The correct answer is then shown and read aloud.

```bash
python library.py --validate               # before any demo
python library.py --suggest                # propose keywords for new entries
python library.py --check en-1 "Sirius"    # try the marking yourself
```

`--validate` is the one that matters: besides missing fields and duplicate ids,
it checks that every keyword actually appears in your own answer. A keyword that
does not would mark a correct student wrong, which is the worst thing this file
can do.

Marking passes at `LIBRARY_PASS_RATIO` (0.6) of an entry's keywords —
deliberately not all of them, because demanding every term punishes a correct
answer phrased differently.

Everything else is unchanged and still validated: Gemma 4, hybrid retrieval,
the Indic glyph repair, grounding with refusal, and the evaluation harness.

---

## Quick start

### On a laptop (Ubuntu or WSL2)

```bash
chmod +x setup.sh
./setup.sh                     # one time, ~30-50 min

source .venv/bin/activate
python web.py                  # browser UI at http://127.0.0.1:7860
python app.py                  # or the desktop window
```

**A shortcut worth installing**, so you are not typing three lines every time:

```bash
echo 'source ~/edubuddy/edubuddy/edubuddy.sh' >> ~/.bashrc
source ~/.bashrc
```

Then `edubuddy` starts the browser UI from anywhere, `edubuddy app` opens the
window, `edubuddy compare` runs the pack comparison, `edubuddy help` lists the
rest. Tab completion included.

`edubuddy deploy` runs the **same install with question packs switched off**,
which is exactly the pre-pack behaviour. There is deliberately no second
installation: two copies would drift apart, and any comparison between them
would then be measuring the drift rather than the packs. The switch is one
environment variable, `EDUBUDDY_QA=0`.

Two interfaces, one engine. `web.py` serves a browser page; `app.py` opens a
Tkinter window. Identical retrieval, question packs, grounding and refusal — the
window is a different presentation, not a different system, and it is not faster.

Which to use:

- **`web.py`** if you need the **microphone**. Browsers hand it over freely on
  `127.0.0.1`; under WSL there is usually no audio device at all, so `app.py`
  falls back to typing. It is also what the Raspberry Pi setup relies on.
- **`app.py`** if you want a native window with no address bar — a cleaner look
  for a demo. `python app.py --no-speak` skips synthesis.

### On a Raspberry Pi 4 or 5 (64-bit Raspberry Pi OS)

> **Setting this up on a Pi for the first time? Follow [SETUP_PI.md](SETUP_PI.md)
> instead.** It covers flashing the OS, per-stage timings, the textbook handover
> and a troubleshooting table. The summary below assumes you already know your
> way around.

Start to finish, from a clean Pi:

```bash
git clone https://github.com/<user>/edubuddy.git
cd edubuddy

chmod +x setup_pi.sh
./setup_pi.sh                  # one time; builds llama.cpp for ARM, 40-70 min

# add textbooks (see "Textbooks are not in this repository" below)
cp /path/to/*.pdf books/
source .venv/bin/activate
python ingest.py --all

EDUBUDDY_PROFILE=pi python web.py
```

`setup_pi.sh` refuses to run on under 4 GB of RAM, warns if you have booted from
a microSD card rather than SSD or USB, and checks the SoC temperature before
starting a long compile. It compiles `llama-cpp-python` from source with NEON
and OpenBLAS enabled — the prebuilt wheels are x86-only, so this step cannot be
skipped, and it is most of the 40-70 minutes.

Then browse from another device on the network to `http://<pi-ip>:7860`
(find the address with `hostname -I`).

**Which profile to use.** `EDUBUDDY_PROFILE=pi` selects the smaller models and
tighter limits. A **Pi 5 with 16 GB** can instead run the desktop profile
unchanged — just omit the variable — which reproduces the validated laptop
results exactly, at roughly 60-90 s per answer. A Pi 4, or an 8 GB Pi 5, should
stay on the `pi` profile.

> **Microphone over the network:** browsers only allow microphone access on a
> secure origin. `127.0.0.1` counts; `http://<pi-ip>` does not, so the mic will
> be blocked when accessing the Pi remotely. Typing works regardless. For voice,
> either enable Chrome's *Insecure origins treated as secure* flag for that
> address, or put the server behind HTTPS.

> **Thermals.** Sustained CPU inference will pin all four cores for the length of
> an answer. A Pi 5 needs active cooling for this; without a fan it will throttle
> and answers get slower, though nothing is damaged. Check with
> `vcgencmd measure_temp` while a question is being answered.

---

## Textbooks are not in this repository

School textbook PDFs are copyrighted, so `books/` ships empty and is excluded by
`.gitignore`. Nothing works until you add your own.

```bash
cp your_chapter.pdf books/
python ingest.py --all         # or just restart the server
python ingest.py --list        # confirm it indexed
```

To reproduce the validation results in `results/` you need the same three
chapters used there — an English astronomy chapter, a Hindi story chapter and a
Tamil Class-3 science chapter. **Ask Arun for these files directly**; the
question set in `testcases.json` is written against them and will not match any
other book.

The filename becomes the citation label, so `science_class7.pdf` produces
citations reading `science_class7 p.42`. Name files accordingly, and keep the
names identical to the originals if you want the recorded results to line up.

---

## The two profiles

One codebase, sized differently per target. Set with `EDUBUDDY_PROFILE`.

| | `desktop` (default) | `pi` |
|---|---|---|
| Language model | Gemma 4 E4B (~3.0 GB) | Gemma 4 E2B (~1.6 GB) |
| Speech recognition | Whisper `small` | Whisper `tiny` |
| Context window | 4096 | 2048 |
| Max answer tokens | 160 | 120 |
| Retrieved chunks | 4 | 2 |
| Threads | cores − 1 | all cores |

Everything else is shared: all three languages, hybrid retrieval, the Indic
text repairs, grounding and refusal, and the evaluation harness.

A **Pi 5 with 16 GB** can run the desktop profile unchanged, which reproduces
the laptop's validated results exactly. A Pi 4 needs the `pi` profile.

Expect roughly **26 s** per grounded answer on a laptop, **60–90 s** on a Pi 5,
and **2–3 min** on a Pi 4.

---

## Adding more textbooks

Drop PDFs into `books/` and they are indexed at startup, as above. Two caveats:

- **Image-only (scanned) PDFs have no text layer** and are skipped with a
  warning. They need an OCR pre-pass first.
- **Indic PDFs are repaired on extraction.** Many Hindi and Tamil textbook PDFs
  store pre-composed glyph clusters that extract with duplicated consonants and
  vowel signs, which silently breaks search. `rag.py` repairs this
  automatically. If a new book still retrieves badly, check the extracted text
  before blaming the retriever — `python ingest.py --search "some phrase"` is
  the fastest way to look.

---

## Checking it works

```bash
python verify.py               # install self-check with load timings
python ingest.py --list        # indexed books and chunk counts
python ingest.py --search "your question"   # test retrieval without the LLM

python diagnose.py             # what retrieval hands the model, per case (seconds)
python research_diagnostics.py # retrieval ablations + metrics  (~2 min)
python benchmark.py            # where the time actually goes, per stage
python benchmark.py --compare-pack --limit 6   # deploy vs dynamic, measured
python evaluate.py --no-tts    # full 34-question end-to-end run (~30 min)
```

Run them in that order. The first three load no language model at all, so they
finish in seconds to minutes and tell you whether a problem is even worth
investigating end-to-end.

`diagnose.py` labels every test case `ALL PRESENT`, `PARTIAL` or `NONE FOUND`,
which separates *"the search found the wrong page"* from *"the model misread the
right page"*. Those two have completely different fixes, and diagnosing the
wrong layer is the most expensive mistake available here.

`research_diagnostics.py` re-measures the whole test set with each design
decision switched off in turn — hybrid retrieval, opening-chunk injection,
n-gram size, the cosine floor, context size — reporting coverage, first-rank and
MRR for each.

`evaluate.py` is the end-to-end run. It scores generated answers automatically
and writes timestamped CSV / Markdown / JSON into `results/` along with a
snapshot of every configuration parameter, so runs are reproducible and two
models can be compared fairly. On a Pi expect this to take a few hours; start
with `--lang en` to sanity-check the install before committing to a full pass.

---

## Files

| File | Purpose |
|---|---|
| `config.py` | All settings: profiles, per-language models, retrieval parameters, tutor prompts |
| `web.py` | FastAPI server — `/transcribe`, `/chat`, `/speak`, `/clear`, `/health`, `/books` |
| `ui.html` | Browser frontend: language selector, citations, silence detection, continuous mode |
| `stt.py` | Speech to text (faster-whisper) |
| `llm.py` | Language model: per-language registry, LRU eviction, sharded downloads |
| `tts.py` | Speech synthesis (MMS-TTS), one cached voice per language |
| `rag.py` | PDF extraction with Indic glyph repair, chunking, hybrid retrieval |
| `ingest.py` | Textbook indexing CLI |
| `main.py` | Terminal mode (`--text`, `--mute`, `--lang`) |
| `verify.py` | Post-install self-check |
| `evaluate.py` | End-to-end validation harness with config snapshotting |
| `diagnose.py` | Retrieval inspector — no model loaded |
| `research_diagnostics.py` | Ablation harness: coverage, first-rank, MRR per design choice |
| `qa_pack.py` | Builds the question pack offline (the model writes the questions) |
| `qa_match.py` | Matches a question against the pack at query time — no model |
| `library.py` | Curated Q&A loading, validation, and keyword marking |
| `library_questions.json` | **Your** question and answer pairs — edit this one |
| `benchmark.py` | Per-stage timings and an `LLM_MAX_TOKENS` sweep |
| `app.py` | Tkinter desktop window — same engine as `web.py` |
| `edubuddy.sh` | Shell shortcut: `edubuddy`, `edubuddy app`, `edubuddy compare`… |
| `testcases.json` | The 34-question validation set (21 short, 10 long, 3 negative controls) |
| `setup.sh` / `setup_pi.sh` | Installers for x86-64 and ARM |
| `SETUP_PI.md` | Step-by-step Raspberry Pi setup guide |
| `books/` `library/` `results/` | Your PDFs, the generated index, evaluation output |

---

## Measured results

### Full validation run — 34 questions, Gemma 4 E4B

| Scope | Cases | Pass | Partial | Fail | Pass rate | Retrieval OK | Median |
|---|---|---|---|---|---|---|---|
| English | 13 | 10 | 2 | 1 | 77% | **100%** | 31.6 s |
| Hindi | 10 | 7 | 2 | 1 | 70% | **100%** | 37.3 s |
| Tamil | 11 | 9 | 2 | 0 | 82% | **100%** | 38.0 s |
| **All** | **34** | **26** | **6** | **2** | **76%** | **100%** | **37.3 s** |

Measured at `RAG_TOP_K = 6`; the shipped default is 4, which is ~11 s faster per
answer at the cost of one Hindi case. Retrieval delivered the correct page for
every book-backed question, so all eight non-passes are generation-side — the
model had the right page and misread it.

### Model comparison — 21 questions, identical settings

Only the model changed between these two runs.

| Scope | Cases | Qwen2.5-3B | Gemma 4 E4B |
|---|---|---|---|
| English | 5 | 100% | 80% |
| Hindi | 5 | 60% | 100% |
| Tamil | 11 | 55% | **82%** |
| **All** | **21** | **67%** | **86%** |
| Median answer | | 46.8 s | **26.2 s** |

Retrieval located the correct page in **100%** of book-backed questions in both
runs. Gemma 4 was adopted on these numbers: more accurate *and* faster, because
its larger vocabulary tokenises Indic scripts more efficiently.

Two caveats worth stating: the Hindi comparison is confounded (the chapter was
replaced between runs), and the English difference is one question out of five,
which is noise rather than evidence. **Tamil is the clean comparison** — same
book, same questions, same settings.

The 76% above is *not* a regression from this 86%. The two measure different
test sets: the English chapter was replaced and ten harder long-answer questions
were added. Tamil scores 82% in both, which is the evidence that nothing
regressed.

Full method, ablations and per-decision measurements are in
`EduBuddy_Diagnostic_Report.pdf`.

---

## Known limitations

- **Indic answer generation** is the weak layer. Retrieval finds the right page
  100% of the time; the model still misreads it on some list-selection questions.
- **One Hindi case is unanswerable at the shipped context size** — its three
  required points span pages 11-14. Coverage is 0% at `k=6`, 67% at `k=12`.
- **One Tamil test case is broken**: no combination of four or fewer chunks
  contains all four expected terms, so it can never exceed PARTIAL. This is a
  fault in the test set, not the system.
- **Latency** is not conversational on CPU-only hardware: ~37 s median on a
  laptop, an estimated 60-90 s on a Pi 5.
- **All Raspberry Pi timings are estimates** from published benchmarks. They
  have not been reproduced on our own hardware. If you are the one running this
  on a Pi — please record real numbers and replace them.
- **Scanned textbooks** are unsupported without an OCR pre-pass.
- **Microphone is blocked over the network** (browser secure-origin rule); see
  the Pi quick start above.
