"""
EduBuddy — central configuration.

Everything tunable lives here: model choices, paths, audio thresholds,
retrieval settings, and the per-language tutor system prompts.
"""
import os
from pathlib import Path

# ---------------------------------------------------------------- paths
BASE_DIR = Path(__file__).resolve().parent
MODELS_DIR = BASE_DIR / "models"
BOOKS_DIR = BASE_DIR / "books"
LIBRARY_DIR = BASE_DIR / "library"

for _d in (MODELS_DIR, BOOKS_DIR, LIBRARY_DIR):
    _d.mkdir(parents=True, exist_ok=True)

# Keep every Hugging Face download inside the project so the whole folder
# is portable and the app stays offline after setup.
os.environ.setdefault("HF_HOME", str(MODELS_DIR / "hf"))
os.environ.setdefault("TRANSFORMERS_NO_ADVISORY_WARNINGS", "1")
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

# Hugging Face's Xet backend (hf-xet) uses a separate chunked CDN that is
# unreliable behind WSL2's NAT — downloads stall at 0% and eventually raise
# ConnectionError from us.aws.cdn.hf.co/xorbs/. The classic HTTP path is slower
# in theory but actually completes, so force it.
os.environ.setdefault("HF_HUB_DISABLE_XET", "1")

# ---------------------------------------------------------------- profile
# "desktop" (default) targets the development laptop: x86-64 with AVX2, 6 cores,
# 10 GB. "pi" targets a Raspberry Pi 4: ARM Cortex-A72, no AVX, 4 slower cores,
# and 8 GB or less. The Pi is roughly 8-15x slower per token, so the profile
# trades model size and answer length for a response time that is merely slow
# rather than unusable.
#
#     EDUBUDDY_PROFILE=pi python web.py
#
# Everything except model sizes and a few limits is shared: retrieval, the Indic
# glyph repair, grounding and the evaluation harness are all architecture-neutral.
PROFILE = os.environ.get("EDUBUDDY_PROFILE", "desktop").lower()
IS_PI = PROFILE == "pi"

# ---------------------------------------------------------------- languages
LANGUAGES = ["en", "hi", "ta"]
LANG_NAMES = {"en": "English", "hi": "Hindi", "ta": "Tamil"}

# ---------------------------------------------------------------- STT
# "small" = best quality/speed balance on CPU. Use "base" to roughly halve latency.
# On the Pi even "base" is too slow, so "tiny" is used; expect noticeably worse
# transcription, particularly for Tamil.
WHISPER_MODEL = "tiny" if IS_PI else "small"
WHISPER_COMPUTE_TYPE = "int8"
WHISPER_DEVICE = "cpu"
WHISPER_DIR = MODELS_DIR / "whisper"
WHISPER_BEAM_SIZE = 1  # greedy; faster on CPU

# ---------------------------------------------------------------- LLM
# A model per language. Each entry is (repo, filename, needs_no_think).
#
# They are NOT all held in memory. Three resident Q4 models plus Whisper, the TTS
# voices, the embedder and three KV caches comes to roughly 9.9 GB against a 9.7 GB
# WSL allocation — the failure mode is a bare "Killed" mid-answer. Instead models
# load on demand and the least-recently-used is evicted once LLM_MAX_RESIDENT is
# exceeded. A tutoring session is almost always single-language, so a switch costs
# one ~3 s load and steady-state memory stays at a single model.
#
# Point several languages at the same entry and it is loaded once and shared.
_Q3B = ("Qwen/Qwen2.5-3B-Instruct-GGUF", "qwen2.5-3b-instruct-q4_k_m.gguf", False)

# Gemma 4 (Apr 2026, Apache 2.0, 140+ languages). Its ~256k-token vocabulary
# tokenises Tamil and Devanagari far more efficiently than Qwen's ~152k, which
# should help both answer quality and speed on Indic text. ~3 GB at Q4.
# Needs llama.cpp from April 2026 or later; an older build reports an unknown
# model architecture on load. _G3_4B is the more mature fallback.
_G4E4B = ("unsloth/gemma-4-E4B-it-GGUF", "gemma-4-E4B-it-Q4_K_M.gguf", False)
_G3_4B = ("unsloth/gemma-3-4b-it-GGUF", "gemma-3-4b-it-Q4_K_M.gguf", False)

# Raspberry Pi: the 2B sibling, ~1.6 GB at Q4. Same family and tokenizer as the
# adopted E4B, so the Indic advantage is retained at roughly half the compute.
# Verify the exact filename before first use:
#   python -c "from huggingface_hub import HfApi; \
#     print([f for f in HfApi().list_repo_files('unsloth/gemma-4-E2B-it-GGUF') \
#            if '4_K_M' in f.upper()])"
_G4E2B = ("unsloth/gemma-4-E2B-it-GGUF", "gemma-4-E2B-it-Q4_K_M.gguf", False)

# Bigger, clearly better at reasoning over textbook excerpts and at Hindi/Tamil.
# ~4.7 GB across two shards; roughly 2-3x slower per answer on CPU.
_Q7B = ("Qwen/Qwen2.5-7B-Instruct-GGUF",
        "qwen2.5-7b-instruct-q4_k_m-00001-of-00002.gguf", False)

# Gemma 4 was adopted on measured results: 86% vs 67% overall and 82% vs 55% on
# Tamil against Qwen2.5-3B, at roughly half the median latency (see the report).
# On the Pi the same family drops to the 2B sibling.
_LLM = _G4E2B if IS_PI else _G4E4B

LLM_MODELS = {
    "en": _LLM,
    "hi": _LLM,
    "ta": _LLM,

    # Tried and reverted: a Tamil-specialised Qwen3-4B fine-tune. Retrieval quality
    # was unaffected (that is handled by hybrid search) and it did not improve Tamil
    # answers, while its replies mixed in Latin text and digits that MMS-TTS cannot
    # speak. Kept here as a documented option — no_think=True is mandatory, since
    # Qwen3 emits a <think> block that otherwise consumes the whole token budget.
    # "ta": ("mradermacher/Qwen3-4B-tamil-16bit-Instruct-GGUF",
    #        "Qwen3-4B-tamil-16bit-Instruct.Q4_K_M.gguf", True),
}

# How many LLMs may stay in memory at once.
#   1 = lowest memory, ~3 s pause when the language changes  (safe on 10 GB)
#   2 = no pause between the two most-used languages          (needs ~12 GB)
LLM_MAX_RESIDENT = 1        # raise to 2 only with >=12 GB; never on a Pi

# Fallback used by verify.py and by setup.sh when pre-downloading.
LLM_REPO, LLM_FILE, LLM_NO_THINK = LLM_MODELS["en"]

LLM_DIR = MODELS_DIR / "llm"
# Smaller context on the Pi: the KV cache is a real fraction of available RAM.
LLM_CTX = 2048 if IS_PI else 4096
# Hard ceiling on answer length. Decode is serial on CPU, so this is roughly a
# linear latency dial: 400 gave ~60 s answers, 160 gives ~20 s. Combined with the
# "2-3 sentences" prompt rule, 160 is ample for English and Hindi.
LLM_MAX_TOKENS = 120 if IS_PI else 160
# Prefill batch. Larger is faster but needs more memory at once; the Pi cannot
# spare it, and a smaller batch also reduces thermal spikes.
LLM_BATCH = 128 if IS_PI else 512
LLM_TEMPERATURE = 0.2      # low: factual recall beats varied phrasing
# Desktop leaves a core for the OS; the Pi has only four and needs them all.
LLM_THREADS = (os.cpu_count() or 4) if IS_PI else max(1, (os.cpu_count() or 4) - 1)
LLM_HISTORY_TURNS = 6       # user+assistant pairs kept in rolling history

# ---------------------------------------------------------------- TTS
TTS_REPOS = {
    "en": "facebook/mms-tts-eng",
    "hi": "facebook/mms-tts-hin",
    "ta": "facebook/mms-tts-tam",
}
TTS_DIR = MODELS_DIR / "tts"
TTS_MAX_CHARS = 600         # long replies are split into sentences before synthesis

# ---------------------------------------------------------------- RAG
EMBED_MODEL = "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"
EMBED_DIR = MODELS_DIR / "embed"
CHUNK_CHARS = 400          # 900 buried key sentences among boilerplate
CHUNK_OVERLAP = 120
# Fewer excerpts on the Pi: prefill dominates latency there, and each chunk is
# ~400 characters of extra prompt to read before a single token is generated.
RAG_TOP_K = 2 if IS_PI else 4
RAG_MIN_SCORE = 0.30        # cosine similarity floor; below this the chunk is ignored

# ------------------------------------------------- hybrid (semantic + lexical)
# On noisy Indic text extracted from print-layout PDFs, embedding scores collapse
# into a narrow band and ranking becomes unreliable. Measured on a Tamil Class-3
# science book (238 chunks), correct-chunk rank by method:
#
#   question                embedding   lexical(n=3)
#   "how long to wash"          1           12
#   "five ways / senses"       31            1
#   "which organ tastes"       18            3
#
# Neither method alone is sufficient; each rescues what the other misses. Results
# are combined with Reciprocal Rank Fusion, which merges rankings rather than raw
# scores, so the two incompatible score scales never need calibrating.
# Character n-grams (not words) are used deliberately: the extracted text has
# spurious spaces inside words, which destroys word tokens but leaves n-grams
# largely intact.
RAG_HYBRID = True
RAG_LEX_NGRAM = 3
RAG_RRF_K = 60              # RRF damping; larger = flatter contribution from tail ranks
RAG_CANDIDATES = 40         # per-method candidate pool before fusion
RAG_LEX_MIN = 0.10          # n-gram overlap that exempts a chunk from RAG_MIN_SCORE

# Always include the opening chunk of the best-matching book. Chapters introduce
# their subject in the first lines and then narrate in the first person, so the
# passage naming the narrator rarely ranks well against a factual question.
RAG_INCLUDE_OPENING = True

# ------------------------------------------------- question packs (instant answers)
# A book is turned into questions once, offline, by the model. At query time a
# student's question is matched against those questions with character n-grams
# and the passage the matched question was written from is returned verbatim,
# with its page citation, in milliseconds. No model runs on this path.
#
# The answer is the textbook's own words, not a cached model reply — which is
# what makes serving one honest. The interface labels it as a direct quote.
#
# Build with:  python qa_pack.py --build
# Probe with:  python qa_pack.py --test "your question"
# ------------------------------------------------- library / study (curated Q&A)
# Hand-written question and answer pairs in library_questions.json, shown in the
# Library tab and quizzed in the Study tab. Nothing here is generated or
# retrieved: the content is what you wrote, and marking is keyword comparison
# with no model in the loop.
#
# Share of an entry's keywords a student must produce to be marked correct.
# Not 1.0 on purpose — demanding every term punishes a correct answer phrased
# differently, and a study tool that calls a right answer wrong teaches the
# wrong lesson. With three keywords, 0.6 means two of them.
LIBRARY_PASS_RATIO = 0.6

# EDUBUDDY_QA=0 turns packs off without editing this file, which is how
# 'edubuddy deploy' reproduces the pre-pack behaviour from the same install.
QA_ENABLED = os.environ.get("EDUBUDDY_QA", "1") not in ("0", "false", "no")
QA_NGRAM = 3
QA_PER_CHUNK = 2            # questions generated per chunk at build time

# The threshold is the safety-critical number. Too low and an off-topic question
# matches on shared function words and is answered with a citation attached,
# which is the exact failure this project treats as worse than no answer.
#
# Measured on a 14-question English pack, 8 genuine paraphrases against 10
# off-topic questions, with IDF-weighted Jaccard:
#
#     threshold   fires on real   fires on off-topic
#       0.35          5 / 8            0 / 10        <- adopted
#       0.40          4 / 8            0 / 10
#       0.45          3 / 8            0 / 10
#       0.50          0 / 8            0 / 10
#
# The two distributions overlap (worst real 0.17, best off-topic 0.30), so no
# threshold separates them perfectly. 0.35 is chosen because the two errors are
# not symmetric: a miss falls through to retrieval and the model, which can
# still answer or refuse, while a false positive ships an unrelated passage with
# a page number attached. Err towards missing.
#
# Re-tune on your own pack before trusting it — `python qa_pack.py --test` on
# questions the books do NOT cover is the check that matters.
QA_MATCH_MIN = 0.35

# A language with fewer entries than this never answers from the pack: in a thin
# pack the nearest question is nearest by accident, not by topic.
QA_MIN_LANG_ENTRIES = 12

# ---------------------------------------------------------------- audio (terminal mode)
SAMPLE_RATE = 16000
FRAME_MS = 30
SILENCE_THRESHOLD = 0.012   # RMS below this counts as silence
SILENCE_DURATION = 1.2      # seconds of silence that ends an utterance
MAX_RECORD_SECONDS = 20
MIN_SPEECH_SECONDS = 0.4

# ---------------------------------------------------------------- server
HOST = "127.0.0.1"
PORT = 7860
AUTO_OPEN_BROWSER = True

# ---------------------------------------------------------------- prompts
# Answer length is the single biggest driver of latency on CPU: every token is
# decoded serially. "Under 90 words" was routinely ignored and produced 60-second
# answers. Two or three sentences is both faster and better suited to speech.
_BASE_RULES = (
    "You are EduBuddy, a friendly school tutor for students aged 10-15. "
    "Your answer will be READ ALOUD, so: use short plain sentences, no markdown, "
    "no bullet points, no asterisks, no emoji, no code blocks. "
    "Answer in AT MOST 2 or 3 short sentences, under 40 words total. "
    "State the answer first, then at most one short line of explanation. "
    "Never repeat yourself and never pad the answer."
)

SYSTEM_PROMPTS = {
    "en": _BASE_RULES + " Always answer in English.",
    "hi": _BASE_RULES + " हमेशा सरल हिंदी में उत्तर दें। देवनागरी लिपि का प्रयोग करें।"
           " हर वाक्य हिंदी में होना चाहिए। अंग्रेज़ी में कुछ भी न लिखें।",
    # The language rule is repeated at the end because the model was observed
    # obeying it for the answer and then appending an English line of
    # explanation. Recency matters in a long prompt. llm.py also strips
    # wrong-script sentences afterwards, since the prompt alone did not hold.
    "ta": _BASE_RULES + " எப்போதும் எளிய தமிழில் பதில் அளிக்கவும். தமிழ் எழுத்துகளைப்"
           " பயன்படுத்தவும். ஒவ்வொரு வாக்கியமும் தமிழில் இருக்க வேண்டும்."
           " ஆங்கிலத்தில் எதையும் எழுத வேண்டாம்.",
}

# Appended when textbook excerpts are supplied.
#
# Retrieval returns the *closest* chunks, never "nothing" — on noisy text it
# happily returns irrelevant passages with high scores. So the model must be
# able to reject its own context. It emits NOT_IN_BOOK_MARKER when the excerpts
# do not answer the question; web.py then strips the marker and drops the
# citations, so an ungrounded answer can never be displayed as if it were sourced.
NOT_IN_BOOK_MARKER = "[NOTINBOOK]"

GROUNDING_PROMPTS = {
    "en": (
        "Textbook excerpts are provided below. The textbook is the ONLY source you "
        "may use, and it overrides anything you believe. Do not add facts, dates, "
        "names or numbers that are not written in the excerpts.\n"
        "When the answer is a specific word, number or name, copy it EXACTLY as "
        "written in the excerpts. If the excerpts give a list of pairs, pick the "
        "one item the question asks about and state it plainly first.\n"
        "Textbook chapters are often narrated in the first person by their "
        "subject. Do not answer as \"I\" or \"me\". If the excerpts state the "
        "narrator's name (for example \"I am Godavari\"), use that name. If they "
        "do NOT name the narrator, say that the excerpts do not say who is "
        "speaking — never substitute another name that merely appears nearby. "
        "In \"I am the second longest river after river Ganga\", Ganga is the "
        "comparison, NOT the speaker.\n"
        f"If the excerpts do not actually answer the question, start your reply "
        f"with {NOT_IN_BOOK_MARKER} and then say briefly what the excerpts do cover. "
        "Being honest that the book does not say is always better than guessing."
    ),
    "hi": (
        "नीचे पाठ्यपुस्तक के अंश दिए गए हैं। केवल इन्हीं का प्रयोग करें। "
        "अंशों में जो नहीं लिखा है, ऐसे तथ्य, तारीख, नाम या संख्या मत जोड़ें।\n"
        "पाठ अक्सर उत्तम पुरुष में लिखे होते हैं। उत्तर में कभी \"मैं\" न लिखें — "
        "अंशों से पहचानें कि कहने वाला कौन है और उसका नाम लिखें।\n"
        f"यदि अंशों में उत्तर नहीं है, तो उत्तर के आरंभ में {NOT_IN_BOOK_MARKER} लिखें "
        "और संक्षेप में बताएं कि अंशों में क्या है। अनुमान लगाने से अच्छा है सच बताना।"
    ),
    "ta": (
        "கீழே பாடநூல் பகுதிகள் உள்ளன. அவற்றை மட்டுமே பயன்படுத்தவும். "
        "பகுதிகளில் இல்லாத தகவல், தேதி, பெயர் அல்லது எண்களை சேர்க்கக் கூடாது.\n"
        "பதில் ஒரு குறிப்பிட்ட சொல், எண் அல்லது பெயராக இருந்தால், பகுதிகளில் "
        "எழுதியுள்ளபடியே அதை அப்படியே எழுதவும். பகுதிகளில் பட்டியல் இருந்தால், "
        "கேள்வி கேட்கும் ஒரு பொருளை மட்டும் தேர்ந்து முதலில் தெளிவாகக் கூறவும்.\n"
        "பாடங்கள் பெரும்பாலும் தன்மையில் எழுதப்படுகின்றன. பதிலில் ஒருபோதும் "
        "\"நான்\" என்று எழுதக் கூடாது — கூறுபவர் யார் என்பதைப் பகுதிகளிலிருந்து "
        "அறிந்து அவரது பெயரைக் குறிப்பிடவும்.\n"
        f"பகுதிகளில் பதில் இல்லையெனில், பதிலின் தொடக்கத்தில் {NOT_IN_BOOK_MARKER} "
        "என எழுதி, பகுதிகளில் என்ன உள்ளது என்பதைச் சுருக்கமாகக் கூறவும். "
        "ஊகிப்பதைவிட உண்மையைச் சொல்வது சிறந்தது."
    ),
}

# Shown (in place of citations) when the model reports the book does not cover it.
NOT_IN_BOOK_MESSAGE = {
    "en": "The textbook does not cover this.",
    "hi": "पाठ्यपुस्तक में यह नहीं है।",
    "ta": "இது பாடநூலில் இல்லை.",
}

NO_SPEECH_MESSAGE = {
    "en": "I did not catch that. Please say it again.",
    "hi": "मुझे सुनाई नहीं दिया। कृपया दोबारा कहिए।",
    "ta": "எனக்குக் கேட்கவில்லை. மீண்டும் சொல்லுங்கள்.",
}


# Whisper regularly reports a neighbouring language for Indic speech — Hindi as
# Urdu most often, since they are acoustically near-identical, and Tamil as one
# of the other southern languages. Falling through to English on those gave a
# Devanagari transcript an English prompt, an English answer and the English
# voice. Fold them onto the nearest language we actually support.
_NEIGHBOURS = {
    "ur": "hi", "mr": "hi", "ne": "hi", "sa": "hi", "bh": "hi", "pa": "hi",
    "ml": "ta", "te": "ta", "kn": "ta", "si": "ta",
}


def normalise_lang(code: str) -> str:
    """Map anything Whisper or the UI hands us onto en/hi/ta."""
    if not code:
        return "en"
    code = code.lower().strip()[:2]
    if code in LANGUAGES:
        return code
    return _NEIGHBOURS.get(code, "en")
