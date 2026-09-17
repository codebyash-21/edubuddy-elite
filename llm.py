"""
Local LLM: llama.cpp with one model per language and LRU eviction.

Each language in config.LLM_MODELS may use a different GGUF, so a Tamil-specialised
model can serve Tamil while a general multilingual model serves English and Hindi.

Models are loaded on demand rather than all at once: three resident Q4 models plus
Whisper, the TTS voices and the embedder exceeds a 10 GB WSL allocation. At most
config.LLM_MAX_RESIDENT stay in memory and the least-recently-used is dropped.
Languages pointing at the same GGUF share one instance, so en+hi cost nothing extra.
"""
from __future__ import annotations

import gc
import re
import threading
import time

import config

_loaded = {}          # model_key -> {"llm":..., "used": float}
_lock = threading.RLock()

# Models whose chat template rejects the "system" role (Gemma family). Learned on
# first failure so the fallback is paid once, not on every message.
_NO_SYSTEM_ROLE = {}


# ----------------------------------------------------------------- model spec
def spec_for(lang: str):
    """Return (repo, filename, no_think) for a language."""
    lang = config.normalise_lang(lang)
    return config.LLM_MODELS.get(lang) or config.LLM_MODELS["en"]


# Large GGUFs are published as shards, e.g. "...-q4_k_m-00001-of-00002.gguf".
# llama.cpp loads them all when given shard 1, but every shard must be present in
# the same directory first — downloading only the named file leaves it broken.
_SHARD_RE = re.compile(r"^(?P<stem>.+)-(?P<idx>\d{5})-of-(?P<total>\d{5})\.gguf$")


def model_path(repo: str = None, filename: str = None) -> str:
    """Local GGUF path, downloading on first use. Handles sharded models."""
    repo = repo or config.LLM_REPO
    filename = filename or config.LLM_FILE
    config.LLM_DIR.mkdir(parents=True, exist_ok=True)

    m = _SHARD_RE.match(filename)
    wanted = [filename]
    if m:
        total = int(m.group("total"))
        stem = m.group("stem")
        wanted = [f"{stem}-{i:05d}-of-{total:05d}.gguf"
                  for i in range(1, total + 1)]

    missing = [f for f in wanted if not (config.LLM_DIR / f).exists()]
    if missing:
        from huggingface_hub import hf_hub_download

        for f in missing:
            print(f"[llm] downloading {f}"
                  f"{f' ({wanted.index(f)+1}/{len(wanted)} shards)' if m else ' (one time)'} ...")
            hf_hub_download(repo_id=repo, filename=f,
                            local_dir=str(config.LLM_DIR))

    # Always hand llama.cpp the first shard; it discovers the rest itself.
    return str(config.LLM_DIR / wanted[0])


def download_all(verbose: bool = True):
    """Fetch every distinct GGUF referenced by LLM_MODELS."""
    seen = set()
    for lang, (repo, filename, _) in config.LLM_MODELS.items():
        if filename in seen:
            continue
        seen.add(filename)
        if verbose:
            print(f"[llm] ensuring {filename} (for {lang}) ...")
        model_path(repo, filename)


# --------------------------------------------------------------- load / evict
def _evict_if_needed():
    """Drop least-recently-used models above the resident cap. Caller holds _lock."""
    limit = max(1, getattr(config, "LLM_MAX_RESIDENT", 1))
    while len(_loaded) > limit:
        oldest = min(_loaded, key=lambda k: _loaded[k]["used"])
        print(f"[llm] evicting {oldest} to stay within "
              f"LLM_MAX_RESIDENT={limit}")
        _loaded.pop(oldest, None)
        gc.collect()          # release llama.cpp's buffers promptly


def load(lang: str = "en", verbose: bool = True):
    """Load (or reuse) the model for one language and return it."""
    repo, filename, _ = spec_for(lang)

    with _lock:
        entry = _loaded.get(filename)
        if entry is not None:
            entry["used"] = time.time()
            # Evict on the cache-hit path too: otherwise lowering
            # LLM_MAX_RESIDENT at runtime would never take effect, since the
            # requested model is already loaded and we would return immediately.
            _evict_if_needed()
            return entry["llm"]

        from llama_cpp import Llama

        path = model_path(repo, filename)
        if verbose:
            print(f"[llm] loading {filename} on {config.LLM_THREADS} threads ...")
        t0 = time.time()
        inst = Llama(
            model_path=path,
            n_ctx=config.LLM_CTX,
            n_threads=config.LLM_THREADS,
            n_batch=getattr(config, "LLM_BATCH", 256),
            verbose=False,
        )
        _loaded[filename] = {"llm": inst, "used": time.time()}
        if verbose:
            print(f"[llm] ready in {time.time() - t0:.1f}s")

        _evict_if_needed()
        return inst


def is_ready(lang: str = None) -> bool:
    with _lock:
        if lang is None:
            return bool(_loaded)
        return spec_for(lang)[1] in _loaded


def resident():
    with _lock:
        return sorted(_loaded)


# --------------------------------------------------------------- text cleanup
# The <think> patterns matter for Qwen3-family models, which emit a reasoning
# block before the answer. Left in, it would be read aloud and would also consume
# the LLM_MAX_TOKENS budget. The unclosed variant catches a reasoning block that
# the token limit truncated, where no closing tag ever arrives.
_CLEAN_PATTERNS = [
    (re.compile(r"<think>.*?</think>", re.S | re.I), " "),
    (re.compile(r"<think>.*$", re.S | re.I), " "),
    (re.compile(r"</?think>", re.I), " "),
    (re.compile(r"```.*?```", re.S), " "),
    (re.compile(r"[*_#`]+"), ""),
    (re.compile(r"^\s*[-•]\s*", re.M), ""),
    (re.compile(r"\s{2,}"), " "),
]


def clean_for_speech(text: str) -> str:
    for pattern, repl in _CLEAN_PATTERNS:
        text = pattern.sub(repl, text)
    return text.strip()


_SENTENCE = re.compile(r"[^.!?।]*[.!?।]|[^.!?।]+")
_LATIN = re.compile(r"[A-Za-z]")
_DEV = re.compile(r"[ऀ-ॿ]")
_TAM = re.compile(r"[஀-௿]")
_SCRIPT_OF = {"hi": _DEV, "ta": _TAM}


def strip_foreign_sentences(text: str, lang: str) -> str:
    """Drop whole sentences written in the wrong script.

    Asked in Tamil, the model answers in Tamil and then appends an English line
    of explanation — "This is listed under the instructions for washing hands."
    The prompt already forbids it and the model does it anyway, so this is the
    deterministic backstop.

    Sentences, not characters: "2 நிமிடங்கள்" must keep its digits, and an
    English answer may legitimately contain a proper noun. A sentence is foreign
    only when it has none of the target script and does have letters of another.

    Never drops the not-in-book marker — removing it would turn a refusal into
    an apparently grounded answer, complete with citations, which is the exact
    dishonesty the marker exists to prevent.
    """
    want = _SCRIPT_OF.get(lang)
    marker = getattr(config, "NOT_IN_BOOK_MARKER", "")

    kept, dropped = [], []
    for raw in _SENTENCE.findall(text):
        s = raw.strip()
        if not s:
            continue
        if marker and marker in s:
            kept.append(raw)
            continue
        if want is None:                       # target is English
            foreign = (_DEV.search(s) or _TAM.search(s)) and not _LATIN.search(s)
        else:
            foreign = _LATIN.search(s) and not want.search(s)
        (dropped if foreign else kept).append(raw)

    if not dropped:
        return text
    out = "".join(kept).strip()
    # If filtering removed everything the model said, it answered entirely in
    # the wrong language. Return the original: a visibly wrong-language answer
    # is a bug the user can see and report, an empty one looks like a crash.
    if not out:
        return text
    print(f"[llm] dropped {len(dropped)} non-{lang} sentence(s) from the reply")
    return out


def build_system_prompt(lang: str, context: str = "") -> str:
    lang = config.normalise_lang(lang)
    prompt = config.SYSTEM_PROMPTS[lang]
    if context:
        prompt += "\n\n" + config.GROUNDING_PROMPTS[lang]
        prompt += "\n\n--- TEXTBOOK EXCERPTS ---\n" + context + "\n--- END EXCERPTS ---"
    # Qwen3 soft switch; harmless text for models that don't recognise it.
    if spec_for(lang)[2]:
        prompt += "\n/no_think"
    return prompt


# ---------------------------------------------------------------- conversation
class Tutor:
    """One conversation. History resets when the language changes, since both the
    reply language and possibly the underlying model change with it."""

    def __init__(self):
        self.history = []
        self._lang = None
        self._lock = threading.Lock()

    def clear(self):
        with self._lock:
            self.history = []
            self._lang = None

    def _trim(self):
        keep = config.LLM_HISTORY_TURNS * 2
        if len(self.history) > keep:
            self.history = self.history[-keep:]

    def ask(self, question: str, lang: str = "en", context: str = "") -> str:
        lang = config.normalise_lang(lang)
        model = load(lang)

        with self._lock:
            if lang != self._lang:
                self.history = []
                self._lang = lang

            system = build_system_prompt(lang, context)
            key = spec_for(lang)[1]

            def call(messages):
                return model.create_chat_completion(
                    messages=messages,
                    max_tokens=config.LLM_MAX_TOKENS,
                    temperature=config.LLM_TEMPERATURE,
                    top_p=0.9,
                    repeat_penalty=1.1,
                )

            with_system = ([{"role": "system", "content": system}]
                           + self.history
                           + [{"role": "user", "content": question}])
            # Gemma-family chat templates reject the "system" role outright. All
            # of the tutor instructions and the textbook excerpts live in that
            # message, so instead fold them into the first user turn.
            merged = (self.history
                      + [{"role": "user", "content": f"{system}\n\n{question}"}])

            if _NO_SYSTEM_ROLE.get(key):
                result = call(merged)
            else:
                try:
                    result = call(with_system)
                except Exception as exc:
                    if "system" not in str(exc).lower():
                        raise
                    print(f"[llm] {key} rejects the system role; "
                          f"folding instructions into the user turn")
                    _NO_SYSTEM_ROLE[key] = True
                    result = call(merged)

            reply = clean_for_speech(
                result["choices"][0]["message"]["content"].strip())
            reply = strip_foreign_sentences(reply, lang)

            self.history.append({"role": "user", "content": question})
            self.history.append({"role": "assistant", "content": reply})
            self._trim()

        return reply


tutor = Tutor()
