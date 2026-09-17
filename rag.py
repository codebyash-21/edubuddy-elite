"""
Textbook library: PDF -> page-aware chunks -> multilingual embeddings ->
cosine retrieval with page citations.

Each book is stored as two files in library/:
    <book_id>.json  chunk text + page numbers
    <book_id>.npy   normalised float32 embedding matrix

Re-indexing a book overwrites its previous entry. A fingerprint (size + mtime)
lets startup skip books that have not changed.
"""
from __future__ import annotations

import json
import re
import threading

import numpy as np

import config

_embedder = None
_lock = threading.Lock()


def load_embedder(verbose: bool = True):
    global _embedder
    if _embedder is not None:
        return _embedder
    with _lock:
        if _embedder is not None:
            return _embedder
        from sentence_transformers import SentenceTransformer

        if verbose:
            print("[rag] loading multilingual embedder ...")
        _embedder = SentenceTransformer(
            config.EMBED_MODEL, cache_folder=str(config.EMBED_DIR)
        )
        if verbose:
            print("[rag] embedder ready")
    return _embedder


def is_ready() -> bool:
    return _embedder is not None


def _book_id(pdf_path) -> str:
    from pathlib import Path

    stem = Path(pdf_path).stem
    return re.sub(r"[^A-Za-z0-9_.-]+", "_", stem).strip("_") or "book"


def _fingerprint(pdf_path) -> str:
    """Identify a book AND the settings it was chunked with.

    The chunk parameters must be part of this. Otherwise changing CHUNK_CHARS
    leaves every existing book looking "unchanged", the re-index is silently
    skipped, and you keep searching a stale index built with the old settings.
    """
    from pathlib import Path

    st = Path(pdf_path).stat()
    return (f"{st.st_size}-{int(st.st_mtime)}"
            f"-c{config.CHUNK_CHARS}-o{config.CHUNK_OVERLAP}")


# --------------------------------------------------------------- Tamil repair
# Many Tamil textbook PDFs (TAU-* fonts) store pre-composed glyph clusters.
# PyMuPDF extracts these with the base consonant and the pulli duplicated, e.g.
#     ககற்்றல் நோFோக்்கங்்கள்   instead of   கற்றல் நோக்கங்கள்
#     தொ<FFFD>ொலைக்்ககாட்சி      instead of   தொலைக்காட்சி
# Content words break while filler words survive, so retrieval quietly fails.
# These substitutions are measurable: on a Class-3 science book they take
# doubled-pulli occurrences from 1709 to 0 and recover words that previously
# never appeared. Only Tamil codepoints are touched, so other scripts are safe.
_TA_PULLI = "்"
_TA_CONS = "க-ஹ"
_TA_VOWEL = "ா-ௌௗ"

_TA_FIXES = [
    (re.compile("�"), ""),                                        # replacement chars
    (re.compile(f"{_TA_PULLI}{{2,}}"), _TA_PULLI),                     # ்் -> ்
    (re.compile(f"({_TA_PULLI})([{_TA_CONS}])\\2"), r"\1\2"),          # ்பப -> ்ப
    (re.compile(f"(?<![{_TA_CONS}{_TA_VOWEL}{_TA_PULLI}])"
                f"([{_TA_CONS}])\\1"), r"\1"),                         # word-initial கக -> க
    (re.compile(f"([{_TA_VOWEL}])\\1"), r"\1"),                        # doubled vowel signs
]


# Devanagari textbook PDFs show the same class of fault: the vowel sign is emitted
# twice, so "बाहर" extracts as "बााहर" and "था" as "थाा". A consonant can never
# legitimately carry two identical vowel signs, so collapsing runs is safe.
# Measured on an NCERT Class-5 Hindi chapter: 626 doubled signs -> 0, and words
# that previously never appeared ("बाहर", "मैदान") are recovered.
_DEV_RANGE = "ऀ-ॿ"
_DEV_MATRA = "ा-ौॢॣ"
_DEV_VIRAMA = "्"

_DEV_FIXES = [
    (re.compile("([" + _DEV_MATRA + "])\\1+"), r"\1"),
    (re.compile(_DEV_VIRAMA + "{2,}"), _DEV_VIRAMA),
    (re.compile("�"), ""),
]


def _repair_text(text: str) -> str:
    """Undo glyph-duplication artefacts from Indic PDF text extraction.

    Each script is only touched when its own characters are present, so English
    text is returned byte-identical.
    """
    if re.search(f"[{_TA_CONS}]", text):
        for pattern, repl in _TA_FIXES:
            text = pattern.sub(repl, text)
    if re.search(f"[{_DEV_RANGE}]", text):
        for pattern, repl in _DEV_FIXES:
            text = pattern.sub(repl, text)
    return text


def _extract_pages(pdf_path):
    """Return [(page_number, text), ...] using PyMuPDF."""
    import fitz  # PyMuPDF

    pages = []
    with fitz.open(pdf_path) as doc:
        for i, page in enumerate(doc):
            text = page.get_text("text") or ""
            text = _repair_text(text)
            text = re.sub(r"[ \t]+", " ", text)
            text = re.sub(r"\n{3,}", "\n\n", text).strip()
            if text:
                pages.append((i + 1, text))
    return pages


def _chunk_pages(pages):
    """Split each page into overlapping chunks, keeping the page number."""
    chunks = []
    size, overlap = config.CHUNK_CHARS, config.CHUNK_OVERLAP
    step = max(1, size - overlap)

    for page_no, text in pages:
        if len(text) <= size:
            chunks.append({"page": page_no, "text": text})
            continue
        for start in range(0, len(text), step):
            piece = text[start:start + size].strip()
            if len(piece) > 80:  # skip slivers
                chunks.append({"page": page_no, "text": piece})
            if start + size >= len(text):
                break
    return chunks


# ------------------------------------------------------- hybrid search helpers
_WS = re.compile(r"\s+")


def _ngrams(text: str, n: int = None) -> set:
    """Character n-grams with whitespace removed.

    Whitespace is stripped on purpose: the extracted text has spurious spaces
    inside words ("பார் வை" for "பார்வை"), which ruins word tokens but leaves
    character n-grams almost untouched.
    """
    n = n or config.RAG_LEX_NGRAM
    s = _WS.sub("", text)
    if len(s) < n:
        return {s} if s else set()
    return {s[i:i + n] for i in range(len(s) - n + 1)}


def _gram_overlap(q_grams: set, text: str) -> float:
    """Fraction of the query's n-grams present in this chunk."""
    if not q_grams:
        return 0.0
    return len(q_grams & _ngrams(text)) / len(q_grams)


def _rrf(cos, lex, candidates: int, k: int):
    """Reciprocal Rank Fusion of two score arrays.

    Fuses *rankings*, not scores, so the cosine scale (compressed into ~0.75-0.87
    on noisy text) and the overlap scale (0-1, mostly near 0) never have to be
    calibrated against each other. Only the top `candidates` of each method
    contribute, which keeps the long tail of near-random scores from adding noise.

    Zero scores are never ranked. Most chunks share no n-grams with a short query,
    so without this, argsort hands high lexical ranks to arbitrary zero-overlap
    chunks and their RRF credit drowns out the genuine match.
    """
    out = np.zeros(len(cos), dtype=np.float32)
    for scores in (cos, lex):
        order = np.argsort(-scores)
        rank = 0
        for idx in order:
            if scores[idx] <= 0:
                break                      # sorted desc: everything after is 0 too
            out[idx] += 1.0 / (k + rank + 1)
            rank += 1
            if rank >= candidates:
                break
    return out


class Library:
    """All indexed textbooks, held in memory for hybrid (semantic + lexical) search."""

    def __init__(self):
        self.books = {}   # book_id -> {"chunks": [...], "vectors": np.ndarray, "fp": str}
        self._lock = threading.Lock()

    # ------------------------------------------------------------ storage
    def _paths(self, book_id):
        return (config.LIBRARY_DIR / f"{book_id}.json",
                config.LIBRARY_DIR / f"{book_id}.npy")

    def load_from_disk(self, verbose: bool = True):
        config.LIBRARY_DIR.mkdir(parents=True, exist_ok=True)
        for meta_path in sorted(config.LIBRARY_DIR.glob("*.json")):
            book_id = meta_path.stem
            _, vec_path = self._paths(book_id)
            if not vec_path.exists():
                continue
            try:
                meta = json.loads(meta_path.read_text(encoding="utf-8"))
                vectors = np.load(vec_path)
                with self._lock:
                    self.books[book_id] = {
                        "chunks": meta["chunks"],
                        "vectors": vectors,
                        "fp": meta.get("fp", ""),
                    }
                if verbose:
                    print(f"[rag] loaded '{book_id}' ({len(meta['chunks'])} chunks)")
            except Exception as exc:
                print(f"[rag] could not load '{book_id}': {exc}")

    def _save(self, book_id, chunks, vectors, fp):
        meta_path, vec_path = self._paths(book_id)
        meta_path.write_text(
            json.dumps({"book_id": book_id, "fp": fp, "chunks": chunks},
                       ensure_ascii=False),
            encoding="utf-8",
        )
        np.save(vec_path, vectors)

    # ------------------------------------------------------------ indexing
    def index_pdf(self, pdf_path, force: bool = False, verbose: bool = True):
        from pathlib import Path

        pdf_path = Path(pdf_path)
        if not pdf_path.exists():
            raise FileNotFoundError(pdf_path)

        book_id = _book_id(pdf_path)
        fp = _fingerprint(pdf_path)

        if not force and book_id in self.books and self.books[book_id]["fp"] == fp:
            if verbose:
                print(f"[rag] '{book_id}' already indexed and unchanged")
            return 0

        if verbose:
            print(f"[rag] indexing {pdf_path.name} ...")

        pages = _extract_pages(pdf_path)
        if not pages:
            print(f"[rag] '{pdf_path.name}' has no text layer — it is probably a "
                  f"scan. Run OCR on it first (see the guide), skipping.")
            return 0

        chunks = _chunk_pages(pages)
        if not chunks:
            return 0

        model = load_embedder(verbose=verbose)
        vectors = model.encode(
            [c["text"] for c in chunks],
            batch_size=32,
            convert_to_numpy=True,
            normalize_embeddings=True,
            show_progress_bar=verbose,
        ).astype(np.float32)

        with self._lock:
            self.books[book_id] = {"chunks": chunks, "vectors": vectors, "fp": fp}
        self._save(book_id, chunks, vectors, fp)

        if verbose:
            print(f"[rag] '{book_id}': {len(chunks)} chunks from {len(pages)} pages")
        return len(chunks)

    def index_books_folder(self, verbose: bool = True):
        """Index every PDF in books/ that is new or has changed."""
        config.BOOKS_DIR.mkdir(parents=True, exist_ok=True)
        pdfs = sorted(config.BOOKS_DIR.glob("*.pdf"))
        if not pdfs:
            if verbose:
                print("[rag] books/ is empty — answers will use general knowledge only")
            return
        for pdf in pdfs:
            try:
                self.index_pdf(pdf, verbose=verbose)
            except Exception as exc:
                print(f"[rag] failed on {pdf.name}: {exc}")

    # ------------------------------------------------------------ retrieval
    def search(self, query: str, top_k=None, min_score=None):
        """Return [{'book','page','text','score'}, ...] best first."""
        top_k = top_k or config.RAG_TOP_K
        min_score = config.RAG_MIN_SCORE if min_score is None else min_score

        with self._lock:
            books = list(self.books.items())
        if not books or not query.strip():
            return []

        model = load_embedder(verbose=False)
        q = model.encode([query], convert_to_numpy=True,
                         normalize_embeddings=True).astype(np.float32)[0]

        q_grams = _ngrams(query) if config.RAG_HYBRID else set()

        hits = []
        for book_id, data in books:
            cos = data["vectors"] @ q             # cosine, vectors are normalised
            if cos.size == 0:
                continue

            if config.RAG_HYBRID and q_grams:
                lex = np.array(
                    [_gram_overlap(q_grams, c["text"]) for c in data["chunks"]],
                    dtype=np.float32)
                fused = _rrf(cos, lex, config.RAG_CANDIDATES, config.RAG_RRF_K)
                order = np.argsort(-fused)[:top_k]
                # Report cosine as the visible score so it stays comparable to
                # the min_score floor and to non-hybrid runs.
                picked = [(int(i), float(cos[i]), float(lex[i]), float(fused[i]))
                          for i in order]
            else:
                take = min(top_k, cos.size)
                idx = np.argpartition(-cos, take - 1)[:take]
                picked = [(int(i), float(cos[i]), 0.0, float(cos[i])) for i in idx]

            for i, score, overlap, rank_score in picked:
                # The cosine floor must not veto a lexical rescue: a chunk that
                # literally contains the query wording is relevant even when the
                # embedding disagrees, which is exactly the case hybrid exists for.
                if score < min_score and overlap < config.RAG_LEX_MIN:
                    continue
                chunk = data["chunks"][i]
                hits.append({"book": book_id, "page": chunk["page"],
                             "text": chunk["text"], "score": score,
                             "rank_score": rank_score})

        # Order by the fused score, NOT by cosine. Sorting by cosine here would
        # undo the fusion: the chunk RRF promoted often has a mediocre cosine, so
        # it would sink below irrelevant chunks and land last in the prompt — the
        # weakest position for the model to attend to.
        hits.sort(key=lambda h: h["rank_score"], reverse=True)
        return hits[:top_k]

    def _opening_chunk(self, book_id):
        """First chunk of a book — where a chapter normally introduces its subject."""
        with self._lock:
            data = self.books.get(book_id)
        if not data or not data["chunks"]:
            return None
        return data["chunks"][0]

    def context_for(self, query: str):
        """Return (context_string, citations) ready for the LLM and the UI."""
        hits = self.search(query)
        if not hits:
            return "", []

        blocks, citations = [], []

        # Textbook chapters introduce their subject in the opening lines, and are
        # often narrated in the first person thereafter ("I am the second longest
        # river..."). Retrieval ranks by similarity to the question, so the passage
        # naming the narrator scores poorly and is left out — leaving the model an
        # excerpt full of "I" with nobody to attach it to. Prepending the opening
        # chunk of the best-matching book costs one chunk and resolves the referent.
        if getattr(config, "RAG_INCLUDE_OPENING", True):
            top_book = hits[0]["book"]
            opening = self._opening_chunk(top_book)
            if opening and not any(h["book"] == top_book
                                   and h["text"] == opening["text"] for h in hits):
                blocks.append(f"[{top_book} p.{opening['page']} — opening]\n"
                              f"{opening['text']}")
                citations.append(f"{top_book} p.{opening['page']}")

        for h in hits:
            blocks.append(f"[{h['book']} p.{h['page']}]\n{h['text']}")
            tag = f"{h['book']} p.{h['page']}"
            if tag not in citations:
                citations.append(tag)
        return "\n\n".join(blocks), citations

    def summary(self):
        with self._lock:
            return [{"book": bid, "chunks": len(d["chunks"])}
                    for bid, d in sorted(self.books.items())]


library = Library()
