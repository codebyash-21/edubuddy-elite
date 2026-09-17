"""EduBuddy as a Tkinter desktop window.

    python app.py                 normal
    python app.py --no-mic        hide the microphone button
    python app.py --no-speak      do not read answers aloud

Same engine as web.py — identical retrieval, question packs, grounding and
refusal — with a native window instead of a browser. Nothing is faster here:
the pipeline is the same Python functions in the same process, and the interface
is a fraction of a second of a reply that takes tens of seconds. It is a
different presentation, not a different system.

Two things this file has to get right.

**The window must never freeze.** A reply takes tens of seconds and Tkinter is
single-threaded, so any model work on the UI thread would lock the window solid
— no repaint, no scrolling, and on Windows a "not responding" title bar in the
middle of a demo. All of it runs on a worker thread which posts results back
through a queue that the UI polls.

**The microphone is not guaranteed.** Under WSL there is usually no audio input
device at all, which is precisely why the browser interface exists. Here the mic
is probed once at startup and, when it is missing, the button is disabled with
the reason shown rather than failing at the moment someone presses it.
"""
from __future__ import annotations

import argparse
import queue
import threading
import time
import tkinter as tk
from tkinter import ttk

import config

BG = "#12161c"
PANEL = "#1a1f27"
INK = "#e8eaee"
MUTED = "#8b93a1"
ACCENT = "#22c98a"
USER = "#2a3442"
WARN = "#d98a4a"

FONT = ("Segoe UI", 11)
FONT_SMALL = ("Segoe UI", 9)
FONT_BOLD = ("Segoe UI", 11, "bold")


class EduBuddyApp:
    def __init__(self, root, allow_mic=True, speak=True):
        self.root = root
        self.speak_back = speak
        self.allow_mic = allow_mic
        self.busy = False
        self.ready = False
        self.mic_error = None
        self.events = queue.Queue()

        root.title("EduBuddy — offline textbook tutor")
        root.geometry("880x660")
        root.minsize(640, 480)
        root.configure(bg=BG)

        self._build_ui()
        self._poll_events()

        threading.Thread(target=self._load_models, daemon=True).start()

    # ------------------------------------------------------------------ ui
    def _build_ui(self):
        top = tk.Frame(self.root, bg=PANEL, padx=14, pady=10)
        top.pack(fill="x")

        tk.Label(top, text="EduBuddy", bg=PANEL, fg=ACCENT,
                 font=("Segoe UI", 15, "bold")).pack(side="left")
        tk.Label(top, text="  offline · en / hi / ta", bg=PANEL, fg=MUTED,
                 font=FONT_SMALL).pack(side="left")


        self.lang = tk.StringVar(value="auto")
        combo = ttk.Combobox(top, textvariable=self.lang, width=12,
                             state="readonly",
                             values=["auto", "english", "hindi", "tamil"])
        combo.pack(side="right")
        tk.Label(top, text="language ", bg=PANEL, fg=MUTED,
                 font=FONT_SMALL).pack(side="right")

        # transcript ---------------------------------------------------
        wrap = tk.Frame(self.root, bg=BG)
        wrap.pack(fill="both", expand=True, padx=14, pady=(12, 6))

        self.chat = tk.Text(wrap, bg=BG, fg=INK, font=FONT, wrap="word",
                            relief="flat", padx=10, pady=10,
                            state="disabled", spacing1=3, spacing3=8)
        bar = ttk.Scrollbar(wrap, command=self.chat.yview)
        self.chat.configure(yscrollcommand=bar.set)
        bar.pack(side="right", fill="y")
        self.chat.pack(side="left", fill="both", expand=True)

        self.chat.tag_configure("user", foreground=INK, background=USER,
                                lmargin1=12, lmargin2=12, rmargin=12,
                                spacing1=6, spacing3=6)
        self.chat.tag_configure("who", foreground=ACCENT, font=FONT_SMALL)
        self.chat.tag_configure("whoq", foreground=MUTED, font=FONT_SMALL)
        self.chat.tag_configure("bot", foreground=INK)
        self.chat.tag_configure("cite", foreground=ACCENT, font=FONT_SMALL)
        self.chat.tag_configure("status", foreground=MUTED, font=FONT_SMALL)
        self.chat.tag_configure("warn", foreground=WARN, font=FONT_SMALL)

        # input --------------------------------------------------------
        bottom = tk.Frame(self.root, bg=BG, padx=14, pady=10)
        bottom.pack(fill="x")

        self.entry = tk.Entry(bottom, bg=PANEL, fg=INK, font=FONT,
                              relief="flat", insertbackground=INK)
        self.entry.pack(side="left", fill="x", expand=True, ipady=8, padx=(0, 8))
        self.entry.bind("<Return>", lambda _e: self._send_typed())

        self.send_btn = tk.Button(bottom, text="Ask", command=self._send_typed,
                                  bg=ACCENT, fg="#04120b", font=FONT_BOLD,
                                  relief="flat", padx=18, pady=4,
                                  activebackground=ACCENT, state="disabled")
        self.send_btn.pack(side="left")

        self.mic_btn = tk.Button(bottom, text="🎤", command=self._start_recording,
                                 bg=PANEL, fg=INK, font=FONT_BOLD,
                                 relief="flat", padx=14, pady=4, state="disabled")
        if self.allow_mic:
            self.mic_btn.pack(side="left", padx=(8, 0))

        # status -------------------------------------------------------
        self.status = tk.Label(self.root, text="starting…", bg=BG, fg=MUTED,
                               font=FONT_SMALL, anchor="w", padx=16, pady=(0))
        self.status.pack(fill="x", pady=(0, 8))

        self._say_status("loading models — this takes a moment on first start")

    # --------------------------------------------------------- transcript
    def _append(self, text, tag=None, newline=True):
        self.chat.configure(state="normal")
        self.chat.insert("end", text + ("\n" if newline else ""), tag or ())
        self.chat.configure(state="disabled")
        self.chat.see("end")

    def _say_status(self, text, warn=False):
        self.status.configure(text=text, fg=WARN if warn else MUTED)

    def _mark(self):
        """Remember where the pending status line starts so it can be replaced."""
        return self.chat.index("end-1c")

    def _replace_from(self, mark, text, tag=None):
        self.chat.configure(state="normal")
        self.chat.delete(mark, "end-1c")
        self.chat.insert("end", text + "\n", tag or ())
        self.chat.configure(state="disabled")
        self.chat.see("end")

    # ------------------------------------------------------------ loading
    def _load_models(self):
        """Import and warm the heavy modules off the UI thread.

        Imported serially and in this order on purpose: concurrent first-imports
        of transformers from several threads produced a partially-initialised
        module and a spurious ImportError that never reproduced sequentially.
        """
        try:
            import rag
            rag.library.load_from_disk(verbose=False)
            rag.load_embedder(verbose=False)
            self.events.put(("status", "retrieval ready — loading the tutor…"))

            import llm
            llm.tutor.ask("hello", lang="en", context="")

            try:
                import qa_match
                qa_match.load(verbose=False)
            except Exception:
                pass

            import tts  # noqa: F401

            books = [b["book"] for b in rag.library.summary()]
            self.events.put(("ready", books))
        except Exception as exc:
            self.events.put(("fatal", str(exc)))

        if self.allow_mic:
            self._probe_mic()

    def _probe_mic(self):
        """Find out now whether recording is possible, not when it is pressed."""
        try:
            import audio_io
            import sounddevice as sd
            if not any(d["max_input_channels"] > 0 for d in sd.query_devices()):
                raise RuntimeError("no input device")
            self.events.put(("mic", None))
        except Exception as exc:
            self.mic_error = str(exc).split("\n")[0][:90]
            self.events.put(("micfail", self.mic_error))

    # -------------------------------------------------------------- events
    def _poll_events(self):
        """Drain worker messages on the UI thread. Tk is not thread-safe."""
        try:
            while True:
                kind, payload = self.events.get_nowait()
                self._handle(kind, payload)
        except queue.Empty:
            pass
        self.root.after(80, self._poll_events)

    def _handle(self, kind, payload):
        if kind == "status":
            self._say_status(payload)
        elif kind == "ready":
            self.ready = True
            books = payload or []
            self._say_status(
                f"ready · {len(books)} textbook{'s' if len(books) != 1 else ''}"
                + (f" ({', '.join(books)})" if books else " — none indexed"))
            self.send_btn.configure(state="normal")
            self.entry.focus_set()
        elif kind == "fatal":
            self._say_status(f"failed to start: {payload}", warn=True)
            self._append(f"Could not start: {payload}", "warn")
        elif kind == "mic":
            self.mic_btn.configure(state="normal")
        elif kind == "micfail":
            self.mic_btn.configure(state="disabled")
            self._append(f"Microphone unavailable — typing still works. ({payload})",
                         "warn")
        elif kind == "progress":
            self._replace_from(self._pending, payload, "status")
        elif kind == "answer":
            self._show_answer(payload)
        elif kind == "error":
            self._replace_from(self._pending, f"Error: {payload}", "warn")
            self._finish()
        elif kind == "heard":
            self.entry.delete(0, "end")
            self._ask(payload)

    def _show_answer(self, data):
        self.chat.configure(state="normal")
        self.chat.delete(self._pending, "end-1c")
        self.chat.configure(state="disabled")

        # A pack answer is the textbook's own wording, returned without the
        # model. Labelled differently because presenting a prewritten passage as
        # EduBuddy's own sentence would be the dishonest way to be fast.
        who = ("From the textbook" if data.get("source") == "textbook"
               else "EduBuddy")
        self._append(who, "who")
        self._append(data["reply"], "bot")
        if data.get("citations"):
            self._append("📖  " + "   ".join(data["citations"]), "cite")
        self._append("")

        if self.speak_back and data["reply"]:
            threading.Thread(target=self._speak, daemon=True,
                             args=(data["reply"], data["lang"])).start()
        self._finish()

    def _finish(self):
        self.busy = False
        self.send_btn.configure(state="normal")
        if self.allow_mic and not self.mic_error:
            self.mic_btn.configure(state="normal")
        self._say_status("ready")

    # ---------------------------------------------------------------- ask
    def _resolve_lang(self, text):
        choice = self.lang.get()
        if choice != "auto":
            return {"english": "en", "hindi": "hi", "tamil": "ta"}[choice]
        import qa_match
        return qa_match.detect_lang(text)

    def _send_typed(self):
        text = self.entry.get().strip()
        if not text or self.busy or not self.ready:
            return
        self.entry.delete(0, "end")
        self._ask(text)

    def _ask(self, text):
        self.busy = True
        self.send_btn.configure(state="disabled")
        self.mic_btn.configure(state="disabled")

        self._append("You", "whoq")
        self._append(text, "user")
        self._pending = self._mark()
        self._append("searching the textbooks…", "status")
        self._say_status("working…")

        threading.Thread(target=self._answer_worker, args=(text,),
                         daemon=True).start()

    def _answer_worker(self, text):
        """All model work happens here — never on the UI thread."""
        try:
            import rag
            import llm
            lang = self._resolve_lang(text)

            # Question pack first: a hit needs no model at all.
            try:
                import qa_match
                hit = qa_match.match(text, lang=lang)
            except Exception:
                hit = None

            if hit:
                self.events.put(("answer", {
                    "reply": hit["answer"], "citations": [hit["citation"]],
                    "lang": lang, "source": "textbook"}))
                return

            context, citations = rag.library.context_for(text)
            # Retrieval finishes in about a second, generation takes tens of
            # seconds. Show what was found rather than holding a blank line.
            self.events.put(("progress",
                             ("found " + " · ".join(citations) + " — writing the answer…")
                             if citations else
                             "nothing matching in the textbooks — checking…"))

            t0 = time.time()
            reply = llm.tutor.ask(text, lang=lang, context=context)

            grounded = True
            if config.NOT_IN_BOOK_MARKER in reply:
                reply = reply.replace(config.NOT_IN_BOOK_MARKER, "").strip()
                note = config.NOT_IN_BOOK_MESSAGE[lang]
                reply = f"{note} {reply}".strip() if reply else note
                # A page number beside an answer the model just disowned is a
                # claim the system knows to be false.
                citations = []
                grounded = False

            print(f"[app] {lang} · {time.time() - t0:.1f}s · "
                  f"{'grounded' if grounded else 'NOT in book'}")
            self.events.put(("answer", {"reply": reply, "citations": citations,
                                        "lang": lang, "source": "model"}))
        except Exception as exc:
            self.events.put(("error", str(exc)))

    # -------------------------------------------------------------- speech
    def _speak(self, text, lang):
        try:
            import audio_io
            import tts
            wav, sample_rate = tts.synth(text, lang=lang)
            if wav is not None and len(wav):
                audio_io.play(wav, sample_rate)
        except Exception as exc:
            # Under WSL there is often no output device either. Losing the voice
            # should not lose the answer, which is already on screen.
            print(f"[app] playback unavailable: {exc}")

    def _start_recording(self):
        if self.busy or not self.ready:
            return
        self.busy = True
        self.mic_btn.configure(state="disabled")
        self.send_btn.configure(state="disabled")
        self._say_status("listening — speak now, then pause")
        threading.Thread(target=self._record_worker, daemon=True).start()

    def _record_worker(self):
        try:
            import audio_io
            import stt
            audio = audio_io.record_utterance(verbose=False)
            if audio is None or not len(audio):
                self.events.put(("status", "did not catch that"))
                self.busy = False
                self.events.put(("mic", None))
                self.send_btn.configure(state="normal")
                return
            choice = self.lang.get()
            forced = None if choice == "auto" else \
                {"english": "en", "hindi": "hi", "tamil": "ta"}[choice]
            # stt.transcribe accepts the float32 array directly — no temp file.
            text, _detected = stt.transcribe(audio, language=forced)
            self.busy = False
            if text and text.strip():
                self.events.put(("heard", text.strip()))
            else:
                self.events.put(("status", "did not catch that"))
                self.events.put(("mic", None))
        except Exception as exc:
            self.busy = False
            self.events.put(("error", f"microphone: {exc}"))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--no-mic", action="store_true")
    ap.add_argument("--no-speak", action="store_true")
    args = ap.parse_args()

    root = tk.Tk()
    try:
        ttk.Style().theme_use("clam")
    except Exception:
        pass
    EduBuddyApp(root, allow_mic=not args.no_mic, speak=not args.no_speak)
    root.mainloop()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
