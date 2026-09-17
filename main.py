"""
Terminal entry point.

    python main.py            voice loop (needs a working microphone)
    python main.py --text     keyboard mode, still speaks the reply
    python main.py --text --mute   keyboard mode, no audio at all
    python main.py --lang hi  force a language instead of auto-detecting

On WSL the microphone is usually not exposed to Linux, so use --text here and
run web.py for real voice conversation (the browser supplies the mic).
"""
from __future__ import annotations

import sys

import config
import llm
import rag
import stt
import tts


def _prepare(use_rag: bool = True):
    print("EduBuddy — offline multilingual voice tutor")
    print("-" * 46)
    rag.library.load_from_disk()
    if use_rag:
        rag.library.index_books_folder()
    stt_needed = "--text" not in sys.argv
    if stt_needed:
        stt.load()
    llm.load()


def _answer(question: str, lang: str, mute: bool):
    context, citations = rag.library.context_for(question)
    reply = llm.tutor.ask(question, lang=lang, context=context)

    print(f"\n🤖 EduBuddy ({config.LANG_NAMES[lang]}): {reply}")
    if citations:
        print(f"   📖 {', '.join(citations)}")

    if not mute:
        try:
            wav, sr = tts.synth(reply, lang)
            import audio_io
            audio_io.play(wav, sr)
        except Exception as exc:
            print(f"   (speech playback unavailable: {exc})")
    return reply


def text_mode(forced_lang, mute):
    print("Type a question in English, Hindi or Tamil. 'quit' to exit, "
          "'clear' to reset the conversation.\n")
    while True:
        try:
            question = input("🧑 You: ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\nBye!")
            return
        if not question:
            continue
        if question.lower() in ("quit", "exit"):
            print("Bye!")
            return
        if question.lower() == "clear":
            llm.tutor.clear()
            print("(conversation cleared)")
            continue

        lang = forced_lang or "en"
        _answer(question, lang, mute)


def voice_mode(forced_lang, mute):
    import audio_io

    print("Speak after the prompt. Ctrl+C to exit.\n")
    while True:
        try:
            audio = audio_io.record_utterance()
            if len(audio) == 0:
                continue

            question, detected = stt.transcribe(audio, language=forced_lang)
            lang = forced_lang or detected
            if not question:
                print("(nothing recognised)")
                continue

            print(f"🧑 You ({config.LANG_NAMES[lang]}): {question}")
            _answer(question, lang, mute)
        except KeyboardInterrupt:
            print("\nBye!")
            return
        except Exception as exc:
            print(f"[error] {exc}")


def main():
    argv = sys.argv[1:]
    mute = "--mute" in argv
    forced_lang = None
    if "--lang" in argv:
        i = argv.index("--lang")
        if i + 1 < len(argv):
            forced_lang = config.normalise_lang(argv[i + 1])

    _prepare()

    if "--text" in argv:
        text_mode(forced_lang, mute)
    else:
        voice_mode(forced_lang, mute)


if __name__ == "__main__":
    main()
