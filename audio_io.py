"""
Microphone capture and playback for terminal mode (main.py).

Energy-based voice activity detection: recording starts when the RMS of a
frame rises above SILENCE_THRESHOLD and stops after SILENCE_DURATION seconds
of quiet.

Only used by the terminal loop. The web UI captures audio in the browser, so
none of this is needed when running web.py — which is why WSL users can skip
PulseAudio setup entirely.
"""
from __future__ import annotations

import sys

import numpy as np

import config


def _sd():
    try:
        import sounddevice as sd
        return sd
    except Exception as exc:  # pragma: no cover - environment dependent
        raise RuntimeError(
            "sounddevice is unavailable. Terminal voice mode needs a working "
            "microphone. On WSL this usually means no audio device is exposed — "
            "use 'python web.py' (browser microphone) instead, or "
            "'python main.py --text' for keyboard mode.\n"
            f"Original error: {exc}"
        ) from exc


def list_devices():
    sd = _sd()
    print(sd.query_devices())


def record_utterance(verbose: bool = True) -> np.ndarray:
    """
    Block until the student speaks and then stops.

    Returns float32 mono audio at config.SAMPLE_RATE, or an empty array if
    nothing was said.
    """
    sd = _sd()

    frame_len = int(config.SAMPLE_RATE * config.FRAME_MS / 1000)
    silence_frames_needed = int(config.SILENCE_DURATION * 1000 / config.FRAME_MS)
    max_frames = int(config.MAX_RECORD_SECONDS * 1000 / config.FRAME_MS)

    collected = []
    started = False
    silent_run = 0

    if verbose:
        print("🎤 listening ... (speak now)", flush=True)

    with sd.InputStream(samplerate=config.SAMPLE_RATE, channels=1,
                        dtype="float32", blocksize=frame_len) as stream:
        for _ in range(max_frames):
            frame, _overflow = stream.read(frame_len)
            frame = frame[:, 0]
            rms = float(np.sqrt(np.mean(np.square(frame))))

            if rms >= config.SILENCE_THRESHOLD:
                if not started and verbose:
                    print("   ... speech detected", flush=True)
                started = True
                silent_run = 0
                collected.append(frame.copy())
            elif started:
                silent_run += 1
                collected.append(frame.copy())
                if silent_run >= silence_frames_needed:
                    break

    if not collected:
        return np.zeros(0, dtype=np.float32)

    audio = np.concatenate(collected)
    if len(audio) < config.MIN_SPEECH_SECONDS * config.SAMPLE_RATE:
        return np.zeros(0, dtype=np.float32)
    return audio.astype(np.float32)


def play(wav: np.ndarray, sample_rate: int, blocking: bool = True):
    """Play a float32 waveform through the default output device."""
    if wav is None or len(wav) == 0:
        return
    sd = _sd()
    sd.play(wav, sample_rate)
    if blocking:
        sd.wait()


def save_wav(path: str, wav: np.ndarray, sample_rate: int):
    import wave

    pcm = (np.clip(wav, -1.0, 1.0) * 32767).astype(np.int16)
    with wave.open(path, "wb") as f:
        f.setnchannels(1)
        f.setsampwidth(2)
        f.setframerate(sample_rate)
        f.writeframes(pcm.tobytes())


if __name__ == "__main__":
    if "--devices" in sys.argv:
        list_devices()
    else:
        print("Recording a test utterance ...")
        a = record_utterance()
        print(f"captured {len(a) / config.SAMPLE_RATE:.1f}s")
