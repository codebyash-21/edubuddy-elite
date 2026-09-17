#!/usr/bin/env bash
#
# EduBuddy one-time installer (Ubuntu / WSL2).
# Needs internet. After this finishes the app runs fully offline.
#
#   chmod +x setup.sh
#   ./setup.sh
#
set -euo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$HERE"

say(){ printf "\n\033[1;36m==> %s\033[0m\n" "$1"; }
warn(){ printf "\033[1;33m!! %s\033[0m\n" "$1"; }

# ------------------------------------------------------------ 0. preflight
say "Preflight"
PYV=$(python3 -c 'import sys; print("%d.%d" % sys.version_info[:2])')
echo "    python3      : $PYV"
echo "    cores        : $(nproc)"
echo "    RAM total    : $(free -h | awk '/^Mem:/{print $2}')"
echo "    free disk    : $(df -h . | awk 'NR==2{print $4}')"

case "$PYV" in
  3.10|3.11|3.12) ;;
  *) warn "Python $PYV is untested here. 3.12 (Ubuntu 24.04 default) is the target." ;;
esac

AVAIL_KB=$(df -k . | awk 'NR==2{print $4}')
if [ "$AVAIL_KB" -lt 12000000 ]; then
  warn "Less than ~12 GB free. Models need ~3.7 GB plus ~3 GB of packages."
  read -r -p "    Continue anyway? [y/N] " ans
  [ "$ans" = "y" ] || [ "$ans" = "Y" ] || { echo "Aborted."; exit 1; }
fi

# ------------------------------------------------------------ 1. system deps
say "Installing system packages (sudo password may be requested)"
sudo apt-get update -qq
sudo apt-get install -y --no-install-recommends \
  python3 python3-venv python3-dev python3-pip \
  build-essential cmake pkg-config git curl ca-certificates \
  ffmpeg libportaudio2 libsndfile1

# ------------------------------------------------------------ 2. virtualenv
say "Creating virtual environment (.venv)"
if [ ! -d .venv ]; then
  python3 -m venv .venv
fi
# shellcheck disable=SC1091
source .venv/bin/activate
python -m pip install --upgrade pip wheel setuptools

# ------------------------------------------------------------ 3. torch (CPU)
# The CPU index matters: plain "pip install torch" pulls ~2.5 GB of nvidia-*
# CUDA dependencies you cannot use on Iris Xe.
say "Installing PyTorch (CPU build, ~200 MB)"
TORCH_INDEX="https://download.pytorch.org/whl/cpu"
if ! pip install "torch==2.5.1" --index-url "$TORCH_INDEX"; then
  warn "torch 2.5.1 unavailable for this Python — installing latest CPU build"
  pip install torch --index-url "$TORCH_INDEX"
fi
python -c "import torch; print('    torch', torch.__version__)"

# ------------------------------------------------------------ 4. llama.cpp
# Only certain versions ship prebuilt cp312 Linux wheels. Verified present at
# time of writing: 0.3.19, 0.3.18, 0.3.2, 0.3.1, 0.3.0. Anything else falls
# back to a 5-10 minute source build, so try the known-good ones in order.
#
# A pip exit code of 0 is NOT enough here. Some wheels on that index are built
# in Alpine containers and are musl-linked; they install cleanly onto Ubuntu but
# then fail at import with:
#     libc.musl-x86_64.so.1: cannot open shared object file
# So every candidate is import-tested, and discarded if it does not actually work.
say "Installing llama-cpp-python"
LCPP_INDEX="https://abetlen.github.io/llama-cpp-python/whl/cpu"

lcpp_works() { python -c "import llama_cpp" >/dev/null 2>&1; }

LCPP_OK=0
for v in 0.3.19 0.3.18 0.3.2; do
  echo "    trying prebuilt wheel $v ..."
  if pip install "llama-cpp-python==$v" --extra-index-url "$LCPP_INDEX" \
       --only-binary llama-cpp-python >/dev/null 2>&1; then
    if lcpp_works; then
      echo "    wheel $v installed and imports cleanly"
      LCPP_OK=1; break
    fi
    warn "wheel $v installed but fails to import (musl-linked) — discarding"
    pip uninstall -y llama-cpp-python >/dev/null 2>&1 || true
  fi
done

if [ "$LCPP_OK" -eq 0 ]; then
  warn "No usable prebuilt wheel — building from source against glibc (5-10 min)"
  pip uninstall -y llama-cpp-python >/dev/null 2>&1 || true
  CMAKE_ARGS="-DGGML_NATIVE=ON" pip install llama-cpp-python --no-binary llama-cpp-python
  if ! lcpp_works; then
    echo "llama-cpp-python is still broken after a source build. Stopping." >&2
    exit 1
  fi
fi
python -c "import llama_cpp; print('    llama_cpp', llama_cpp.__version__)"

# ------------------------------------------------------------ 5. everything else
say "Installing Python packages"
pip install -r requirements.txt

# ------------------------------------------------------------ 6. models
say "Downloading models (~3.7 GB, one time)"
python - <<'PY'
import config  # sets HF_HOME inside the project

print("\n[1/4] Whisper small (int8) ...")
from faster_whisper import WhisperModel
WhisperModel(config.WHISPER_MODEL,
             device=config.WHISPER_DEVICE,
             compute_type=config.WHISPER_COMPUTE_TYPE,
             download_root=str(config.WHISPER_DIR))
print("      done")

print("\n[2/4] language models (one per language, ~2-2.5 GB each) ...")
import llm
llm.download_all()
print("      done")

print("\n[3/4] MMS-TTS voices (eng / hin / tam) ...")
from transformers import AutoTokenizer, VitsModel
for lang, repo in config.TTS_REPOS.items():
    AutoTokenizer.from_pretrained(repo, cache_dir=str(config.TTS_DIR))
    VitsModel.from_pretrained(repo, cache_dir=str(config.TTS_DIR))
    print(f"      {lang}: {repo}")

print("\n[4/4] Multilingual sentence embedder ...")
from sentence_transformers import SentenceTransformer
SentenceTransformer(config.EMBED_MODEL, cache_folder=str(config.EMBED_DIR))
print("      done")
PY

# ------------------------------------------------------------ 7. index books
say "Indexing any textbooks already in books/"
python ingest.py --all || warn "No textbooks indexed yet (this is fine)"

# ------------------------------------------------------------ 8. self-check
say "Running self-check"
if ! python verify.py; then
  warn "Self-check reported problems — see the output above and Part 6 of the guide"
  exit 1
fi

# ------------------------------------------------------------ done
DISK=$(du -sh models 2>/dev/null | cut -f1 || echo "?")
cat <<EOF

$(printf "\033[1;32m")✔ EduBuddy is installed. Models on disk: ${DISK}$(printf "\033[0m")

Start it:
    cd "$HERE"
    source .venv/bin/activate
    python web.py

Then open  http://127.0.0.1:7860  in your Windows browser.

Add textbooks any time:
    cp /path/to/science_class7.pdf books/
    (they are re-indexed automatically next time you start web.py)

Other modes:
    python main.py --text     keyboard only
    python main.py            terminal voice loop (needs a Linux-visible mic)

EOF
