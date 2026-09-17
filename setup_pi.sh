#!/usr/bin/env bash
#
# EduBuddy installer for Raspberry Pi 4 / 5 (Raspberry Pi OS 64-bit, Debian
# bookworm or later). Needs internet; the app runs offline afterwards.
#
#   chmod +x setup_pi.sh
#   ./setup_pi.sh
#
# Differences from setup.sh, all forced by ARM:
#   * No PyTorch CPU index — that index is x86-only. aarch64 wheels come from PyPI.
#   * No prebuilt llama-cpp-python wheels for ARM, so it always builds from
#     source with NEON enabled. Expect 20-40 minutes on a Pi 4.
#   * Model choices come from EDUBUDDY_PROFILE=pi in config.py.
#
set -euo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$HERE"
export EDUBUDDY_PROFILE=pi

say(){  printf "\n\033[1;36m==> %s\033[0m\n" "$1"; }
warn(){ printf "\033[1;33m!! %s\033[0m\n" "$1"; }
die(){  printf "\033[1;31mXX %s\033[0m\n" "$1"; exit 1; }

# ------------------------------------------------------------ 0. preflight
say "Preflight"
ARCH=$(uname -m)
echo "    arch         : $ARCH"
[ "$ARCH" = "aarch64" ] || die "Expected aarch64. A 32-bit Raspberry Pi OS cannot
    run these models — reinstall with the 64-bit image."

PYV=$(python3 -c 'import sys; print("%d.%d" % sys.version_info[:2])')
CORES=$(nproc)
RAM_MB=$(free -m | awk '/^Mem:/{print $2}')
DISK_GB=$(df -BG --output=avail . | tail -1 | tr -dc '0-9')
echo "    python3      : $PYV"
echo "    cores        : $CORES"
echo "    RAM          : ${RAM_MB} MB"
echo "    free disk    : ${DISK_GB} GB"

if [ "$RAM_MB" -lt 3500 ]; then
  die "Under 4 GB of RAM. A local LLM plus speech recognition plus retrieval will
    not fit. An 8 GB Pi is recommended; 4 GB is the practical minimum."
elif [ "$RAM_MB" -lt 7000 ]; then
  warn "4 GB detected. This works but leaves little headroom. Keep
    LLM_MAX_RESIDENT=1, close other applications, and expect swapping if you
    open a browser on the Pi itself."
fi

[ "$DISK_GB" -ge 8 ] || die "Need at least 8 GB free; models alone are ~2.5 GB."

# Running from microSD makes model loading painfully slow.
ROOT_DEV=$(findmnt -no SOURCE / || true)
case "$ROOT_DEV" in
  /dev/mmcblk*) warn "Running from microSD. Loading a 1.6 GB model will take
    around a minute each time. A USB 3.0 SSD is strongly recommended." ;;
esac

# Sustained inference will thermal-throttle a Pi without cooling.
if command -v vcgencmd >/dev/null 2>&1; then
  TEMP=$(vcgencmd measure_temp 2>/dev/null | tr -dc '0-9.' | cut -d. -f1 || echo 0)
  echo "    SoC temp     : ${TEMP}C"
  [ "${TEMP:-0}" -lt 60 ] || warn "Already ${TEMP}C at idle. Sustained inference
    will throttle without a heatsink and fan."
fi

# ------------------------------------------------------------ 1. system deps
say "Installing system packages (sudo password may be requested)"
sudo apt-get update -qq
sudo apt-get install -y --no-install-recommends \
  python3 python3-venv python3-dev python3-pip \
  build-essential cmake pkg-config git curl ca-certificates \
  ffmpeg libportaudio2 libsndfile1 libopenblas-dev

# ------------------------------------------------------------ 2. virtualenv
say "Creating virtual environment (.venv)"
[ -d .venv ] || python3 -m venv .venv
# shellcheck disable=SC1091
source .venv/bin/activate
python -m pip install --upgrade pip wheel setuptools

# ------------------------------------------------------------ 3. torch (ARM)
# The CPU index used on x86 has no aarch64 builds; PyPI does.
say "Installing PyTorch for aarch64 (this is a large download)"
pip install torch --index-url https://pypi.org/simple
python -c "import torch; print('    torch', torch.__version__)"

# ------------------------------------------------------------ 4. llama.cpp
say "Building llama-cpp-python for ARM (20-40 minutes on a Pi 4)"
if ! python -c "import llama_cpp" >/dev/null 2>&1; then
  # GGML_NATIVE lets the compiler target this exact CPU (NEON, dotprod on Pi 5).
  CMAKE_ARGS="-DGGML_NATIVE=ON -DGGML_BLAS=ON -DGGML_BLAS_VENDOR=OpenBLAS" \
    pip install llama-cpp-python --no-binary llama-cpp-python
fi
python -c "import llama_cpp; print('    llama_cpp', llama_cpp.__version__)" \
  || die "llama-cpp-python built but will not import."

# ------------------------------------------------------------ 5. the rest
say "Installing Python packages"
pip install -r requirements.txt

# ------------------------------------------------------------ 6. models
say "Downloading models (~2.5 GB, one time)"
python - <<'PY'
import config
print(f"    profile={config.PROFILE}  llm={config.LLM_FILE}  whisper={config.WHISPER_MODEL}")

print("\n[1/4] Whisper", config.WHISPER_MODEL, "...")
from faster_whisper import WhisperModel
WhisperModel(config.WHISPER_MODEL, device=config.WHISPER_DEVICE,
             compute_type=config.WHISPER_COMPUTE_TYPE,
             download_root=str(config.WHISPER_DIR))

print("\n[2/4] language model ...")
import llm
llm.download_all()

print("\n[3/4] MMS-TTS voices (eng / hin / tam) ...")
from transformers import AutoTokenizer, VitsModel
for lang, repo in config.TTS_REPOS.items():
    AutoTokenizer.from_pretrained(repo, cache_dir=str(config.TTS_DIR))
    VitsModel.from_pretrained(repo, cache_dir=str(config.TTS_DIR))
    print("     ", lang, repo)

print("\n[4/4] multilingual embedder ...")
from sentence_transformers import SentenceTransformer
SentenceTransformer(config.EMBED_MODEL, cache_folder=str(config.EMBED_DIR))
PY

# ------------------------------------------------------------ 7. index books
say "Indexing any textbooks in books/"
python ingest.py --all || warn "No textbooks indexed yet (this is fine)"

# ------------------------------------------------------------ 8. self-check
say "Running self-check"
python verify.py || warn "Self-check reported problems — see above"

DISK=$(du -sh models 2>/dev/null | cut -f1 || echo "?")
cat <<EOF

$(printf "\033[1;32m")EduBuddy is installed on this Pi. Models on disk: ${DISK}$(printf "\033[0m")

Start it:
    cd "$HERE"
    source .venv/bin/activate
    EDUBUDDY_PROFILE=pi python web.py

The Pi has no screen in most setups, so open the interface from another device
on the same network. Find the Pi's address with:  hostname -I
Then browse to  http://<pi-address>:7860

IMPORTANT: microphone capture needs a secure context. That is automatic on
127.0.0.1, but NOT over the network, so the mic will be blocked on http://<ip>.
Either use Chrome's "Insecure origins treated as secure" flag for that address,
or put the server behind HTTPS. Typing questions works either way.

Expect roughly 1-3 minutes per grounded answer on a Pi 4.
EOF
