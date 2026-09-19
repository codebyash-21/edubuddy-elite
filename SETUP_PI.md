# Setting up EduBuddy Dynamic on a Raspberry Pi

Step by step, from a Pi with nothing on it to a working tutor. No prior
experience with this project assumed.

**Total time: about 1.5 to 2 hours**, but only ~15 minutes of that is you typing.
The rest is downloads and one long compile that runs unattended.

> **You are the first person to run this on real Pi hardware.** Everything here
> has been tested on a laptop; the Pi path is built from the ARM documentation
> and has not been executed end to end. So a step may need adjusting. If
> something differs from what is written here, that is worth reporting back —
> see [Step 8](#step-8-report-back).

---

## Before you start

| You need | Notes |
|---|---|
| Raspberry Pi 4 or 5 | **Pi 5 with 8 GB or 16 GB is strongly preferred.** A Pi 4 works but answers take 2-3 minutes each. |
| Active cooling | A heatsink **and** fan. Sustained inference pins all four cores; without a fan the Pi throttles and gets slower. Not dangerous, just slow. |
| USB 3.0 SSD, or a fast microSD | Loading a 1.6 GB model from microSD adds ~1 minute every time. An SSD is a large, cheap improvement. |
| 16 GB+ free storage | Models alone are ~2.5 GB; the installer refuses to run below 8 GB free. |
| Official 27 W USB-C power supply (Pi 5) | An underpowered supply causes random reboots under load, which look like software crashes but are not. |
| Internet, during setup only | Once installed, EduBuddy runs fully offline. |
| The three textbook PDFs | **Not in this repository** — ask Arun. See [Step 5](#step-5-add-the-textbooks). |

**RAM decides which profile you run**, and that decides answer quality:

| Your Pi | Command to start it | What you get |
|---|---|---|
| Pi 5, 16 GB | `python web.py` | The **full desktop profile** — reproduces the validated laptop results exactly. This is the interesting configuration. |
| Pi 5, 8 GB | `EDUBUDDY_PROFILE=pi python web.py` | Smaller model, shorter context. Faster, somewhat less accurate. |
| Pi 4, 4-8 GB | `EDUBUDDY_PROFILE=pi python web.py` | Same as above, but noticeably slower. |

---

## Step 1 — Install 64-bit Raspberry Pi OS

**This must be the 64-bit image.** The installer checks and will refuse to
continue on 32-bit, because the model runtime has no 32-bit ARM build.

1. Install **Raspberry Pi Imager** on your computer: <https://www.raspberrypi.com/software/>
2. Choose your Pi model, then **Raspberry Pi OS (64-bit)** — the standard
   desktop version is fine, Lite also works if you prefer headless
3. Click the gear / **Edit Settings** before writing, and set:
   - a hostname, e.g. `edubuddy`
   - **Enable SSH** — you will want this
   - your Wi-Fi name and password
   - a username and password (remember these)
4. Write the image, put the card or SSD in the Pi, power on

Already have 64-bit Raspberry Pi OS running? Skip to Step 2. Confirm with:

```bash
uname -m
```

You want `aarch64`. If it says `armv7l`, that is the 32-bit OS and you must
reflash.

---

## Step 2 — Connect to the Pi

Either plug in a monitor and keyboard, or connect from another computer:

```bash
ssh <your-username>@edubuddy.local
```

If `edubuddy.local` does not resolve, find the Pi's IP from your router's device
list and use that instead: `ssh username@192.168.1.42`.

Then bring the system up to date:

```bash
sudo apt update && sudo apt upgrade -y
```

This can take 10-20 minutes on a fresh image. It is worth doing before the
install rather than discovering a package conflict halfway through.

---

## Step 3 — Get the code

Copy the clone URL from the green **Code** button on the repository page, then:

```bash
cd ~
git clone https://github.com/codebyash-21/edubuddy-elite.git
cd edubuddy-elite
```

If `git` is missing: `sudo apt install -y git`, then retry.

---

## Step 4 — Run the installer

```bash
chmod +x setup_pi.sh
./setup_pi.sh
```

It will ask for your `sudo` password early on, then run unattended. **Do not
close the terminal.** If you are on SSH and worried about the connection
dropping, run it inside `tmux` instead so it survives a disconnect:

```bash
sudo apt install -y tmux
tmux new -s edubuddy
./setup_pi.sh
# detach any time with Ctrl+B then D; come back with: tmux attach -t edubuddy
```

### What it does, and how long each stage takes

| Stage | Roughly | What you will see |
|---|---|---|
| Preflight | instant | Prints arch, cores, RAM, free disk, SoC temperature. **Read the warnings** — they are about microSD and cooling. |
| System packages | 2-5 min | `apt` installing build tools, ffmpeg, OpenBLAS. |
| Virtual environment | <1 min | Creates `.venv/`. |
| PyTorch for ARM | 5-15 min | A large download. Uses PyPI, not the x86 CPU index. |
| **Building llama-cpp-python** | **20-40 min** | **The long one.** Compiles from source with NEON and OpenBLAS. There are no prebuilt ARM wheels, so this cannot be skipped. |
| Remaining packages | 3-8 min | Whisper, transformers, sentence-transformers, FastAPI. |
| Downloading models | 15-40 min | ~2.5 GB: Whisper, the language model, three TTS voices, the embedder. |
| Indexing + self-check | 1-2 min | Will say "No textbooks indexed" — **that is expected**, you add them next. |

**During the compile the terminal may look frozen for several minutes at a
time.** It is not. Compilation prints sporadically. Leave it alone. In another
terminal you can watch it working:

```bash
top          # you should see cc1plus using CPU
vcgencmd measure_temp
```

When it finishes you will see a green **"EduBuddy is installed on this Pi"** with
the model size on disk.

---

## Step 5 — Add the textbooks

The PDFs are copyrighted, so they are deliberately **not** in this repository.
Ask Arun for the three chapters — an English astronomy chapter, a Hindi story
chapter, and a Tamil Class-3 science chapter.

**Keep the filenames exactly as he sends them.** The filename becomes the
citation label, and the 34 test questions in `testcases.json` are written
against those specific chapters. Rename them and the validation run will not
match.

Copy them onto the Pi. From your own computer:

```bash
scp beyond_earth.pdf hindi.pdf tamil.pdf <username>@edubuddy.local:~/edubuddy/books/
```

Or just use a USB stick and a file manager. Then, on the Pi:

```bash
cd ~/edubuddy
source .venv/bin/activate
python ingest.py --all
python ingest.py --list
```

`--list` should print three books with a few hundred chunks between them. If a
book is missing, it is almost certainly a scanned PDF with no text layer — those
are skipped with a warning and need OCR first.

---

## Step 6 — Start it

```bash
cd ~/edubuddy
source .venv/bin/activate

# Pi 5 with 16 GB — the full profile:
python web.py

# Pi 5 with 8 GB, or any Pi 4:
EDUBUDDY_PROFILE=pi python web.py
```

First start takes a minute or two while models load into memory. Wait for the
line saying the server is listening on port 7860.

To find the Pi's address:

```bash
hostname -I
```

---

## Step 7 — Open it in a browser

From any device on the same network:

```
http://<pi-address>:7860
```

**Typing questions works immediately.** Try one in each language to confirm the
three books are being retrieved — you should see page citations under each
answer.

### The microphone will be blocked, and this is expected

Browsers only allow microphone access on a *secure origin*. `127.0.0.1` counts
as secure; a LAN address like `http://192.168.1.42` does not. So voice input is
blocked when you connect over the network. This is a browser rule, not a bug in
EduBuddy.

Three ways round it, easiest first:

1. **Use a browser on the Pi itself** and go to `http://127.0.0.1:7860`. The mic
   works with no configuration. Slower, since the desktop competes for RAM.
2. **Whitelist the address in Chrome** on your laptop. Open
   `chrome://flags/#unsafely-treat-insecure-origin-as-secure`, add
   `http://192.168.1.42:7860` (your actual Pi address), set the flag to
   **Enabled**, restart Chrome.
3. **Put it behind HTTPS** with a self-signed certificate. Most work; only worth
   it for a permanent installation.

---

## Troubleshooting

| Symptom | What it means | What to do |
|---|---|---|
| `Expected aarch64` and the script exits | You are on 32-bit Raspberry Pi OS | Reflash with the 64-bit image (Step 1) |
| `Under 4 GB of RAM` and it exits | Pi with too little memory | Needs a 4 GB Pi minimum; 8 GB+ recommended |
| `Need at least 8 GB free` | Not enough disk | Free space, or use a larger card/SSD |
| Compile fails with `c++: fatal error: Killed` | Ran out of RAM while compiling | Add swap: `sudo dphys-swapfile swapoff`, set `CONF_SWAPSIZE=2048` in `/etc/dphys-swapfile`, `sudo dphys-swapfile setup && sudo dphys-swapfile swapon`, then rerun |
| Model download stalls at 0% | Hugging Face CDN issue | `HF_HUB_DISABLE_XET=1` is already set in `config.py`; if it still stalls, Ctrl+C and rerun — downloads resume |
| `apt` is extremely slow | Mirror or IPv6 problem | `sudo apt -o Acquire::ForceIPv4=true update` |
| Server starts but answers say the textbook does not cover it | Books not indexed | `python ingest.py --list` — if empty, redo Step 5 |
| `Address already in use` | An earlier instance is still running | `pkill -f web.py` then start again |
| Answers take 3+ minutes | Thermal throttling, or microSD | Check `vcgencmd measure_temp` while answering; over 80C means it is throttling. Add a fan. |
| Random reboots under load | Underpowered supply | Use the official 27 W supply |

To separate a **search** problem from a **model** problem without waiting for
full answers, run the retrieval inspector — it loads no language model and
finishes in seconds:

```bash
python diagnose.py
```

Every case labelled `ALL PRESENT` means retrieval is working and any wrong
answer is the model's doing. `NONE FOUND` means the search failed, which is a
different problem with a different fix.

---

## Step 8 — Report back

This is the part that matters most, because **every Raspberry Pi figure in our
report is currently an estimate from published benchmarks, not a measurement.**
You are in a position to replace them with real numbers.

Please record and send back:

1. **Your exact hardware** — Pi model, RAM, storage type, cooling, power supply
2. **Which profile you ran** (`pi` or desktop) and whether it was stable
3. **Time per answer.** Ask 3-4 questions and note how long each takes from
   pressing send to the answer appearing
4. **Peak temperature** during a long answer: `vcgencmd measure_temp`
5. **Peak memory**: run `free -h` while an answer is being generated
6. **Anything in this guide that was wrong or unclear** — you are the first
   person through it

If you have a few hours to spare, the full validation run gives directly
comparable numbers:

```bash
source .venv/bin/activate
python evaluate.py --no-tts --lang en     # start with English to check it works
python evaluate.py --no-tts               # all 34 questions
```

Results land in `results/` as timestamped CSV, Markdown and JSON, each with a
snapshot of the configuration used. Send the whole folder back — comparing your
Pi run against the laptop run, on identical questions, is a genuinely useful
result rather than a demo.
