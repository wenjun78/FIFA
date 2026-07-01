# RefVision — Officiating Console

Pose-only referee decision-support for football. Upload a clip (or paste a YouTube link),
and RefVision finds the **contact moment** from body pose, lets the official **confirm or
correct the exact point of contact** with a click, grades the challenge (and flags possible
dives), and lays its read — no foul / foul / caution / sending-off — beside the referee's
on-field signal, flagging anything close for **review**.

Runs on **CPU only** — no GPU, no external model APIs. The pose model downloads once on the
first run. A hosted overview page lives at `docs/index.html` (GitHub Pages).

> Decision-support, not an automated referee: it measures, marks and compares — the official
> makes the call.

## Run

```bash
python -m venv .venv
.venv\Scripts\Activate.ps1        # Windows (PowerShell)
source .venv/bin/activate          # macOS / Linux
python -m pip install -r requirements.txt
python -m streamlit run app.py     # → http://localhost:8501
```

On Windows, `python -m streamlit run app.py` avoids PATH issues with Microsoft Store Python.
The pose model (~9 MB) downloads automatically on the first analysis, so the first run needs
a connection. **No ffmpeg required.**

`streamlit-image-coordinates` is included in `requirements.txt` and lets you place the contact
point by **clicking the frame**; if it's ever missing, the app falls back to sliders.

## Using the console

Two tabs:

**Crop & mark** — scrub to the contact frame (scrollbar or type the time), click the frame to
place the contact marker, set the **± replay window**, then **Confirm** (jumps you to the
decision).

**Decision** — the confirmed marker shown as a **±n replay that pauses ~2 s on the contact
frame**, the pose measurements (contact, speed, boot elevation, studs, plus simulation and
holding checks), the pose call with confidence, a **possible-dive** warning when simulation is
suspected, and the comparison against the referee's signal.

## How it works

1. **Input** — an uploaded file, or a YouTube URL fetched with `yt-dlp` (progressive stream,
   trimmed in OpenCV — use short highlight clips, not whole matches).
2. **Pose** — MediaPipe Tasks `PoseLandmarker` across the sampled frames (multi-person);
   players separated by shirt colour so the two in the challenge are tracked apart.
3. **Engine** (`engine.py`) — picks the **contact onset** (earliest in-contact frame while both
   players are still upright, so it reports the strike, not the collapse) and measures across the
   approach window: contact confirmation and height, closing speed (scaled by each player's own
   shoulder-to-ankle height in frame, so it reads without a fixed-camera assumption), boot
   elevation, and a studs proxy from foot orientation. It maps those to a call with calibrated
   confidence and **abstains** when contact is marginal. In parallel it screens for **simulation
   (a dive with no confirmed contact)**, and for **holding** (hand-on-torso) and **aerial duels**
   (including an arm/elbow to the head) across all players — flagging these for review.
4. **Mark & correct** — the engine's estimated contact point is only a starting position. In
   the **Crop & mark** tab the official clicks the true point of contact to confirm or correct
   it (essential when pose can't resolve a fused scramble), then sets the **±n** replay window.
   RefVision builds a windowed replay that holds ~2 s on the contact frame.
5. **Decision card** — you set what the referee signalled; the console flags **agreement /
   divergence / review**.

Tune `CONTACT_FRAC`, the speed bands and `RED_PTS` / `YEL_PTS` in `engine.py` to your footage.

## Honest limits

- 2D pose loses limbs under occlusion and motion blur; a clean side-on clip assesses far better
  than a congested one. Fused scrambles (e.g. two goalkeepers in a clinch) are the ceiling of
  single-view pose — which is exactly why the operator places the marker by hand.
- Speed is an estimate; 2D can't recover true depth or camera angle.
- Downloading match footage may be against YouTube's ToS; fine for a research demo, worth knowing
  for anything you'd ship.

## Files

- `app.py` — Streamlit UI and orchestration
- `engine.py` — the foul-scoring engine (pure Python, no heavy deps)
- `pose.py` — MediaPipe multi-person pose extraction, overlay drawing, replay-GIF assembly
- `video_io.py` — upload / YouTube input
- `docs/index.html` — GitHub Pages overview site
- `requirements.txt` — dependencies
