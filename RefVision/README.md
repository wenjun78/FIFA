# RefVision — Officiating Console

Pose-only referee decision-support for football. Upload a clip (or paste a YouTube link),
and RefVision finds the **contact moment** from body pose, lets the official **confirm or
correct the exact point of contact** with a click, grades the challenge (and flags possible
dives), and presents its read — no foul / foul / caution / sending-off — beside the referee's
on-field signal, flagging anything close for **review**.

Runs on **CPU only** — no GPU and no external model APIs. The pose model downloads
automatically on the first run.

> Decision-support, not an automated referee: it measures, marks and compares — the official
> makes the call.

## Run

```bash
git clone https://github.com/wenjun78/FIFA.git
cd FIFA/RefVision
python -m pip install -r requirements.txt
python -m streamlit run app.py
```

The console opens at `http://localhost:8501`. On Windows, the `python -m` form avoids PATH
issues. The pose model (~9 MB) downloads on the first analysis, so the first run needs an
internet connection. No ffmpeg required.

## Using the console

**Crop & mark** — scrub to the contact frame (scrollbar or type the time), click the frame to
place the contact marker, set the **± replay window**, then **Confirm**.

**Decision** — the confirmed marker shown as a **±n replay that pauses ~2 s on the contact
frame**, the pose measurements (contact, speed, boot elevation, studs, plus simulation and
holding checks), the pose call with confidence, a possible-dive warning when simulation is
suspected, and the comparison against the referee's signal.

## How it works

1. **Input** — an uploaded file, or a YouTube URL fetched with `yt-dlp` and trimmed in OpenCV
   (use short highlight clips).
2. **Pose** — MediaPipe Tasks `PoseLandmarker` runs across the sampled frames; players are
   separated by shirt colour so the two in the challenge are tracked apart.
3. **Engine** (`engine.py`) — selects the **contact onset** (the earliest in-contact frame while
   both players are still upright, so it reports the strike rather than the collapse) and measures
   across the approach window: contact confirmation and height, closing speed (scaled by each
   player's own shoulder-to-ankle height in frame, so it reads without a fixed-camera assumption),
   boot elevation, and a studs proxy from foot orientation. It maps those to a call with calibrated
   confidence and abstains when contact is marginal. In parallel it screens for simulation (a dive
   with no confirmed contact), and for holding (hand-on-torso) and aerial duels (including an arm or
   elbow to the head) across all players, flagging these for review.
4. **Mark & correct** — the engine's estimated contact point is a starting position only. In the
   **Crop & mark** tab the official clicks the true point of contact to confirm or correct it, then
   sets the **±n** replay window. RefVision builds a windowed replay that holds ~2 s on the contact
   frame.
5. **Decision** — the referee's on-field signal is recorded and the console flags **agreement,
   divergence, or review**.

Thresholds in `engine.py` (`CONTACT_FRAC`, the speed bands, `RED_PTS` / `YEL_PTS`) can be tuned to
your footage.

## Limitations

- 2D pose loses limbs under occlusion and motion blur; a clean side-on clip reads far better than a
  congested one. Fused scrambles (for example, two goalkeepers in a clinch) are the ceiling of
  single-view pose — which is why the official places the marker by hand.
- Closing speed is an estimate; 2D cannot recover true depth or camera angle.
- Downloading match footage may violate YouTube's Terms of Service; use footage you have the rights to.

## Files

- `app.py` — Streamlit UI and orchestration
- `engine.py` — the foul-scoring engine (pure Python)
- `pose.py` — MediaPipe multi-person pose extraction, overlay drawing, and replay-GIF assembly
- `video_io.py` — clip and YouTube input
- `docs/index.html` — project overview page (GitHub Pages)
- `requirements.txt` — dependencies
