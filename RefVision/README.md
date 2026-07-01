# RefVision — Officiating Console (Streamlit)

The foul engine, now able to assess a real clip. Upload a video **or paste a YouTube
link**, trim to the challenge, and get an evidence-based call (no-card / yellow / red)
that's compared against what the referee signalled — with the divergence flag firing
on real footage. The trained foul-severity classifier + Grad-CAM is an optional
production layer you plug in.

## Run

```bash
pip install -r requirements.txt
python -m streamlit run app.py
```

On Windows, `python -m streamlit run app.py` avoids PATH issues.

The pose model (~30 MB) downloads automatically on the first analysis, so the first run
needs an internet connection. **No ffmpeg required** — YouTube clips download as a single
progressive stream and are trimmed in OpenCV. (For a very long source video you could
install ffmpeg to fetch only a segment, but the default path needs nothing extra; just
use short highlight clips.)

## How it works

1. **Input** — an uploaded file, or a YouTube URL fetched with `yt-dlp` (downloaded as a
   single progressive stream, then trimmed to your start/end window in OpenCV — so use
   short highlight clips, not whole matches).
2. **Pose** — MediaPipe Tasks `PoseLandmarker` with `num_poses=2` runs across the frames.
3. **Engine** (`engine.py`) — finds the contact moment (closest approach between the two
   players' lower limbs) and measures, **across the approach window rather than a single
   frame**: contact confirmation, contact height, challenge speed (scaled by the tackler's
   own shoulder-to-ankle height in frame, so it reads in km/h without a fixed camera
   assumption), **boot elevation** (a raised, airborne high lunge — caught even when contact
   lands low), and a studs proxy from foot orientation. It maps those to a call with
   calibrated confidence, and **abstains** when contact is marginal.
4. **Replay** — a single side-by-side video is rendered (real footage + overlay on the
   left, skeleton-only on the right) so you can watch what the system saw, in sync. Written
   as H.264 via the pip-bundled ffmpeg, so it plays in the browser with no system install.
5. **Decision card** — you set what the referee signalled; the console flags
   **agreement / divergence / review** and recommends escalation when the evidence
   supports a different sanction.

The thresholds in `engine.py` are identical to the browser build, so calls match across
both. Tune `CONTACT_FRAC`, the speed bands and `RED_PTS`/`YEL_PTS` to your footage.

## Honest limits

- 2D pose loses limbs under occlusion and motion blur, so a clean side-on clip assesses
  far better than a congested one. The app tells you how many frames tracked both players.
- Speed is an estimate — 2D can't recover true depth or camera angle.
- Downloading match footage is against YouTube's ToS; fine for a research demo, worth
  knowing for anything you'd ship.

## Wiring in the trained classifier + Grad-CAM

`classifier.py` is the integration point for your SoccerNet-trained model and your
medical-imaging XAI work. It's optional — the app runs pose-only until you supply weights.

1. Implement `load_model(weights_path)` with your architecture + `load_state_dict`.
2. Implement `_preprocess(frames_bgr)` to match your training input.
3. Set `TARGET_LAYER_NAME` to the conv layer Grad-CAM should attribute over.
4. `pip install torch grad-cam`, then pass the weights path in the sidebar.

When weights are present, the console runs the classifier on the contact-window frames
and renders the Grad-CAM heatmap on the contact frame — the "why" a referee can read.

## Files

- `app.py` — Streamlit UI and orchestration
- `engine.py` — the foul-scoring engine (pure Python, no heavy deps)
- `pose.py` — MediaPipe two-person pose extraction + overlay drawing
- `video_io.py` — upload / YouTube input
- `classifier.py` — optional trained classifier + Grad-CAM hook
