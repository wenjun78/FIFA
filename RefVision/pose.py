"""
Two-person pose extraction with MediaPipe Tasks PoseLandmarker.

Downloads the pose model on first run. Processes frames within an optional
[start, end] window, sampling so long clips stay fast. Returns:
  frames:      [{"t": seconds, "poses": [[(x,y,vis)...33], ...]}]
  bgr_frames:  {frame_time_seconds: numpy BGR image}  (kept sparse for overlay)
"""
import os
import urllib.request

POSE_MODEL_URL = ("https://storage.googleapis.com/mediapipe-models/pose_landmarker/"
                  "pose_landmarker_full/float16/1/pose_landmarker_full.task")
MODEL_PATH = os.path.join(os.path.dirname(__file__), "pose_landmarker_full.task")


def ensure_model():
    if not os.path.exists(MODEL_PATH):
        urllib.request.urlretrieve(POSE_MODEL_URL, MODEL_PATH)
    return MODEL_PATH


def _torso_color(bgr, pose):
    """Median BGR colour of a player's torso (shoulders->hips) — a shirt-colour proxy
    used to separate the two teams. Returns None if the torso isn't clearly visible."""
    import numpy as np
    h, w = bgr.shape[:2]
    pts = [pose[i] for i in (11, 12, 23, 24)]  # shoulders + hips
    if min(p[2] for p in pts) < 0.3:
        return None
    xs = sorted(int(p[0] * w) for p in pts)
    ys = sorted(int(p[1] * h) for p in pts)
    x0, x1, y0, y1 = xs[0], xs[-1], ys[0], ys[-1]
    if (x1 - x0) < 4 or (y1 - y0) < 4:
        return None
    px, py = int((x1 - x0) * 0.2), int((y1 - y0) * 0.15)  # shrink to central torso
    x0, x1, y0, y1 = x0 + px, x1 - px, y0 + py, y1 - py
    x0, y0 = max(0, x0), max(0, y0)
    x1, y1 = min(w, x1), min(h, y1)
    if x1 <= x0 or y1 <= y0:
        return None
    patch = bgr[y0:y1, x0:x1].reshape(-1, 3)
    return tuple(float(c) for c in np.median(patch, axis=0))


def extract_poses(video_path, start=None, end=None, max_frames=300, progress=None, num_poses=5, preview_cb=None):
    import cv2
    import mediapipe as mp
    from mediapipe.tasks import python as mp_python
    from mediapipe.tasks.python import vision

    ensure_model()
    options = vision.PoseLandmarkerOptions(
        base_options=mp_python.BaseOptions(model_asset_path=MODEL_PATH),
        running_mode=vision.RunningMode.VIDEO,
        num_poses=num_poses,
        min_pose_detection_confidence=0.45,
        min_tracking_confidence=0.45,
        min_pose_presence_confidence=0.45,
    )
    landmarker = vision.PoseLandmarker.create_from_options(options)

    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise RuntimeError("Could not open the video file.")
    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
    dur = total / fps if total else None

    s = float(start) if start is not None else 0.0
    e = float(end) if (end is not None) else (dur if dur else 1e9)
    if start is not None:
        cap.set(cv2.CAP_PROP_POS_MSEC, s * 1000.0)

    # sample stride so we span the WHOLE window with <= max_frames frames.
    # ceil (not floor) — floor lets a clip just over the cap truncate to the first
    # max_frames frames (e.g. only the first 10s of a 15s clip), dropping the tail.
    win_frames = (e - s) * fps if dur else max_frames
    stride = max(1, int(-(-win_frames // max_frames))) if win_frames > max_frames else 1

    frames, bgr_frames = [], {}
    i = 0
    ts_ms = 0
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        t = cap.get(cv2.CAP_PROP_POS_MSEC) / 1000.0
        if t < s:
            continue
        if t > e:
            break
        if i % stride != 0:
            i += 1
            continue
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        mp_img = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
        ts_ms += int(1000 / fps) + 1  # must be strictly increasing
        res = landmarker.detect_for_video(mp_img, ts_ms)
        poses = []
        for lm in (res.pose_landmarks or []):
            poses.append([(p.x, p.y, getattr(p, "visibility", 1.0)) for p in lm])
        colors = [_torso_color(frame, p) for p in poses]
        frames.append({"t": round(t, 3), "poses": poses, "colors": colors})
        bgr_frames[round(t, 3)] = frame.copy()
        if preview_cb is not None:
            preview_cb(round(t, 3), frame, poses)
        i += 1
        if progress is not None and win_frames:
            progress(min(1.0, (t - s) / max(e - s, 1e-3)))
        if len(frames) >= max_frames:
            break

    cap.release()
    landmarker.close()
    return frames, bgr_frames


def _draw_on(img, poses, contact_pt=None, pose_colors=None):
    """Draw the skeletons (and optional contact ring) onto the given BGR image.
    pose_colors: optional list of BGR colours, one per pose (team colours). Falls
    back to the accent/grey pair when not supplied."""
    import cv2
    from engine import CONNECTIONS
    h, w = img.shape[:2]
    fallback = [(196, 176, 67), (223, 217, 205)]  # BGR: accent, opponent grey
    for i, p in enumerate(poses):
        c = pose_colors[i] if (pose_colors is not None and i < len(pose_colors) and pose_colors[i] is not None) \
            else fallback[i % 2]
        for a, b in CONNECTIONS:
            pa, pb = p[a], p[b]
            if (pa[2] < 0.3) or (pb[2] < 0.3):
                continue
            cv2.line(img, (int(pa[0] * w), int(pa[1] * h)),
                     (int(pb[0] * w), int(pb[1] * h)), c, 2, cv2.LINE_AA)
        for k in (11, 12, 23, 24, 25, 26, 27, 28, 31, 32):
            pt = p[k]
            if pt[2] < 0.3:
                continue
            cv2.circle(img, (int(pt[0] * w), int(pt[1] * h)), 3, c, -1, cv2.LINE_AA)
    if contact_pt is not None:
        cx, cy = int(contact_pt[0] * w), int(contact_pt[1] * h)
        cv2.circle(img, (cx, cy), 13, (99, 85, 226), 2, cv2.LINE_AA)  # BGR red
        cv2.circle(img, (cx, cy), 3, (99, 85, 226), -1, cv2.LINE_AA)
    return img


def draw_overlay(bgr, poses, contact_pt=None, pose_colors=None):
    """Skeletons drawn on a copy of the real frame."""
    return _draw_on(bgr.copy(), poses, contact_pt, pose_colors)


def skeleton_frame(shape_hw, poses, contact_pt=None, pose_colors=None):
    """Skeletons drawn on a clean dark canvas — the skeleton-only view."""
    import numpy as np
    h, w = int(shape_hw[0]), int(shape_hw[1])
    canvas = np.full((h, w, 3), (20, 26, 14), dtype=np.uint8)  # BGR dark slate
    return _draw_on(canvas, poses, contact_pt, pose_colors)


# ---- team-colour assignment (shirt colour -> fixed skeleton colour) ----
SKEL_PALETTE = [(245, 165, 50), (50, 90, 245), (60, 215, 245), (220, 90, 210), (230, 200, 60)]
# BGR: blue, red, yellow, magenta, teal  (team A, team B, ref/other, ...)


def team_palette(frames, k=3):
    """Cluster every shirt colour in the clip into <=k teams (stable, global) and
    assign each a fixed skeleton colour. Returns (centroids_bgr, palette_bgr), both
    ordered by how many players fall in each team (most common team first)."""
    import numpy as np
    import cv2
    cols = [c for f in frames for c in (f.get("colors") or []) if c is not None]
    distinct = {tuple(round(v) for v in c) for c in cols}
    if len(cols) < 2 or len(distinct) < 2:
        return [], []
    k = min(k, len(distinct))
    data = np.array(cols, dtype=np.float32)
    crit = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 25, 1.0)
    _, labels, centers = cv2.kmeans(data, k, None, crit, 5, cv2.KMEANS_PP_CENTERS)
    labels = labels.flatten()
    order = sorted(range(k), key=lambda c: -int((labels == c).sum()))
    centroids = [tuple(float(v) for v in centers[c]) for c in order]
    palette = [SKEL_PALETTE[i % len(SKEL_PALETTE)] for i in range(len(centroids))]
    return centroids, palette


def skel_colors_for(colors, centroids, palette, default=(196, 176, 67)):
    """Map each pose's shirt colour to its team skeleton colour (per-frame; use
    lock_team_colors for a flicker-free, per-player-stable assignment instead)."""
    if not centroids:
        return [default] * len(colors)
    out = []
    for c in colors:
        if c is None:
            out.append(default)
            continue
        bi = min(range(len(centroids)),
                 key=lambda i: (c[0] - centroids[i][0]) ** 2 + (c[1] - centroids[i][1]) ** 2 + (c[2] - centroids[i][2]) ** 2)
        out.append(palette[bi])
    return out


def _ensure_tracks(frames, max_jump=0.16):
    """Give every pose a stable track id. If the frames already carry track_ids
    (YOLO/ByteTrack backend) keep them; otherwise assign ids greedily by nearest
    hip-centre across consecutive frames (MediaPipe backend)."""
    if any(f.get("track_ids") for f in frames):
        return
    def hipc(p):
        return ((p[23][0] + p[24][0]) / 2.0, (p[23][1] + p[24][1]) / 2.0)
    next_id = 0
    prev = []  # (track_id, hip_centre)
    for f in frames:
        poses = f["poses"]
        ids = [-1] * len(poses)
        used = set()
        cur = []
        for i, p in enumerate(poses):
            hc = hipc(p)
            best, bd = None, max_jump
            for (tid, php) in prev:
                if tid in used:
                    continue
                d = ((hc[0] - php[0]) ** 2 + (hc[1] - php[1]) ** 2) ** 0.5
                if d < bd:
                    bd, best = d, tid
            if best is None:
                best, next_id = next_id, next_id + 1
            used.add(best)
            ids[i] = best
            cur.append((best, hc))
        f["track_ids"] = ids
        prev = cur


def lock_team_colors(frames, centroids, palette, default=(196, 176, 67)):
    """Write frame['skel']: a per-pose skeleton BGR that is LOCKED per player for
    the whole clip (majority vote over each track's shirt-colour samples), so a
    single player's skeleton colour never flickers frame-to-frame."""
    if not centroids:
        for f in frames:
            f["skel"] = [default] * len(f["poses"])
        return
    from collections import defaultdict, Counter
    _ensure_tracks(frames)

    def team_of(c):
        if c is None:
            return None
        return min(range(len(centroids)),
                   key=lambda i: (c[0] - centroids[i][0]) ** 2 + (c[1] - centroids[i][1]) ** 2 + (c[2] - centroids[i][2]) ** 2)

    votes = defaultdict(Counter)
    for f in frames:
        cols = f.get("colors") or []
        tids = f.get("track_ids") or [-1] * len(f["poses"])
        for i in range(len(f["poses"])):
            t = team_of(cols[i] if i < len(cols) else None)
            if t is not None and i < len(tids):
                votes[tids[i]][t] += 1
    locked = {tid: cnt.most_common(1)[0][0] for tid, cnt in votes.items()}
    for f in frames:
        tids = f.get("track_ids") or [-1] * len(f["poses"])
        skel = []
        for i in range(len(f["poses"])):
            team = locked.get(tids[i] if i < len(tids) else -1)
            skel.append(palette[team] if team is not None else default)
        f["skel"] = skel


def build_sidebyside(frames, bgr_frames, contact_t=None, out_path=None, max_w=460,
                     centroids=None, palette=None):
    """
    Render a single playable video: real footage + overlay on the left, the
    skeleton-only view on the right, in sync. Written as H.264 via imageio-ffmpeg
    (pip-installed binary, no system ffmpeg), so it plays in the browser.
    Returns the mp4 path, or None if there's nothing to render.
    """
    import os
    import tempfile
    import numpy as np
    import cv2
    import imageio.v2 as imageio

    times = sorted(bgr_frames.keys())
    if not times:
        return None
    poses_at = {round(f["t"], 3): f["poses"] for f in frames}
    colors_at = {round(f["t"], 3): (f.get("colors") or []) for f in frames}
    skel_at = {round(f["t"], 3): f.get("skel") for f in frames}
    if out_path is None:
        out_path = os.path.join(tempfile.mkdtemp(prefix="refvision_vid_"), "sidebyside.mp4")

    span = times[-1] - times[0]
    # play back in real time: duration = n_frames / fps  ->  fps = n_frames / span,
    # so the replay runs the same length as the source window (no time-stretch).
    fps = max(4.0, min(60.0, len(times) / span)) if span > 0 else 12.0

    # format="FFMPEG" + yuv420p forces a browser-playable H.264 file and prevents imageio
    # from silently picking a still-image (e.g. TIFF) writer that rejects fps.
    writer = imageio.get_writer(out_path, format="FFMPEG", mode="I", fps=fps,
                                codec="libx264", quality=8, pixelformat="yuv420p",
                                macro_block_size=16)
    try:
        for t in times:
            bgr = bgr_frames[t]
            h, w = bgr.shape[:2]
            if w > max_w:
                s = max_w / w
                bgr = cv2.resize(bgr, (max_w, int(h * s)))
            h, w = bgr.shape[:2]
            poses = poses_at.get(round(t, 3), [])
            pcol = skel_at.get(round(t, 3)) or skel_colors_for(colors_at.get(round(t, 3), []), centroids, palette)
            cp = None  # per-frame contact point isn't tracked; ring shown in the still
            left = _draw_on(bgr.copy(), poses, cp, pcol)
            right = skeleton_frame((h, w), poses, cp, pcol)
            divider = np.full((h, 6, 3), (196, 176, 67), dtype=np.uint8)
            combo = np.hstack([left, divider, right])
            writer.append_data(cv2.cvtColor(combo, cv2.COLOR_BGR2RGB))
    finally:
        writer.close()
    return out_path


def build_hover_gif(frames, bgr_frames, contact_t, contact_pt=None, span=3.0, max_frames=32, max_w=600,
                    centroids=None, palette=None, draw_skeleton=True):
    """Animated GIF spanning [contact-span, contact+span] with the contact at the
    midpoint. Single panel: real footage with the pose overlay drawn on it (the
    contact point is marked on frames near contact). Built with Pillow (no ffmpeg).
    Returns (gif_bytes, poster_png_bytes)."""
    import io
    import cv2
    import numpy as np
    from PIL import Image
    if contact_t is None:
        return None, None
    times = sorted(t for t in bgr_frames if (contact_t - span) <= t <= (contact_t + span))
    if not times:
        times = sorted(bgr_frames.keys())
    if len(times) > max_frames:
        step = len(times) / max_frames
        times = [times[int(i * step)] for i in range(max_frames)]
    poses_at = {round(f["t"], 3): f["poses"] for f in frames}
    colors_at = {round(f["t"], 3): (f.get("colors") or []) for f in frames}
    skel_at = {round(f["t"], 3): f.get("skel") for f in frames}
    imgs = []
    for t in times:
        bgr = bgr_frames[t]
        h, w = bgr.shape[:2]
        if w > max_w:
            s = max_w / w
            bgr = cv2.resize(bgr, (max_w, int(h * s)))
        poses = poses_at.get(round(t, 3), [])
        pcol = skel_at.get(round(t, 3)) or skel_colors_for(colors_at.get(round(t, 3), []), centroids, palette)
        cp = contact_pt if (contact_pt is not None and abs(t - contact_t) < 0.08) else None
        frame = _draw_on(bgr.copy(), poses if draw_skeleton else [], cp, pcol)
        imgs.append(Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)))
    if not imgs:
        return None, None
    cidx = min(range(len(times)), key=lambda i: abs(times[i] - contact_t))
    gbuf = io.BytesIO()
    imgs[0].save(gbuf, format="GIF", save_all=True, append_images=imgs[1:],
                 duration=80, loop=0, disposal=2)
    pbuf = io.BytesIO()
    imgs[cidx].save(pbuf, format="PNG")
    return gbuf.getvalue(), pbuf.getvalue()


def build_marked_gif(bgr_frames, contact_t, contact_pt, span=2.0, max_frames=32, max_w=600,
                     hold_ms=2000, play_ms=80):
    """Footage replay over [contact_t-span, contact_t+span]. The operator's contact marker
    pops up ONLY on the contact frame, which is HELD by repeating it so playback pauses
    ~hold_ms then resumes. Uses a single uniform frame duration (works on every Pillow build,
    unlike a per-frame duration list). No skeleton. Returns (gif_bytes, poster_png_bytes)."""
    import io
    import cv2
    from PIL import Image
    if contact_t is None or contact_pt is None or not bgr_frames:
        return None, None
    times = sorted(t for t in bgr_frames if (contact_t - span) <= t <= (contact_t + span))
    if not times:
        times = sorted(bgr_frames.keys())
    if len(times) > max_frames:
        step = len(times) / max_frames
        times = [times[int(i * step)] for i in range(max_frames)]
    cidx = min(range(len(times)), key=lambda i: abs(times[i] - contact_t))
    reps = max(1, int(round(hold_ms / max(play_ms, 1))))   # repeat contact frame -> pause
    imgs, poster = [], None
    for i, t in enumerate(times):
        bgr = bgr_frames[t]
        h, w = bgr.shape[:2]
        if w > max_w:
            s = max_w / w
            bgr = cv2.resize(bgr, (max_w, int(h * s)))
        cp = contact_pt if i == cidx else None          # ring only at the contact instant
        im = Image.fromarray(cv2.cvtColor(_draw_on(bgr.copy(), [], cp), cv2.COLOR_BGR2RGB))
        if i == cidx:
            poster = im
            imgs.extend([im] * reps)                     # hold the contact frame
        else:
            imgs.append(im)
    if not imgs:
        return None, None
    gbuf = io.BytesIO()
    imgs[0].save(gbuf, format="GIF", save_all=True, append_images=imgs[1:],
                 duration=play_ms, loop=0, disposal=2)   # uniform int duration — robust
    pbuf = io.BytesIO()
    (poster or imgs[0]).save(pbuf, format="PNG")
    return gbuf.getvalue(), pbuf.getvalue()


def build_duel_frames(frames, tackler_color, opponent_color, tol=85.0):
    """Filter each frame's detected poses down to just the two duelling players,
    matched by shirt colour (tackler first -> accent, opponent -> grey), so the
    drawn skeleton sticks to the same two players across the clip instead of
    flickering onto bystanders. Falls back to the first two poses when colours
    are unavailable for a frame."""
    def cdist(a, b):
        if a is None or b is None:
            return 1e9
        return ((a[0] - b[0]) ** 2 + (a[1] - b[1]) ** 2 + (a[2] - b[2]) ** 2) ** 0.5
    out = []
    for f in frames:
        poses = f["poses"]
        colors = f.get("colors") or [None] * len(poses)
        duel = []
        if tackler_color is not None and opponent_color is not None and poses:
            ti = min(range(len(poses)),
                     key=lambda i: cdist(colors[i] if i < len(colors) else None, tackler_color))
            oi = min(range(len(poses)),
                     key=lambda i: cdist(colors[i] if i < len(colors) else None, opponent_color))
            if cdist(colors[ti] if ti < len(colors) else None, tackler_color) < tol:
                duel.append(poses[ti])
            if oi != ti and cdist(colors[oi] if oi < len(colors) else None, opponent_color) < tol:
                duel.append(poses[oi])
        if not duel:
            duel = poses[:2]
        out.append({"t": f["t"], "poses": duel})
    return out
