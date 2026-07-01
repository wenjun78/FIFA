"""
RefVision — Officiating Console (Streamlit)
Pick a source in the sidebar, mark the contact point on a ± window replay,
and get an evidence-based call compared against what the referee signalled.
Pose-only: MediaPipe locates the contact moment and measures the call.
Run:  python -m streamlit run app.py
"""
import time
import base64
import streamlit as st
import numpy as np

import engine
import video_io
import pose as posemod

try:
    from streamlit_image_coordinates import streamlit_image_coordinates as st_img_coords
    HAVE_CLICK = True
except Exception:
    HAVE_CLICK = False

st.set_page_config(page_title="RefVision — Officiating Console", layout="wide")

# ---------- styling ----------
st.markdown("""
<style>
  .block-container{max-width:1100px}
  .kicker{font-size:11px;letter-spacing:.28em;text-transform:uppercase;color:#43b0c4;font-weight:700}
  .chip{display:inline-flex;align-items:center;gap:9px;border-radius:4px;padding:9px 14px;
    font-weight:800;font-size:16px;border:1px solid transparent}
  .chip .sq{width:12px;height:17px;border-radius:2px;display:inline-block}
  .chip.red{background:rgba(226,85,99,.12);border-color:#e25563;color:#f4b6bd}.chip.red .sq{background:#e25563}
  .chip.yellow{background:rgba(242,193,78,.12);border-color:#f2c14e;color:#f6dca0}.chip.yellow .sq{background:#f2c14e}
  .chip.none{background:rgba(255,255,255,.05);border-color:#3a464d;color:#8aa0ab}.chip.none .sq{background:#5f7480}
  .chip.foul{background:rgba(108,142,191,.14);border-color:#6c8ebf;color:#bcd0ea}.chip.foul .sq{background:#6c8ebf}
  .chip.review{background:rgba(67,176,196,.16);border-color:#43b0c4;color:#bfe6ee}.chip.review .sq{background:#43b0c4}
  .verdict{padding:13px 15px;border-radius:4px;display:flex;align-items:center;gap:12px;margin-top:6px}
  .verdict .tag{font-size:11px;letter-spacing:.14em;text-transform:uppercase;font-weight:700;
    padding:6px 11px;border-radius:4px;white-space:nowrap}
  .verdict.agree{background:rgba(95,178,122,.08)}.verdict.agree .tag{background:rgba(95,178,122,.16);color:#5fb27a;border:1px solid rgba(95,178,122,.4)}
  .verdict.flag{background:rgba(67,176,196,.14)}.verdict.flag .tag{background:rgba(67,176,196,.2);color:#43b0c4;border:1px dashed #43b0c4}
  .verdict .msg b{color:#43b0c4}.verdict.agree .msg b{color:#5fb27a}
  .ev td{padding:7px 10px;border-bottom:1px solid rgba(255,255,255,.07);font-size:13.5px}
  .ev .q{font-size:10px;letter-spacing:.1em;text-transform:uppercase}
  .q.hi{color:#e25563}.q.mid{color:#f2c14e}.q.lo{color:#8aa0ab}.q.ok{color:#5fb27a}
  .gifhover{position:relative;display:block;max-width:720px}
  .gifhover img{width:100%;display:block;border-radius:6px;border:1px solid rgba(255,255,255,.12)}
  .gifhover .anim{position:absolute;left:0;top:0;opacity:0;transition:opacity .12s}
  .gifhover:hover .anim{opacity:1}
  .gifhint{font-size:11.5px;color:#8aa0ab;margin-top:5px}
</style>""", unsafe_allow_html=True)

CALL = {"red": ("red", "Red — sending-off"), "yellow": ("yellow", "Yellow — caution"),
        "foul": ("foul", "Foul"), "none": ("none", "No foul — play on"),
        "review": ("review", "Review — uncertain")}
RANK = {"none": 0, "foul": 1, "yellow": 2, "red": 3}


def chip(call):
    cls, txt = CALL[call]
    return f'<span class="chip {cls}"><span class="sq"></span>{txt}</span>'


def verdict_html(call, signalled, review):
    if review:
        sig = CALL[signalled][1].split(" — ")[0].lower()
        return (f'<div class="verdict flag"><span class="tag">Flag · review</span>'
                f'<span class="msg">Engine is uncertain; referee signalled <b>{sig}</b>. '
                f'<b>Recommend review</b> before the call stands.</span></div>')
    if call == signalled:
        return ('<div class="verdict agree"><span class="tag">Agreement</span>'
                '<span class="msg">Evidence matches the signalled call. <b>No review needed.</b></span></div>')
    if RANK[call] > RANK[signalled]:
        return ('<div class="verdict flag"><span class="tag">Divergence</span>'
                '<span class="msg">Evidence supports a <b>stronger sanction</b> than signalled. '
                '<b>Recommend VAR review.</b></span></div>')
    return ('<div class="verdict flag"><span class="tag">Divergence</span>'
            '<span class="msg">Evidence supports a <b>lesser sanction</b> than signalled. '
            '<b>Recommend review.</b></span></div>')


# ---------- analysis runner ----------
def run_analysis(path, win_start, win_end, max_frames, land="Decision"):
    import cv2
    cap = cv2.VideoCapture(path)
    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    total = cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0
    dur = (total / fps) if total else None
    cap.release()

    with st.status("Analysing…", expanded=True) as status:
        status.update(label="Loading pose model & reading frames…")
        bar = st.progress(0.0)
        prev_ph = st.empty()
        _pc = {"i": 0}

        def _preview(t, frame_bgr, poses):
            _pc["i"] += 1
            if _pc["i"] % 6:
                return
            prev_ph.image(frame_bgr[:, :, ::-1], caption=f"processing… {t:.2f}s",
                          use_container_width=True)

        frames, bgr = posemod.extract_poses(
            path, start=win_start, end=win_end, max_frames=max_frames,
            progress=lambda v: bar.progress(v), preview_cb=_preview)
        prev_ph.empty()
        status.update(label="Scoring…")

        # Team colours: cluster shirt colours once (stable across the clip). Done BEFORE
        # scoring so the engine can restrict the duel to the two main teams (ref excluded).
        team_centroids, team_palette = posemod.team_palette(frames)
        result = engine.analyse_from_tracks(frames, team_centroids=team_centroids)

        def _ds(img, maxw=480):
            h, w = img.shape[:2]
            return img if w <= maxw else cv2.resize(img, (maxw, int(h * maxw / w)))
        bgr_small = {t: _ds(im) for t, im in bgr.items()}

        two = sum(1 for f in frames if len(f["poses"]) >= 2)
        full_dur = dur or (max(bgr.keys()) if bgr else 1.0)
        st.session_state.update({
            "video_path": path, "frames": frames,
            "bgr": bgr_small, "result": result,
            "team_centroids": team_centroids, "team_palette": team_palette,
            "frames_n": len(frames), "two_n": two, "duration": full_dur,
            "win": (win_start if win_start is not None else 0.0,
                    win_end if win_end is not None else full_dur),
            "diag_timeline": [(f["t"], len(f["poses"])) for f in frames],
            "diag_max": max((len(f["poses"]) for f in frames), default=0),
            "contact_pt": None, "hover_gif": None, "hover_poster": None,
            "manual_contact_pt": None, "manual_contact_t": None,
            "pending_contact_pt": None, "pending_contact_t": None,
            "manual_gif": None, "manual_poster": None,
            "manual_span": None, "manual_gif_err": None,
        })
        if "error" not in result:
            ct = result["contact_t"]
            st.session_state["contact_pt"] = result["contact_pt"]
            try:
                g, p = posemod.build_hover_gif(
                    frames, bgr, ct, contact_pt=result["contact_pt"], span=3.0,
                    draw_skeleton=False)
                st.session_state["hover_gif"], st.session_state["hover_poster"] = g, p
            except Exception:
                pass
            st.session_state["tabseg"] = land
            st.session_state["active_tab"] = land
        else:
            st.session_state["tabseg"] = "Crop & mark"
            st.session_state["active_tab"] = "Crop & mark"
        status.update(label="Done", state="complete")


# ---------- header ----------
st.markdown('<p class="kicker">RefVision · Officiating Console</p>', unsafe_allow_html=True)
st.title("Foul Engine")
st.caption("Two-player pose finds the contact moment and grades the challenge — speed, boot "
           "elevation, studs, contact height — and screens for holding and aerial duels, then "
           "weighs the call against the referee's signal.")

# ---------- sidebar: source / settings ----------
with st.sidebar:
    st.subheader("Source")
    mode = st.radio("Input", ["Upload a clip", "YouTube link"], label_visibility="collapsed")
    upload, url = None, None
    if mode == "Upload a clip":
        upload = st.file_uploader("Video file", type=["mp4", "mov", "webm", "mkv", "avi"])
    else:
        url = st.text_input("YouTube URL", placeholder="https://youtube.com/watch?v=…")
        st.caption("Paste a short highlight — mark the contact point in the Crop & mark tab after.")

    st.subheader("Settings")
    max_frames = st.slider("Max frames to process", 500, 700, 600, 50)
    st.caption("MediaPipe pose on every sampled frame. More frames = better contact "
               "localisation, slightly slower.")

    go = st.button("Analyse clip", type="primary", use_container_width=True)

# ---------- triggers ----------
if go:
    for k in ("scrub", "cropwin"):
        st.session_state.pop(k, None)
    try:
        if mode == "Upload a clip":
            if not upload:
                st.warning("Choose a video file first.")
                st.stop()
            run_analysis(video_io.from_upload(upload), None, None, max_frames, land="Crop & mark")
        else:
            if not url:
                st.warning("Paste a YouTube link first.")
                st.stop()
            run_analysis(video_io.from_youtube(url), None, None, max_frames, land="Crop & mark")
    except RuntimeError as e:
        st.error(str(e))
        st.stop()


# ---------- tab renderers ----------
def render_crop():
    st.markdown("**Crop & mark** · scrub to the contact frame, click the contact point, set the "
                "± replay window, then **Confirm** → **Decision**")
    if st.session_state.get("video_path"):
        try:
            st.video(st.session_state["video_path"])
        except Exception:
            st.caption("Couldn't embed the source video; marking still works below.")
    res = st.session_state.get("result")
    if res and "error" in res:
        two, n = st.session_state.get("two_n", 0), st.session_state.get("frames_n", 1)
        st.warning(f"Two players were never detected together ({two} of {n} frames had two poses). "
                   "Try the slow-mo replay segment — it tracks much better than live play.")
        return
    bgr = st.session_state.get("bgr")
    if not bgr or not res:
        return
    _mark_contact_ui(bgr, res)


def _scrub_with_keyin(label, options, default, base_key, step=0.1, fmt="%.2f"):
    """A scrollbar (select_slider) plus a type-in number box for the same value, kept in
    sync. Returns the selected option. The number box snaps to the nearest option."""
    sld_k, num_k = base_key + "_sld", base_key + "_num"
    if st.session_state.get(sld_k) not in options:          # init, or reset if stale (new clip)
        st.session_state[sld_k] = default
        st.session_state[num_k] = float(default)

    def _from_sld():
        st.session_state[num_k] = float(st.session_state[sld_k])

    def _from_num():
        nearest = min(options, key=lambda o: abs(o - st.session_state[num_k]))
        st.session_state[sld_k] = nearest
        st.session_state[num_k] = float(nearest)

    c1, c2 = st.columns([3, 1])
    c1.select_slider(label, options=options, key=sld_k, on_change=_from_sld)
    c2.number_input("key in", min_value=float(min(options)), max_value=float(max(options)),
                    step=step, format=fmt, key=num_k, on_change=_from_num,
                    label_visibility="collapsed")
    return st.session_state[sld_k]


def _mark_contact_ui(bgr, result):
    tkeys = sorted(bgr.keys())
    opts = [round(t, 2) for t in tkeys]
    ct = result.get("contact_t", opts[0] if opts else 0.0)
    dflt = st.session_state.get("pending_contact_t")
    if dflt not in opts:
        dflt = round(min(tkeys, key=lambda k: abs(k - ct)), 2)
    ti = _scrub_with_keyin("Frame (s) · replay centre", opts, dflt, "mark_frame", step=0.1, fmt="%.2f")
    st.session_state["pending_contact_t"] = ti
    fk = min(bgr.keys(), key=lambda k: abs(k - ti))
    frame = bgr[fk]

    st.markdown("**Contact point**")
    pend = (st.session_state.get("pending_contact_pt")
            or st.session_state.get("manual_contact_pt")
            or result.get("contact_pt") or (0.5, 0.5))
    if HAVE_CLICK:
        # click-to-place only — no X/Y sliders (their persistent keys were overwriting the click).
        from PIL import Image
        ring = posemod.draw_overlay(frame, [], pend)
        pil = Image.fromarray(ring[:, :, ::-1])
        st.caption("Click on the image to place the contact point.")
        coords = st_img_coords(pil, key="mark_click")
        if coords:
            w = coords.get("width") or pil.width
            h = coords.get("height") or pil.height
            newpt = (coords["x"] / w, coords["y"] / h)
            if abs(newpt[0] - pend[0]) > 1e-4 or abs(newpt[1] - pend[1]) > 1e-4:
                st.session_state["pending_contact_pt"] = newpt
                st.rerun()
    else:
        # fallback only when the click component isn't installed
        st.image(posemod.draw_overlay(frame, [], pend)[:, :, ::-1], use_container_width=True,
                 caption=f"{ti:.2f}s")
        st.info("Install `streamlit-image-coordinates` to place the point by clicking — sliders meanwhile.")
        cax = st.slider("Horizontal %", 0, 100, int(round(pend[0] * 100)), key="mark_x")
        cay = st.slider("Vertical %", 0, 100, int(round(pend[1] * 100)), key="mark_y")
        st.session_state["pending_contact_pt"] = (cax / 100.0, cay / 100.0)

    # replay window (± seconds around the frame above) — the marker pops up at its centre
    nsec = _scrub_with_keyin("Replay window (± seconds)", [0.5, 1.0, 1.5, 2.0, 3.0, 5.0],
                             2.0, "crop_n", step=0.5, fmt="%.1f")
    lo, hi = max(0.0, ti - nsec), ti + nsec
    st.caption(f"Window: {lo:.1f}s – {hi:.1f}s")

    b1, b2 = st.columns(2)
    if b1.button("✓ Confirm contact point", type="primary", use_container_width=True):
        pt = st.session_state.get("pending_contact_pt", pend)
        st.session_state["manual_contact_pt"] = pt
        st.session_state["manual_contact_t"] = ti
        st.session_state["manual_span"] = nsec
        try:  # ±n replay; the marker pops up at the contact frame and holds ~2 s
            g, p = posemod.build_marked_gif(bgr, ti, pt, span=nsec)
            st.session_state["manual_gif_err"] = None
        except Exception as e:
            g, p = None, None
            st.session_state["manual_gif_err"] = f"{type(e).__name__}: {e}"
        st.session_state["manual_gif"], st.session_state["manual_poster"] = g, p
        st.session_state["_goto_decision"] = True
        st.rerun()
    if b2.button("Reset to engine estimate", use_container_width=True):
        for k in ("pending_contact_pt", "manual_contact_pt", "manual_contact_t",
                  "manual_gif", "manual_poster", "manual_span", "manual_gif_err"):
            st.session_state.pop(k, None)
        st.rerun()
    if st.session_state.get("manual_contact_pt") is not None:
        st.caption(f"Contact point confirmed — the Decision tab replays ±{nsec:g}s and pauses on contact.")


def render_decision(frames, bgr, result):
    if "error" in result:
        st.warning("No assessment — two players were never tracked together. Try a clip trimmed "
                   "to the slow-mo replay, which tracks much better than live play.")
        return
    ct = result["contact_t"]
    mpt = st.session_state.get("manual_contact_pt")
    mt = st.session_state.get("manual_contact_t")
    mgif, mpost = st.session_state.get("manual_gif"), st.session_state.get("manual_poster")
    tt = mt if mt is not None else ct
    nrep = st.session_state.get("manual_span", 2.0)
    if mpt is not None and mgif and mpost:
        # operator-confirmed point; ±window replay, marker pops up at contact and holds ~2 s
        g = base64.b64encode(mgif).decode()
        p = base64.b64encode(mpost).decode()
        st.markdown(
            f'<div class="gifhover"><img class="poster" src="data:image/png;base64,{p}">'
            f'<img class="anim" src="data:image/gif;base64,{g}"></div>'
            f'<div class="gifhint">±{nrep:g}s replay · marker pops up at contact @ {tt:.2f}s and '
            f'pauses ~2 s · hover to play · confirmed by operator</div>',
            unsafe_allow_html=True)
    elif mpt is not None and bgr:
        # confirmed point but the replay didn't build — fall back to a single still
        err = st.session_state.get("manual_gif_err")
        if err:
            st.caption(f"⚠ ±{nrep:g}s replay couldn't be built ({err}) — showing the contact still.")
        key = min(bgr.keys(), key=lambda k: abs(k - tt))
        img = posemod.draw_overlay(bgr[key], [], mpt)
        st.image(img[:, :, ::-1], caption=f"contact @ {tt:.2f}s · confirmed by operator",
                 use_container_width=True)
    elif st.session_state.get("hover_gif") and st.session_state.get("hover_poster"):
        g = base64.b64encode(st.session_state["hover_gif"]).decode()
        p = base64.b64encode(st.session_state["hover_poster"]).decode()
        st.markdown(
            f'<div class="gifhover"><img class="poster" src="data:image/png;base64,{p}">'
            f'<img class="anim" src="data:image/gif;base64,{g}"></div>'
            f'<div class="gifhint">contact @ {ct:.2f}s · hover to play (±3 s) · '
            f'mark it yourself in the Crop &amp; mark tab</div>',
            unsafe_allow_html=True)
    elif bgr:
        key = min(bgr.keys(), key=lambda k: abs(k - ct))
        still = posemod.draw_overlay(bgr[key], [], result["contact_pt"])  # footage + contact ring, no skeleton
        st.image(still[:, :, ::-1], caption=f"contact @ {ct:.2f}s", use_container_width=True)

    st.write("")
    col_l, col_r = st.columns([1, 1])
    with col_l:
        st.markdown("**Pose measurements**")
        rows = "".join(
            f'<tr><td style="color:#8aa0ab">{lbl}</td>'
            f'<td style="text-align:right;font-weight:600">{val}'
            f'<div class="q {q}">{qt}</div></td></tr>'
            for (lbl, val, qt, q) in result["evidence"])
        st.markdown(f'<table class="ev" style="width:100%">{rows}</table>', unsafe_allow_html=True)
    with col_r:
        st.markdown("**Pose call**")
        st.markdown(chip(result["call"]), unsafe_allow_html=True)
        ft = result.get("foul_type")
        if ft:
            st.caption(f"Foul type: {ft}")
        st.progress(result["conf"] / 100.0, text=f"Confidence {result['conf']}%")
        if result["review"]:
            st.caption("⚐ " + result["reason"])
        sim = result.get("simulation") or {}
        if sim.get("suspected"):
            st.warning("**Possible simulation (dive).** " + sim.get("detail", ""))

    st.divider()
    st.markdown("**Decision card**")
    signalled = st.radio("Referee signalled", ["none", "foul", "yellow", "red"],
                         format_func=lambda c: CALL[c][1].split(" — ")[0], horizontal=True)
    decision_call = result["call"]
    e1, e2, e3 = st.columns([1, 0.2, 1])
    e1.markdown("Decision (pose engine)")
    e1.markdown(chip(decision_call), unsafe_allow_html=True)
    e3.markdown("Signalled (referee)")
    e3.markdown(chip(signalled), unsafe_allow_html=True)
    st.markdown(verdict_html(decision_call, signalled, result["review"]), unsafe_allow_html=True)


# ---------- results: tabs ----------
result = st.session_state.get("result")
if result:
    # a Confirm elsewhere may request the Decision tab; apply it BEFORE the tabseg
    # widget is created (Streamlit forbids modifying a widget's state key afterwards)
    if st.session_state.pop("_goto_decision", False):
        st.session_state["tabseg"] = "Decision"
    options = ["Crop & mark", "Decision"]
    if hasattr(st, "segmented_control"):
        sel = st.segmented_control("view", options, key="tabseg", label_visibility="collapsed")
    else:
        sel = st.radio("view", options, key="tabseg", horizontal=True, label_visibility="collapsed")
    if sel not in options:
        sel = st.session_state.get("active_tab", "Decision")
    st.session_state["active_tab"] = sel
    st.divider()
    frames = st.session_state.get("frames")
    bgr = st.session_state.get("bgr")
    if sel == "Crop & mark":
        render_crop()
    else:
        render_decision(frames, bgr, result)
else:
    st.info("Pick a source in the sidebar and hit **Analyse clip**. The pose model downloads on the "
            "first run, so the first analysis needs a connection.")

st.divider()
st.caption("Speed is scaled by each player's own shoulder-to-ankle height in frame; boot elevation "
           "and studs are read across the approach window, not a single frame. Leg challenges are "
           "graded; holding and aerial duels are flagged for review. Most reliable on cleanly-framed "
           "or slow-mo footage.")
