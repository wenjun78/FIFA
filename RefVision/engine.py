"""
RefVision foul engine — scoring over pose tracks.

Identical thresholds to the browser build (foul-upload.html). Everything here is
computed from the pose geometry: contact distance between the two players' lower
limbs, challenge speed scaled by the tackler's own body height in frame, contact
height, and a studs proxy from foot orientation. Nothing is hardcoded per clip.
"""
import math

# ---- thresholds (keep in sync with the JS build) ----
RED_PTS, YEL_PTS = 4.0, 1.2
CONTACT_GATE = 0.5
BOUND_MARGIN = 0.4
REVIEW_LO, REVIEW_HI = 0.45, 0.60
BODY_M = 1.35           # shoulder->ankle ~ 1.35 m, used to scale speed to km/h
CONTACT_FRAC = 0.18     # limbs within 18% of body-height count as contact

# ---- BlazePose 33-point indices ----
IDX = dict(Lsh=11, Rsh=12, Lel=13, Rel=14, Lwr=15, Rwr=16, Lhip=23, Rhip=24,
           Lkn=25, Rkn=26, Lank=27, Rank=28, Lheel=29, Rheel=30, Lfoot=31, Rfoot=32)
LOWER = [IDX[k] for k in ("Lkn", "Rkn", "Lank", "Rank", "Lheel", "Rheel", "Lfoot", "Rfoot")]
FEET = {IDX[k] for k in ("Lank", "Rank", "Lheel", "Rheel", "Lfoot", "Rfoot")}
CONNECTIONS = [(11, 12), (11, 23), (12, 24), (23, 24), (11, 13), (13, 15), (12, 14),
               (14, 16), (23, 25), (25, 27), (27, 29), (29, 31), (27, 31), (24, 26),
               (26, 28), (28, 30), (30, 32), (28, 32)]


def _clamp(v, a, b):
    return max(a, min(b, v))


def _dist(p, q):
    return math.hypot(p[0] - q[0], p[1] - q[1])


def _avg(a, b):
    return ((a[0] + b[0]) / 2.0, (a[1] + b[1]) / 2.0)


def _torso_angle(pose):
    """Angle of the torso away from vertical: ~0deg standing upright, ~90deg lying
    horizontal (on the ground). Used to detect a player going to ground."""
    try:
        sh = _avg(pose[IDX["Lsh"]], pose[IDX["Rsh"]])
        hip = _avg(pose[IDX["Lhip"]], pose[IDX["Rhip"]])
    except Exception:
        return None
    dx = abs(sh[0] - hip[0])
    dy = abs(hip[1] - sh[1])
    return math.degrees(math.atan2(dx, max(dy, 1e-4)))


def _torso_center(p):
    return _avg(_avg(p[IDX["Lsh"]], p[IDX["Rsh"]]), _avg(p[IDX["Lhip"]], p[IDX["Rhip"]]))


def _body_h_of(p):
    sh = _avg(p[IDX["Lsh"]], p[IDX["Rsh"]])
    ak = _avg(p[IDX["Lank"]], p[IDX["Rank"]])
    return max(abs(sh[1] - ak[1]), 0.06)


def _detect_holding(trk_a, trk_b):
    """A hand of one player sustained on the other's torso (shirt-pull / grab). trk_* are
    [(t, pose), ...]. Pose-only screening, not a graded card call.
    Returns (suspected, t, contact_point) — the point where the hand meets the body."""
    pa = {round(t, 3): p for t, p in trk_a}
    pb = {round(t, 3): p for t, p in trk_b}
    common = sorted(set(pa) & set(pb))
    if len(common) < 4:
        return (False, None, None)
    grabbed = []
    for t in common:
        A, B = pa[t], pb[t]
        bh = (_body_h_of(A) + _body_h_of(B)) / 2.0
        thr = 0.38 * bh                       # a hand within ~0.38 body-heights of the torso centre
        tcA, tcB = _torso_center(A), _torso_center(B)
        cand = [(_dist(A[IDX["Lwr"]], tcB), A[IDX["Lwr"]], tcB),
                (_dist(A[IDX["Rwr"]], tcB), A[IDX["Rwr"]], tcB),
                (_dist(B[IDX["Lwr"]], tcA), B[IDX["Lwr"]], tcA),
                (_dist(B[IDX["Rwr"]], tcA), B[IDX["Rwr"]], tcA)]
        d, hand, torso = min(cand, key=lambda x: x[0])
        if d < thr:
            grabbed.append((t, _avg(hand, torso)))    # contact sits between the hand and the body
    sustained = len(grabbed) >= 3 and len(grabbed) >= 0.25 * len(common)
    if not sustained:
        return (False, None, None)
    mt, mpt = grabbed[len(grabbed) // 2]
    return (True, mt, mpt)


def _detect_aerial(trk_a, trk_b, contact_t):
    """Both players airborne with upper-body contact — a jumping / heading duel. Pose-only
    screening. Returns (suspected, t, elbow_to_head)."""
    pa = {round(t, 3): p for t, p in trk_a}
    pb = {round(t, 3): p for t, p in trk_b}
    common = sorted(set(pa) & set(pb))
    if len(common) < 4:
        return (False, None, False)

    def jumped(pts):
        hips = [_avg(p[IDX["Lhip"]], p[IDX["Rhip"]])[1] for p in pts]
        bh = sum(_body_h_of(p) for p in pts) / len(pts)
        return (max(hips) - min(hips)) > 0.35 * bh     # hips rose by >35% of a body height

    ja = jumped(list(pa.values()))
    jb = jumped(list(pb.values()))

    tc = min(common, key=lambda t: abs(t - contact_t))
    A, B = pa[tc], pb[tc]
    bh = (_body_h_of(A) + _body_h_of(B)) / 2.0
    UPPER = [IDX["Lsh"], IDX["Rsh"], IDX["Lel"], IDX["Rel"], IDX["Lwr"], IDX["Rwr"], 0]  # +nose (idx 0)
    up_d, ua, ub = min(((_dist(A[i], B[j]), A[i], B[j]) for i in UPPER for j in UPPER),
                       key=lambda x: x[0])
    upper_contact = up_d < 0.55 * bh
    aerial = (ja or jb) and upper_contact
    contact_point = _avg(ua, ub) if aerial else None

    elbow_head, head_pt = False, None
    for P, Q in ((A, B), (B, A)):
        for s in ("L", "R"):
            el, sh = P[IDX[s + "el"]], P[IDX[s + "sh"]]
            if el[1] < sh[1] - 0.02 and _dist(el, Q[0]) < 0.45 * bh:   # raised arm near opp head
                elbow_head, head_pt = True, _avg(el, Q[0])
    if elbow_head:
        aerial = True                       # an arm to the head is a challenge regardless of jump
        contact_point = head_pt             # and the contact belongs at the head, not the torso
    return (aerial, tc, elbow_head, contact_point)


def analyse_from_tracks(frames, team_centroids=None):
    """
    frames: list of {"t": seconds, "poses": [ pose, ... ]}
            where pose is a list of 33 (x, y, visibility) tuples, normalised to the image.
    team_centroids: optional list of team shirt colours (BGR) ordered most-common first,
            from pose.team_palette. When given, the duel pair is restricted to the two
            MAIN teams (clusters 0 and 1) so the referee (a separate, rarer cluster) is
            never picked as a foul participant.
    Returns a dict with the call, confidence, evidence rows and the contact frame,
    or {"error": "no_two_poses"} when two players were never detected together.
    """
    TEAM_DIST = 45.0  # min BGR distance between shirt colours to read as opposing teams

    def _coldist(a, b):
        if a is None or b is None:
            return None
        return ((a[0] - b[0]) ** 2 + (a[1] - b[1]) ** 2 + (a[2] - b[2]) ** 2) ** 0.5

    def _team_of(c):
        if not team_centroids or c is None:
            return None
        return min(range(len(team_centroids)),
                   key=lambda k: (c[0] - team_centroids[k][0]) ** 2 + (c[1] - team_centroids[k][1]) ** 2 + (c[2] - team_centroids[k][2]) ** 2)

    GROUND_DEG = 55.0   # torso angle past which a player is essentially on the ground

    # representative body height (normalized coords) to scale the contact-onset threshold
    _bhs = []
    for f in frames:
        for p in f["poses"]:
            try:
                sh = _avg(p[IDX["Lsh"]], p[IDX["Rsh"]])
                ak = _avg(p[IDX["Lank"]], p[IDX["Rank"]])
                h = abs(sh[1] - ak[1])
                if h > 0.03:
                    _bhs.append(h)
            except Exception:
                pass
    BH = sorted(_bhs)[len(_bhs) // 2] if _bhs else 0.25
    ONSET_D = CONTACT_FRAC * BH * 1.3   # studs within ~this distance = contact established

    best = None         # overall closest lower-limb pair (last-ditch fallback)
    best_cross = None   # closest pair on DIFFERENT shirt colours (colour-only fallback)
    best_duel = None    # closest pair across the two MAIN teams, ref excluded (preferred)
    # Per-frame closest record for each category, EXCLUDING frames where both players are
    # already on the ground. We pick the ONSET of contact from these, not the global minimum:
    # on a tackle the limbs keep closing as the players collapse together, so the tightest
    # tangle lands a few tenths AFTER the actual strike. The strike is the first frame the
    # studs reach the leg.
    duel_up_by_frame, cross_up_by_frame, any_up_by_frame = [], [], []
    for f in frames:
        poses = f["poses"]
        if len(poses) < 2:
            continue
        colors = f.get("colors") or [None] * len(poses)
        angs = [_torso_angle(p) for p in poses]
        fr_duel = fr_cross = fr_any = None
        for i in range(len(poses)):
            for j in range(i + 1, len(poses)):
                A, B = poses[i], poses[j]
                ci = colors[i] if i < len(colors) else None
                cj = colors[j] if j < len(colors) else None
                cd = _coldist(ci, cj)
                cross = cd is not None and cd >= TEAM_DIST
                # a true duel = one player from team 0 and one from team 1 (ref = cluster >=2, excluded)
                duel = {_team_of(ci), _team_of(cj)} == {0, 1}
                ai, aj = angs[i], angs[j]
                both_down = (ai is not None and aj is not None
                             and ai > GROUND_DEG and aj > GROUND_DEG)
                for ia in LOWER:
                    for ib in LOWER:
                        d = _dist(A[ia], B[ib])
                        rec = dict(d=d, t=f["t"], A=A, B=B, ia=ia, ib=ib, colA=ci, colB=cj)
                        if best is None or d < best["d"]:
                            best = rec
                        if cross and (best_cross is None or d < best_cross["d"]):
                            best_cross = rec
                        if duel and (best_duel is None or d < best_duel["d"]):
                            best_duel = rec
                        if not both_down:
                            if fr_any is None or d < fr_any["d"]:
                                fr_any = rec
                            if cross and (fr_cross is None or d < fr_cross["d"]):
                                fr_cross = rec
                            if duel and (fr_duel is None or d < fr_duel["d"]):
                                fr_duel = rec
        if fr_duel:
            duel_up_by_frame.append(fr_duel)
        if fr_cross:
            cross_up_by_frame.append(fr_cross)
        if fr_any:
            any_up_by_frame.append(fr_any)

    def _onset(by_frame):
        """The strike: earliest frame where studs are already within contact range, looking
        back up to 0.5s from the tightest tangle so we catch the hit, not the collapse."""
        if not by_frame:
            return None
        tight = min(by_frame, key=lambda r: r["d"])
        tc = tight["t"]
        window = [r for r in by_frame
                  if r["d"] <= ONSET_D and (tc - 0.5) <= r["t"] <= tc + 1e-6]
        return min(window, key=lambda r: r["t"]) if window else tight

    # Prefer the standing-contact ONSET, team-matched first, then colour-only, then the
    # on-ground pick if the whole incident was already grounded.
    best = (_onset(duel_up_by_frame) or _onset(cross_up_by_frame) or _onset(any_up_by_frame)
            or best_duel or best_cross or best)
    if best is None:
        return {"error": "no_two_poses"}

    a_is_foot, b_is_foot = best["ia"] in FEET, best["ib"] in FEET
    if a_is_foot and not b_is_foot:
        TK, OP, tk_foot = best["A"], best["B"], best["ia"]
    elif b_is_foot and not a_is_foot:
        TK, OP, tk_foot = best["B"], best["A"], best["ib"]
    else:
        TK, OP, tk_foot = best["A"], best["B"], best["ia"]
    if TK is best["A"]:
        tackler_color, opponent_color = best.get("colA"), best.get("colB")
    else:
        tackler_color, opponent_color = best.get("colB"), best.get("colA")

    sh = _avg(TK[IDX["Lsh"]], TK[IDX["Rsh"]])
    ank = _avg(TK[IDX["Lank"]], TK[IDX["Rank"]])
    body_h = max(abs(sh[1] - ank[1]), 0.06)

    contact_norm = best["d"] / body_h
    contact_score = _clamp((CONTACT_FRAC + 0.04 - contact_norm) / CONTACT_FRAC, 0, 1)
    other = best["B"][best["ib"]] if best["ia"] == tk_foot else best["A"][best["ia"]]
    contact_pt = _avg(TK[tk_foot], other)

    op_ank = _avg(OP[IDX["Lank"]], OP[IDX["Rank"]])
    op_hip = _avg(OP[IDX["Lhip"]], OP[IDX["Rhip"]])
    h_frac = _clamp((op_ank[1] - contact_pt[1]) / max(op_ank[1] - op_hip[1], 0.05), 0, 1)
    if h_frac < 0.20:
        height_pts, region = 0, "ankle (low)"
    elif h_frac < 0.55:
        height_pts, region = 1, "shin (mid)"
    else:
        height_pts, region = 2, "knee / thigh (high)"

    # challenge speed from the tackling foot around the contact moment
    near = [f for f in frames if len(f["poses"]) >= 1 and abs(f["t"] - best["t"]) <= 0.18]
    kmh = 0.0
    if len(near) >= 2:
        tk_ank_ref = _avg(TK[IDX["Lank"]], TK[IDX["Rank"]])
        path, dt, prev = 0.0, 0.0, None
        for f in near:
            pose, pd = f["poses"][0], float("inf")
            for p in f["poses"]:
                d = _dist(_avg(p[IDX["Lank"]], p[IDX["Rank"]]), tk_ank_ref)
                if d < pd:
                    pd, pose = d, p
            foot = pose[tk_foot]
            if prev is not None:
                path += _dist(prev[0], foot)
                dt += max(f["t"] - prev[1], 1e-3)
            prev = (foot, f["t"])
        if dt > 0:
            metres = (path / body_h) * BODY_M
            kmh = (metres / dt) * 3.6
    speed_pts = _clamp((kmh - 8) / 7, 0, 3.2)
    speed_q = "high" if kmh > 24 else ("moderate" if kmh >= 14 else "low")

    # --- peak danger across the approach window, not just the contact frame ---
    # The menace in a lunge (a raised, studs-up boot) usually peaks a beat BEFORE
    # contact, when the boot is already back down. So scan a short window around the
    # contact moment and keep the WORST reading instead of the single contact frame.
    tk_ank_ref = _avg(TK[IDX["Lank"]], TK[IDX["Rank"]])
    op_cx = _avg(OP[IDX["Lhip"]], OP[IDX["Rhip"]])[0]
    window = [f for f in frames
              if (best["t"] - 0.8) <= f["t"] <= (best["t"] + 0.15) and len(f["poses"]) >= 1]
    studs_up = False
    high_boot = 0.0
    for f in window:
        pose, pd = f["poses"][0], float("inf")
        for p in f["poses"]:
            d = _dist(_avg(p[IDX["Lank"]], p[IDX["Rank"]]), tk_ank_ref)
            if d < pd:
                pd, pose = d, p
        # boot elevation: the higher (lunging) foot lifted toward knee->hip height
        knee_y = min(pose[IDX["Lkn"]][1], pose[IDX["Rkn"]][1])
        hip_y = _avg(pose[IDX["Lhip"]], pose[IDX["Rhip"]])[1]
        foot_y = min(pose[IDX["Lfoot"]][1], pose[IDX["Rfoot"]][1])
        elev = (knee_y - foot_y) / max(knee_y - hip_y, 0.04)   # 0 at knee, ~1 at hip
        high_boot = max(high_boot, _clamp(elev, 0.0, 1.5))
        # studs-up on either foot anywhere in the window
        for side in ("L", "R"):
            toe, ankle = pose[IDX[side + "foot"]], pose[IDX[side + "ank"]]
            if toe[1] < ankle[1] - 0.01 and \
               math.copysign(1, op_cx - ankle[0]) == math.copysign(1, toe[0] - ankle[0]):
                studs_up = True
    high_boot_pts = round(high_boot * 2)        # a hip-high boot adds ~2 points

    foul_confirmed = contact_score >= CONTACT_GATE
    # take the WORSE of contact-height vs boot-elevation, so a low contact delivered by
    # a high airborne lunge is still scored as the dangerous challenge it is.
    severity = speed_pts + (2 if studs_up else 0) + max(height_pts, high_boot_pts)
    contact_marginal = REVIEW_LO <= contact_score < REVIEW_HI
    near_boundary = min(abs(severity - RED_PTS), abs(severity - YEL_PTS)) < BOUND_MARGIN

    review, reason = False, ""
    if contact_marginal:
        review, call, conf = True, "review", 57
        reason = ("Contact is marginal — the engine cannot confirm a foul to the bar it "
                  "requires. Recommend review rather than a guess.")
    elif not foul_confirmed:
        call, conf = "none", _clamp(72 + (CONTACT_GATE - contact_score) * 60, 72, 95)
    elif severity >= RED_PTS:
        call, conf = "red", _clamp(72 + (severity - RED_PTS) * 10, 72, 96)
        if near_boundary:
            review, reason, conf = True, "Severity sits on the red/yellow line. Recommend review.", min(conf, 64)
    elif severity >= YEL_PTS:
        call = "yellow"
        conf = _clamp(72 + min(severity - YEL_PTS, RED_PTS - severity) * 10, 72, 93)
        if near_boundary:
            review, reason, conf = True, "Severity sits on a card boundary. Recommend review.", min(conf, 64)
    else:
        # contact WAS confirmed (foul_confirmed) but severity is below a caution:
        # a foul worth a free kick, no card — distinct from "no foul at all".
        call, conf = "foul", _clamp(72 + (YEL_PTS - severity) * 14, 72, 93)

    # --- simulation (dive) check: a player goes to ground with no confirmed contact ---
    def _cd(a, b):
        d = _coldist(a, b)
        return 1e9 if d is None else d

    def _track(color):
        if color is None:
            return []
        out = []
        for f in frames:
            ps = f["poses"]
            cs = f.get("colors") or [None] * len(ps)
            if not ps:
                continue
            bi = min(range(len(ps)), key=lambda i: _cd(cs[i] if i < len(cs) else None, color))
            if _cd(cs[bi] if bi < len(cs) else None, color) < 95.0:
                out.append((f["t"], ps[bi]))
        return out

    def _went_down(trk):
        seq = [(t, _torso_angle(p)) for (t, p) in trk]
        seq = sorted(((t, a) for (t, a) in seq if a is not None), key=lambda x: x[0])
        if len(seq) < 4:
            return False
        peak_i = max(range(len(seq)), key=lambda i: seq[i][1])
        if seq[peak_i][1] < 58.0:                      # never becomes horizontal enough
            return False
        before = [a for (_, a) in seq[:peak_i]]
        if not before or min(before) >= 38.0:          # was never clearly upright beforehand
            return False
        if sum(1 for (_, a) in seq if a > 55.0) < 3:    # a momentary spike, not a sustained fall
            return False
        return True

    op_down = _went_down(_track(opponent_color))
    tk_down = _went_down(_track(tackler_color))
    # A dive requires the two players to have been clearly APART at their closest approach
    # (they never engaged): contact not confirmed, not even marginal, and > ~55% of a body-
    # height between them. This stops a genuine contact foul whose contact gate was merely
    # borderline from being mislabelled as simulation.
    players_apart = contact_norm > 0.55
    sim_suspected = ((not foul_confirmed) and (not contact_marginal)
                     and players_apart and (op_down or tk_down))
    if sim_suspected:
        sim_detail = ("A player went to ground with no confirmed contact — possible simulation "
                      "(cautionable as a dive under Law 12). Recommend review.")
    elif foul_confirmed:
        sim_detail = "Contact was confirmed — consistent with a genuine challenge, not a dive."
    else:
        sim_detail = "No simulation indicators — no clear go-to-ground without contact."
    simulation = {"suspected": sim_suspected, "detail": sim_detail}

    # ---- other foul types: pose-only screening alongside the leg-challenge call ----
    # The holding/aerial screen runs on the pair with the closest UPPER-BODY contact across
    # ALL players (any shirt colour), not the leg-duel pair. This stops goalkeepers — whose
    # rare kit colours get clustered into 'other' with the referee — from being excluded.
    UPPER_IDX = [IDX["Lsh"], IDX["Rsh"], IDX["Lel"], IDX["Rel"], IDX["Lwr"], IDX["Rwr"], 0]
    best_upper = None
    for f in frames:
        ps = f["poses"]
        if len(ps) < 2:
            continue
        cs = f.get("colors") or [None] * len(ps)
        for i in range(len(ps)):
            for j in range(i + 1, len(ps)):
                cd = _coldist(cs[i], cs[j])
                if cd is None or cd < TEAM_DIST:        # opposing-colour pairs only
                    continue
                d = min(_dist(ps[i][a], ps[j][b]) for a in UPPER_IDX for b in UPPER_IDX)
                if best_upper is None or d < best_upper["d"]:
                    best_upper = dict(d=d, t=f["t"], ci=cs[i], cj=cs[j])
    if best_upper and best_upper["ci"] is not None and best_upper["cj"] is not None:
        scr_a, scr_b, scr_t = _track(best_upper["ci"]), _track(best_upper["cj"]), best_upper["t"]
    else:
        scr_a, scr_b, scr_t = _track(tackler_color), _track(opponent_color), best["t"]
    hold_susp, hold_t, hold_pt = _detect_holding(scr_a, scr_b)
    aer_susp, aer_t, aer_elbow, aer_pt = _detect_aerial(scr_a, scr_b, scr_t)
    # for an upper-body foul the marker belongs on the hand/torso (holding) or the aerial
    # collision point — not the leg-contact point the tackle pass computed.
    contact_t_final = best["t"]
    # surface the upper-body foul; arm-to-head aerial is the most specific/serious signal, so
    # it wins the marker and the label over a generic hold.
    if aer_susp and aer_elbow and aer_pt is not None:
        contact_pt, contact_t_final, foul_type = aer_pt, aer_t, "Aerial challenge · arm to head"
    elif hold_susp and hold_pt is not None:
        contact_pt, contact_t_final, foul_type = hold_pt, hold_t, "Holding"
    elif aer_susp and aer_pt is not None:
        contact_pt, contact_t_final, foul_type = aer_pt, aer_t, "Aerial challenge"
    elif foul_confirmed:
        foul_type = "Leg challenge / tackle"
    else:
        foul_type = "No foul"
    # a detected aerial/holding pattern the leg pipeline scored as nothing is still a foul —
    # bump to 'foul' and flag for review (pose can't grade these from a single view).
    if (hold_susp or aer_susp) and call == "none":
        call, review = "foul", True
        conf = min(conf, 60)
        reason = (foul_type + " detected from pose — severity isn't gradable from a single "
                  "view; recommend review.")
    if aer_susp and aer_elbow:
        review = True
        conf = min(conf, 58)
        reason = ("Possible arm/elbow to the head in an aerial duel — potential serious foul "
                  "play; recommend review.")

    boot_q = "high" if high_boot >= 0.6 else "low"
    evidence = [
        ("Point of contact",
         ("Opponent " + region) if (foul_confirmed or contact_marginal) else "No clear contact",
         region.split(" ")[0].upper() if (foul_confirmed or contact_marginal) else "—",
         "hi" if height_pts == 2 else "mid" if height_pts == 1 else "lo"),
        ("Contact confirmed",
         f"{contact_norm * 100:.0f}% of body-height apart",
         "CONFIRMED" if foul_confirmed else "MARGINAL" if contact_marginal else "NONE",
         "hi" if foul_confirmed else "mid" if contact_marginal else "lo"),
        ("Challenge speed",
         f"{kmh:.0f} km/h (est.)" if kmh > 0 else "not measurable",
         speed_q.upper() if kmh > 0 else "—",
         "hi" if speed_q == "high" else "mid" if speed_q == "moderate" else "lo"),
        ("Boot elevation",
         "Raised — high lunge" if boot_q == "high" else "Low / grounded",
         boot_q.upper(), "hi" if boot_q == "high" else "lo"),
        ("Studs showing",
         "Yes — sole toward opponent" if studs_up else "No — foot flat",
         "STUDS UP" if studs_up else "NO",
         "hi" if studs_up else "ok"),
        ("Simulation (diving)",
         "Possible dive — went down, no contact" if sim_suspected else "No dive indicators",
         "REVIEW" if sim_suspected else "OK",
         "mid" if sim_suspected else "ok"),
        ("Holding (grab)",
         "Sustained hand on opponent" if hold_susp else "None detected",
         "REVIEW" if hold_susp else "OK",
         "mid" if hold_susp else "ok"),
    ]

    return dict(call=call, conf=round(conf), review=review, reason=reason, evidence=evidence,
                contact_pt=contact_pt, contact_t=contact_t_final, poses=[best["A"], best["B"]],
                kmh=kmh, studs_up=studs_up, contact_norm=contact_norm,
                tackler_color=tackler_color, opponent_color=opponent_color,
                simulation=simulation, foul_type=foul_type)
