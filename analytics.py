"""
analytics.py - aggregation engine (the privacy boundary).

Input : per-frame detections (ephemeral tracker id, foot point, centre point).
State : zone occupancy and dwell, counters (lines + door zones), heat grids,
        per-minute series, alerts, anonymous event log.
Output: aggregate snapshots only. Track IDs and coordinates never leave this module.

Robustness layers
  1. Point smoothing      EMA on every track's foot/centre point -> no jitter flips.
  2. Track stitching      when the tracker loses someone (fast motion, occlusion) and
                          a new ID appears where the old one was heading, the new ID
                          inherits the old state. Motion-only: no appearance features.
  3. Dead-band lines      a crossing is a change of *confirmed side* (outside a band
                          around the line), so fast walkers that jump across the line
                          between two frames still count and loiterers on the line don't.
  4. Door zones           a person who first appears inside a door zone came in;
                          one who finally disappears inside it went out. Needs only one
                          sighting in the doorway, so very fast entries/exits still count.
  5. Confirmations        tracks must persist N frames before they count; visits shorter
                          than min_dwell are ignored; zone exits have a grace period;
                          alerts need to persist before opening/closing.
"""
from __future__ import annotations

import math
import threading
import time
from collections import deque
from dataclasses import dataclass, field
from statistics import median
from typing import Callable, Optional

import numpy as np

DWELL_BINS = [(0, 30, "<30s"), (30, 60, "30–60s"), (60, 120, "1–2m"),
              (120, 300, "2–5m"), (300, 600, "5–10m"), (600, math.inf, "10m+")]
RETENTION_S = 24 * 3600


@dataclass
class Detection:
    track_id: Optional[int]
    x: float                      # foot point (bottom-centre), normalised 0..1
    y: float
    cx: float = 0.0               # box centre, normalised
    cy: float = 0.0
    box: tuple = (0, 0, 0, 0)     # normalised x1,y1,x2,y2 (local anonymised view only)
    conf: float = 1.0


# ---------------------------------------------------------------- geometry
def point_in_poly(x: float, y: float, poly) -> bool:
    inside, n, j = False, len(poly), len(poly) - 1
    for i in range(n):
        xi, yi = poly[i]
        xj, yj = poly[j]
        if (yi > y) != (yj > y) and x < (xj - xi) * (y - yi) / ((yj - yi) or 1e-12) + xi:
            inside = not inside
        j = i
    return inside


def line_frame(a, b, p, aspect: float):
    """Signed distance (in frame-height units, >0 = IN side) and projection t along A->B."""
    ax, ay, bx, by, px, py = a[0] * aspect, a[1], b[0] * aspect, b[1], p[0] * aspect, p[1]
    dx, dy = bx - ax, by - ay
    L2 = dx * dx + dy * dy or 1e-12
    d = (dx * (py - ay) - dy * (px - ax)) / math.sqrt(L2)
    t = ((px - ax) * dx + (py - ay) * dy) / L2
    return d, t


def pct(values, q):
    if not values:
        return None
    s = sorted(values)
    k = (len(s) - 1) * q
    f, c = math.floor(k), math.ceil(k)
    return s[f] if f == c else s[f] + (s[c] - s[f]) * (k - f)


# ------------------------------------------------------------------- state
@dataclass
class ZoneStat:
    occupants: dict = field(default_factory=dict)          # canon id -> enter (mono)
    dwells: deque = field(default_factory=deque)            # (t_exit mono, dwell_s)
    visits: int = 0
    peak: int = 0
    buf: deque = field(default_factory=lambda: deque(maxlen=9))


@dataclass
class CounterStat:
    n_in: int = 0
    n_out: int = 0
    events: deque = field(default_factory=deque)            # (t mono, +1 in / -1 out)


class Engine:
    ALERT_RAISE_S = 5.0
    ALERT_CLEAR_S = 10.0

    def __init__(self, config: dict, on_minute: Optional[Callable[[dict, dict], None]] = None):
        self.lock = threading.RLock()
        self.on_minute = on_minute
        self.zones, self.lines = [], []
        self.zstat: dict[str, ZoneStat] = {}
        self.cstat: dict[str, CounterStat] = {}
        self.aspect = 16 / 9
        self.reset()
        self.apply_config(config)

    # ------------------------------------------------------------ config
    def apply_config(self, cfg: dict) -> None:
        with self.lock:
            a = cfg.get("analytics", {})
            self.window_s = float(a.get("stats_window_min", 15)) * 60
            self.min_dwell_s = float(a.get("min_dwell_s", 2))
            self.exit_grace_s = float(a.get("exit_grace_s", 1.0))
            self.lost_s = float(a.get("lost_track_s", 0.7))        # unseen this long -> limbo
            self.stitch_s = float(a.get("stitch_window_s", 2.5))   # limbo lifetime
            self.door_stitch_s = float(a.get("door_stitch_s", 0.6))  # re-link window when lost in a doorway
            self.recover_s = float(a.get("recover_s", 3.0))        # tracker may still recover its own id this long
            self.min_frames = int(a.get("min_track_frames", 3))
            self.smooth = float(a.get("smoothing", 0.55))           # EMA weight of the new sample
            self.decay_k = math.log(2) / float(a.get("heat_half_life_s", 30))
            self.k_min = int(cfg.get("privacy", {}).get("k_min", 0))
            self.zones = [z for z in cfg.get("zones", []) if len(z.get("points", [])) >= 3]
            old_pts = {l["id"]: l["points"] for l in getattr(self, "lines", [])}
            self.lines = [l for l in cfg.get("lines", []) if len(l.get("points", [])) == 2]
            moved = {l["id"] for l in self.lines if old_pts.get(l["id"], l["points"]) != l["points"]}
            self.doors = [z for z in self.zones if z.get("type") == "door"]
            self.zstat = {z["id"]: self.zstat.get(z["id"], ZoneStat()) for z in self.zones}
            ids = [l["id"] for l in self.lines] + [z["id"] for z in self.doors]
            self.cstat = {i: self.cstat.get(i, CounterStat()) for i in ids}
            zids = set(self.zstat)
            for tr in list(self.tracks.values()) + list(self.limbo.values()):
                tr["zones"] = {k: v for k, v in tr["zones"].items() if k in zids}
                tr["side"] = {k: v for k, v in tr["side"].items() if k in self.cstat and k not in moved}

    def reset(self) -> None:
        with self.lock:
            self.gw, self.gh = 64, 36
            self.heat_live = np.zeros((self.gh, self.gw), np.float32)
            self.heat_total = np.zeros((self.gh, self.gw), np.float32)
            self.tracks: dict[int, dict] = {}       # canon id -> state
            self.limbo: dict[int, dict] = {}        # canon id -> state (recently lost)
            self.alias: dict[int, int] = {}         # raw tracker id -> canon id
            self.visitors = 0
            self.stitches = 0
            self.zstat = {k: ZoneStat() for k in self.zstat}
            self.cstat = {k: CounterStat() for k in self.cstat}
            self.series: deque = deque(maxlen=24 * 60)
            # keep the open minute: its counts already happened and may already be on the hub
            self.bucket: Optional[dict] = getattr(self, "bucket", None)
            self.event_seq = getattr(self, "event_seq", 0)
            self.events: deque = deque(maxlen=200)
            self.pending_events: list = []
            self.alerts_active: dict = {}
            self.alerts_log: deque = deque(maxlen=50)
            self.alert_on: dict = {}
            self.alert_off: dict = {}
            self.people_buf: deque = deque(maxlen=9)
            self.people_hist: deque = deque(maxlen=240)
            self.frame_times: deque = deque(maxlen=40)
            self.last_t: Optional[float] = None
            self.started = time.time()

    def forget_tracks(self) -> None:
        """Call when the tracker restarts (camera switch): finalise everyone, drop id mapping."""
        with self.lock:
            b = self._bucket(time.time())
            for cid, tr in list(self.tracks.items()) + list(self.limbo.items()):
                self._finalise(cid, tr, b, count_doors=False)
            self.tracks, self.limbo, self.alias = {}, {}, {}
            for st in self.zstat.values():
                st.occupants.clear()

    # ------------------------------------------------------------ helpers
    def _event(self, text: str, kind: str) -> None:
        self.event_seq += 1
        e = {"id": self.event_seq, "t": round(time.time(), 1), "kind": kind, "text": text}
        self.events.appendleft(e)
        self.pending_events.append(e)

    def events_since(self, seq: int) -> list:
        with self.lock:
            return [dict(e) for e in reversed(self.events) if e.get("id", 0) > seq]

    def drain_events(self) -> list:
        with self.lock:
            out, self.pending_events = self.pending_events, []
            return out

    def _names(self) -> dict:
        return {"z": {z["id"]: z["name"] for z in self.zones},
                "c": {**{l["id"]: (l["name"], "line") for l in self.lines},
                      **{z["id"]: (z["name"], "door") for z in self.doors}}}

    def _bucket(self, wall: float) -> dict:
        key = int(wall // 60) * 60
        b = self.bucket
        if b is None or b["t"] != key:
            if b is not None:
                self.series.append(b)
                if self.on_minute:
                    try:
                        self.on_minute(b, self._names())
                    except Exception as e:  # logging must never stop analytics
                        print("minute log error:", e)
            b = self.bucket = {"t": key, "n": 0, "p_sum": 0.0, "p_max": 0, "z": {}, "c": {}}
        for z in self.zones:
            b["z"].setdefault(z["id"], [0.0, 0, 0, 0.0, 0])   # occ_sum, occ_max, visits, dwell_sum, dwell_n
        for cid in self.cstat:
            b["c"].setdefault(cid, [0, 0])                      # in, out
        return b

    def _count(self, cid: str, into: bool, b: dict, label: str) -> None:
        st = self.cstat.get(cid)
        if st is None:
            return
        if into:
            st.n_in += 1
            b["c"][cid][0] += 1
        else:
            st.n_out += 1
            b["c"][cid][1] += 1
        st.events.append((time.monotonic(), 1 if into else -1))
        self._event(label, "count")

    def _close_visit(self, zid: str, enter: float, t_exit: float, b: Optional[dict] = None) -> None:
        st = self.zstat.get(zid)
        if st is None:
            return
        dwell = t_exit - enter
        if dwell < self.min_dwell_s:
            return
        st.visits += 1
        st.dwells.append((t_exit, dwell))
        b = b if b is not None and b is self.bucket else self._bucket(time.time())
        b["z"][zid][2] += 1
        b["z"][zid][3] += dwell
        b["z"][zid][4] += 1
        name = next((z["name"] for z in self.zones if z["id"] == zid), zid)
        self._event(f"{name}: visit ended after {fmt_s(dwell)}", "visit")

    def _finalise(self, cid: int, tr: dict, b: dict, count_doors: bool = True) -> None:
        """Track is gone for good: close visits, count door exits."""
        for zid, (enter, left) in tr["zones"].items():
            self._close_visit(zid, enter, left if left is not None else tr["seen"], b)
            if zid in self.zstat:
                self.zstat[zid].occupants.pop(cid, None)
        if count_doors and tr["frames"] >= self.min_frames:
            # any door touched during the last ~6 sightings: they walked out through it
            last = set().union(*tr["recent"]) if tr["recent"] else set()
            for z in self.doors:
                if z["id"] in last:
                    self._count(z["id"], False, b, f"{z['name']}: {z.get('out_label', 'Out').lower()}")

    def _doors_touched(self, d: Detection) -> set:
        """Doors touched by the person's ground contact (3 samples along the box bottom)."""
        if not self.doors:
            return set()
        x1, _, x2, y2 = d.box if d.box != (0, 0, 0, 0) else (d.x, 0, d.x, d.y)
        w = x2 - x1
        pts = [(d.x, d.y), (x1 + 0.15 * w, y2), (x2 - 0.15 * w, y2)]
        return {z["id"] for z in self.doors if any(point_in_poly(px, py, z["points"]) for px, py in pts)}

    @staticmethod
    def _zone_pt(z: dict, foot, ctr):
        """Feet by default; body centre for seated people, desks, close-up or top-down views."""
        return ctr if z.get("anchor") == "center" else foot

    def _in_door(self, p) -> bool:
        return any(point_in_poly(p[0], p[1], z["points"]) for z in self.doors)

    def _dist(self, p, q) -> float:
        return math.hypot((p[0] - q[0]) * self.aspect, p[1] - q[1])

    # ------------------------------------------------------------ ingest
    def update(self, dets: list[Detection], aspect: float = 16 / 9, t: Optional[float] = None) -> None:
        with self.lock:
            t = time.monotonic() if t is None else t
            self.aspect = aspect
            wall = time.time()
            dt = 0.0 if self.last_t is None else min(t - self.last_t, 1.0)
            self.last_t = t
            self.frame_times.append(t)
            b = self._bucket(wall)

            # heat + raw occupancy (raw detections: independent of tracker quality)
            if dt > 0:
                self.heat_live *= math.exp(-self.decay_k * dt)
            counts = {z["id"]: 0 for z in self.zones}
            for d in dets:
                gx = min(self.gw - 1, max(0, int(d.x * self.gw)))
                gy = min(self.gh - 1, max(0, int(d.y * self.gh)))
                self.heat_live[gy, gx] += dt
                self.heat_total[gy, gx] += dt
                for z in self.zones:
                    px, py = self._zone_pt(z, (d.x, d.y), (d.cx, d.cy))
                    if point_in_poly(px, py, z["points"]):
                        counts[z["id"]] += 1
            self.people_buf.append(len(dets))
            for zid, c in counts.items():
                self.zstat[zid].buf.append(c)

            # tracked behaviour: known ids first, so new ids can only stitch onto
            # tracks that were NOT seen in this frame
            tracked = [d for d in dets if d.track_id is not None]
            for d in sorted(tracked, key=lambda d: d.track_id not in self.alias):
                self._track(d, t, b)

            # lost -> limbo -> finalised
            for cid in [k for k, v in self.tracks.items() if t - v["seen"] > self.lost_s]:
                tr = self.limbo[cid] = self.tracks.pop(cid)
                # lost inside a doorway = almost certainly walked out: re-link a NEW tracker id only
                # briefly, so the next person through the same door is not mistaken for them
                tr["ttl"] = self.door_stitch_s if self._in_door(tr["fp"]) else self.stitch_s
            # finalise only once the tracker can no longer recover its own id either
            for cid in [k for k, v in self.limbo.items() if t - v["seen"] > max(v.get("ttl", self.stitch_s), self.recover_s)]:
                self._finalise(cid, self.limbo.pop(cid), b)
            if len(self.alias) > 5000:  # keep the alias map bounded
                live = set(self.tracks) | set(self.limbo)
                self.alias = {k: v for k, v in self.alias.items() if v in live}

    def _stitch(self, foot, t) -> Optional[int]:
        best, best_d = None, 1e9
        # limbo tracks, plus active ones unseen for 0.2 s+ (never steal a track that missed one frame)
        pool = list(self.limbo.items()) + [(k, v) for k, v in self.tracks.items() if t - v["seen"] >= 0.2]
        new_in_door = self._in_door(foot)
        for cid, tr in pool:
            gap = t - tr["seen"]
            if gap > tr.get("ttl", self.stitch_s) and cid in self.limbo:
                continue   # past its re-link window (kept only so the tracker can recover its own id)
            if new_in_door and gap > 0.35 and self._in_door(tr["fp"]):
                continue   # someone left through this door; a new arrival there is a different person
            g = min(gap, 1.0)
            pred = (tr["fp"][0] + tr["v"][0] * g, tr["fp"][1] + tr["v"][1] * g)
            speed = math.hypot(tr["v"][0] * self.aspect, tr["v"][1])
            # scale-invariant gate: ~half a body height, growing with speed and time lost
            radius = max(0.06, 0.5 * tr["h"]) + 0.5 * speed * g + 0.08 * min(gap, self.stitch_s)
            d = min(self._dist(pred, foot), self._dist(tr["fp"], foot) * 1.25)
            if d < radius and d < best_d:
                best, best_d = cid, d
        return best

    def _track(self, d: Detection, t: float, b: dict) -> None:
        raw = d.track_id
        foot, ctr = (d.x, d.y), (d.cx, d.cy)
        cid = self.alias.get(raw)
        if cid is None:
            cid = self._stitch(foot, t)
            if cid is not None:                              # same person, new tracker id
                if cid in self.limbo:
                    self.tracks[cid] = self.limbo.pop(cid)
                # retire the old tracker id(s) so they can't drive this state in parallel
                self.alias = {k: v for k, v in self.alias.items() if v != cid}
                self.stitches += 1
            else:
                cid = raw if raw not in self.tracks and raw not in self.limbo else -raw - 10 ** 6
            self.alias[raw] = cid
        elif cid in self.limbo:                              # tracker recovered the same id
            self.tracks[cid] = self.limbo.pop(cid)

        bh = max(0.02, d.box[3] - d.box[1]) if d.box != (0, 0, 0, 0) else 0.3
        tr = self.tracks.get(cid)
        if tr is None:
            tr = self.tracks[cid] = {"fp": foot, "cp": ctr, "v": (0.0, 0.0), "h": bh, "seen": t, "first": t,
                                     "frames": 0, "born": foot, "confirmed": False,
                                     "early": set(), "recent": deque(maxlen=6), "door_in": False,
                                     "zones": {}, "side": {}}
        else:
            gap = max(t - tr["seen"], 1e-3)
            # adaptive smoothing: damp small jitter, follow real (large) moves immediately
            move = self._dist(foot, tr["fp"])
            a = min(1.0, self.smooth + move / 0.04)
            nfp = (tr["fp"][0] + a * (foot[0] - tr["fp"][0]), tr["fp"][1] + a * (foot[1] - tr["fp"][1]))
            ncp = (tr["cp"][0] + a * (ctr[0] - tr["cp"][0]), tr["cp"][1] + a * (ctr[1] - tr["cp"][1]))
            vel = ((foot[0] - tr["fp"][0]) / gap, (foot[1] - tr["fp"][1]) / gap)
            tr["v"] = (0.4 * tr["v"][0] + 0.6 * vel[0], 0.4 * tr["v"][1] + 0.6 * vel[1])
            tr["fp"], tr["cp"] = nfp, ncp
            tr["h"] = 0.7 * tr["h"] + 0.3 * bh
        tr["seen"] = t
        tr["frames"] += 1
        touched = self._doors_touched(d)
        tr["recent"].append(touched)
        if tr["frames"] <= 6:
            tr["early"] |= touched

        # confirmation: visitor count
        if not tr["confirmed"] and tr["frames"] >= self.min_frames:
            tr["confirmed"] = True
            self.visitors += 1
        # door entry: touched a doorway during its first ~6 sightings (never for re-linked tracks)
        if tr["confirmed"] and not tr["door_in"] and tr["frames"] <= 6 and tr["early"]:
            tr["door_in"] = True
            for z in self.doors:
                if z["id"] in tr["early"]:
                    self._count(z["id"], True, b, f"{z['name']}: {z.get('in_label', 'In').lower()}")

        # counting lines: change of confirmed side, outside the dead-band
        for ln in self.lines:
            p = tr["cp"] if ln.get("anchor") == "center" else tr["fp"]
            dist, along = line_frame(ln["points"][0], ln["points"][1], p, self.aspect)
            if along < -0.2 or along > 1.2:                  # walked around the end: forget side
                tr["side"].pop(ln["id"], None)
                continue
            if along < -0.05 or along > 1.05:
                continue
            band = float(ln.get("band", 0.03))
            s = 1 if dist > band else -1 if dist < -band else 0
            if s == 0:
                continue
            prev = tr["side"].get(ln["id"])
            tr["side"][ln["id"]] = s
            if prev is not None and prev != s and tr["frames"] >= 2:
                into = s > 0
                lbl = ln.get("in_label", "In") if into else ln.get("out_label", "Out")
                self._count(ln["id"], into, b, f"{ln['name']}: {lbl.lower()}")

        # zone membership with exit grace
        inside = set()
        for z in self.zones:
            if z.get("type") == "door":
                continue
            px, py = self._zone_pt(z, tr["fp"], tr["cp"])
            if point_in_poly(px, py, z["points"]):
                inside.add(z["id"])
        zones = tr["zones"]
        for zid in inside:
            if zid in zones:
                zones[zid][1] = None
            else:
                zones[zid] = [t, None]
                self.zstat[zid].occupants[cid] = t
        for zid in list(zones):
            if zid in inside:
                continue
            enter, left = zones[zid]
            if left is None:
                zones[zid][1] = t
            elif t - left > self.exit_grace_s:
                self._close_visit(zid, enter, left, b)
                self.zstat[zid].occupants.pop(cid, None)
                del zones[zid]

    # ---------------------------------------------------------- snapshot
    def _k(self, n):
        return None if (self.k_min and 0 < n < self.k_min) else n

    def snapshot(self) -> dict:
        with self.lock:
            t, wall = time.monotonic(), time.time()
            b = self._bucket(wall)
            people = int(round(median(self.people_buf))) if self.people_buf else 0
            b["n"] += 1
            b["p_sum"] += people
            b["p_max"] = max(b["p_max"], people)
            self.people_hist.append([round(wall, 1), people])

            zones_out, longest_wait, all_dw = [], None, []
            for z in self.zones:
                st = self.zstat[z["id"]]
                while st.dwells and t - st.dwells[0][0] > RETENTION_S:
                    st.dwells.popleft()
                win = [d for te, d in st.dwells if t - te <= self.window_s]
                if z.get("type") != "door":
                    all_dw += win
                count = int(round(median(st.buf))) if st.buf else 0
                st.peak = max(st.peak, count)
                bz = b["z"][z["id"]]
                bz[0] += count
                bz[1] = max(bz[1], count)
                cur = [t - e for e in st.occupants.values()]
                longest = max(cur) if cur else None
                if z.get("type") == "queue" and longest is not None:
                    longest_wait = max(longest_wait or 0, longest)
                cap = z.get("capacity") or None
                alert_s = z.get("wait_alert_s") or None
                status = "ok"
                if (cap and count > cap) or (alert_s and longest and longest > alert_s):
                    status = "critical"
                elif (cap and count >= 0.8 * cap) or (alert_s and longest and longest >= 0.75 * alert_s):
                    status = "warning"
                self._alerts(z, count, cap, longest, alert_s)
                zones_out.append({
                    "id": z["id"], "name": z["name"], "type": z.get("type", "area"), "area": z.get("area") or "",
                    "count": self._k(count), "peak": st.peak, "visits": st.visits,
                    "visits_window": len(win),
                    "throughput_hr": round(len(win) / self.window_s * 3600, 1),
                    "avg_dwell_s": r1(sum(win) / len(win)) if win else None,
                    "median_dwell_s": r1(pct(win, 0.5)), "p90_dwell_s": r1(pct(win, 0.9)),
                    "max_dwell_s": r1(max(win)) if win else None,
                    "longest_current_s": r1(longest), "capacity": cap,
                    "utilisation": round(count / cap, 2) if cap and self._k(count) is not None else None,
                    "wait_alert_s": alert_s, "status": status,
                })

            counters, tot_in, tot_out = [], 0, 0
            for kind, items in (("line", self.lines), ("door", self.doors)):
                for c in items:
                    st = self.cstat[c["id"]]
                    while st.events and t - st.events[0][0] > RETENTION_S:
                        st.events.popleft()
                    w_in = sum(1 for te, d in st.events if d > 0 and t - te <= self.window_s)
                    w_out = sum(1 for te, d in st.events if d < 0 and t - te <= self.window_s)
                    role = counter_role(c)
                    site = role == "perimeter"
                    if site:                       # only perimeter counters feed the site Entries / Exits
                        tot_in += st.n_in
                        tot_out += st.n_out
                    counters.append({
                        "id": c["id"], "name": c["name"], "kind": kind, "site": site, "role": role,
                        "from_area": c.get("from_area") or "", "to_area": c.get("to_area") or "",
                        "in_label": c.get("in_label", "In"), "out_label": c.get("out_label", "Out"),
                        "in": st.n_in, "out": st.n_out, "net": st.n_in - st.n_out,
                        "in_window": w_in, "out_window": w_out,
                        "in_hr": round(w_in / self.window_s * 3600, 1),
                        "out_hr": round(w_out / self.window_s * 3600, 1),
                    })

            fps = 0.0
            if len(self.frame_times) > 1:
                span = self.frame_times[-1] - self.frame_times[0]
                fps = (len(self.frame_times) - 1) / span if span > 0 else 0.0

            return {
                "ts": wall, "fps": round(fps, 1), "since": self.started,
                "window_min": round(self.window_s / 60),
                "kpis": {
                    "in_view": self._k(people), "visitors": self.visitors,
                    "entries": tot_in, "exits": tot_out,
                    "avg_dwell_s": r1(sum(all_dw) / len(all_dw)) if all_dw else None,
                    "longest_wait_s": r1(longest_wait),
                    "active_alerts": len(self.alerts_active),
                },
                "tracking": {"active": len(self.tracks), "recovering": len(self.limbo), "stitched": self.stitches},
                "zones": zones_out, "counters": counters,
                "heat": {"w": self.gw, "h": self.gh,
                         "live": norm(self.heat_live), "total": norm(self.heat_total),
                         "live_peak_s": r1(float(self.heat_live.max())), "total_peak_s": r1(float(self.heat_total.max()))},
                "people_hist": list(self.people_hist),
                # copies: these dicts keep changing after the snapshot is handed to other threads
                "alerts": [dict(a) for a in self.alerts_active.values()],
                "alerts_log": [dict(a) for a in list(self.alerts_log)[:20]],
                "events": [dict(e) for e in list(self.events)[:50]],
            }

    def _alerts(self, z, count, cap, longest, alert_s) -> None:
        conds = {
            "capacity": (cap is not None and count > cap,
                         f"{z['name']}: over capacity" + ("" if self.k_min else f" ({count}/{cap})")),
            "wait": (alert_s is not None and longest is not None and longest > alert_s,
                     f"{z['name']}: wait above {fmt_s(alert_s or 0)}"),
        }
        now = time.time()
        for kind, (on, text) in conds.items():
            key = f"{z['id']}:{kind}"
            active = key in self.alerts_active
            if on:
                self.alert_off.pop(key, None)
                if active:
                    self.alerts_active[key]["text"] = text
                    continue
                first = self.alert_on.setdefault(key, now)
                if now - first >= self.ALERT_RAISE_S:
                    self.alert_on.pop(key, None)
                    self.alerts_active[key] = {"key": key, "zone": z["name"], "kind": kind,
                                               "text": text, "since": round(first, 1)}
                    self._event(text, "alert")
            else:
                self.alert_on.pop(key, None)
                if not active:
                    continue
                first = self.alert_off.setdefault(key, now)
                if now - first >= self.ALERT_CLEAR_S:
                    self.alert_off.pop(key, None)
                    a = self.alerts_active.pop(key)
                    a["ended"] = round(first, 1)
                    self.alerts_log.appendleft(a)
                    self._event(f"{a['text']} - cleared", "alert")

    # ------------------------------------------------------------ series
    def series_range(self, minutes: int) -> dict:
        with self.lock:
            cutoff = time.time() - minutes * 60
            buckets = [b for b in list(self.series) + ([self.bucket] if self.bucket else []) if b["t"] >= cutoff - 60]
            t = time.monotonic()
            hist = {}
            for z in self.zones:
                ds = [d for te, d in self.zstat[z["id"]].dwells if t - te <= minutes * 60]
                hist[z["id"]] = [sum(1 for d in ds if lo <= d < hi) for lo, hi, _ in DWELL_BINS]
            return {
                "minutes": minutes,
                "zones": [{"id": z["id"], "name": z["name"], "type": z.get("type")} for z in self.zones],
                "counters": [{"id": l["id"], "name": l["name"], "kind": "line"} for l in self.lines] +
                            [{"id": z["id"], "name": z["name"], "kind": "door"} for z in self.doors],
                "buckets": [{
                    "t": b["t"],
                    "people_avg": round(b["p_sum"] / b["n"], 2) if b["n"] else 0,
                    "people_max": b["p_max"],
                    "z": {k: {"occ_avg": round(v[0] / b["n"], 2) if b["n"] else 0, "occ_max": v[1],
                              "visits": v[2], "dwell_sum": round(v[3], 1), "dwell_n": v[4]}
                          for k, v in b["z"].items()},
                    "c": {k: {"in": v[0], "out": v[1]} for k, v in b.get("c", {}).items()},
                } for b in buckets],
                "dwell_bins": [lbl for _, _, lbl in DWELL_BINS],
                "dwell_hist": hist,
            }

    def minute_record(self, b: dict) -> dict:
        """A self-describing per-minute aggregate (names, roles, areas included) for the hub."""
        with self.lock:
            n = b["n"] or 1
            zmap = {z["id"]: z for z in self.zones}
            cmap = {**{l["id"]: ("line", l) for l in self.lines}, **{z["id"]: ("door", z) for z in self.doors}}
            return {
                "t": b["t"], "people_avg": round(b["p_sum"] / n, 2), "people_max": b["p_max"],
                "zones": [{"id": k, "name": zmap[k]["name"], "type": zmap[k].get("type", "area"),
                           "area": zmap[k].get("area") or "", "occ_avg": round(v[0] / n, 2), "occ_max": v[1],
                           "visits": v[2], "dwell_sum": round(v[3], 1), "dwell_n": v[4]}
                          for k, v in b["z"].items() if k in zmap],
                "counters": [{"id": k, "name": cmap[k][1]["name"], "kind": cmap[k][0], "role": counter_role(cmap[k][1]),
                              "from_area": cmap[k][1].get("from_area") or "", "to_area": cmap[k][1].get("to_area") or "",
                              "in": v[0], "out": v[1]}
                             for k, v in b.get("c", {}).items() if k in cmap],
            }

    def current_minute(self) -> Optional[dict]:
        with self.lock:
            return self.minute_record(self.bucket) if self.bucket else None

    def labels_for(self, dets: list[Detection]) -> dict:
        """Per-box dwell label for the local anonymised view."""
        with self.lock:
            t, out = time.monotonic(), {}
            for d in dets:
                cid = self.alias.get(d.track_id) if d.track_id is not None else None
                tr = self.tracks.get(cid) if cid is not None else None
                if tr and tr["zones"]:
                    enter = min(v[0] for v in tr["zones"].values())
                    out[d.track_id] = fmt_s(t - enter)
            return out

    # -------------------------------------------------------- persistence
    def dump_state(self) -> dict:
        """Aggregates only (no tracks, no positions) - safe to write to disk."""
        with self.lock:
            t = time.monotonic()
            return {
                "version": 3, "saved_at": time.time(), "started": self.started,
                "visitors": self.visitors,
                "series": list(self.series) + ([self.bucket] if self.bucket else []),
                "zones": {k: {"visits": s.visits, "peak": s.peak,
                              "dwells": [[round(t - te, 1), round(d, 1)] for te, d in s.dwells]}
                          for k, s in self.zstat.items()},
                "counters": {k: {"in": s.n_in, "out": s.n_out,
                                 "events": [[round(t - te, 1), d] for te, d in s.events]}
                             for k, s in self.cstat.items()},
                "heat_total": [round(float(v), 2) for v in self.heat_total.ravel()],
                "events": list(self.events)[:100],
                "open_minute_t": self.bucket["t"] if self.bucket else None,
                "alerts_log": list(self.alerts_log),
            }

    def load_state(self, s: dict) -> bool:
        with self.lock:
            if s.get("version") != 3 or time.time() - s.get("saved_at", 0) > RETENTION_S:
                return False
            t = time.monotonic()
            self.started = s.get("started", self.started)
            self.visitors = s.get("visitors", 0)
            cutoff = time.time() - RETENTION_S
            bs = [b for b in s.get("series", []) if b["t"] >= cutoff]
            cur_key = int(time.time() // 60) * 60
            self.bucket = bs.pop() if bs and bs[-1]["t"] == cur_key else None
            self.series = deque(bs, maxlen=24 * 60)
            # the minute that was open when the app stopped never reached the CSV/hub: send it now
            if bs and s.get("open_minute_t") == bs[-1]["t"] and self.on_minute:
                try:
                    self.on_minute(bs[-1], self._names())
                except Exception as e:
                    print("minute log error:", e)
            self.event_seq = max([e.get("id", 0) for e in s.get("events", [])] + [0])
            for k, v in s.get("zones", {}).items():
                if k in self.zstat:
                    st = self.zstat[k]
                    st.visits, st.peak = v.get("visits", 0), v.get("peak", 0)
                    st.dwells = deque((t - age, d) for age, d in v.get("dwells", []) if age < RETENTION_S)
            for k, v in s.get("counters", {}).items():
                if k in self.cstat:
                    st = self.cstat[k]
                    st.n_in, st.n_out = v.get("in", 0), v.get("out", 0)
                    st.events = deque((t - age, d) for age, d in v.get("events", []) if age < RETENTION_S)
            ht = s.get("heat_total")
            if ht and len(ht) == self.gw * self.gh:
                self.heat_total = np.array(ht, np.float32).reshape(self.gh, self.gw)
            self.events = deque(s.get("events", []), maxlen=200)
            self.alerts_log = deque(s.get("alerts_log", []), maxlen=50)
            return True


# --------------------------------------------------------------- helpers
def counter_role(c: dict) -> str:
    """perimeter = venue entrance (feeds Entries/Exits); internal = between areas; ignore = shown only."""
    r = c.get("role")
    if r in ("perimeter", "internal", "ignore"):
        return r
    return "perimeter" if c.get("site", True) else "internal"


def r1(v):
    return None if v is None else round(float(v), 1)


def norm(g: np.ndarray) -> list:
    peak = float(g.max())
    if peak <= 1e-6:
        return [0] * g.size
    return [round(float(v), 3) for v in (g / peak).ravel()]


def fmt_s(s: float) -> str:
    s = int(round(s))
    return f"{s}s" if s < 60 else f"{s // 60}m {s % 60:02d}s"
