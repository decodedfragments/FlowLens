"""
hub/hub.py - FlowLens Hub: one site view across many camera nodes.

    python hub/hub.py                         # http://localhost:9000, data in hub/data
    python hub/hub.py --port 9000 --pin 4821  # ask viewers for a PIN

Nodes push aggregates only (counts, waits, counters, a fading occupancy grid, zone
geometry, anonymous event text). The hub never receives video, images, track ids or
per-person coordinates, so it can run on the event LAN or in the cloud.

Event maths
  Entries / exits      = sum of per-minute counts from counters marked "site entrance"
  Inside now           = entries - exits since the event started (never below 0)
  Area occupancy       = live people in the zones tagged with that area
  Flows                = "between areas" counters: arrow direction from -> to
Overlapping cameras never add raw counts together; that is what counter roles are for.
"""
from __future__ import annotations

import argparse
import hashlib
import hmac
import asyncio
import base64
import csv
import io
import json
import logging
import math
import re
import secrets
import socket
import sqlite3
import threading
import time
from datetime import datetime
from pathlib import Path

import numpy as np
import uvicorn
from fastapi import Body, FastAPI, Request, WebSocket, WebSocketDisconnect
from fastapi.responses import FileResponse, JSONResponse, Response

ROOT = Path(__file__).parent
log = logging.getLogger("hub")
ONLINE_S, STALE_S = 8.0, 30.0
PLAN_COLS = 96


# --------------------------------------------------------------------- store
class Store:
    def __init__(self, data: Path):
        self.data = data
        data.mkdir(parents=True, exist_ok=True)
        self.cfg_file = data / "hub_config.json"
        self.lock = threading.RLock()
        self.db = sqlite3.connect(data / "hub.db", check_same_thread=False)
        self.db.execute("PRAGMA journal_mode=WAL")
        self.db.executescript("""
            CREATE TABLE IF NOT EXISTS minutes (node_id TEXT, t INTEGER, rec TEXT, PRIMARY KEY (node_id, t));
            CREATE TABLE IF NOT EXISTS events (node_id TEXT, t REAL, kind TEXT, text TEXT, UNIQUE (node_id, t, text));
            CREATE TABLE IF NOT EXISTS nodes (id TEXT PRIMARY KEY, name TEXT, site TEXT, first_seen REAL, last_seen REAL);
            CREATE INDEX IF NOT EXISTS ev_t ON events (t);
        """)
        self.cfg = self._load_cfg()

    def _load_cfg(self) -> dict:
        cfg = {"event_name": "My site", "token": secrets.token_urlsafe(18), "pin": "", "pin_disabled": False,
               "started_at": int(time.time() // 60) * 60, "floorplan": None, "nodes": {}}
        if self.cfg_file.exists():
            for f in (self.cfg_file, self.cfg_file.with_suffix(".tmp")):
                try:
                    cfg.update(json.loads(f.read_text(encoding="utf-8")))
                    break
                except Exception as e:
                    log.error("%s unreadable (%s)", f.name, e)
            else:
                # never silently start over: a new token would disconnect every camera and the
                # event totals would restart - make the operator decide
                bad = self.cfg_file.with_name(f"hub_config.broken-{int(time.time())}.json")
                self.cfg_file.replace(bad)
                raise SystemExit(f"hub_config.json was damaged and has been moved to {bad.name}. "
                                 "Restore a backup, or start again to create a fresh hub (new token).")
        if not cfg.get("pin") and not cfg.get("pin_disabled"):
            cfg["pin"] = f"{secrets.randbelow(10 ** 6):06d}"     # other devices need it; this computer doesn't
        self.save_cfg(cfg)
        return cfg

    def save_cfg(self, cfg: dict | None = None) -> None:
        with self.lock:
            cfg = cfg or self.cfg
            tmp = self.cfg_file.with_suffix(".tmp")
            tmp.write_text(json.dumps(cfg, indent=2), encoding="utf-8")
            tmp.replace(self.cfg_file)

    def exec(self, sql: str, args=()) -> list:
        with self.lock:
            cur = self.db.execute(sql, args)
            rows = cur.fetchall()
            self.db.commit()
            return rows

    def many(self, sql: str, rows: list) -> None:
        with self.lock:
            self.db.executemany(sql, rows)
            self.db.commit()


# ---------------------------------------------------------------------- hub
class Hub:
    def __init__(self, store: Store):
        self.s = store
        self.live: dict[str, dict] = {}          # node id -> {"p": payload, "recv": t}
        self.partial: dict[str, dict] = {}       # node id -> {minute t: record} (current, not yet stored)
        self.lock = threading.RLock()
        self._agg = None                         # cache of stored-minute aggregates
        self.state: dict = {}
        self._state_at = 0.0
        self._series_cache: dict = {}

    # ------------------------------------------------------------- ingest
    def ingest_live(self, p: dict) -> None:
        p = clean_live(p)
        node = p["node"]
        nid = node["id"]
        now = time.time()
        ref = p["ts"] or now                       # node clock: laptops at events are often not NTP-synced
        stale = []
        with self.lock:
            self.live[nid] = {"p": p, "recv": now}
            m = p.get("minute")
            if m:
                part = self.partial.setdefault(nid, {})
                part[m["t"]] = m
                for k in [k for k in part if k < ref - 900]:
                    stale.append((nid, k, part.pop(k)))
        for n, t, rec in stale:                    # a minute that never arrived as a record: keep the partial
            self._store(n, [(t, rec)], replace=False)
        self.s.exec("INSERT INTO nodes (id, name, site, first_seen, last_seen) VALUES (?,?,?,?,?) "
                    "ON CONFLICT(id) DO UPDATE SET name=excluded.name, site=excluded.site, last_seen=excluded.last_seen",
                    (nid, node["name"], node["site"], now, now))
        evs = [(nid, e["t"], e["kind"], e["text"]) for e in p["events"]]
        if evs:
            self.s.many("INSERT OR IGNORE INTO events VALUES (?,?,?,?)", evs)

    def ingest_minutes(self, node: dict, recs: list) -> int:
        nid = clean_id(node["id"])
        rows = [(r["t"], r) for r in (clean_minute(x) for x in recs) if r]
        self._store(nid, rows, replace=True)
        return len(rows)

    def _store(self, nid: str, rows: list, replace: bool) -> None:
        """Write minute records and keep the event totals cache in step (no full rescans)."""
        with self.lock:
            agg = self._agg
            for t, rec in rows:
                old = self.s.exec("SELECT rec FROM minutes WHERE node_id=? AND t=?", (nid, t))
                if old and not replace:
                    continue
                self.s.exec("INSERT OR REPLACE INTO minutes VALUES (?,?,?)", (nid, t, json.dumps(rec)))
                if agg is not None and t >= agg["start"]:
                    if old:
                        self._agg_add(agg, nid, json.loads(old[0][0]), -1)
                    self._agg_add(agg, nid, rec, +1)
                    agg["keys"].add((nid, t))
                self.partial.get(nid, {}).pop(t, None)
            self._series_cache.clear()

    @staticmethod
    def _agg_add(agg: dict, nid: str, rec: dict, sign: int) -> None:
        for c in rec.get("counters", []):
            a = agg["stored"].setdefault((nid, c["id"]), {**c, "in": 0, "out": 0})
            if sign > 0:
                a.update({x: c.get(x, a.get(x)) for x in ("name", "role", "from_area", "to_area", "kind")})
            a["in"] += sign * c.get("in", 0)
            a["out"] += sign * c.get("out", 0)

    # ---------------------------------------------------------- minutes
    def minute_records(self, since: int) -> dict:
        """{(node, t): record} from storage plus live partial minutes not yet stored."""
        out = {}
        for nid, t, rec in self.s.exec("SELECT node_id, t, rec FROM minutes WHERE t >= ?", (since,)):
            out[(nid, t)] = json.loads(rec)
        with self.lock:
            for nid, part in self.partial.items():
                for t, rec in part.items():
                    if t >= since and (nid, t) not in out:
                        out[(nid, t)] = rec
        return out

    def totals(self) -> dict:
        """Event totals per counter (node, counter id) since the event started."""
        start = self.s.cfg["started_at"]
        with self.lock:
            if self._agg is None or self._agg["start"] != start:
                stored = {}
                for nid, t, rec in self.s.exec("SELECT node_id, t, rec FROM minutes WHERE t >= ?", (start,)):
                    for c in json.loads(rec).get("counters", []):
                        k = (nid, c["id"])
                        a = stored.setdefault(k, {**c, "in": 0, "out": 0})
                        a.update({x: c.get(x, a.get(x)) for x in ("name", "role", "from_area", "to_area", "kind")})
                        a["in"] += c.get("in", 0); a["out"] += c.get("out", 0)
                keys = {(nid, t) for nid, t in self.s.exec("SELECT node_id, t FROM minutes WHERE t >= ?", (start,))}
                self._agg = {"start": start, "stored": stored, "keys": keys}
            tot = {k: dict(v) for k, v in self._agg["stored"].items()}
            for nid, part in self.partial.items():
                for t, rec in part.items():
                    if t < start or (nid, t) in self._agg["keys"]:
                        continue
                    for c in rec.get("counters", []):
                        a = tot.setdefault((nid, c["id"]), {**c, "in": 0, "out": 0})
                        a["in"] += c.get("in", 0); a["out"] += c.get("out", 0)
        return tot

    # ------------------------------------------------------------- state
    def node_rows(self) -> list:
        return self.s.exec("SELECT id, name, site, first_seen, last_seen FROM nodes ORDER BY name")

    def build_state(self) -> dict:
        with self.lock:
            if time.time() - self._state_at < 0.8 and self.state:
                return self.state
        st = self._build_state()
        with self.lock:
            self.state, self._state_at = st, time.time()
        return st

    def _build_state(self) -> dict:
        now = time.time()
        cfg = self.s.cfg
        with self.lock:
            live = {k: dict(v) for k, v in self.live.items()}
        names = {}
        cameras, alerts = [], []
        for nid, name, site, first, last in self.node_rows():
            names[nid] = name
            lv = live.get(nid)
            age = now - (lv["recv"] if lv else last)
            status = "online" if age < ONLINE_S else "stale" if age < STALE_S else "offline"
            p = lv["p"] if lv else {}
            cam = p.get("camera") or {}
            k = p.get("kpis") or {}
            cameras.append({"id": nid, "name": name, "site": site, "status": status, "age_s": round(age, 1),
                            "camera": cam.get("status"), "device": cam.get("device"), "fps": p.get("fps"),
                            "cam_fps": cam.get("cam_fps"), "in_view": k.get("in_view") if status != "offline" else None,
                            "mapped": bool((p.get("floor") or {}).get("H")),
                            "pos": (cfg["nodes"].get(nid) or {}).get("pos")})
            if status == "offline" and now - last < 6 * 3600:
                alerts.append({"level": "critical", "text": f"{name}: offline for {fmt_s(age)}", "since": last})
            elif status == "stale":
                alerts.append({"level": "warning", "text": f"{name}: no data for {fmt_s(age)}", "since": last})
            if status == "online" and (cam.get("status") == "connecting" or cam.get("detector") not in (None, "ok")):
                alerts.append({"level": "warning", "text": f"{name}: starting up", "since": now})
            elif status == "online" and cam.get("status") not in (None, "ok"):
                alerts.append({"level": "critical", "text": f"{name}: camera problem ({cam.get('status')})", "since": now})
            if status == "online" and p.get("fps") is not None and cam.get("detector") == "ok" and p["fps"] < 4:
                alerts.append({"level": "warning", "text": f"{name}: detection slow ({p['fps']} fps)", "since": now})
            for a in ((p.get("alerts") or []) if status != "offline" else []):
                alerts.append({"level": "critical", "text": f"{name} · {a.get('text')}", "since": a.get("since", now)})

        online = [c for c in cameras if c["status"] != "offline"]
        tot = self.totals()
        entries = sum(v["in"] for v in tot.values() if v.get("role", "perimeter") == "perimeter")
        exits = sum(v["out"] for v in tot.values() if v.get("role", "perimeter") == "perimeter")

        # live zones -> areas and queues
        areas: dict[str, dict] = {}
        queues = []
        in_view = 0
        for c in online:
            p = live.get(c["id"], {}).get("p") or {}
            in_view += (p.get("kpis") or {}).get("in_view") or 0
            for z in p.get("zones") or []:
                if z.get("type") == "door":
                    continue
                an = z.get("area") or c["name"]
                a = areas.setdefault(an, {"name": an, "now": 0, "capacity": 0, "zones": 0, "flow_in": 0, "flow_out": 0, "cameras": set()})
                a["now"] += z.get("count") or 0
                a["capacity"] += z.get("capacity") or 0
                a["zones"] += 1
                a["cameras"].add(c["name"])
                if z.get("type") == "queue":
                    queues.append({"camera": c["name"], "name": z["name"], "area": an, "count": z.get("count"),
                                   "longest_s": z.get("longest_current_s"), "avg_s": z.get("avg_dwell_s"),
                                   "p90_s": z.get("p90_dwell_s"), "visits_hr": z.get("throughput_hr"),
                                   "status": z.get("status", "ok"), "alert_s": z.get("wait_alert_s")})
        # flows and entrances from event totals
        flows: dict[tuple, int] = {}
        entrances = []
        for (nid, cid), v in tot.items():
            role = v.get("role", "perimeter")
            cam_name = names.get(nid, nid)
            if role == "perimeter":
                to = v.get("to_area") or "Site"
                flows[("Outside", to)] = flows.get(("Outside", to), 0) + v["in"]
                flows[(to, "Outside")] = flows.get((to, "Outside"), 0) + v["out"]
                entrances.append({"camera": cam_name, "name": v.get("name"), "kind": v.get("kind"), "in": v["in"], "out": v["out"], "to_area": to})
                a = areas.setdefault(to, {"name": to, "now": 0, "capacity": 0, "zones": 0, "flow_in": 0, "flow_out": 0, "cameras": set()}) if to != "Site" else None
                if a is not None:
                    a["flow_in"] += v["in"]; a["flow_out"] += v["out"]
            elif role == "internal":
                fa, ta = v.get("from_area") or "?", v.get("to_area") or "?"
                flows[(fa, ta)] = flows.get((fa, ta), 0) + v["in"]
                flows[(ta, fa)] = flows.get((ta, fa), 0) + v["out"]
                for an, fin, fout in ((ta, v["in"], v["out"]), (fa, v["out"], v["in"])):
                    if an == "?":
                        continue
                    a = areas.setdefault(an, {"name": an, "now": 0, "capacity": 0, "zones": 0, "flow_in": 0, "flow_out": 0, "cameras": set()})
                    a["flow_in"] += fin; a["flow_out"] += fout
        for a in areas.values():
            a["cameras"] = sorted(a["cameras"])
            a["utilisation"] = round(a["now"] / a["capacity"], 2) if a["capacity"] else None
        queues.sort(key=lambda q: (-(q["longest_s"] or 0)))
        longest = max([q["longest_s"] or 0 for q in queues] or [0]) or None
        waits = [(q["avg_s"], 1) for q in queues if q["avg_s"] is not None]

        events = [{"t": t, "kind": k, "text": f"{names.get(n, n)} · {tx}"} for n, t, k, tx in
                  self.s.exec("SELECT node_id, t, kind, text FROM events ORDER BY t DESC LIMIT 80")]
        alerts.sort(key=lambda a: (a["level"] != "critical", -a["since"]))

        state = {
            "ts": now, "event": {"name": cfg["event_name"], "started_at": cfg["started_at"]},
            "kpis": {"inside": max(0, entries - exits), "entries": entries, "exits": exits, "in_view": in_view,
                     "longest_wait_s": longest, "avg_wait_s": round(sum(w for w, _ in waits) / len(waits), 1) if waits else None,
                     "alerts": len(alerts), "cameras_online": sum(1 for c in cameras if c["status"] == "online"),
                     "cameras": len(cameras)},
            "cameras": cameras, "alerts": alerts[:40], "areas": sorted(areas.values(), key=lambda a: a["name"]),
            "queues": queues, "flows": [{"from": a, "to": b, "count": n} for (a, b), n in sorted(flows.items(), key=lambda x: -x[1]) if n],
            "entrances": sorted(entrances, key=lambda e: -(e["in"] + e["out"])), "events": events,
            "floor": self.floor_state(live, cameras),
        }
        self.state = state
        return state

    # -------------------------------------------------------------- floor
    def floor_state(self, live: dict, cameras: list) -> dict | None:
        fp = self.s.cfg.get("floorplan")
        if not fp:
            return None
        pw = PLAN_COLS
        ph = max(8, round(PLAN_COLS * fp["h"] / fp["w"]))
        grid = np.zeros((ph, pw), np.float32)
        shapes, coverage = [], []
        for c in cameras:
            if c["status"] == "offline":
                continue
            p = live.get(c["id"], {}).get("p") or {}
            H = (p.get("floor") or {}).get("H")
            if not H:
                continue
            H = np.array(H, np.float64)
            heat = p.get("heat") or {}
            if heat.get("live"):
                hw, hh = heat["w"], heat["h"]
                try:
                    v = np.array(heat["live"], np.float32).reshape(hh, hw) * float(heat.get("live_peak_s") or 1)
                except (ValueError, TypeError):
                    v = np.zeros((1, 1), np.float32)
                ys, xs = np.nonzero(v > 0.01)
                if len(xs):
                    pts = warp(H, np.stack([(xs + .5) / hw, (ys + .5) / hh], 1))
                    ok = (pts[:, 0] >= 0) & (pts[:, 0] < 1) & (pts[:, 1] >= 0) & (pts[:, 1] < 1)
                    gx = (pts[ok, 0] * pw).astype(int); gy = (pts[ok, 1] * ph).astype(int)
                    np.add.at(grid, (gy, gx), v[ys[ok], xs[ok]])
            foot = warp(H, np.array([[0, .45], [1, .45], [1, 1], [0, 1]], np.float64))
            if np.isfinite(foot).all():
                coverage.append({"camera": c["name"], "id": c["id"], "points": foot.round(4).tolist()})
            for z in p.get("zones") or []:
                if z.get("points"):
                    shapes.append({"camera": c["name"], "kind": "zone", "type": z.get("type"), "name": z["name"],
                                   "count": z.get("count"), "status": z.get("status"), "points": warp(H, np.array(z["points"])).round(4).tolist()})
            for k in p.get("counters") or []:
                if k.get("kind") == "line" and k.get("points"):
                    shapes.append({"camera": c["name"], "kind": "line", "name": k["name"], "in": k.get("in"), "out": k.get("out"),
                                   "role": k.get("role"), "points": warp(H, np.array(k["points"])).round(4).tolist()})
        peak = float(grid.max())
        return {"w": fp["w"], "h": fp["h"], "version": fp.get("version", 0),
                "heat": {"w": pw, "h": ph, "data": [round(float(x), 3) for x in (grid / peak if peak > 0 else grid).ravel()]},
                "shapes": [s for s in shapes if np.isfinite(np.array(s["points"])).all()], "coverage": coverage}

    # ------------------------------------------------------------- series
    def series(self, minutes: int) -> dict:
        key = (minutes, self.s.cfg["started_at"], int(time.time() // 10))
        with self.lock:
            if key in self._series_cache:
                return self._series_cache[key]
        out = self._series(minutes)
        with self.lock:
            self._series_cache = {key: out}
        return out

    def _series(self, minutes: int) -> dict:
        start = self.s.cfg["started_at"]
        since = max(start, int((time.time() - minutes * 60) // 60) * 60)
        names = {r[0]: r[1] for r in self.node_rows()}
        recs = self.minute_records(start)
        by_t: dict[int, dict] = {}
        cum_in = cum_out = 0
        for (nid, t), r in sorted(recs.items(), key=lambda x: x[0][1]):
            b = by_t.setdefault(t, {"t": t, "entries": 0, "exits": 0, "queue_people": 0.0, "seen": 0.0, "areas": {}, "cams": {}})
            for c in r.get("counters", []):
                if c.get("role", "perimeter") == "perimeter":
                    b["entries"] += c["in"]; b["exits"] += c["out"]
            for z in r.get("zones", []):
                if z.get("type") == "door":
                    continue
                if z.get("type") == "queue":
                    b["queue_people"] += z.get("occ_avg", 0)
                an = z.get("area") or names.get(nid, nid)
                b["areas"][an] = round(b["areas"].get(an, 0) + z.get("occ_avg", 0), 2)
            b["seen"] += r.get("people_avg", 0)
            b["cams"][names.get(nid, nid)] = r.get("people_avg", 0)
        out = []
        for t in sorted(by_t):
            b = by_t[t]
            cum_in += b["entries"]; cum_out += b["exits"]
            b["inside"] = max(0, cum_in - cum_out)
            b["queue_people"] = round(b["queue_people"], 2); b["seen"] = round(b["seen"], 2)
            if t >= since:
                out.append(b)
        return {"minutes": minutes, "started_at": start, "buckets": out}



# ------------------------------------------------------------ input hygiene
# Camera payloads are data from the network: coerce every field to the type the
# dashboard expects, so a bad or malicious node can neither break the hub nor
# inject markup into the operator's browser.
ID_RE = re.compile(r"[^A-Za-z0-9_-]")
STATUSES, ZTYPES, KINDS, ROLES = {"ok", "warning", "critical"}, {"queue", "service", "area", "door"}, {"line", "door"}, {"perimeter", "internal", "ignore"}


def clean_id(v) -> str:
    v = ID_RE.sub("", str(v))[:40]
    if not v:
        raise ValueError("bad id")
    return v


def _s(v, n=60) -> str:
    return str(v if v is not None else "")[:n]


def _n(v, lo=0.0, hi=1e9):
    try:
        f = float(v)
        return None if not math.isfinite(f) else max(lo, min(hi, f))
    except (TypeError, ValueError):
        return None


def _i(v) -> int:
    f = _n(v, 0, 1e9)
    return int(f) if f is not None else 0


def _pts(v):
    try:
        out = [[float(x), float(y)] for x, y in v][:64]
        return out if out and all(map(math.isfinite, sum(out, []))) else None
    except (TypeError, ValueError):
        return None


def _one(v, allowed, default):
    return v if v in allowed else default


def clean_counter(c: dict) -> dict:
    return {"id": clean_id(c.get("id")), "name": _s(c.get("name")), "kind": _one(c.get("kind"), KINDS, "line"),
            "role": _one(c.get("role"), ROLES, "perimeter"), "from_area": _s(c.get("from_area"), 40),
            "to_area": _s(c.get("to_area"), 40), "in": _i(c.get("in")), "out": _i(c.get("out"))}


def clean_minute(r: dict):
    try:
        return {"t": int(r["t"]) // 60 * 60, "people_avg": _n(r.get("people_avg")) or 0, "people_max": _i(r.get("people_max")),
                "zones": [{"id": clean_id(z.get("id")), "name": _s(z.get("name")), "type": _one(z.get("type"), ZTYPES, "area"),
                           "area": _s(z.get("area"), 40), "occ_avg": _n(z.get("occ_avg")) or 0, "occ_max": _i(z.get("occ_max")),
                           "visits": _i(z.get("visits")), "dwell_sum": _n(z.get("dwell_sum")) or 0, "dwell_n": _i(z.get("dwell_n"))}
                          for z in (r.get("zones") or [])[:64]],
                "counters": [clean_counter(c) for c in (r.get("counters") or [])[:64]]}
    except (KeyError, TypeError, ValueError):
        return None


def clean_live(p: dict) -> dict:
    node = p.get("node") or {}
    k = p.get("kpis") or {}
    cam = p.get("camera") or {}
    heat = p.get("heat") or {}
    hw, hh = _i(heat.get("w")), _i(heat.get("h"))
    live = heat.get("live")
    if not (isinstance(live, list) and 0 < hw <= 256 and 0 < hh <= 256 and len(live) == hw * hh):
        live = None
    H = (p.get("floor") or {}).get("H")
    try:
        H = [[float(x) for x in row] for row in H][:3] if H else None
        if H and (len(H) != 3 or any(len(r) != 3 for r in H)):
            H = None
    except (TypeError, ValueError):
        H = None
    return {
        "node": {"id": clean_id(node.get("id")), "name": _s(node.get("name")) or "Camera", "site": _s(node.get("site"))},
        "ts": _n(p.get("ts")), "fps": _n(p.get("fps"), 0, 1000),
        "camera": {"status": _s(cam.get("status"), 120), "w": _i(cam.get("w")), "h": _i(cam.get("h")),
                   "cam_fps": _n(cam.get("cam_fps"), 0, 1000), "device": _s(cam.get("device")), "detector": _s(cam.get("detector"), 160)},
        "kpis": {"in_view": None if k.get("in_view") is None else _i(k.get("in_view"))},
        "zones": [{"id": clean_id(z.get("id")), "name": _s(z.get("name")), "type": _one(z.get("type"), ZTYPES, "area"),
                   "area": _s(z.get("area"), 40), "count": None if z.get("count") is None else _i(z.get("count")),
                   "capacity": _i(z.get("capacity")) or None, "status": _one(z.get("status"), STATUSES, "ok"),
                   "longest_current_s": _n(z.get("longest_current_s")), "avg_dwell_s": _n(z.get("avg_dwell_s")),
                   "p90_dwell_s": _n(z.get("p90_dwell_s")), "throughput_hr": _n(z.get("throughput_hr")),
                   "wait_alert_s": _n(z.get("wait_alert_s")), "points": _pts(z.get("points"))}
                  for z in (p.get("zones") or [])[:64]],
        "counters": [{**clean_counter(c), "points": _pts(c.get("points"))} for c in (p.get("counters") or [])[:64]],
        "heat": {"w": hw, "h": hh, "live": live, "live_peak_s": _n(heat.get("live_peak_s"))},
        "minute": clean_minute(p["minute"]) if p.get("minute") else None,
        "floor": {"H": H},
        "alerts": [{"text": _s(a.get("text"), 200), "since": _n(a.get("since"))} for a in (p.get("alerts") or [])[:20]],
        "events": [{"t": _n(e.get("t")) or 0, "kind": _s(e.get("kind"), 20), "text": _s(e.get("text"), 200)} for e in (p.get("events") or [])[:200]],
    }


def csv_safe(v):
    return "'" + v if isinstance(v, str) and v[:1] in ("=", "+", "-", "@") else v


def warp(H: np.ndarray, pts: np.ndarray) -> np.ndarray:
    p = np.c_[pts, np.ones(len(pts))] @ H.T
    with np.errstate(divide="ignore", invalid="ignore"):
        return p[:, :2] / p[:, 2:3]


def fmt_s(s: float) -> str:
    s = int(s)
    return f"{s}s" if s < 60 else f"{s // 60}m" if s < 3600 else f"{s // 3600}h {s % 3600 // 60}m"


def lan_ip() -> str:
    try:
        s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        s.connect(("10.255.255.255", 1))
        ip = s.getsockname()[0]
        s.close()
        return ip
    except OSError:
        return "127.0.0.1"


# ---------------------------------------------------------------------- app
def build_app(hub: Hub, store: Store) -> FastAPI:
    app = FastAPI(title="FlowLens Hub")

    def token_ok(req) -> bool:
        h = req.headers.get("authorization", "")
        return h.startswith("Bearer ") and secrets.compare_digest(h[7:].strip(), store.cfg["token"])

    def is_local(req) -> bool:
        return (req.client.host if req.client else "") in ("127.0.0.1", "::1", "localhost")

    def pin_cookie() -> str:
        return hmac.new(store.cfg["token"].encode(), ("pin:" + store.cfg.get("pin", "")).encode(), hashlib.sha256).hexdigest()[:32]

    def pin_ok(req) -> tuple[bool, bool]:
        """(allowed, set_cookie). The hub computer itself never needs the PIN."""
        pin = store.cfg.get("pin")
        if not pin or is_local(req):
            return True, False
        if secrets.compare_digest(req.cookies.get("flowlens_hub", ""), pin_cookie()):
            return True, False
        h = req.headers.get("authorization", "")
        if h.startswith("Basic "):
            try:
                return secrets.compare_digest(base64.b64decode(h[6:]).decode().split(":", 1)[1], pin), True
            except Exception:
                return False, False
        return False, False

    def host_ok(req) -> bool:
        """DNS-rebinding guard: only IP addresses, localhost and this machine's own name."""
        host = (req.headers.get("host") or "").rsplit(":", 1)[0].strip("[]").lower()
        names = {"localhost", socket.gethostname().lower(), socket.gethostname().lower() + ".local"}
        return host in names or bool(re.match(r"^[0-9.]+$|^[0-9a-f:]+$", host))

    @app.middleware("http")
    async def guard(request: Request, call_next):
        path = request.url.path
        if not host_ok(request):
            return JSONResponse({"error": "open the hub by its IP address or localhost"}, status_code=403)
        if path.startswith("/api/ingest/"):
            if request.method != "POST" or not token_ok(request):
                return JSONResponse({"error": "invalid token"}, status_code=401)
            return await call_next(request)
        if path == "/api/floorplan" and request.method == "GET" and token_ok(request):
            return await call_next(request)
        ok, set_cookie = pin_ok(request)
        if not ok:
            return Response("PIN required - use any user name and the hub PIN", status_code=401,
                            headers={"WWW-Authenticate": 'Basic realm="FlowLens Hub"'})
        if request.method in ("POST", "PUT", "DELETE") and request.headers.get("x-flowlens") != "1":
            return JSONResponse({"error": "missing X-FlowLens header"}, status_code=403)   # CSRF guard
        resp = await call_next(request)
        if set_cookie:
            resp.set_cookie("flowlens_hub", pin_cookie(), httponly=True, samesite="strict", max_age=7 * 86400)
        return resp

    @app.get("/")
    def index():
        return FileResponse(ROOT / "static" / "index.html", headers={"Cache-Control": "no-store"})

    @app.post("/api/ingest/live")
    def ingest_live(body: dict = Body(...)):
        try:
            hub.ingest_live(body)
        except (KeyError, TypeError, ValueError) as e:
            return JSONResponse({"error": f"bad payload: {e}"}, status_code=400)
        return {"ok": True}

    @app.post("/api/ingest/minutes")
    def ingest_minutes(body: dict = Body(...)):
        try:
            n = hub.ingest_minutes(body["node"], body.get("minutes") or [])
        except (KeyError, TypeError, ValueError) as e:
            return JSONResponse({"error": f"bad payload: {e}"}, status_code=400)
        return {"ok": True, "stored": n}

    @app.get("/api/state")
    def state():
        return hub.build_state()

    @app.get("/api/series")
    def series(minutes: int = 60):
        return hub.series(max(1, min(7 * 1440, minutes)))

    @app.get("/api/settings")
    def get_settings(request: Request):
        c = store.cfg
        host = request.headers.get("host", "").split(":")[0]
        port = request.url.port or 9000
        local = is_local(request)
        return {"event_name": c["event_name"], "local": local,
                "token": c["token"] if local else None,        # secrets only on the hub computer itself
                "pin": c.get("pin") if local else None, "pin_set": bool(c.get("pin")),
                "started_at": c["started_at"], "floorplan": c.get("floorplan"),
                "hub_url": f"http://{lan_ip() if host in ('localhost', '127.0.0.1') else host}:{port}"}

    @app.put("/api/settings")
    def put_settings(request: Request, body: dict = Body(...)):
        c = store.cfg
        if "event_name" in body:
            c["event_name"] = str(body["event_name"] or "My site").strip()[:80]
        if ("new_token" in body or "pin" in body) and not is_local(request):
            return JSONResponse({"error": "change the token or PIN on the hub computer itself"}, status_code=403)
        if body.get("new_token"):
            c["token"] = secrets.token_urlsafe(18)
        if "pin" in body:
            c["pin"] = re.sub(r"\s", "", str(body["pin"] or ""))[:32]
            c["pin_disabled"] = not c["pin"]
        store.save_cfg()
        return {"ok": True}

    @app.post("/api/event/reset")
    def reset_event():
        store.cfg["started_at"] = int(time.time() // 60) * 60
        store.save_cfg()
        with hub.lock:
            hub._agg = None
        return {"ok": True, "started_at": store.cfg["started_at"]}

    @app.get("/api/floorplan")
    def floorplan():
        fp = store.cfg.get("floorplan")
        if not fp or not (store.data / fp["file"]).exists():
            return JSONResponse({"error": "no floor plan"}, status_code=404)
        return FileResponse(store.data / fp["file"], media_type=fp["mime"],
                            headers={"Cache-Control": "no-store", "Content-Security-Policy": "sandbox", "X-Content-Type-Options": "nosniff"})

    @app.put("/api/floorplan")
    def put_floorplan(body: dict = Body(...)):
        m = re.match(r"^data:(image/(png|jpeg|webp));base64,(.+)$", body.get("data_url", ""), re.S)
        if not m:   # no SVG: it can carry script and is served from this origin
            return JSONResponse({"error": "send a PNG, JPG or WebP image (export SVG/PDF plans as PNG)"}, status_code=400)
        raw = base64.b64decode(m.group(3))
        if len(raw) > 12_000_000:
            return JSONResponse({"error": "image is larger than 12 MB"}, status_code=400)
        w, h = int(body.get("w") or 0), int(body.get("h") or 0)
        if w <= 0 or h <= 0:
            return JSONResponse({"error": "image size missing"}, status_code=400)
        ext = {"png": "png", "jpeg": "jpg", "webp": "webp"}[m.group(2)]
        old = store.cfg.get("floorplan")
        name = f"floorplan.{ext}"
        (store.data / name).write_bytes(raw)
        if old and old["file"] != name and (store.data / old["file"]).exists():
            (store.data / old["file"]).unlink()
        store.cfg["floorplan"] = {"file": name, "mime": m.group(1), "w": w, "h": h, "version": int(time.time())}
        store.save_cfg()
        return store.cfg["floorplan"]

    @app.delete("/api/floorplan")
    def delete_floorplan():
        old = store.cfg.get("floorplan")
        if old and (store.data / old["file"]).exists():
            (store.data / old["file"]).unlink()
        store.cfg["floorplan"] = None
        store.save_cfg()
        return {"ok": True}

    @app.put("/api/nodes/{nid}")
    def put_node(nid: str, body: dict = Body(...)):
        n = store.cfg["nodes"].setdefault(nid[:40], {})
        if "pos" in body:
            n["pos"] = [round(min(1, max(0, float(v))), 4) for v in body["pos"]][:2] if body["pos"] else None
        store.save_cfg()
        return n

    @app.delete("/api/nodes/{nid}")
    def forget_node(nid: str):
        """Remove a camera that is no longer used (its history stays in exports)."""
        store.exec("DELETE FROM nodes WHERE id = ?", (nid,))
        store.cfg["nodes"].pop(nid, None)
        store.save_cfg()
        with hub.lock:
            hub.live.pop(nid, None)
        return {"ok": True}

    @app.get("/api/export.csv")
    def export(scope: str = "event"):
        names = {r[0]: r[1] for r in hub.node_rows()}
        buf = io.StringIO()
        raw = csv.writer(buf)

        class w:  # formula-safe rows (camera and zone names come from the network)
            writerow = staticmethod(lambda row: raw.writerow([csv_safe(c) for c in row]))
        if scope == "cameras":
            w.writerow(["minute", "camera", "kind", "name", "area/role", "occupancy_avg", "occupancy_max", "visits", "avg_dwell_s", "in", "out"])
            for (nid, t), r in sorted(hub.minute_records(store.cfg["started_at"]).items(), key=lambda x: (x[0][1], x[0][0])):
                m = datetime.fromtimestamp(t).strftime("%Y-%m-%d %H:%M")
                cam = names.get(nid, nid)
                w.writerow([m, cam, "scene", "All people", "", r.get("people_avg"), r.get("people_max"), "", "", "", ""])
                for z in r.get("zones", []):
                    w.writerow([m, cam, "zone", z["name"], z.get("area", ""), z["occ_avg"], z["occ_max"], z["visits"],
                                round(z["dwell_sum"] / z["dwell_n"], 1) if z.get("dwell_n") else "", "", ""])
                for c in r.get("counters", []):
                    w.writerow([m, cam, c.get("kind"), c["name"], c.get("role"), "", "", "", "", c["in"], c["out"]])
        else:
            s = hub.series(7 * 1440)
            areas = sorted({a for b in s["buckets"] for a in b["areas"]})
            w.writerow(["minute", "entries", "exits", "inside", "people_seen_by_cameras", "people_in_queues"] + [f"{a} occupancy_avg" for a in areas])
            for b in s["buckets"]:
                w.writerow([datetime.fromtimestamp(b["t"]).strftime("%Y-%m-%d %H:%M"), b["entries"], b["exits"], b["inside"],
                            b["seen"], b["queue_people"]] + [b["areas"].get(a, "") for a in areas])
        fn = f"flowlens-hub_{scope}_{datetime.now():%Y%m%d_%H%M}.csv"
        return Response(buf.getvalue(), media_type="text/csv", headers={"Content-Disposition": f'attachment; filename="{fn}"'})

    @app.websocket("/ws")
    async def ws(sock: WebSocket):
        # the HTTP middleware does not see websockets: check host + PIN here too
        local = (sock.client.host if sock.client else "") in ("127.0.0.1", "::1")
        pin = store.cfg.get("pin")
        if not host_ok(sock) or (pin and not local and not secrets.compare_digest(sock.cookies.get("flowlens_hub", ""), pin_cookie())):
            await sock.close(code=4401)
            return
        await sock.accept()
        try:
            while True:
                st = await asyncio.to_thread(hub.build_state)
                await sock.send_json(st)
                await asyncio.sleep(1.0)
        except (WebSocketDisconnect, RuntimeError):
            pass

    return app


def main():
    ap = argparse.ArgumentParser(description="FlowLens Hub - one site view across many cameras")
    ap.add_argument("--host", default="0.0.0.0", help="0.0.0.0 lets cameras on the network reach the hub")
    ap.add_argument("--port", type=int, default=9000)
    ap.add_argument("--data", default=str(ROOT / "data"))
    ap.add_argument("--pin", default=None, help="require this PIN to open the dashboard")
    ap.add_argument("--event", default=None, help="event / site name")
    a = ap.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s  %(message)s", datefmt="%H:%M:%S")
    store = Store(Path(a.data))
    if a.pin is not None:
        store.cfg["pin"] = a.pin
    if a.event:
        store.cfg["event_name"] = a.event
    store.save_cfg()
    # keep 30 days of minutes and events
    store.exec("DELETE FROM minutes WHERE t < ?", (int(time.time()) - 30 * 86400,))
    store.exec("DELETE FROM events WHERE t < ?", (time.time() - 30 * 86400,))
    hub = Hub(store)
    url = f"http://{lan_ip()}:{a.port}"
    log.info("FlowLens Hub: open http://localhost:%d  (cameras connect to %s)", a.port, url)
    log.info("Access token for cameras: %s", store.cfg["token"])
    if store.cfg.get("pin"):
        log.info("PIN for viewing on other devices: %s  (not needed on this computer)", store.cfg["pin"])
    uvicorn.run(build_app(hub, store), host=a.host, port=a.port, log_level="warning")


if __name__ == "__main__":
    main()
