"""
node_link.py - uplink from a FlowLens camera node to a FlowLens Hub.

What leaves the node: aggregates only (counts, waits, counters, a fading 64x36
occupancy grid, zone/line geometry, anonymous event text). Never frames, ids or
per-person coordinates.

Two channels, both plain HTTPS/HTTP POST with a bearer token (works through
proxies and firewalls, no extra dependencies):
  live    every 2 s  -> /api/ingest/live      (latest state; dropped if the hub is down)
  minutes  each min  -> /api/ingest/minutes   (store-and-forward: queued on disk in
                                               logs/outbox.jsonl until the hub confirms)
"""
from __future__ import annotations

import json
import logging
import threading
import time
import urllib.error
import urllib.request
from pathlib import Path

log = logging.getLogger("qa")
VERSION = "2.0"
MAX_OUTBOX = 10080          # one week of minutes
# never route hub traffic (and its token) through a Windows/IE system proxy
_OPENER = urllib.request.build_opener(urllib.request.ProxyHandler({}))


class HubLink:
    LIVE_EVERY_S = 2.0

    def __init__(self, runtime, outbox: Path):
        self.rt = runtime
        self.outbox = outbox
        self.lock = threading.Lock()
        self.status = "off"
        self.last_ok = None
        self.last_error = None
        self.acked_event = runtime.engine.event_seq   # events up to here are on the hub already
        self._queued = self._count_lines()
        self._stop = False
        threading.Thread(target=self._run, daemon=True, name="hub-link").start()

    # ---------------------------------------------------------------- state
    @property
    def cfg(self) -> dict:
        return self.rt.cfg.get("hub", {})

    def enabled(self) -> bool:
        return bool(self.cfg.get("enabled") and self.cfg.get("url"))

    def _count_lines(self) -> int:
        try:
            with open(self.outbox, encoding="utf-8") as fh:
                return sum(1 for l in fh if l.strip())
        except OSError:
            return 0

    def queued(self) -> int:
        return self._queued

    def info(self) -> dict:
        return {"enabled": self.enabled(), "url": self.cfg.get("url", ""), "status": self.status,
                "last_ok": self.last_ok, "last_error": self.last_error, "queued_minutes": self.queued()}

    # --------------------------------------------------------------- minutes
    def queue_minute(self, record: dict) -> None:
        """Called at each minute close. Kept on disk until the hub has it (max ~7 days)."""
        if not self.enabled():
            return
        try:
            with self.lock:
                if self._queued >= MAX_OUTBOX:      # hub unreachable for a week: drop the oldest
                    lines = [l for l in self.outbox.read_text(encoding="utf-8").splitlines() if l.strip()]
                    self._write_lines(lines[-(MAX_OUTBOX - 1):])
                lead = ""
                if self.outbox.exists() and self.outbox.stat().st_size:
                    with open(self.outbox, "rb") as fh:       # a torn last line (power cut) must not
                        fh.seek(-1, 2)                         # swallow the next record
                        lead = "" if fh.read(1) == b"\n" else "\n"
                with open(self.outbox, "a", encoding="utf-8") as fh:
                    fh.write(lead + json.dumps(record) + "\n")
                self._queued += 1
        except OSError as e:
            log.warning("outbox write failed: %s", e)

    def _write_lines(self, lines: list) -> None:
        tmp = self.outbox.with_suffix(".tmp")
        tmp.write_text("".join(x + "\n" for x in lines), encoding="utf-8")
        tmp.replace(self.outbox)
        self._queued = len(lines)

    def _flush_outbox(self) -> None:
        with self.lock:
            if not self.outbox.exists():
                return
            lines = [l for l in self.outbox.read_text(encoding="utf-8").splitlines() if l.strip()]
        if not lines:
            return
        recs = []
        for l in lines[-10080:]:
            try:
                recs.append(json.loads(l))
            except ValueError:
                pass
        for i in range(0, len(recs), 200):
            self._post("/api/ingest/minutes", {"node": self.rt.node_meta(), "minutes": recs[i:i + 200]})
        with self.lock:   # remove only what was sent; keep anything queued meanwhile
            now = [l for l in self.outbox.read_text(encoding="utf-8").splitlines() if l.strip()]
            self._write_lines(now[len(lines):])

    # ------------------------------------------------------------------ live
    def _live_payload(self) -> dict:
        s = self.rt.snapshot or {}
        cfg = self.rt.cfg
        geo_z = {z["id"]: z["points"] for z in cfg.get("zones", [])}
        geo_l = {l["id"]: l["points"] for l in cfg.get("lines", [])}
        events = self.rt.engine.events_since(self.acked_event)[-100:]
        heat = s.get("heat") or {}
        k_min = int((cfg.get("privacy") or {}).get("k_min") or 0)
        in_view = (s.get("kpis") or {}).get("in_view")
        if k_min and (in_view is None or in_view < k_min):
            heat = {"w": heat.get("w"), "h": heat.get("h")}   # too few people: no grid that could trace one
        return {
            "node": self.rt.node_meta(), "ts": s.get("ts", time.time()), "fps": s.get("fps"),
            "camera": {k: (s.get("camera") or {}).get(k) for k in ("status", "w", "h", "cam_fps", "device", "detector")},
            "kpis": s.get("kpis"), "tracking": s.get("tracking"), "window_min": s.get("window_min"),
            "zones": [{**z, "points": geo_z.get(z["id"])} for z in s.get("zones", [])],
            "counters": [{**c, "points": geo_l.get(c["id"]) or geo_z.get(c["id"])} for c in s.get("counters", [])],
            "heat": {"w": heat.get("w"), "h": heat.get("h"), "live": heat.get("live"), "live_peak_s": heat.get("live_peak_s")},
            "minute": self.rt.engine.current_minute(),
            "floor": {"H": (cfg.get("floor") or {}).get("H")},
            "alerts": s.get("alerts", []),
            "events": events,
        }

    # ------------------------------------------------------------------ http
    def _post(self, path: str, body: dict) -> dict:
        url = self.cfg["url"].rstrip("/") + path
        data = json.dumps(body).encode("utf-8")
        req = urllib.request.Request(url, data=data, method="POST", headers={
            "Content-Type": "application/json", "Authorization": "Bearer " + str(self.cfg.get("token", ""))})
        with _OPENER.open(req, timeout=8) as r:
            return json.loads(r.read() or b"{}")

    def get_bytes(self, path: str) -> tuple[bytes, str]:
        url = self.cfg["url"].rstrip("/") + path
        req = urllib.request.Request(url, headers={"Authorization": "Bearer " + str(self.cfg.get("token", ""))})
        with _OPENER.open(req, timeout=10) as r:
            return r.read(), r.headers.get("Content-Type", "application/octet-stream")

    def _run(self) -> None:
        backoff = self.LIVE_EVERY_S
        while not self._stop:
            if not self.enabled():
                self.status = "off"
                time.sleep(1)
                continue
            try:
                payload = self._live_payload()
                self._post("/api/ingest/live", payload)
                if payload["events"]:            # advance only once the hub has them
                    self.acked_event = max(e["id"] for e in payload["events"])
                self._flush_outbox()
                self.status, self.last_ok, self.last_error = "connected", time.time(), None
                backoff = self.LIVE_EVERY_S
            except urllib.error.HTTPError as e:
                self.status = "rejected" if e.code in (401, 403) else "error"
                self.last_error = f"hub answered {e.code}" + (" - check the token" if e.code in (401, 403) else "")
                backoff = min(backoff * 2, 30)
            except Exception as e:  # network down, hub restarting: retry with backoff
                self.status = "unreachable"
                self.last_error = f"{type(e).__name__}: {e}"[:200]
                backoff = min(backoff * 2, 30)
            time.sleep(backoff)
