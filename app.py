"""
app.py - FlowLens: privacy-first queue & occupancy analytics.

    python app.py                       # webcam 0 -> http://localhost:8000
    python app.py --source 1            # another camera
    python app.py --source rtsp://...   # IP camera
    python app.py --model yolo11s.pt    # more accurate detector (slower)

Data path:  camera -> RAM frame -> YOLO(person) + ByteTrack -> Engine (aggregates)
            -> /ws JSON (2 Hz) + auto-saved CSV logs + state file (aggregates only)
Local view: the same RAM frame with people anonymised BEFORE encoding, MJPEG to this
            machine only. Nothing visual is ever written to disk.
"""
import argparse
import asyncio
import copy
import csv
import io
import json
import logging
import re
import threading
import time
import traceback
from datetime import datetime
from logging.handlers import RotatingFileHandler
from pathlib import Path

import uvicorn
from fastapi import Body, FastAPI, Request, WebSocket, WebSocketDisconnect
from fastapi.responses import FileResponse, JSONResponse, Response, StreamingResponse

from analytics import Engine

ROOT = Path(__file__).parent
CONFIG_FILE = ROOT / "config.json"     # re-pointed by --data (several nodes on one machine)
LOG_DIR = ROOT / "logs"
STATE_FILE = LOG_DIR / "state.json"
OUTBOX_FILE = LOG_DIR / "outbox.jsonl"
VIDEO_MODES = ("blur", "pixelate", "mask", "off")
ZONE_TYPES = ("queue", "service", "area", "door")

log = logging.getLogger("qa")
log.setLevel(logging.INFO)


def setup_paths(data_dir: Path) -> None:
    """Config, logs and state live in one data folder (default: next to app.py)."""
    global CONFIG_FILE, LOG_DIR, STATE_FILE, OUTBOX_FILE
    data_dir.mkdir(parents=True, exist_ok=True)
    CONFIG_FILE = data_dir / "config.json"
    LOG_DIR = data_dir / "logs"
    LOG_DIR.mkdir(exist_ok=True)
    STATE_FILE = LOG_DIR / "state.json"
    OUTBOX_FILE = LOG_DIR / "outbox.jsonl"
    fh = RotatingFileHandler(LOG_DIR / "app.log", maxBytes=2_000_000, backupCount=3, encoding="utf-8")
    fh.setFormatter(logging.Formatter("%(asctime)s %(levelname)s %(message)s"))
    sh = logging.StreamHandler()
    sh.setFormatter(logging.Formatter("%(asctime)s  %(message)s", "%H:%M:%S"))
    log.handlers[:] = [fh, sh]

DEFAULT_CONFIG = {
    "camera": {"source": 0, "width": 1280, "height": 720},
    "privacy": {"video_mode": "blur", "k_min": 0},
    "analytics": {"stats_window_min": 15, "min_dwell_s": 2, "heat_half_life_s": 30},
    "logging": {"stats_csv": True, "events_csv": True},
    "node": {"id": "", "name": "Camera 1", "site": ""},
    "hub": {"enabled": False, "url": "", "token": ""},
    "floor": {"H": None, "cam_pts": [], "plan_pts": []},
    "zones": [],
    "lines": [],
}
ROLES = ("perimeter", "internal", "ignore")


def _role(c: dict) -> str:
    if c.get("role") in ROLES:
        return c["role"]
    return "perimeter" if c.get("site", True) else "internal"


def _area(v) -> str:
    return str(v or "").strip()[:40]


# ------------------------------------------------------------------ config
def load_config() -> dict:
    cfg = copy.deepcopy(DEFAULT_CONFIG)
    if CONFIG_FILE.exists():
        try:
            saved = json.loads(CONFIG_FILE.read_text(encoding="utf-8"))
        except Exception as e:
            log.error("config.json unreadable (%s) - keeping a copy and starting with defaults", e)
            CONFIG_FILE.replace(CONFIG_FILE.with_suffix(f".broken-{int(time.time())}.json"))
            return cfg
        for k, v in saved.items():
            cfg[k] = {**cfg[k], **v} if isinstance(cfg.get(k), dict) and isinstance(v, dict) else v
        if cfg["privacy"].get("video_mode") not in VIDEO_MODES:
            cfg["privacy"]["video_mode"] = "blur"
    return cfg


def save_config(cfg: dict) -> None:
    tmp = CONFIG_FILE.with_suffix(".tmp")
    tmp.write_text(json.dumps(cfg, indent=2), encoding="utf-8")
    tmp.replace(CONFIG_FILE)


def clean_points(pts, n_min, n_max=None):
    out = [[round(min(1, max(0, float(x))), 4), round(min(1, max(0, float(y))), 4)] for x, y in pts]
    if len(out) < n_min or (n_max and len(out) > n_max):
        raise ValueError("bad point count")
    return out


def _opt_int(v):
    return int(v) if v not in (None, "", 0, "0") else None


ID_RE = re.compile(r"^[A-Za-z0-9_-]{1,40}$")
COLOR_RE = re.compile(r"^#[0-9a-fA-F]{6}$")


def _id(v) -> str:
    v = str(v)
    if not ID_RE.match(v):
        raise ValueError(f"bad id {v[:20]!r}")
    return v


def _color(v, default: str) -> str:
    return v if isinstance(v, str) and COLOR_RE.match(v) else default


def csv_safe(v):
    """Stop names like '=cmd' being run as formulas when a CSV is opened in Excel."""
    return "'" + v if isinstance(v, str) and v[:1] in ("=", "+", "-", "@") else v


def validate(patch: dict, cfg: dict) -> dict:
    new = copy.deepcopy(cfg)
    if "zones" in patch:
        new["zones"] = [{
            "id": _id(z["id"]), "name": str(z.get("name") or "Zone")[:60],
            "type": z.get("type") if z.get("type") in ZONE_TYPES else "area",
            "color": _color(z.get("color"), "#3987e5"),
            "anchor": "center" if z.get("anchor") == "center" else "feet",
            "points": clean_points(z["points"], 3),
            "capacity": _opt_int(z.get("capacity")),
            "wait_alert_s": _opt_int(z.get("wait_alert_s")),
            "in_label": str(z.get("in_label") or "In")[:20],
            "out_label": str(z.get("out_label") or "Out")[:20],
            "area": _area(z.get("area")),
            "role": _role(z), "site": _role(z) == "perimeter",
            "from_area": _area(z.get("from_area")), "to_area": _area(z.get("to_area")),
        } for z in patch["zones"]]
    if "lines" in patch:
        new["lines"] = [{
            "id": _id(l["id"]), "name": str(l.get("name") or "Line")[:60],
            "color": _color(l.get("color"), "#d95926"),
            "points": clean_points(l["points"], 2, 2),
            "in_label": str(l.get("in_label") or "In")[:20],
            "out_label": str(l.get("out_label") or "Out")[:20],
            "anchor": "center" if l.get("anchor") == "center" else "feet",
            "band": max(0.005, min(0.15, float(l.get("band", 0.03)))),
            "role": _role(l), "site": _role(l) == "perimeter",
            "from_area": _area(l.get("from_area")), "to_area": _area(l.get("to_area")),
        } for l in patch["lines"]]
    if "privacy" in patch:
        p = patch["privacy"]
        if p.get("video_mode") in VIDEO_MODES:
            new["privacy"]["video_mode"] = p["video_mode"]
        if "k_min" in p:
            new["privacy"]["k_min"] = max(0, min(10, int(p["k_min"] or 0)))
    if "analytics" in patch:
        a = patch["analytics"]
        for key, lo, hi in (("stats_window_min", 1, 240), ("min_dwell_s", 0, 60), ("heat_half_life_s", 5, 600)):
            if key in a:
                new["analytics"][key] = max(lo, min(hi, float(a[key])))
    if "logging" in patch:
        for key in ("stats_csv", "events_csv"):
            if key in patch["logging"]:
                new["logging"][key] = bool(patch["logging"][key])
    if "node" in patch:
        for key in ("name", "site"):
            if key in patch["node"]:
                new["node"][key] = str(patch["node"][key] or "").strip()[:60] or ("Camera" if key == "name" else "")
    if "hub" in patch:
        h = patch["hub"]
        if "url" in h:
            u = str(h["url"] or "").strip().rstrip("/")
            if u and not re.match(r"^https?://", u):
                u = "http://" + u
            new["hub"]["url"] = u[:200]
        if "token" in h:
            new["hub"]["token"] = str(h["token"] or "").strip()[:200]
        if "enabled" in h:
            new["hub"]["enabled"] = bool(h["enabled"])
    if "camera" in patch and "source" in patch["camera"]:
        s = str(patch["camera"]["source"]).strip()
        new["camera"]["source"] = int(s) if s.isdigit() else s
    return new


# ------------------------------------------------------------------ logs
class CsvLogger:
    """Auto-saves per-minute stats and anonymous events to daily CSV files."""

    def __init__(self, rt):
        self.rt = rt
        self.lock = threading.Lock()
        self.last_stats = None
        self.last_event = None
        self.rows_today = 0

    @staticmethod
    def _append(path: Path, header: list, rows: list):
        rows = [[csv_safe(c) for c in r] for r in rows]
        for target in (path, path.with_name(path.stem + "_while-open-in-excel" + path.suffix)):
            try:
                new = not target.exists()
                with open(target, "a", newline="", encoding="utf-8") as fh:
                    w = csv.writer(fh)
                    if new:
                        w.writerow(header)
                    w.writerows(rows)
                return
            except PermissionError:   # the file is open in Excel (Windows locks it): use the side file
                continue
        raise PermissionError(f"{path.name} is locked")

    def minute(self, bucket: dict, names: dict) -> None:
        if not self.rt.cfg["logging"].get("stats_csv", True):
            return
        stamp = datetime.fromtimestamp(bucket["t"])
        n = bucket["n"] or 1
        m = stamp.strftime("%Y-%m-%d %H:%M")
        rows = [[m, "scene", "All people", round(bucket["p_sum"] / n, 2), bucket["p_max"], "", "", "", ""]]
        for zid, v in bucket["z"].items():
            rows.append([m, "zone", names["z"].get(zid, zid), round(v[0] / n, 2), v[1], v[2],
                         round(v[3] / v[4], 1) if v[4] else "", "", ""])
        for cid, v in bucket.get("c", {}).items():
            name, kind = names["c"].get(cid, (cid, "counter"))
            rows.append([m, kind, name, "", "", "", "", v[0], v[1]])
        with self.lock:
            self._append(LOG_DIR / f"stats_{stamp:%Y-%m-%d}.csv",
                         ["minute", "kind", "name", "occupancy_avg", "occupancy_max",
                          "visits_completed", "avg_dwell_s", "in", "out"], rows)
            self.last_stats = time.time()

    def events(self, evs: list) -> None:
        if not evs or not self.rt.cfg["logging"].get("events_csv", True):
            return
        by_day: dict = {}
        for e in evs:
            ts = datetime.fromtimestamp(e["t"])
            by_day.setdefault(ts.strftime("%Y-%m-%d"), []).append(
                [ts.strftime("%Y-%m-%d %H:%M:%S"), e["kind"], e["text"]])
        with self.lock:
            for day, rows in by_day.items():
                self._append(LOG_DIR / f"events_{day}.csv", ["time", "kind", "event"], rows)
            self.last_event = time.time()


# ---------------------------------------------------------------- runtime
class Runtime:
    def __init__(self, args):
        self.args = args
        self.cfg = load_config()
        if args.source is not None:
            self.cfg["camera"]["source"] = args.source
        if not self.cfg["node"].get("id"):
            import uuid
            self.cfg["node"]["id"] = "n_" + uuid.uuid4().hex[:10]
            save_config(self.cfg)
        self.csv = CsvLogger(self)
        self.engine = Engine(self.cfg, on_minute=self.on_minute)
        from node_link import HubLink
        self.hub = HubLink(self, OUTBOX_FILE)
        if STATE_FILE.exists():
            try:
                if self.engine.load_state(json.loads(STATE_FILE.read_text(encoding="utf-8"))):
                    log.info("restored statistics from %s", STATE_FILE.name)
            except Exception as e:
                log.warning("could not restore state: %s", e)
        self.snapshot: dict = {}
        self.jpeg = None
        self.viewers = 0
        self.grabber = None
        self.detector = None
        self.detector_status = "loading model"
        self.errors = 0
        self.last_error = None
        self.running = True
        self.last_state_save = 0.0
        self.cfg_lock = threading.RLock()
        self.pending_camera = False

    def start_camera(self):
        from vision import CameraGrabber
        if self.grabber:
            self.grabber.stop()
        c = self.cfg["camera"]
        self.grabber = CameraGrabber(c["source"], c.get("width", 1280), c.get("height", 720))
        if self.detector:
            self.detector.reset_tracker()
        self.engine.forget_tracks()

    def update_config(self, patch: dict) -> dict:
        with self.cfg_lock:
            new = validate(patch, self.cfg)
            source_changed = new["camera"]["source"] != self.cfg["camera"]["source"]
            self.engine.apply_config(new)
            self.cfg = new
            try:
                save_config(new)
            except OSError as e:          # antivirus / locked file: keep running, retry on next change
                log.warning("config not saved to disk: %s", e)
            if source_changed:
                self.pending_camera = True   # the inference loop switches cameras (thread-safe)
        log.info("config updated (%s)", ", ".join(patch.keys()))
        return new

    def set_floor(self, floor: dict) -> dict:
        with self.cfg_lock:
            new = copy.deepcopy(self.cfg)
            new["floor"] = floor
            self.cfg = new
            save_config(new)
            return floor

    def save_state(self):
        try:
            tmp = STATE_FILE.with_suffix(".tmp")
            tmp.write_text(json.dumps(self.engine.dump_state()), encoding="utf-8")
            tmp.replace(STATE_FILE)
            self.last_state_save = time.time()
        except Exception as e:
            log.warning("state save failed: %s", e)

    def node_meta(self) -> dict:
        n = self.cfg["node"]
        return {"id": n["id"], "name": n.get("name") or "Camera", "site": n.get("site", ""), "version": "2.0"}

    def on_minute(self, bucket: dict, names: dict) -> None:
        # each sink on its own: a CSV open in Excel must never cost the hub its minute (or vice versa)
        try:
            self.hub.queue_minute(self.engine.minute_record(bucket))
        except Exception as e:
            log.warning("hub queue failed: %s", e)
        try:
            self.csv.minute(bucket, names)
        except Exception as e:
            log.warning("stats CSV not written: %s", e)

    def publish(self):
        snap = self.engine.snapshot()
        w, h = self.grabber.size if self.grabber else (0, 0)
        d = self.detector
        snap["camera"] = {"status": self.grabber.status if self.grabber else "starting", "w": w, "h": h,
                          "source": str(self.cfg["camera"]["source"]),
                          "cam_fps": round(self.grabber.cam_fps, 1) if self.grabber else 0,
                          "detector": self.detector_status,
                          "device": d.device_name if d else None,
                          "infer_ms": round(d.ms, 1) if d else None,
                          "model": self.args.model, "errors": self.errors, "last_error": self.last_error}
        snap["privacy"] = self.cfg["privacy"]
        snap["logging"] = {**self.cfg["logging"], "last_stats": self.csv.last_stats,
                           "last_event": self.csv.last_event, "last_state": self.last_state_save,
                           "folder": str(LOG_DIR)}
        snap["node"] = self.node_meta()
        snap["hub"] = self.hub.info()
        snap["floor"] = {"calibrated": bool((self.cfg.get("floor") or {}).get("H"))}
        self.snapshot = snap

    def loop(self):
        from vision import YoloDetector, render_view
        self.start_camera()
        self.publish()
        try:
            self.detector = YoloDetector(self.args.model, self.args.conf, self.args.imgsz, self.args.device)
            self.detector_status = "ok"
        except Exception as e:
            self.detector_status = f"model failed to load: {e}"
            log.exception("model load failed")
            while self.running:            # keep the dashboard up so the error is visible
                self.publish()
                time.sleep(1)
            return
        seq, last_snap, last_frame = 0, 0.0, time.monotonic()
        while self.running:
            try:
                if self.pending_camera:
                    self.pending_camera = False
                    self.start_camera()
                    seq = 0
                frame, seq = self.grabber.next(seq, timeout=0.5)
                if frame is None and time.monotonic() - last_frame > 1.0:
                    # no frames (camera unplugged / reconnecting): keep time moving so people
                    # age out of zones instead of freezing, and nobody is re-linked across the gap
                    w, h = self.grabber.size
                    self.engine.update([], aspect=w / max(1, h))
                if frame is not None:
                    last_frame = time.monotonic()
                    h, w = frame.shape[:2]
                    dets = self.detector(frame)
                    self.engine.update(dets, aspect=w / h)
                    mode = self.cfg["privacy"]["video_mode"]
                    if self.viewers > 0 and mode != "off":
                        self.jpeg = render_view(frame, dets, mode, self.engine.labels_for(dets))
                    elif mode == "off":
                        self.jpeg = None
                    del frame  # the frame never outlives this iteration
                now = time.monotonic()
                if now - last_snap >= 0.5:
                    last_snap = now
                    self.publish()
                    evs = self.engine.drain_events()
                    try:
                        self.csv.events(evs)
                    except Exception as e:
                        log.warning("events CSV not written: %s", e)
                if time.time() - self.last_state_save > 30:
                    self.save_state()
            except Exception as e:  # never let one bad frame kill the service
                self.errors += 1
                self.last_error = f"{type(e).__name__}: {e}"
                log.error("loop error: %s\n%s", e, traceback.format_exc())
                time.sleep(0.2)


# ----------------------------------------------------------------- server
SAFE_LOG = re.compile(r"^(stats|events)_\d{4}-\d{2}-\d{2}\.csv$|^app\.log$")


def build_app(rt: Runtime) -> FastAPI:
    app = FastAPI(title="FlowLens")
    local_only = args_host_is_local(rt.args.host)

    @app.middleware("http")
    async def guard(request: Request, call_next):
        host = (request.headers.get("host") or "").rsplit(":", 1)[0].strip("[]").lower()
        if local_only and host not in ("localhost", "127.0.0.1", "::1"):
            return JSONResponse({"error": "use http://localhost"}, status_code=403)   # DNS-rebinding guard
        if request.method in ("POST", "PUT", "DELETE") and request.headers.get("x-flowlens") != "1":
            return JSONResponse({"error": "missing X-FlowLens header"}, status_code=403)  # CSRF guard
        return await call_next(request)

    @app.get("/")
    def index():
        return FileResponse(ROOT / "static" / "index.html", headers={"Cache-Control": "no-store"})

    @app.get("/api/config")
    def get_config():
        return rt.cfg

    @app.put("/api/config")
    def put_config(patch: dict = Body(...)):
        try:
            return rt.update_config(patch)
        except (KeyError, ValueError, TypeError) as e:
            return JSONResponse({"error": f"invalid config: {e}"}, status_code=400)

    @app.post("/api/reset")
    def reset():
        rt.engine.reset()
        rt.engine.apply_config(rt.cfg)
        rt.save_state()
        log.info("statistics reset from dashboard")
        return {"ok": True}

    @app.get("/api/snapshot")
    def snapshot():
        return rt.snapshot

    @app.get("/api/series")
    def series(minutes: int = 60):
        return rt.engine.series_range(max(1, min(1440, minutes)))

    @app.get("/api/logs")
    def logs():
        files = sorted((p for p in LOG_DIR.iterdir() if SAFE_LOG.match(p.name)),
                       key=lambda p: p.stat().st_mtime, reverse=True)
        return [{"name": p.name, "bytes": p.stat().st_size, "modified": p.stat().st_mtime} for p in files[:60]]

    @app.get("/api/logs/{name}")
    def log_file(name: str):
        if not SAFE_LOG.match(name) or not (LOG_DIR / name).exists():
            return JSONResponse({"error": "not found"}, status_code=404)
        return FileResponse(LOG_DIR / name, filename=name)

    @app.get("/api/export.csv")
    def export(minutes: int = 1440):
        s = rt.engine.series_range(max(1, min(1440, minutes)))
        buf = io.StringIO()
        w = csv.writer(buf)
        head = ["minute", "people_avg", "people_max"]
        for z in s["zones"]:
            head += [f"{z['name']} occupancy_avg", f"{z['name']} occupancy_max",
                     f"{z['name']} visits", f"{z['name']} avg_dwell_s"]
        for c in s["counters"]:
            head += [f"{c['name']} in", f"{c['name']} out"]
        w.writerow(head)
        for b in s["buckets"]:
            row = [datetime.fromtimestamp(b["t"]).strftime("%Y-%m-%d %H:%M"), b["people_avg"], b["people_max"]]
            for z in s["zones"]:
                v = b["z"].get(z["id"])
                row += ([v["occ_avg"], v["occ_max"], v["visits"],
                         round(v["dwell_sum"] / v["dwell_n"], 1) if v["dwell_n"] else ""] if v else ["", "", "", ""])
            for c in s["counters"]:
                v = b["c"].get(c["id"])
                row += [v["in"], v["out"]] if v else ["", ""]
            w.writerow(row)
        name = f"queue-analytics_{datetime.now():%Y%m%d_%H%M}.csv"
        return Response(buf.getvalue(), media_type="text/csv",
                        headers={"Content-Disposition": f'attachment; filename="{name}"'})

    @app.get("/api/hub/floorplan")
    def hub_floorplan():
        """Proxy the hub's floor plan image for calibration (the camera view never leaves this machine)."""
        if not rt.hub.enabled():
            return JSONResponse({"error": "connect this camera to a hub first"}, status_code=400)
        try:
            data, ctype = rt.hub.get_bytes("/api/floorplan")
            return Response(data, media_type=ctype, headers={"Cache-Control": "no-store"})
        except Exception as e:
            return JSONResponse({"error": f"hub has no floor plan or is unreachable ({e})"}, status_code=404)

    @app.put("/api/floor")
    def set_floor(body: dict = Body(...)):
        """4+ point pairs: camera view (0..1) -> floor plan (0..1). Stores the homography."""
        try:
            import cv2
            import numpy as np
            cam = np.float32([[float(x), float(y)] for x, y in body["cam_pts"]])
            plan = np.float32([[float(x), float(y)] for x, y in body["plan_pts"]])
            if len(cam) < 4 or len(cam) != len(plan):
                raise ValueError("need 4 matching point pairs")
            H, _ = cv2.findHomography(cam, plan, 0) if len(cam) > 4 else (cv2.getPerspectiveTransform(cam, plan), None)
            if H is None or not np.isfinite(H).all():
                raise ValueError("points are degenerate - spread them out")
            Hn = (H / H[2, 2]).round(8).tolist()
        except (KeyError, ValueError, TypeError) as e:
            return JSONResponse({"error": str(e)}, status_code=400)
        floor = rt.set_floor({"H": Hn, "cam_pts": cam.round(4).tolist(), "plan_pts": plan.round(4).tolist()})
        log.info("floor plan mapping saved")
        return floor

    @app.delete("/api/floor")
    def clear_floor():
        return rt.set_floor({"H": None, "cam_pts": [], "plan_pts": []})

    @app.get("/api/privacy")
    def privacy():
        return {
            "face_recognition": False, "identity_or_appearance_models": False,
            "track_stitching": "motion only (position + velocity), no appearance features",
            "video_recorded": False, "frames_written_to_disk": False,
            "live_view": f"{rt.cfg['privacy']['video_mode']} (people anonymised before encoding, served to {rt.args.host} only)",
            "saved_to_disk": "per-minute aggregates, anonymous event text, aggregate state - no images, ids or coordinates",
            "k_min": rt.cfg["privacy"]["k_min"], "processing": "on-device",
        }

    @app.websocket("/ws")
    async def ws(sock: WebSocket):
        await sock.accept()
        try:
            while True:
                await sock.send_json(rt.snapshot)
                await asyncio.sleep(0.5)
        except (WebSocketDisconnect, RuntimeError):
            pass

    @app.get("/video.mjpg")
    async def video(request: Request):
        async def gen():
            rt.viewers += 1
            last = None
            try:
                while not await request.is_disconnected():
                    jpg = rt.jpeg
                    if jpg is not None and jpg is not last:
                        last = jpg
                        yield (b"--frame\r\nContent-Type: image/jpeg\r\nContent-Length: "
                               + str(len(jpg)).encode() + b"\r\n\r\n" + jpg + b"\r\n")
                    await asyncio.sleep(0.03)
            finally:
                rt.viewers -= 1
        return StreamingResponse(gen(), media_type="multipart/x-mixed-replace; boundary=frame",
                                 headers={"Cache-Control": "no-store"})

    return app


def args_host_is_local(host: str) -> bool:
    return host in ("127.0.0.1", "localhost", "::1")


def port_free(host: str, port: int) -> bool:
    import socket
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sk:
        try:
            sk.bind(("127.0.0.1" if host in ("localhost", "::1") else host, port))
            return True
        except OSError:
            return False


def parse_args():
    p = argparse.ArgumentParser(description="Privacy-first queue & occupancy analytics")
    p.add_argument("--source", default=None, help="camera index, RTSP/HTTP URL or video file (overrides config.json)")
    p.add_argument("--model", default="yolo11n.pt", help="yolo11n.pt (fast) / yolo11s.pt (more accurate)")
    p.add_argument("--conf", type=float, default=0.35, help="min confidence for untracked detections")
    p.add_argument("--imgsz", type=int, default=640)
    p.add_argument("--device", default="auto", help="auto, cpu, or 0 for the first NVIDIA GPU")
    p.add_argument("--host", default="127.0.0.1", help="127.0.0.1 keeps the dashboard on this machine")
    p.add_argument("--port", type=int, default=8000)
    p.add_argument("--data", default=None, help="folder for config + logs (one per camera when several run on one machine)")
    p.add_argument("--name", default=None, help="camera name shown on the hub, e.g. 'Hall A - entrance'")
    p.add_argument("--hub", default=None, help="hub address, e.g. http://192.168.1.20:9000")
    p.add_argument("--hub-token", default=None, help="the hub's access token")
    a = p.parse_args()
    if a.source is not None and str(a.source).isdigit():
        a.source = int(a.source)
    return a


if __name__ == "__main__":
    args = parse_args()
    setup_paths(Path(args.data).resolve() if args.data else ROOT)
    if not port_free(args.host, args.port):
        log.error("Port %d is already in use - another FlowLens camera is probably running. "
                  "Start this one with a different --port (and its own --data folder).", args.port)
        raise SystemExit(2)
    rt = Runtime(args)
    patch = {}
    if args.name:
        patch["node"] = {"name": args.name}
    if args.hub:
        patch["hub"] = {"url": args.hub, "enabled": True, **({"token": args.hub_token} if args.hub_token else {})}
    elif args.hub_token:
        patch["hub"] = {"token": args.hub_token}
    if patch:
        rt.update_config(patch)
    server = uvicorn.Server(uvicorn.Config(build_app(rt), host=args.host, port=args.port, log_level="warning"))
    threading.Thread(target=server.run, daemon=True, name="http").start()
    log.info("Dashboard: http://localhost:%d   (Ctrl+C to stop)", args.port)
    try:
        rt.loop()
    except KeyboardInterrupt:
        pass
    finally:
        rt.running = False
        try:
            rt.csv.events(rt.engine.drain_events())
        except Exception as e:
            log.warning("events CSV not written: %s", e)
        try:
            m = rt.engine.current_minute()
            if m:
                rt.hub.queue_minute(m)   # the hub replaces it if this minute continues after a restart
        except Exception:
            pass
        rt.save_state()
        if rt.grabber:
            rt.grabber.stop()
        log.info("stopped - statistics saved")
