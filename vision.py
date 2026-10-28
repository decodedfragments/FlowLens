"""
vision.py - on-device sensing.

CameraGrabber : background thread holding only the newest frame (RAM), auto-reconnect,
                MJPG negotiation for full frame rate on USB/laptop webcams.
YoloDetector  : YOLO person class + tuned ByteTrack. Auto GPU (+FP16), low-confidence
                second-stage association so fast/blurred walkers keep their track.
render_view   : anonymised local view - people blurred / pixelated / masked before encoding.
"""
from __future__ import annotations

import logging
import os
import sys
import threading
import time
from pathlib import Path

import cv2
import numpy as np

from analytics import Detection

log = logging.getLogger("qa")
PERSON_CLASS = 0  # COCO "person"
TRACKER_CFG = str(Path(__file__).parent / "tracker.yaml")


class CameraGrabber:
    def __init__(self, source, width: int = 1280, height: int = 720):
        self.source, self.width, self.height = source, width, height
        self.frame = None
        self.seq = 0
        self.status = "connecting"
        self.size = (width, height)
        self.cam_fps = 0.0
        self._times: list = []
        self.cond = threading.Condition()
        self._stop = False
        self.thread = threading.Thread(target=self._run, daemon=True, name="camera")
        self.thread.start()

    def _open(self):
        src = self.source
        if isinstance(src, int) and sys.platform.startswith("win"):
            cap = cv2.VideoCapture(src, cv2.CAP_DSHOW)
        else:
            cap = cv2.VideoCapture(src)
        if isinstance(src, int):
            cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*"MJPG"))  # 720p@30 on most webcams
            cap.set(cv2.CAP_PROP_FRAME_WIDTH, self.width)
            cap.set(cv2.CAP_PROP_FRAME_HEIGHT, self.height)
            cap.set(cv2.CAP_PROP_FPS, 30)
            cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
        return cap

    def _run(self):
        is_file = isinstance(self.source, str) and os.path.isfile(self.source)
        backoff = 1.0
        while not self._stop:
            try:
                cap = self._open()
            except Exception as e:
                cap = None
                log.warning("camera open error: %s", e)
            if cap is None or not cap.isOpened():
                self.status = f"cannot open camera {self.source!r} - retrying"
                time.sleep(backoff)
                backoff = min(backoff * 2, 10)
                continue
            backoff = 1.0
            file_fps = cap.get(cv2.CAP_PROP_FPS) or 25
            self.status = "ok"
            log.info("camera %r opened", self.source)
            fails = 0
            while not self._stop:
                ok, frame = cap.read()
                if not ok or frame is None:
                    if is_file:
                        cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
                        continue
                    fails += 1
                    if fails > 50:
                        self.status = "camera stopped delivering frames - reconnecting"
                        log.warning(self.status)
                        break
                    time.sleep(0.02)
                    continue
                fails = 0
                now = time.monotonic()
                self._times = [x for x in self._times[-29:] if now - x < 2] + [now]
                if len(self._times) > 1:
                    self.cam_fps = (len(self._times) - 1) / (self._times[-1] - self._times[0] or 1)
                with self.cond:
                    self.frame = frame
                    self.size = (frame.shape[1], frame.shape[0])
                    self.seq += 1
                    self.cond.notify_all()
                if is_file:
                    time.sleep(1 / min(file_fps, 30))
            cap.release()

    def next(self, last_seq: int, timeout: float = 1.0):
        with self.cond:
            self.cond.wait_for(lambda: self.seq != last_seq or self._stop, timeout=timeout)
            if self.seq == last_seq or self.frame is None:
                return None, last_seq
            return self.frame, self.seq

    def stop(self):
        self._stop = True
        with self.cond:
            self.cond.notify_all()


class YoloDetector:
    def __init__(self, model: str = "yolo11n.pt", conf: float = 0.35,
                 imgsz: int = 640, device: str | None = None):
        import torch
        from ultralytics import YOLO
        if device in (None, "", "auto"):
            device = "0" if torch.cuda.is_available() else "cpu"
        self.device = device
        self.half = device != "cpu" and torch.cuda.is_available()
        # FP16 on the GPU: newer Ultralytics calls it quantize=16 (half= is deprecated), older ones half=True
        try:
            from ultralytics.cfg import DEFAULT_CFG_DICT
            new_api = "quantize" in DEFAULT_CFG_DICT
        except Exception:
            new_api = False
        self.fp16 = ({"quantize": 16} if new_api else {"half": True}) if self.half else {}
        self.device_name = torch.cuda.get_device_name(0) if self.half else "CPU"
        self.model = YOLO(model)
        self.conf, self.imgsz = conf, imgsz
        self.ms = 0.0
        self.warmup()
        log.info("detector %s on %s (fp16=%s)", model, self.device_name, self.half)

    def warmup(self):
        dummy = np.zeros((self.imgsz, self.imgsz, 3), np.uint8)
        self.model.predict(dummy, device=self.device, verbose=False, imgsz=self.imgsz,
                           **self.fp16)

    def reset_tracker(self):
        """Drop tracker state (e.g. after a camera switch)."""
        try:
            for t in getattr(self.model.predictor, "trackers", []) or []:
                t.reset()
        except Exception:
            pass

    def __call__(self, frame) -> list[Detection]:
        t0 = time.perf_counter()
        # conf 0.1 lets ByteTrack's second stage re-associate blurred / partly hidden people;
        # untracked low-score boxes are filtered out below.
        r = self.model.track(
            frame, persist=True, classes=[PERSON_CLASS], conf=0.1, iou=0.6,
            imgsz=self.imgsz, device=self.device, tracker=TRACKER_CFG, verbose=False,
            **self.fp16,
        )[0]
        self.ms = 0.8 * self.ms + 0.2 * (time.perf_counter() - t0) * 1000 if self.ms else (time.perf_counter() - t0) * 1000
        if r.boxes is None or len(r.boxes) == 0:
            return []
        h, w = frame.shape[:2]
        xyxy = r.boxes.xyxy.cpu().numpy()
        confs = r.boxes.conf.cpu().numpy()
        ids = r.boxes.id.int().cpu().tolist() if r.boxes.id is not None else [None] * len(xyxy)
        out = []
        for (x1, y1, x2, y2), c, tid in zip(xyxy, confs, ids):
            if tid is None and c < self.conf:
                continue
            if (x2 - x1) < 6 or (y2 - y1) < 12:          # implausibly tiny boxes
                continue
            out.append(Detection(tid, float((x1 + x2) / 2 / w), float(y2 / h),
                                 float((x1 + x2) / 2 / w), float((y1 + y2) / 2 / h),
                                 (x1 / w, y1 / h, x2 / w, y2 / h), float(c)))
        return out


def _anonymise(frame, x1, y1, x2, y2, mode):
    roi = frame[y1:y2, x1:x2]
    if roi.size == 0:
        return
    if mode == "mask":
        overlay = roi.copy()
        overlay[:] = (58, 56, 54)
        frame[y1:y2, x1:x2] = cv2.addWeighted(overlay, 0.92, roi, 0.08, 0)
        return
    if mode == "pixelate":
        bw = 8
    else:  # blur: pixelate hard first (irreversible), then smooth so it reads as a soft silhouette
        bw = 6
    bh = max(1, round(bw * roi.shape[0] / max(1, roi.shape[1])))
    small = cv2.resize(roi, (bw, bh), interpolation=cv2.INTER_AREA)
    if mode == "pixelate":
        frame[y1:y2, x1:x2] = cv2.resize(small, (roi.shape[1], roi.shape[0]), interpolation=cv2.INTER_NEAREST)
    else:
        up = cv2.resize(small, (roi.shape[1], roi.shape[0]), interpolation=cv2.INTER_CUBIC)
        k = max(3, (min(roi.shape[:2]) // 6) | 1)
        frame[y1:y2, x1:x2] = cv2.GaussianBlur(up, (k, k), 0)


def _brackets(img, x1, y1, x2, y2, color, t=2):
    L = max(6, min(22, (x2 - x1) // 4, (y2 - y1) // 4))
    for (px, py, dx, dy) in ((x1, y1, 1, 1), (x2, y1, -1, 1), (x1, y2, 1, -1), (x2, y2, -1, -1)):
        cv2.line(img, (px, py), (px + dx * L, py), color, t, cv2.LINE_AA)
        cv2.line(img, (px, py), (px, py + dy * L), color, t, cv2.LINE_AA)


def render_view(frame, dets: list[Detection], mode: str, labels: dict,
                max_w: int = 1280, quality: int = 78):
    """Anonymise people in a (downscaled) copy and JPEG-encode it. Nothing is stored."""
    h, w = frame.shape[:2]
    if w > max_w:
        frame = cv2.resize(frame, (max_w, int(h * max_w / w)), interpolation=cv2.INTER_AREA)
    else:
        frame = frame.copy()
    h, w = frame.shape[:2]
    boxes = []
    for d in dets:
        x1, y1, x2, y2 = d.box
        # pad 6% so hair / shoulders at the box edge are also covered
        pw, ph = (x2 - x1) * 0.06, (y2 - y1) * 0.04
        X1, Y1 = max(0, int((x1 - pw) * w)), max(0, int((y1 - ph) * h))
        X2, Y2 = min(w, int((x2 + pw) * w)), min(h, int((y2 + ph) * h))
        _anonymise(frame, X1, Y1, X2, Y2, mode)
        boxes.append((d, X1, Y1, X2, Y2))
    white = (240, 239, 236)
    for d, X1, Y1, X2, Y2 in boxes:
        _brackets(frame, X1, Y1, X2, Y2, white, 2)
        fx, fy = int(d.x * w), int(d.y * h)
        cv2.circle(frame, (fx, fy), 5, (20, 20, 20), -1, cv2.LINE_AA)
        cv2.circle(frame, (fx, fy), 4, (229, 135, 57), -1, cv2.LINE_AA)
        lbl = labels.get(d.track_id)
        if lbl:
            fs = 0.5
            (tw, th), _ = cv2.getTextSize(lbl, cv2.FONT_HERSHEY_SIMPLEX, fs, 1)
            bx, by = X1, max(th + 10, Y1 - 6)
            cv2.rectangle(frame, (bx, by - th - 8), (bx + tw + 12, by + 2), (22, 22, 22), -1)
            cv2.putText(frame, lbl, (bx + 6, by - 3), cv2.FONT_HERSHEY_SIMPLEX, fs, white, 1, cv2.LINE_AA)
    ok, buf = cv2.imencode(".jpg", frame, [cv2.IMWRITE_JPEG_QUALITY, quality])
    return buf.tobytes() if ok else None
