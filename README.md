# FlowLens

**See where queues form. Never who is in them.**

FlowLens is privacy-first people analytics for events, retail and venues. Each camera runs YOLO11 person detection on its own computer and counts:

- entries and exits at doors and lines
- queue lengths and waits
- dwell time and occupancy per zone

A **FlowLens Hub** combines any number of cameras into one site dashboard. Video never leaves the camera computer. Only counts and aggregates do.

- **Camera node** (`app.py`): live anonymised view, draw doors, lines and zones, analytics, CSV logs.
- **Hub** (`hub/hub.py`): people inside the site, areas, queues, movement between areas, alerts, trends, an optional floor plan.
- **Browser demo** (`docs/demo/`): try it with a webcam, running entirely in the browser.
- **Landing page** (`docs/index.html`): published with GitHub Pages.

```
 camera computer(s)                                      hub (event LAN or cloud)
 ┌───────────────────────────────┐   counts, waits,     ┌────────────────────────────┐
 │ webcam / IP camera            │   heat grid, events  │ FlowLens Hub               │
 │  → YOLO11 person + ByteTrack  │ ───────────────────► │  site KPIs · areas · queues│
 │  → doors · lines · zones      │   every 2 s + per-   │  flows · alerts · trends   │
 │  → local dashboard + CSV      │   minute records     │  floor plan (optional)     │
 │ video never leaves this box   │   (buffered offline) │  SQLite · CSV export       │
 └───────────────────────────────┘                      └────────────────────────────┘
```

---

## 1. Install (Windows, once per computer)

```powershell
cd path\to\flowlens
py -m venv .venv
.\.venv\Scripts\Activate.ps1          # if blocked: Set-ExecutionPolicy -Scope CurrentUser RemoteSigned
pip install -r requirements.txt
```

**NVIDIA GPU (recommended):** `pip install torch torchvision --index-url https://download.pytorch.org/whl/cu124 --force-reinstall`. FlowLens uses the GPU automatically, with FP16.

## 2. Single camera

```powershell
python app.py                      # webcam 0 → http://localhost:8000
python app.py --source 1           # USB camera
python app.py --source rtsp://user:pass@192.168.1.31/stream1   # IP camera
```

Draw on the live view:

- **Door**: outline a doorway or frame edge. A person first seen there counts as *Entered*; a person last seen there counts as *Left*. This works for very fast walkers.
- **Line**: draw it across an aisle. A crossing counts when the person changes side outside a dead-band.
- **Zone**: outline a queue, service point or area to measure occupancy and dwell. Set **Count a person by their** to *Body centre* for seated people or desks, where the feet are hidden or out of frame. *Feet* (the default) suits standing queues.

Set each door and line's **Counts as** role:

| Role | Use for | Feeds |
|---|---|---|
| **Site entrance** | the real ways in and out of the venue or store | Entries, Exits, *Inside now* |
| **Between areas** (From → To) | doorways inside the site, e.g. Lobby → Hall A | movement between areas, area flows |
| **Ignore** | anything you only want to watch | nothing |

Give queue and area zones an **Area** name (Hall A, Checkout, Reception…) so the hub can group them.

## 3. Many cameras + hub (events, multi-room stores)

**On the hub computer:**

```powershell
python hub\hub.py --event "Auto Expo 2026"          # or double-click start-hub.bat
```

It prints the dashboard address (`http://localhost:9000`), the network address cameras should use, and an **access token**. Both are also under **Settings** on the hub page.

**On each camera computer**, open FlowLens → **Site hub**, enter a name, the hub address and the token, tick *Send data to the hub*, and save. Or start it with:

```powershell
python app.py --name "Hall A - entrance" --hub http://192.168.1.20:9000 --hub-token <token>
```

**Several IP cameras on one machine:** give each its own port and data folder. `deploy/event-4-cameras.example.bat` is a ready template.

```powershell
python app.py --source rtsp://.../cam1 --port 8001 --data cams\cam1 --name "Main entrance" --hub ... --hub-token ...
python app.py --source rtsp://.../cam2 --port 8002 --data cams\cam2 --name "Hall A"        --hub ... --hub-token ...
```

As a rule of thumb, an RTX laptop GPU runs about 4–6 cameras of `yolo11n` at 15+ fps each.

### Hub dashboard

| Panel | What it shows |
|---|---|
| KPIs | **Inside now** (site entrances in − out), entries, exits, people seen by cameras, longest queue wait, active alerts |
| Site map / Areas | the floor plan with a live heatmap, camera coverage, zones and lines, *or* (no plan) a board of areas with occupancy and capacity bars |
| Cameras | online / stale / offline, fps, CPU or GPU, people in view |
| Alerts | camera offline or failing, slow detection, queue waits and capacities over their limits |
| Queues | every queue on every camera, ranked by longest wait |
| Movement between areas | Outside → Entrance, Entrance → Hall A… since the event started |
| Trends | people inside, entries/exits, occupancy by area, people in queues (1 h – 7 days) |
| Export | event-wide CSV (per minute) and per-camera CSV |

### Floor plan (optional)

1. Hub → **Settings → Floor plan** and upload any image: a venue plan, a store layout, a booth map or a sketch.
2. On each camera page → **Floor plan mapping** → click 4 spots on the floor in the camera view, then the same 4 spots on the plan. Use more points for better accuracy.
3. That camera's heatmap, zones and lines now appear in place on the site map. Cameras you can't map can be dropped as markers with **Place cameras**.

Without a floor plan, everything else works and the hub shows the areas board.

### Reliability

- Each camera keeps working on its own if the hub or network is down. Per-minute records queue in `logs/outbox.jsonl` and upload when the hub is back.
- Camera disconnects reconnect automatically. Statistics survive restarts (`logs/state.json`), and daily CSVs are written on every node.
- Hub: SQLite with 30 days of history. **Start new event** resets the counts without deleting history. Cameras authenticate with a token. Other devices that open the dashboard need a PIN, which is generated automatically and printed at startup (or set it with `--pin 4821`). The hub computer itself never needs the PIN. Floor plans accept PNG, JPG or WebP.
- Each node's controls are reachable only from its own computer. Starting a second node on a port that is already in use stops with a clear message.
- Counting is protected against double counting by the counter roles. People are never re-identified across cameras.

## 4. Browser demo

`docs/demo/index.html` runs YOLO11n (ONNX, 320 px) in the browser with ONNX Runtime Web, using WebGPU when available and WebAssembly otherwise. It has one door, one line and one queue, and you can drag their corners. It works with a webcam or a local video file. Nothing is uploaded. It is served by GitHub Pages at `https://<you>.github.io/flowlens/demo/`.

## Privacy contract

- Person detection only. No face recognition, no appearance features. Lost tracks are re-linked by motion (position and speed).
- No recording. Frames live in RAM for one loop iteration. The live view anonymises people (blur, pixelate or mask) **before** encoding, and it is served to `127.0.0.1` only by default.
- Nodes save and send aggregates only: counts, waits, a fading 64×36 heat grid, zone geometry, and anonymous event text such as "Main door: entered".
- The hub stores only those aggregates. Optional *hide counts below k* suppresses small counts.

## Files

| Path | Role |
|---|---|
| `app.py` | camera node: CLI, inference loop, API, config, CSV logging, floor mapping |
| `vision.py` | camera grabber, YOLO detector and tracker, anonymised view |
| `analytics.py` | counting engine: smoothing, re-linking, doors, lines, zones, alerts, series |
| `node_link.py` | uplink to the hub (live + store-and-forward minutes) |
| `tracker.yaml` | ByteTrack tuning |
| `static/index.html` | camera dashboard |
| `hub/hub.py`, `hub/static/index.html` | hub service and dashboard |
| `docs/` | landing page and browser demo (GitHub Pages) |
| `start-hub.bat`, `start-camera.bat`, `deploy/` | Windows launchers and an event template |

## License

FlowLens uses [Ultralytics YOLO](https://github.com/ultralytics/ultralytics) (AGPL-3.0), so FlowLens is released under **AGPL-3.0** (see `LICENSE`). Closed-source commercial use needs an Ultralytics Enterprise licence, or a detector with a permissive licence.

---
Built as Day 19 of a daily AI build series.
