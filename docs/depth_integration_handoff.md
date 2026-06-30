# Handoff: Unitree G1 head-cam DEPTH stream — wire spec + integration guide

**For:** the agent integrating depth into the ZMQ image client in the other repo.
**Context:** The G1 image_server running on the robot's PC2 publishes a head-camera **depth** stream over ZMQ on a dedicated port. This was verified empirically against the live robot (PC2 IP `192.168.123.164`) on 2026-06-23. The depth publisher exists on PC2 only — the vendored/on-disk `image_server.py` in our repos has the depth *capture* scaffolding but no publish thread, so don't expect the source to show it. Trust the wire probe.

> **STATUS (2026-06-29 update): DONE — depth is integrated AND consumed.** The client-side
> subscriber described below is built (`image_server/image_client.py`: zmq-backend 2nd SUB +
> `get_depth_frame()`), and depth now drives a real downstream consumer: the **depth-ESDF
> collision world** (`g1_classical_manip/motion/collision_world.py`: `EsdfMapper` → cuRobo
> `Mapper` → ESDF `VoxelGrid`), fed to the native `plan_grasp` so the grasp **approach** routes
> around the object/table. A live cuRobo `RobotSegmenter` self-filter (hand-active, masks the
> robot at its measured arm+finger q) keeps the robot's own arm out of the fused world
> (`CuroboArmPlanner.robot_depth_filter` / `update_grasp_world` in `motion/curobo_planner.py`,
> gated `configs/planner.yaml: grasp.collision_world`, default OFF). Inspect it in isolation with
> `scripts/12_check_world.py` (the new top rung of the bring-up ladder). One open issue surfaced:
> the dynamic approach route can outrun the controller's tracking-error abort at full-speed
> playback — see `docs/trajectory_speed_tracking.md` (workaround `09_graspgen --speed 0.5`).
>
> The wire spec / probe / integration guidance below is the original handoff and remains accurate
> as the **real-robot** port (`56555`, ZED, 720×1280). The **sim** path (Isaac `front_camera`
> `distance_to_image_plane`) is the same code on a separate ZMQ PUB **`:55556`** at 480×640 mm —
> see `configs/camera_sim.yaml: stream.depth_port`.

---

## TL;DR wire spec (head depth)

| Property | Value |
| --- | --- |
| Transport | ZMQ **PUB/SUB**, subscribe-all (`SUBSCRIBE=""`), single-part `send()`/`recv()` |
| Host | PC2 IP (robot), e.g. `192.168.123.164` |
| **Depth port** | **`56555`** (dedicated; color is on a different port) |
| Payload | raw `np.float32` from `.tobytes()` — **3,686,400 bytes/frame**, fixed size |
| Shape | **`(720, 1280)`** — single depth map (NOT binocular-stitched) |
| Units | **millimeters** |
| Invalid pixels | **NaN / inf** (~12% typical: sky / occlusion / out-of-range) — must be masked |
| Rate | ~30 fps, cadence-aligned with the color stream |

**Decode is one line:**
```python
depth_mm = np.frombuffer(buf, dtype=np.float32).reshape(720, 1280)  # NaN/inf == invalid
```

### Companion color stream (for reference / alignment)
| Port | Payload | Shape |
| --- | --- | --- |
| `55555` | JPEG (well-formed, ~415 KB, variable size = compressed) | binocular **720×2560** (left+right side-by-side) |
| `55556` / `55557` | wrist cams (JPEG) | 480×640 each (were silent/disconnected during probe) |

**Resolution mismatch — important for RGBD alignment:** color is binocular 720×2560, depth is a single 720×1280 map = **one eye** (almost certainly the **left**). For pixel-correspondence, pair depth against the **left half** of the color frame (`color[:, :1280]`), not the full stitch.

---

## How to repeat the probe (do this first to confirm against the target robot)

The probe matches the client protocol exactly (SUB, `SUBSCRIBE=""`, `RCVHWM=1`, `LINGER=0`, plain `connect`) and uses `recv_multipart()` so it can't miss a multipart payload. Requires only `pyzmq` + `numpy`.

### Step 1 — enumerate publishing ports & detect format
Save as `zmq_probe.py` and run `python zmq_probe.py --host <PC2_IP>`:

```python
#!/usr/bin/env python3
"""Probe Unitree image_server ZMQ PUB ports: liveness, parts/msg, byte size, format, fps."""
import argparse, time, zmq

KNOWN_SHAPES = {"head_mono_720x1280": (720, 1280),
                "head_binoc_720x2560": (720, 2560),
                "wrist_480x640": (480, 640)}
DEFAULT_PORTS = [55555, 55556, 55557, 56555]

def detect_format(buf: bytes) -> str:
    n = len(buf)
    if n >= 3 and buf[:3] == b"\xff\xd8\xff":
        return "JPEG" + (" (well-formed)" if buf[-2:] == b"\xff\xd9" else " (truncated?)")
    if n >= 4 and buf[:4] == b"\x89PNG":
        return "PNG (likely compressed depth)"
    matches = []
    for name, (h, w) in KNOWN_SHAPES.items():
        if n == h*w*4: matches.append(f"raw float32 {name} ({h}x{w}x4)")
        if n == h*w*2: matches.append(f"raw uint16/z16 {name} ({h}x{w}x2)")
        if n == h*w:   matches.append(f"raw mono8 {name} ({h}x{w})")
        if n == h*w*3: matches.append(f"raw bgr8 {name} ({h}x{w}x3)")
    return "RAW -> " + " | ".join(matches) if matches else f"unknown (first bytes {buf[:8].hex()})"

def probe_port(ctx, host, port, frames, timeout_ms):
    s = ctx.socket(zmq.SUB); s.setsockopt(zmq.RCVHWM, 1); s.setsockopt(zmq.LINGER, 0)
    s.connect(f"tcp://{host}:{port}"); s.setsockopt_string(zmq.SUBSCRIBE, "")
    poller = zmq.Poller(); poller.register(s, zmq.POLLIN)
    samples, deadline = [], time.time() + timeout_ms/1000.0
    while len(samples) < frames and time.time() < deadline:
        if s in dict(poller.poll(timeout=timeout_ms)): samples.append((time.time(), s.recv_multipart()))
        else: break
    s.close()
    if not samples:
        print(f"  port {port}: SILENT ({timeout_ms} ms)"); return
    fps = (len(samples)-1)/(samples[-1][0]-samples[0][0]) if len(samples) >= 2 and samples[-1][0] > samples[0][0] else None
    parts = samples[0][1]
    print(f"  port {port}: PUBLISHING | {len(samples)} frame(s) | {len(parts)} part(s)/msg | {f'{fps:.1f} fps' if fps else 'fps n/a'}")
    for i, p in enumerate(parts):
        print(f"      part[{i}]: {len(p):>9,d} bytes | {detect_format(p)}")
    sizes = [sum(len(p) for p in pr) for _, pr in samples]
    print(f"      total-bytes {'varies '+format(min(sizes),',')+'..'+format(max(sizes),',')+' (compressed)' if len(set(sizes))>1 else 'constant '+format(sizes[0],',')+' (raw/fixed)'}")

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--host", required=True); ap.add_argument("--ports", type=int, nargs="*", default=DEFAULT_PORTS)
    ap.add_argument("--scan", type=int, nargs=2, metavar=("LO","HI"))
    ap.add_argument("--frames", type=int, default=20); ap.add_argument("--timeout", type=int, default=2000)
    a = ap.parse_args()
    ports = list(range(a.scan[0], a.scan[1]+1)) if a.scan else a.ports
    print(f"Probing {a.host} on {len(ports)} port(s)\n")
    ctx = zmq.Context()
    try:
        for p in ports: probe_port(ctx, a.host, p, a.frames, a.timeout)
    finally: ctx.term()

if __name__ == "__main__": main()
```

Expected output for a depth-enabled robot:
```
  port 55555: PUBLISHING | 20 frame(s) | 1 part(s)/msg | 30.1 fps
      part[0]:   415,020 bytes | JPEG (well-formed)
      total-bytes varies 414,298..415,185 (compressed)
  port 56555: PUBLISHING | 20 frame(s) | 1 part(s)/msg | 29.8 fps
      part[0]: 3,686,400 bytes | RAW -> raw float32 head_mono_720x1280 (720x1280x4) | raw uint16/z16 head_binoc_720x2560 (720x2560x2)
      total-bytes constant 3,686,400 (raw/fixed)
```
Note `3,686,400` matches **both** float32@720×1280 and uint16@720×2560 — so disambiguate with step 2.

### Step 2 — confirm dtype = float32 / units = mm
```python
import zmq, numpy as np
ctx = zmq.Context(); s = ctx.socket(zmq.SUB)
s.setsockopt(zmq.RCVHWM,1); s.setsockopt(zmq.LINGER,0)
s.connect("tcp://<PC2_IP>:56555"); s.setsockopt_string(zmq.SUBSCRIBE, "")
poller = zmq.Poller(); poller.register(s, zmq.POLLIN)
buf = s.recv() if dict(poller.poll(3000)).get(s) else None
f = np.frombuffer(buf, dtype=np.float32); finite = np.isfinite(f)
print(f.size == 720*1280, finite.mean(), f[finite].min(), np.median(f[finite]), f[finite].max())
# Expect: True, ~0.88, ~166, ~1258, ~3164  -> 921,600 float32 values, ~12% NaN/inf, values in mm
```
float32 is confirmed by: exact 921,600 count, presence of NaN/inf (raw uint16 has none), and finite values in a sane mm range. The uint16 reinterpretation gives saturated noise — wrong.

---

## Integration guidance for the client

**[DONE 2026-06-29]** All steps below are implemented in `image_server/image_client.py` (zmq
backend: 2nd SUB on `depth_port`, single-part `recv()`, `np.frombuffer(..., float32)` decode,
optional `get_depth_frame()` returning mm with NaN/inf preserved, graceful absence via a
`depth_port`/`depth=` tri-state gate). Kept here as the spec of record. The depth path mirrors the
existing color path — same SUB socket pattern, just a different port and decode. Concretely:

1. **Config:** the depth port already lives in `cam_config_client.yaml` under `head_camera` as `zmq_depth_port: 56555` (and `enable_depth: true`). Read it from there; don't hardcode.
2. **Subscriber:** add a SUB subscriber thread on `zmq_depth_port` exactly like the color subscriber (RCVHWM=1, LINGER=0, SUBSCRIBE=""). It's a single-part `recv()` — no multipart handling needed.
3. **Decode:** `np.frombuffer(buf, np.float32).reshape(720, 1280)`. Do NOT JPEG-decode this path.
4. **Frame object:** add a `depth` field alongside the existing color fields (e.g. extend the per-camera image struct / `__slots__`). Keep it optional so non-depth robots/ports degrade gracefully (subscriber returns nothing → leave depth `None`).
5. **Invalid mask:** NaN/inf are invalid. Decide a policy before feeding a model — e.g. `np.nan_to_num(depth, nan=0, posinf=0, neginf=0)` plus a separate validity mask, or clip to a working range. Don't silently pass NaNs into a network.
6. **Units:** values are **millimeters**. Normalize/scale to whatever the consumer expects (meters = `/1000`).
7. **RGBD alignment:** depth is the **left eye** at 720×1280; color is binocular 720×2560. Use `color[:, :1280]` for correspondence. Confirm left-vs-right empirically (wave a hand) before relying on it.
8. **Graceful absence:** wrist ports and even depth may be silent on a given robot/session (wrist cams were disconnected during this probe). Treat a silent port as "feature off," not an error.

## Caveats
- Verified on one robot/session (PC2 `192.168.123.164`, 2026-06-23). Re-run the probe against the actual target robot before integrating — port and format are stable in the config but the *publisher* lives on PC2 and may differ by robot firmware/setup.
- The head cam also advertises WebRTC (`enable_webrtc: true`, h264, ports ~60001–60003). This depth stream is the **ZMQ** path; if a robot lacks the ZMQ depth publisher, depth may instead be on WebRTC — out of scope here.
- "left eye" for the depth↔color mapping is inferred, not confirmed — verify visually.
