"""
Real-time anomaly detection streaming server (MJPEG over HTTP).

Usage:
    python infer_stream.py \
        --input  ./input.mp4 \
        --msflow_ckpt work_dirs/.../last.pt \
        --rf_ckpt     work_dirs/.../rf_last.pt

Then open http://localhost:5000 in your browser.
"""
from __future__ import annotations

import argparse
import datetime
import os
import threading
import time
from typing import List

import av
import cv2
import numpy as np
import torch
from flask import Flask, Response, render_template_string
from torchvision import transforms as T
from torchvision.transforms import InterpolationMode

import default as c
from rectified_flow_train_posco import msflow_forward, rf_transport, minmax_norm
from visualize_bboxes import (
    build_msflow, build_rf_from_batch,
    anomaly_map_to_bboxes, get_final_localization_map,
)

# ── constants ────────────────────────────────────────────────────────────────
IMG_MEAN = [0.485, 0.456, 0.406]
IMG_STD  = [0.229, 0.224, 0.225]
_MEAN    = torch.tensor(IMG_MEAN).view(3, 1, 1)
_STD     = torch.tensor(IMG_STD).view(3, 1, 1)

# ── HTML page ────────────────────────────────────────────────────────────────
HTML = """
<!DOCTYPE html>
<html>
<head>
  <title>MSFlow Anomaly Detection</title>
  <style>
    body { background:#111; color:#eee; font-family:monospace;
           display:flex; flex-direction:column; align-items:center; padding:20px; }
    h2   { margin-bottom:8px; }
    img  { border:2px solid #444; max-width:100%; }
    #stats { margin-top:12px; font-size:14px; color:#aef; }
  </style>
</head>
<body>
  <h2>MSFlow + RF — Live Anomaly Detection</h2>
  <img src="/video_feed" />
  <div id="stats">Connecting…</div>
  <script>
    // Poll /stats every second and update the overlay
    setInterval(() => {
      fetch('/stats').then(r => r.json()).then(d => {
        document.getElementById('stats').textContent =
          `Inference FPS: ${d.fps.toFixed(1)}  |  Frames processed: ${d.frames}  |  Latency: ${d.latency_ms.toFixed(1)} ms`;
      });
    }, 1000);
  </script>
</body>
</html>
"""

# ── global state shared between inference thread and Flask ───────────────────
_frame_lock   = threading.Lock()
_latest_frame: bytes | None = None   # JPEG bytes of the most recent annotated frame

_stats_lock   = threading.Lock()
_stats = {"fps": 0.0, "frames": 0, "latency_ms": 0.0}


def set_latest_frame(jpeg_bytes: bytes):
    global _latest_frame
    with _frame_lock:
        _latest_frame = jpeg_bytes


def get_latest_frame() -> bytes | None:
    with _frame_lock:
        return _latest_frame


def update_stats(fps: float, frames: int, latency_ms: float):
    with _stats_lock:
        _stats["fps"]        = fps
        _stats["frames"]     = frames
        _stats["latency_ms"] = latency_ms


# ── preprocessing helpers ────────────────────────────────────────────────────
def build_preprocess(input_size):
    # T.Resize and T.Normalize work on both PIL images and float tensors
    return T.Compose([
        T.Resize(input_size, InterpolationMode.BILINEAR),
        T.Normalize(IMG_MEAN, IMG_STD),
    ])


def frame_bgr_to_tensor(frame_bgr: np.ndarray, preprocess) -> torch.Tensor:
    # BGR → RGB, HWC uint8 → CHW float tensor, then normalize+resize
    rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
    t = torch.from_numpy(rgb).permute(2, 0, 1).float().div(255.0)  # (3,H,W) float
    return preprocess(t)  # Resize + Normalize work on tensors too


def tensor_to_bgr_uint8(t: torch.Tensor) -> np.ndarray:
    img = ((t.cpu() * _STD + _MEAN).clamp(0, 1) * 255).byte()
    return cv2.cvtColor(img.permute(1, 2, 0).numpy(), cv2.COLOR_RGB2BGR)


def draw_bboxes_cv2(frame: np.ndarray, bboxes,
                    color=(0, 0, 255), thickness=3) -> np.ndarray:
    out = frame.copy()
    for x0, y0, x1, y1 in bboxes:
        cv2.rectangle(out, (x0, y0), (x1, y1), color, thickness)
    return out


def encode_jpeg(frame_bgr: np.ndarray, quality: int = 80) -> bytes:
    ok, buf = cv2.imencode('.jpg', frame_bgr, [cv2.IMWRITE_JPEG_QUALITY, quality])
    return buf.tobytes() if ok else b''


# ── inference thread ─────────────────────────────────────────────────────────
def open_container(path: str):
    """Open a video file or RTSP stream via PyAV."""
    is_rtsp = path.startswith('rtsp://') or path.startswith('rtsps://')
    options = {'rtsp_transport': 'tcp'} if is_rtsp else {}
    container = av.open(path, options=options)
    return container, is_rtsp


def iter_frames(container) -> np.ndarray:
    """Yield decoded BGR frames from a PyAV container."""
    stream = container.streams.video[0]
    stream.thread_type = 'AUTO'   # enable multi-threaded decoding
    for packet in container.demux(stream):
        for av_frame in packet.decode():
            yield av_frame.to_ndarray(format='bgr24')


# sentinel to signal decode thread termination
_DECODE_DONE = object()


def decode_thread_fn(args, preprocess, frame_queue: "queue.Queue", is_rtsp: bool):
    """Continuously decodes frames and puts tensors into frame_queue."""
    import queue as _queue
    while True:
        container, _ = open_container(args.input)
        for frame_bgr in iter_frames(container):
            tensor = frame_bgr_to_tensor(frame_bgr, preprocess)
            # block if inference can't keep up (max 32 frames buffered)
            frame_queue.put(tensor)
        container.close()
        if is_rtsp:
            break
    frame_queue.put(_DECODE_DONE)


def inference_loop(args, device):
    """Producer-consumer: decode thread fills queue, inference drains it."""
    import queue as _queue

    # ── model init ───────────────────────────────────────────────────────────
    extractor, parallel_flows, fusion_flow = build_msflow(c, args.msflow_ckpt)
    preprocess = build_preprocess(c.input_size)

    # peek at first frame to get channel dims for rf_model
    container, is_rtsp = open_container(args.input)
    stream  = container.streams.video[0]
    src_fps = float(stream.average_rate) if stream.average_rate else 0.0
    print(f"[INFO] Source: {args.input}  ({stream.frames} frames @ {src_fps:.1f} fps)")
    frame0  = next(iter_frames(container))
    container.close()

    dummy = frame_bgr_to_tensor(frame0, preprocess).unsqueeze(0).to(device)
    with torch.no_grad():
        _, z_fused_list, _ = msflow_forward(
            c, extractor, parallel_flows, fusion_flow, dummy, return_pre_fusion=True)
    rf_model = build_rf_from_batch(
        device, args.rf_ckpt, z_fused_list, args.rf_tdims, args.rf_depths)

    print("[INFO] Models ready. Starting inference…")

    # ── start decode thread ──────────────────────────────────────────────────
    frame_queue: _queue.Queue = _queue.Queue(maxsize=32)
    dt = threading.Thread(
        target=decode_thread_fn,
        args=(args, preprocess, frame_queue, is_rtsp),
        daemon=True,
    )
    dt.start()

    # ── inference loop ───────────────────────────────────────────────────────
    processed    = 0
    t_loop_start = time.time()
    batch_tensors: List[torch.Tensor] = []

    def flush_batch():
        nonlocal processed
        if not batch_tensors:
            return
        t0   = time.time()
        imgs = torch.stack(batch_tensors).to(device, non_blocking=True)
        final_maps = get_final_localization_map(
            c, extractor, parallel_flows, fusion_flow,
            rf_model, imgs, args.rf_steps,
        )
        latency_ms = (time.time() - t0) * 1000 / len(batch_tensors)

        for b in range(len(batch_tensors)):
            amap = final_maps[b]
            if torch.is_tensor(amap):
                amap = amap.detach().cpu().numpy()
            frame_512 = tensor_to_bgr_uint8(batch_tensors[b])
            bboxes    = anomaly_map_to_bboxes(amap, threshold=args.threshold, min_area=args.min_area)
            annotated = draw_bboxes_cv2(frame_512, bboxes)
            set_latest_frame(encode_jpeg(annotated, quality=args.jpeg_quality))

        processed += len(batch_tensors)
        elapsed    = time.time() - t_loop_start
        update_stats(processed / max(elapsed, 1e-6), processed, latency_ms)
        batch_tensors.clear()

    while True:
        try:
            item = frame_queue.get(timeout=5.0)
        except Exception:
            continue

        if item is _DECODE_DONE:
            flush_batch()
            break

        batch_tensors.append(item)
        if len(batch_tensors) == args.batch_size:
            flush_batch()

    print("[INFO] Inference thread finished.")


# ── Flask app ────────────────────────────────────────────────────────────────
app = Flask(__name__)


@app.route('/')
def index():
    return render_template_string(HTML)


@app.route('/video_feed')
def video_feed():
    def generate():
        while True:
            frame = get_latest_frame()
            if frame is None:
                time.sleep(0.01)
                continue
            yield (b'--frame\r\n'
                   b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')
            time.sleep(0.01)   # ~100 fps ceiling for the HTTP stream

    return Response(generate(),
                    mimetype='multipart/x-mixed-replace; boundary=frame')


@app.route('/stats')
def stats():
    import json
    with _stats_lock:
        return app.response_class(
            response=json.dumps(_stats),
            mimetype='application/json',
        )


# ── entry point ──────────────────────────────────────────────────────────────
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--input',        type=str, default="posco/CH001_01.avi")
    parser.add_argument('--msflow_ckpt',  type=str,
                        default='work_dirs/msflow_wide_resnet50_2_avgpool_pl258/posco/posco/last.pt')
    parser.add_argument('--rf_ckpt',      type=str,
                        default='work_dirs/rf_on_msflow_wide_resnet50_2_avgpool_pl258/posco/posco/rf_last.pt')
    parser.add_argument('--threshold',    type=float, default=2.5)
    parser.add_argument('--min_area',     type=int,   default=80)
    parser.add_argument('--batch_size',   type=int,   default=8)
    parser.add_argument('--rf_steps',     type=int,   default=1)
    parser.add_argument('--gpu',          type=str,   default='0')
    parser.add_argument('--rf-tdims',     type=int,   nargs='+', default=[128, 128])
    parser.add_argument('--rf-depths',    type=int,   nargs='+', default=[3, 3])
    parser.add_argument('--port',         type=int,   default=5000)
    parser.add_argument('--jpeg_quality', type=int,   default=80,
                        help='JPEG quality for streaming (1-100)')
    args = parser.parse_args()

    os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu
    c.device          = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    c.input_size      = (512, 512)
    c.img_mean        = IMG_MEAN
    c.img_std         = IMG_STD
    c.extractor       = 'wide_resnet50_2'
    c.pool_type       = 'avg'
    c.parallel_blocks = [2, 5, 8]
    c.c_conds         = [64, 64, 64]
    c.clamp_alpha     = 1.9

    is_rtsp = args.input.startswith("rtsp://")
    if not is_rtsp:
        assert os.path.isfile(args.input), f"Input not found: {args.input}"
    assert os.path.isfile(args.msflow_ckpt), f"MSFlow ckpt not found: {args.msflow_ckpt}"
    assert os.path.isfile(args.rf_ckpt),     f"RF ckpt not found: {args.rf_ckpt}"

    # inference runs in a background thread; Flask serves in the main thread
    t = threading.Thread(target=inference_loop, args=(args, c.device), daemon=True)
    t.start()

    print(f"[INFO] Open http://localhost:{args.port} in your browser")
    app.run(host='0.0.0.0', port=args.port, threaded=True)


if __name__ == '__main__':
    main()