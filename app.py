"""Gradio demo: webcam streaming, image uploads, and short video replay.

Baseline mode is a non-ML visual demonstration. CNN mode requires a model
trained by this project's train.py and exported to models/steering.onnx.
"""
from __future__ import annotations

import os
from functools import lru_cache
from pathlib import Path
from tempfile import NamedTemporaryFile

import cv2
import gradio as gr
import numpy as np
from PIL import Image

from src.data import OUTPUT_SCALE_DEGREES, image_to_tensor
from src.vision import draw_dashboard, estimate_road_angle

ROOT = Path(__file__).resolve().parent
MODEL_FILE = ROOT / "models" / "steering.onnx"
SAMPLE_IMAGE = ROOT / "demo" / "road-straight.jpg"
SAMPLE_RESULT = ROOT / "demo" / "annotated-straight.jpg"
SAMPLE_VIDEO = ROOT / "demo" / "road-sample.mp4"
SAMPLE_REPLAY = ROOT / "demo" / "road-annotated.mp4"
BASELINE = "Lane geometry (untrained illustration)"
TRAINED = "CNN steering prediction (trained model)"


@lru_cache(maxsize=1)
def load_model():
    if not MODEL_FILE.is_file():
        return None, "No trained CNN found. Train your model and add models/steering.onnx."
    try:
        import onnxruntime as ort
        session = ort.InferenceSession(str(MODEL_FILE), providers=["CPUExecutionProvider"])
        return session, None
    except Exception as exc:
        return None, f"Could not load the ONNX model: {type(exc).__name__}: {exc}"


def estimate(frame: np.ndarray, mode: str) -> tuple[float | None, str]:
    if mode == TRAINED:
        session, message = load_model()
        if session is None:
            return None, message
        try:
            arr = image_to_tensor(Image.fromarray(frame.astype(np.uint8), mode="RGB"))
            value = session.run(None, {session.get_inputs()[0].name: arr})[0]
            angle = float(value.reshape(-1)[0]) * OUTPUT_SCALE_DEGREES
            if not np.isfinite(angle):
                return None, "Model returned a non-finite result"
            return angle, "Trained CNN prediction (not a vehicle control signal)"
        except Exception as exc:
            return None, f"Inference failed: {type(exc).__name__}: {exc}"
    return estimate_road_angle(frame)


def run_frame(frame: np.ndarray | None, mode: str):
    if frame is None:
        return None, "Choose a source image or enable your webcam."
    if frame.ndim == 2:
        frame = np.stack([frame] * 3, axis=-1)
    frame = np.ascontiguousarray(frame[:, :, :3], dtype=np.uint8)
    angle, context = estimate(frame, mode)
    output = draw_dashboard(frame, angle, context)
    detail = ("**No numerical estimate available.** " + context if angle is None else
              f"**{angle:+.2f}°** — {context}")
    return output, detail


def replay_video(video_path: str | None, mode: str):
    if not video_path:
        return None, "Upload a short video first."
    if mode == TRAINED:
        model, message = load_model()
        if model is None:
            return None, message
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        return None, "Could not open the uploaded video."
    temp_path = None
    writer = None
    try:
        fps = cap.get(cv2.CAP_PROP_FPS)
        if not 1 <= fps <= 120:
            fps = 25.0
        temp = NamedTemporaryFile(suffix=".mp4", delete=False)
        temp_path = temp.name
        temp.close()
        total = 0
        predicted = 0
        while total < 300:  # duration/compute cap: never process an unbounded video
            ok, bgr = cap.read()
            if not ok:
                break
            h, w = bgr.shape[:2]
            scale = min(1.0, 720.0 / max(w, 1))
            new_w = max(360, int(w * scale))
            new_h = max(240, int(h * scale))
            new_w += new_w % 2
            new_h += new_h % 2
            bgr = cv2.resize(bgr, (new_w, new_h))
            rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
            angle, context = estimate(rgb, mode)
            out = draw_dashboard(rgb, angle, context)
            output_bgr = cv2.cvtColor(out, cv2.COLOR_RGB2BGR)
            if writer is None:
                height, width = output_bgr.shape[:2]
                writer = cv2.VideoWriter(temp_path, cv2.VideoWriter_fourcc(*"mp4v"),
                                         fps, (width, height))
                if not writer.isOpened():
                    return None, "Video output codec unavailable on this server."
            writer.write(output_bgr)
            predicted += int(angle is not None)
            total += 1
        if total == 0:
            return None, "The uploaded video contained no readable frames."
    finally:
        cap.release()
        if writer is not None:
            writer.release()
    return temp_path, (f"Processed **{total} frames** (first 300 maximum); "
                       f"obtained estimates for **{predicted} frames**. "
                       f"Mode: {mode}. Demo-only; not validated for driving.")


with gr.Blocks(title="RoadSense | Steering Prediction") as demo:
    gr.Markdown("""
    # 🚘 RoadSense — Steering Prediction Lab
    Explore road-frame steering estimates using an **original CNN implementation** or a
    **transparent lane-geometry demonstration**. This is a research/education demo,
    **not a self-driving vehicle**. Never use it to control a car.

    **Training status:** The lane-geometry option works without training. The CNN
    requires **your own trained** `models/steering.onnx` file. Predictions from the
    lane-geometry option are illustrative and are **not neural-network predictions**.
    """)
    mode = gr.Radio([BASELINE, TRAINED], value=BASELINE, label="Prediction method")
    gr.Markdown("**Try it instantly:** Sample road images and an eight-second processed video are preloaded. They were generated synthetically and analysed with the untrained lane-geometry baseline. No real driving data or trained CNN is included.")
    with gr.Tabs():
        with gr.Tab("📷 Webcam stream"):
            with gr.Row():
                webcam = gr.Image(sources=["webcam"], type="numpy", streaming=True,
                                  label="Road camera (browser permission required)")
                annotated_stream = gr.Image(type="numpy", interactive=False,
                                            label="Steering visualisation")
            stream_status = gr.Markdown("Enable your webcam to start the preview.")
            webcam.stream(fn=run_frame, inputs=[webcam, mode],
                          outputs=[annotated_stream, stream_status],
                          stream_every=0.25, time_limit=120, concurrency_limit=1)
        with gr.Tab("🖼️ Single image"):
            with gr.Row():
                uploaded = gr.Image(value=str(SAMPLE_IMAGE), type="numpy", sources=["upload"], label="Upload a road image")
                image_output = gr.Image(value=str(SAMPLE_RESULT), type="numpy", interactive=False, label="Annotated result (example)")
            image_button = gr.Button("Analyse image", variant="primary")
            image_status = gr.Markdown("**Sample result: near 0°** — computed from synthetic lane lines (not machine learning). Click Analyse image or upload your own photograph.")
            gr.Examples(examples=[[str(ROOT / "demo" / f"road-{turn}.jpg")] for turn in ("left", "straight", "right")], inputs=uploaded, label="Try: left · straight · right")
            image_button.click(fn=run_frame, inputs=[uploaded, mode],
                               outputs=[image_output, image_status])
        with gr.Tab("🎬 Video replay"):
            gr.Markdown("Upload a short MP4 video. For resource control, only the first **300 frames** are processed.")
            video_input = gr.Video(value=str(SAMPLE_VIDEO), sources=["upload"], label="Road video (synthetic sample loaded)")
            video_button = gr.Button("Analyse video", variant="primary")
            video_output = gr.Video(value=str(SAMPLE_REPLAY), label="Processed video (already generated)")
            video_status = gr.Markdown("**Ready-to-watch example:** 160 synthetic frames processed by the lane-geometry baseline. Click Analyse video to run the processing pipeline yourself.")
            video_button.click(fn=replay_video, inputs=[video_input, mode],
                               outputs=[video_output, video_status])
    gr.Markdown("""
    ---
    **What is predicted?** Steering angle based on visual road information.
    A trained model cannot infer safe driving actions from a single frame, and
    no physical vehicle-control outputs are provided.

    **Design basis:** NVIDIA, *End to End Learning for Self-Driving Cars* (2016).
    See the README for data provenance, evaluation limitations and setup.
    """)


if __name__ == "__main__":
    demo.launch(
        server_name=os.environ.get("GRADIO_SERVER_NAME", "0.0.0.0"),
        server_port=int(os.environ.get("PORT", os.environ.get("GRADIO_SERVER_PORT", "7860"))),
        share=os.environ.get("GRADIO_SHARE", "0") == "1",
    )
