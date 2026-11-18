# RoadSense source-code guide

This guide is a **map of the executable source files**, not evidence that any one person wrote them. Read the source files in an editor or on GitHub; the ZIP is just a transport format.

## Where to find the code

| Project file | Language | What it implements | Corresponding part in the reference GitHub project |
| --- | --- | --- | --- |
| `src/network.py` | Python / PyTorch | `SteeringNet`, a convolutional neural network predicting a continuous steering angle | Model architecture (`model.py`) |
| `train.py` | Python / PyTorch | Data splits, training loop, validation, held-out evaluation, checkpoint saving, ONNX export | Training (`train.py`) |
| `src/data.py` | Python | Driving-image index parsing, crop/resize/normalise preprocessing | Data preparation (`driving_data.py`) |
| `app.py` | Python / Gradio | Browser interface; image, webcam and video processing | Model running and UI (`run.py`, `run_dataset.py`) |
| `src/vision.py` | Python / OpenCV | Lane-geometry heuristic and frame overlays (not ML) | Separate baseline; not equivalent to the CNN |
| `docs/sim.js` | JavaScript | Moving-car graphics, route-following logic, interaction handlers, steering display | Standalone, browser-only demo; not CNN inference |
| `docs/index.html` | HTML | Simulator dashboard, controls and canvas | Browser presentation |
| `docs/style.css` | CSS | Website visual style and responsive layout | Browser presentation |
| `docs/vision-replay.html` | HTML / JavaScript | Recorded frame-by-frame OpenCV baseline results | Recorded demonstration |
| `demo/generate_samples.py` | Python | Generates synthetic test roads; not actual driving evidence | Test-fixture generation |
| `tests/test_core.py` | Python | Tests model/data/vision fundamentals | Software checks |
| `tests/test_site_structure.py` | Python | Tests static site structure | Software checks |

`docs/road-annotated.mp4`, the JPG files and GIF previews are **demo media**, not the source implementation. `docs/angles.json` contains a recorded angle trace. `models/` has **no trained CNN checkpoint** in this starter; `data/` is not distributed.

## What happens in the working website

1. `docs/index.html` loads `docs/sim.js` and `docs/style.css`.
2. JavaScript builds road geometry and draws road, scenery and car in a Canvas.
3. The `requestAnimationFrame` loop updates the virtual distance and simulated steering angle.
4. A route-following controller adjusts the simulated lateral position; no neural network is used.
5. Choosing **Processed road replay** changes the view to recorded OpenCV output, with angles loaded from `angles.json` or an embedded trace.

## What happens if a real steering model is trained

1. `src/data.py:read_index` reads labelled images and steering angles.
2. `src/data.py:image_to_tensor` crops the lower 150 pixels, resizes to **200 × 66**, and converts pixels into a normalised RGB tensor.
3. `train.py:chronological_splits` creates training, validation and test segments with a frame gap.
4. `src/network.py:SteeringNet` maps **3 × 66 × 200** image tensors to one continuous steering prediction. The scalar output is multiplied by 45 to obtain degrees.
5. `train.py` minimises mean-squared error using AdamW, selects the checkpoint by validation MAE, evaluates the held-out split, saves metrics and attempts ONNX export.
6. `app.py` can load the exported ONNX model and calculate predictions for incoming images and videos.

**Limitations**: The current ZIP does not include driving data, a trained model, evidence of real-data generalisation, or safety validation. The synthetic footage and working Canvas simulation are visual demonstrations only. They are **not proof of autonomous driving**.

## Reproduce existing checks

From the project root:

```bash
python -m venv .venv
# Windows PowerShell: .\.venv\Scripts\Activate.ps1
pip install -r requirements.txt -r requirements-train.txt
python -m pytest tests -q
python app.py
```

For model training, obtain a permitted compatible dataset (not included) and follow `README.md` for `python train.py ...`. Do not report model accuracy until actual training and held-out testing are completed. The original environment tests passed **7/7** on 18 November 2026; rerun them on your computer and record the output.

## How an examiner can inspect the code

- Open the GitHub repo's **Code** tab. Click any file above; source code is displayed with line numbers.
- Select **History** or **Blame** on a file to inspect its recorded revisions.
- Review the `DEVELOPMENT_RECORD.md` for actual experiments, problems and modifications, along with the associated genuine commits.
- Ask the maintainer to explain a key function and demonstrate a real modification, test and result.

GitHub commit timestamps and file histories demonstrate recorded development activity; they cannot alone verify who personally composed the code.
