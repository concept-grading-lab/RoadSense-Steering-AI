---
title: RoadSense Steering Prediction Lab
colorFrom: blue
colorTo: cyan
sdk: gradio
app_file: app.py
pinned: false
---

# RoadSense — steering prediction lab

**Live interactive website:** [https://concept-grading-lab.github.io/RoadSense-Steering-AI/](https://concept-grading-lab.github.io/RoadSense-Steering-AI/)  
**Source code:** [https://github.com/concept-grading-lab/RoadSense-Steering-AI](https://github.com/concept-grading-lab/RoadSense-Steering-AI)

> **Project status:** An interactive driving simulation and a separate OpenCV lane-geometry demonstration have been implemented and tested. The repository also includes a PyTorch convolutional neural network (CNN), dataset-processing code, training/evaluation scripts, and a Gradio inference application. **A CNN trained and evaluated on real labelled driving data is not yet included**, and the GitHub Pages animation is not controlled by a neural network.
> This is an **educational software and computer-vision demonstration**, not a real-world autonomous vehicle or a safe vehicle-control system.

## Overview

RoadSense explores two different ways to visualise steering behaviour: (1) a browser-based driving simulation, where a car moves through generated road geometry using a simple programmed route-following controller, and (2) a Python/OpenCV demonstration, where lane features are extracted from images to produce illustrative steering-angle estimates. A separate machine-learning pipeline is provided for a future supervised learning experiment using a CNN trained on image–steering-angle pairs.
Its underlying camera-to-steering research concept is informed by [Bojarski et al. (2016)](https://arxiv.org/abs/1604.07316).

## Live demonstration

Open the [RoadSense website](https://concept-grading-lab.github.io/RoadSense-Steering-AI/) to explore the browser-based simulator.

The site provides:

- A car that animates along an illustrative road.
- Road/route selection, speed control and pause/resume.
- Manual steering with the available on-screen or keyboard controls.
- Steering-angle, speed and distance indicators.
- A **Vision replay** mode showing previously processed synthetic road footage.

**How it works:** `docs/sim.js` updates the generated road and car state within an animation loop. Its automatic route-following behaviour is programmed geometry, **not learned CNN inference**. The replay is recorded OpenCV output, **not new inference running inside GitHub Pages**.

## Technologies

| Technology | Role |
| --- | --- |
| Python | Data preparation, image processing, model training and inference application |
| PyTorch | CNN architecture and supervised steering-angle regression |
| OpenCV | Edge detection, Hough-line-based road analysis and video annotation |
| NumPy / Pillow | Image arrays, resizing and pixel normalisation |
| ONNX / ONNX Runtime | Planned portable deployment path for a trained CNN |
| Gradio | Local Python interface for image, webcam and video inputs |
| HTML / CSS / JavaScript Canvas | Interactive public driving simulation |
| GitHub / GitHub Pages | Source hosting and static website publication |

## Project structure

```text
RoadSense-Steering-AI/
├── README.md                  # Project overview and reproducibility instructions
├── app.py                     # Gradio: webcam, image and video analysis
├── train.py                   # PyTorch training, validation, testing, ONNX export
├── requirements.txt          # Runtime dependencies
├── requirements-train.txt    # Additional training dependencies
├── src/
│   ├── data.py                # Dataset parser and image preprocessing
│   ├── network.py             # SteeringNet CNN definition
│   └── vision.py              # OpenCV lane-geometry baseline and overlays
├── docs/
│   ├── index.html             # GitHub Pages moving-car simulator
│   ├── sim.js                 # Animation, route controls and steering logic
│   ├── style.css              # Website presentation
│   ├── vision-replay.html      # Recorded vision-demo viewer
│   ├── angles.json            # Synthetic recorded steering trace
│   └── road-annotated.mp4     # Recorded synthetic demonstration
├── demo/                      # Synthetic sample media and generator
├── models/                    # Trained model files go here (not included)
├── tests/                     # Automated software checks
├── SOURCE_CODE_GUIDE.md       # File-by-file technical explanation
├── DEVELOPMENT_RECORD.md     # Record of changes and experiments
└── AI_USE_AND_PROVENANCE.md   # Attribution and AI assistance

# What is new and what should be credited?

The underlying idea of end-to-end steering prediction is established research, not a new invention. Credit research and dataset sources:

- Bojarski et al. (2016), [End to End Learning for Self-Driving Cars](https://arxiv.org/abs/1604.07316).
- [Sully Chen — driving datasets](https://github.com/SullyChen/driving-datasets).

The reference repository has an MIT licence, but no source files from it are incorporated here.

