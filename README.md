---
title: RoadSense Steering Prediction Lab
emoji: 🚘
colorFrom: blue
colorTo: cyan
sdk: gradio
app_file: app.py
pinned: false
---

# RoadSense — steering prediction lab



# What is new and what should be credited?

The underlying idea of end-to-end steering prediction is established research, not a new invention. Credit research and dataset sources:

- Bojarski et al. (2016), [End to End Learning for Self-Driving Cars](https://arxiv.org/abs/1604.07316).
- [Sully Chen — driving datasets](https://github.com/SullyChen/driving-datasets).

The reference repository has an MIT licence, but no source files from it are
incorporated here. If you later copy any of its code, preserve the required
copyright and permission notices. Add **your own** licence once you choose its
terms; the user of this starter is responsible for checking any chosen data,
assets, and models before redistribution.

## Roadmap

- Add a numerical ground-truth comparison dashboard.
- Train on additional approved datasets and report per-domain metrics.
- Add uncertainty / out-of-distribution warnings.
- Record a short example video of the model operating on offline footage.
- Use a driving simulator to examine closed-loop behaviour (not a real vehicle).
