# Development and validation record

This file separates **what is known** from **future work to be recorded**. Do not invent earlier development steps or create artificial commit dates to suggest work that did not occur.

## Initial baseline — 18 November 2026

- AI-assisted starter generated in a ChatGPT session in response to a request for a new non-fork steering-prediction project.
- Delivered: browser-based moving-car simulation, recorded synthetic-road OpenCV replay, Gradio interface, PyTorch model/training source, and unit tests.
- Environment check: `python -m pytest tests -q` -> **7 passed** in the project preparation environment on 18 November 2026.
- **Not completed**: training the CNN on a real labelled driving dataset; external test-set evaluation; real-vehicle or real-world safety verification.

## Your actual development log (fill in only after completing the work)

| Date | What I personally changed or tested | Files touched | Test/data and results | Git commit link |
| --- | --- | --- | --- | --- |
| (add date) | (describe precise change and rationale) | (paths) | (measured result) | (actual URL) |

## Suggested next engineering tasks

1. Explain and document the model layers in `src/network.py` and their dimensions, then add a unit test to verify the output tensor shape.
2. Create and measure a camera-to-angle training experiment using a properly licensed labelled driving dataset; record split sizes, training settings, seed, and validation/test MAE and RMSE.
3. Compare the trained model predictions with the OpenCV heuristic on held-out recordings; include failure examples.
4. Extend the website to display **actual model inference** from a suitable server, distinguishing predictions from the route-following animation.
5. Record screenshots, console logs, short screen captures, and genuine commits describing each change.

These are **recommendations**, not claims that any such work is already completed.
