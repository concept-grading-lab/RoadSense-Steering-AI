# Source provenance and AI assistance

**Initial project creation (18 November 2026):** The RoadSense starter code, interactive moving-car simulator, example synthetic footage and project documentation were generated through an OpenAI ChatGPT collaboration at the user's request. The user requested a **new project rather than a GitHub fork** of `adityaguptai/Self-Driving-Car-`. The source in this starter is a new implementation inspired by the general end-to-end steering-prediction research idea; it is not a claim of a new research method.

**Authorship caution:** A project being uploaded to a GitHub account does not establish that its owner personally wrote all the source files. The initial code was produced with AI assistance, and it would be inaccurate to declare that the repository owner wrote every line independently. The owner should independently inspect, test, correct and extend it, record those activities, and comply with the disclosure rules applicable to their university, course or examination.

**Current technical boundary:** The animated Canvas demo uses a simple route-following controller, not a neural-network model. The OpenCV baseline video uses synthetic sample footage. The repository includes a PyTorch CNN training pipeline, but the supplied source package does not contain a trained real-data model or independently verified performance measurements.

## Attribution

- Conceptual inspiration: [Aditya Gupta — Self-Driving-Car-](https://github.com/adityaguptai/Self-Driving-Car-).
- CNN steering-prediction research: [Bojarski et al. (2016), *End to End Learning for Self-Driving Cars*](https://arxiv.org/abs/1604.07316).
- Suggested separate driving dataset: [Sully Chen — driving-datasets](https://github.com/SullyChen/driving-datasets). Dataset not included.
- Initial code and documentation: AI-assisted generation through ChatGPT, 18 November 2026. Future contributors and changes should be documented accurately in Git commits and the development record.

## Suggested truthful disclosure (edit to reflect actual work)

> The initial RoadSense application source code, browser-based simulator and documentation were generated with assistance from OpenAI ChatGPT on 18 November 2026. I reviewed the code and, where documented in the repository's development log and commits, independently tested, modified and extended it. I have distinguished the browser-only simulation from the separate CNN training pipeline and have not presented synthetic demonstrations as real-world trained-model performance.

Only retain the clause about independent testing/modification when it is factually true. Check your institution's applicable AI policy and assessment conditions. Keep the chat record or relevant extracts as evidence if permitted.
