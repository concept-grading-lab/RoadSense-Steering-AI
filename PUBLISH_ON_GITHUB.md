# Publish RoadSense under your own GitHub account

This is a **new, independently implemented repository**, not a fork of the original project.

## What visitors will see

When someone opens your **GitHub Pages URL**, the front page (`docs/index.html`) automatically runs the animated road and moving car in their browser. They can change routes, pause, change speed, use arrow-key manual steering, and switch to the separately recorded computer-vision playback. GitHub Pages hosts **HTML, CSS, JavaScript, JPEG, and MP4** here. It does **not** execute Python or serve a running CNN.

## Publish to GitHub

1. Sign into GitHub as `concept-grading-lab` (or sign into the account that owns/manages that profile, with permission to create repositories).
2. Go to <https://github.com/new>. Under the **Owner** dropdown choose the desired account or organisation. Set repository name `RoadSense-Steering-AI` and visibility **Public**. **Do not** enable “Add a README”, “Add .gitignore”, or “Choose a license” while creating the empty repository (these files already exist or can be added later).
3. Extract the downloaded ZIP. Open a terminal **inside the extracted project root**, where `README.md`, `app.py`, `docs/` and `src/` are visible.
4. Use the following commands, **provided your GitHub owner is `concept-grading-lab`**:

```bash
git init
git branch -M main
git add .
git commit -m "Create independently built RoadSense live driving demo"
git remote add origin https://github.com/concept-grading-lab/RoadSense-Steering-AI.git
git push -u origin main
```

GitHub will prompt you to authenticate if necessary. Never paste an access token or password into a chat.

5. On the new repository, visit **Settings → Pages**. Under **Build and deployment**, choose **Deploy from a branch**, select `main`, choose `/docs`, then **Save**. GitHub will build the site.

Once published, the project website should be:

`https://concept-grading-lab.github.io/RoadSense-Steering-AI/`

This URL is an **expected URL pattern**, **not a verified live website** until you finish pushing the repository and enabling Pages. GitHub provides the actual deployment URL in the repository's Pages settings. If your GitHub Owner is different, replace `concept-grading-lab` in both commands and URLs.

## What opens where?

| Address | What it does |
| --- | --- |
| `https://github.com/concept-grading-lab/RoadSense-Steering-AI` | Source code and animated README preview. |
| `https://concept-grading-lab.github.io/RoadSense-Steering-AI/` | **The interactive moving-car simulation**. Opens automatically. |
| `https://concept-grading-lab.github.io/RoadSense-Steering-AI/vision-replay.html` | Recorded OpenCV results and steering timeline. |

## To run the Python application

Python/Gradio requires a separate application host (such as a suitable Render instance) and an independently trained ONNX checkpoint for the CNN mode. See the project README. The GitHub Pages car animation is a **simulation**; the vision replay is a **recording of baseline computer-vision results**, not a CNN prediction on a live road.

## Source, rights, and safety

The original project is used only as a conceptual reference: <https://github.com/adityaguptai/Self-Driving-Car->. The standalone RoadSense browser simulator and Python model pipeline were independently implemented. The NVIDIA DAVE-2 paper is <https://arxiv.org/abs/1604.07316>. This is an educational visualisation only; never connect it to a physical vehicle.

References: [GitHub Pages publishing source](https://docs.github.com/en/pages/getting-started-with-github-pages/configuring-a-publishing-source-for-your-github-pages-site) and [creating a new GitHub repository](https://docs.github.com/en/repositories/creating-and-managing-repositories/creating-a-new-repository).
