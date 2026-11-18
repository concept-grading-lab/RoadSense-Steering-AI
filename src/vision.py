"""A transparent, non-learning lane geometry baseline and frame annotations."""
from __future__ import annotations

from math import cos, radians, sin

import cv2
import numpy as np


def estimate_road_angle(frame_rgb: np.ndarray) -> tuple[float | None, str]:
    """Estimate lateral lane-centre displacement; NOT an autonomous steering model.

    This visual baseline is only a demonstrator while the CNN is untrained.
    Positive angles mean the visible lane centre lies to the image's right.
    """
    h, w = frame_rgb.shape[:2]
    if h < 80 or w < 100:
        return None, "Frame too small to estimate lane edges"

    small = cv2.resize(frame_rgb, (640, 360), interpolation=cv2.INTER_AREA)
    gray = cv2.cvtColor(small, cv2.COLOR_RGB2GRAY)
    gray = cv2.GaussianBlur(gray, (5, 5), 0)
    edges = cv2.Canny(gray, 55, 145)
    mask = np.zeros_like(edges)
    polygon = np.array([[(65, 350), (250, 200), (390, 200), (575, 350)]], dtype=np.int32)
    cv2.fillPoly(mask, polygon, 255)
    candidates = cv2.HoughLinesP(
        cv2.bitwise_and(edges, mask), 1, np.pi / 180, 35,
        minLineLength=32, maxLineGap=40,
    )
    if candidates is None:
        return None, "No usable lane edges detected"

    left: list[tuple[float, float]] = []
    right: list[tuple[float, float]] = []
    target_y = 345.0
    for line in candidates[:, 0]:
        x1, y1, x2, y2 = (float(z) for z in line)
        dx = x2 - x1
        if abs(dx) < 2:
            continue
        slope = (y2 - y1) / dx
        if not 0.55 <= abs(slope) <= 4.0:
            continue
        bottom_x = x1 + (target_y - y1) / slope
        if not (0 <= bottom_x <= 640):
            continue
        weight = float(np.hypot(dx, y2 - y1))
        (left if slope < 0 else right).append((bottom_x, weight))

    if not left or not right:
        return None, "Both lane boundaries were not detected"
    left_x = float(np.average([x for x, _ in left], weights=[wt for _, wt in left]))
    right_x = float(np.average([x for x, _ in right], weights=[wt for _, wt in right]))
    if not (left_x < 320 < right_x) or right_x - left_x < 110:
        return None, "Lane geometry was ambiguous"
    offset = ((left_x + right_x) / 2 - 320.0) / 320.0
    angle = float(np.clip(offset * 50.0, -35.0, 35.0))
    return angle, "Illustrative lane-centre estimate (not machine learning)"


def draw_dashboard(frame_rgb: np.ndarray, angle: float | None, caption: str) -> np.ndarray:
    """Overlay a readable steering wheel indicator on an RGB image."""
    img = cv2.cvtColor(frame_rgb.copy(), cv2.COLOR_RGB2BGR)
    h, w = img.shape[:2]
    if w < 360:
        scale = 360 / w
        img = cv2.resize(img, (360, max(200, int(h * scale))))
        h, w = img.shape[:2]

    bottom_y = h - 15
    header_h = min(82, h // 3)
    cv2.rectangle(img, (0, 0), (w, header_h), (12, 22, 36), thickness=-1)
    title = "No steering estimate" if angle is None else f"Estimated steering: {angle:+.1f} deg"
    cv2.putText(img, title, (14, 32), cv2.FONT_HERSHEY_SIMPLEX, 0.76, (255, 255, 255), 2, cv2.LINE_AA)
    caption_short = caption[:65]
    cv2.putText(img, caption_short, (14, 58), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (170, 225, 255), 1, cv2.LINE_AA)

    cx = max(60, w - 76)
    cy = min(h - 75, max(header_h + 65, bottom_y - 65))
    radius = min(50, max(28, w // 10))
    cv2.circle(img, (cx, cy), radius, (250, 250, 250), thickness=4)
    if angle is not None:
        theta = radians(float(np.clip(angle, -80, 80)))
        end = (int(cx + radius * 0.85 * sin(theta)), int(cy - radius * 0.85 * cos(theta)))
        cv2.line(img, (cx, cy), end, (45, 230, 150), thickness=5, lineType=cv2.LINE_AA)
        cv2.circle(img, (cx, cy), 5, (255, 255, 255), thickness=-1)
    return cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
