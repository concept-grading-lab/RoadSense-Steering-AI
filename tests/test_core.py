"""Minimal unit tests; run with `python -m unittest discover -s tests`."""
import tempfile
import unittest
from pathlib import Path

import numpy as np
from PIL import Image

from src.data import HEIGHT, WIDTH, image_to_tensor, read_index
from src.vision import draw_dashboard, estimate_road_angle


class CoreTests(unittest.TestCase):
    def test_tensor_shape_and_range(self):
        image = Image.new("RGB", (640, 480), (100, 200, 40))
        result = image_to_tensor(image)
        self.assertEqual(result.shape, (1, 3, HEIGHT, WIDTH))
        self.assertEqual(result.dtype, np.float32)
        self.assertTrue(np.all(result >= 0) and np.all(result <= 1))

    def test_dataset_index(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            Image.new("RGB", (200, 150)).save(root / "road.jpg")
            (root / "data.txt").write_text("road.jpg -3.5,2018-01-01 12:00:00\n")
            items = read_index(root)
            self.assertEqual(len(items), 1)
            self.assertEqual(items[0].angle_degrees, -3.5)

    def test_original_synthetic_scenarios_have_expected_direction(self):
        from src.vision import estimate_road_angle
        import cv2
        root = Path(__file__).resolve().parents[1] / "demo"
        angles = {}
        for name in ("left", "straight", "right"):
            frame = cv2.cvtColor(cv2.imread(str(root / f"road-{name}.jpg")), cv2.COLOR_BGR2RGB)
            angle, _ = estimate_road_angle(frame)
            self.assertIsNotNone(angle)
            angles[name] = angle
        self.assertLess(angles["left"], -1.0)
        self.assertGreater(angles["right"], 1.0)
        self.assertLess(abs(angles["straight"]), 1.0)

    def test_empty_road_returns_none_not_fake_steering(self):
        blank = np.zeros((360, 640, 3), dtype=np.uint8)
        result, _ = estimate_road_angle(blank)
        self.assertIsNone(result)
        annotated = draw_dashboard(blank, result, "No lanes")
        self.assertEqual(annotated.shape, blank.shape)


if __name__ == "__main__":
    unittest.main()
