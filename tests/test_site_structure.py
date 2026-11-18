"""Offline publication checks for the self-contained GitHub Pages website."""
from pathlib import Path
import json

DOCS = Path(__file__).resolve().parents[1] / 'docs'


def test_site_has_required_assets():
    expected = ['index.html', 'style.css', 'sim.js', 'vision-replay.html',
                'road-annotated.mp4', 'road-left.jpg', 'road-right.jpg',
                'road-straight.jpg', 'angles.json', '.nojekyll']
    assert all((DOCS / file).exists() for file in expected)


def test_main_page_launches_simulator():
    content = (DOCS / 'index.html').read_text(encoding='utf-8')
    assert 'id="road"' in content
    assert '<script src="sim.js"' in content
    assert 'id="btnVideo"' in content
    assert 'id="speedSlider"' in content


def test_replay_angle_trace_has_one_value_per_frame():
    trace = json.loads((DOCS / 'angles.json').read_text(encoding='utf-8'))
    assert len(trace) == 160
    assert all(isinstance(x, (int, float)) for x in trace)
