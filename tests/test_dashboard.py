import os

import pytest

pytest.importorskip("streamlit")

from streamlit.testing.v1 import AppTest


def _app_path():
    return os.path.join(
        os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "app.py"
    )


def test_dashboard_runs_without_exception():
    app = AppTest.from_file(_app_path())
    app.run(timeout=180)
    assert not app.exception


def test_dashboard_shows_title_and_tabs():
    app = AppTest.from_file(_app_path())
    app.run(timeout=180)
    titles = [t.value for t in app.title]
    assert any("Drift Batch Blame" in t for t in titles)
    assert len(app.tabs) == 3
