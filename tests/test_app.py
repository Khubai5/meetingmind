import json
from pathlib import Path

import pandas as pd
from streamlit.testing.v1 import AppTest

from schemas import ActionItem, MeetingAnalysis

APP = str(Path(__file__).resolve().parents[1] / "app.py")


def test_empty_and_missing_key(monkeypatch):
    monkeypatch.setenv("GROQ_API_KEY", "")
    app = AppTest.from_file(APP).run()
    assert not app.exception
    app.button[1].click().run()
    assert "Please enter" in app.error[0].value
    app.button[0].click().run()
    assert "Ayesha:" in app.text_area[0].value
    app.button[1].click().run(timeout=20)
    assert not app.exception
    assert "GROQ_API_KEY" in app.error[0].value
    assert len(app.dataframe) >= 2


def test_results_exports_and_invalidation(monkeypatch):
    monkeypatch.setenv("GROQ_API_KEY", "test-key")
    result = MeetingAnalysis(action_items=[ActionItem(task="Write tests", owner=None,
                             deadline=None, evidence="I'll write tests.")], decisions=[])
    calls = []

    def extract(*args):
        calls.append(args)
        return result, []

    monkeypatch.setattr("extractor.extract_meeting", extract)
    app = AppTest.from_file(APP).run()
    app.text_area[0].input("Ayesha: I'll write tests.").run()
    app.button[1].click().run(timeout=20)
    assert not app.exception
    assert [m.value for m in app.metric] == ["1", "0", "0"]
    assert app.dataframe[0].value.iloc[0]["owner"] == "Unassigned"
    assert len(app.get("download_button")) == 2
    app.run()
    assert len(calls) == 1
    assert json.loads(result.model_dump_json())["action_items"][0]["owner"] is None
    frame = pd.DataFrame([a.model_dump() for a in result.action_items])
    assert frame.to_csv(index=False).splitlines()[0] == "task,owner,deadline,evidence"
    app.text_area[0].input("Changed transcript.").run()
    assert not app.metric
    assert not app.get("download_button")
    assert "linguistic" not in app.session_state
