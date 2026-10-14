import pytest
from pydantic import ValidationError

from schemas import ActionItem, MeetingAnalysis


def test_nullable_and_roundtrip():
    item = ActionItem(task="Write tests", owner=None, deadline=None, evidence="Write tests.")
    result = MeetingAnalysis(action_items=[item], decisions=[])
    assert MeetingAnalysis.model_validate_json(result.model_dump_json()) == result
    assert result.model_dump()["action_items"][0]["owner"] is None


@pytest.mark.parametrize("change", [{"task": " "}, {"owner": 3}, {"extra": "bad"}])
def test_invalid_fields(change):
    data = dict(task="Test", owner=None, deadline=None, evidence="Test.")
    with pytest.raises(ValidationError):
        ActionItem(**(data | change))


def test_fields_required():
    with pytest.raises(ValidationError):
        ActionItem(task="Test", evidence="Test.")
    with pytest.raises(ValidationError):
        MeetingAnalysis.model_validate_json('{}')
