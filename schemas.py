"""Validated information extracted from a meeting."""

from pydantic import BaseModel, ConfigDict, field_validator


class ExtractionModel(BaseModel):
    model_config = ConfigDict(extra="forbid", strict=True)

    @field_validator("*", mode="after")
    @classmethod
    def reject_blank_strings(cls, value):
        if isinstance(value, str) and not value.strip():
            raise ValueError("Text fields must not be blank")
        return value


class ActionItem(ExtractionModel):
    task: str
    owner: str | None
    deadline: str | None
    evidence: str


class Decision(ExtractionModel):
    decision: str
    evidence: str


class MeetingAnalysis(ExtractionModel):
    action_items: list[ActionItem]
    decisions: list[Decision]
