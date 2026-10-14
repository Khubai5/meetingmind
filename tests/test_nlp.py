import pytest

from nlp_utils import analyze_transcript, load_nlp, match_evidence


@pytest.fixture(scope="module")
def nlp():
    return load_nlp()


def test_empty(nlp):
    with pytest.raises(ValueError, match="transcript"):
        analyze_transcript(" \n ", nlp)


def test_linguistic_processing(nlp):
    text = "Ayesha: I am writing tests by Friday.\r\nRahul: We approved the release."
    result = analyze_transcript(text, nlp)
    assert [s["speaker"] for s in result["sentences"]] == ["Ayesha", "Rahul"]
    assert all(text[s["start"]:s["end"]] == s["text"] for s in result["sentences"])
    writing = next(t for t in result["tokens"] if t["token"] == "writing")
    assert writing["lemma"] == "write"
    assert writing["pos"] == "VERB"
    assert writing["dependency"]
    assert any(e["text"] == "Friday" and e["label"] == "DATE" for e in result["entities"])


def test_evidence(nlp):
    text = "Ayesha: I'll write  tests by Friday. We approved the release."
    sentences = analyze_transcript(text, nlp)["sentences"]
    assert match_evidence("I'll write tests by Friday.", text, sentences) == "I'll write  tests by Friday."
    assert match_evidence("Ayesha: I'll write tests by Friday.", text, sentences)
    assert match_evidence("We approved the release.", text, sentences)
    assert match_evidence("write tests", text, sentences) is None
    assert match_evidence("I'll write tests by Monday.", text, sentences) is None
    assert match_evidence("", text, sentences) is None
