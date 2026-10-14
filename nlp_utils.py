"""Linguistic analysis with original transcript offsets and speaker context."""

import re

import spacy


def load_nlp():
    try:
        return spacy.load("en_core_web_sm")
    except OSError as exc:
        raise RuntimeError(
            "spaCy English model is missing. Run: python -m spacy download en_core_web_sm"
        ) from exc


def analyze_transcript(transcript: str, nlp) -> dict:
    if not transcript.strip():
        raise ValueError("Please enter a meeting transcript.")
    sentences, tokens, entities = [], [], []
    offset = 0
    speaker = None
    # Process individual lines so a speaker turn cannot merge into the prior turn.
    for line in transcript.splitlines(keepends=True):
        match = re.match(r"^\s*([^:\r\n]{1,80}):[ \t]*", line)
        content_start = match.end() if match else 0
        if match:
            speaker = match.group(1).strip()
        doc = nlp(line[content_start:])
        base = offset + content_start
        for sent in doc.sents:
            raw = sent.text
            if not raw.strip():
                continue
            start = base + sent.start_char + len(raw) - len(raw.lstrip())
            end = base + sent.end_char - (len(raw) - len(raw.rstrip()))
            sentences.append({"text": transcript[start:end], "speaker": speaker,
                              "start": start, "end": end})
        tokens.extend({"token": t.text, "lemma": t.lemma_, "pos": t.pos_,
                       "dependency": t.dep_, "head": t.head.text,
                       "start": base + t.idx} for t in doc if not t.is_space)
        entities.extend({"text": e.text, "label": e.label_,
                         "start": base + e.start_char, "end": base + e.end_char}
                        for e in doc.ents)
        offset += len(line)
    return {"sentences": sentences, "tokens": tokens, "entities": entities}


def match_evidence(evidence: str, transcript: str, sentences: list[dict]) -> str | None:
    """Accept a complete source sentence, optionally prefixed by its speaker."""
    normalize = lambda text: " ".join(text.split())
    wanted = normalize(evidence)
    if not wanted:
        return None
    for sentence in sentences:
        start, end = sentence["start"], sentence["end"]
        original = transcript[start:end]
        if normalize(original) == wanted:
            return original
        line_start = transcript.rfind("\n", 0, start) + 1
        prefix = transcript[line_start:start]
        if re.fullmatch(r"\s*[^:\r\n]{1,80}:[ \t]*", prefix):
            labeled = transcript[line_start:end].strip()
            if normalize(labeled) == wanted:
                return labeled
    return None
