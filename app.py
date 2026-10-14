"""MeetingMind Streamlit dashboard. Run with: streamlit run app.py."""

import os
from pathlib import Path

import pandas as pd
import streamlit as st
from dotenv import load_dotenv

from extractor import MODEL, ExtractionError, extract_meeting
from nlp_utils import analyze_transcript, load_nlp

ROOT = Path(__file__).resolve().parent
load_dotenv(ROOT / ".env")
st.set_page_config(page_title="MeetingMind", page_icon="📝", layout="wide")


@st.cache_resource
def get_nlp():
    return load_nlp()


def clear_results():
    for key in ("analysis", "linguistic", "warnings", "analyzed_text"):
        st.session_state.pop(key, None)


def load_sample():
    st.session_state.transcript = (ROOT / "sample_transcript.txt").read_text(encoding="utf-8")
    clear_results()


def load_upload():
    clear_results()
    st.session_state.pop("upload_error", None)
    uploaded = st.session_state.upload
    if uploaded is not None:
        try:
            st.session_state.transcript = uploaded.getvalue().decode("utf-8-sig")
        except UnicodeDecodeError:
            st.session_state.upload_error = "Cannot read this file. Save it as UTF-8 plain text and upload again."


with st.sidebar:
    st.title("📝 MeetingMind")
    st.write("Intelligent Meeting Conversation Analyzer")
    st.caption("A BTech NLP mini project")
    st.divider()
    st.markdown("**NLP techniques**\n\nSentence segmentation · Tokenization · Lemmatization · POS tagging · Named entities · Dependency parsing")
    st.markdown("**Linguistic model:** `en_core_web_sm`")
    st.markdown(f"**Extraction model:** `{MODEL}` via Groq")
    st.caption("spaCy analyzes language; the pretrained LLM extracts tasks and decisions.")

st.title("MeetingMind")
st.write("Turn meeting conversations into tasks, owners, deadlines, and confirmed decisions.")
st.header("1. Meeting Input")
st.caption("English transcripts only. Clicking Analyze sends the transcript and linguistic context to Groq.")
st.file_uploader("Upload a UTF-8 TXT transcript", type=["txt"], key="upload", on_change=load_upload)
if st.session_state.get("upload_error"):
    st.error(st.session_state.upload_error)
st.button("Load sample", on_click=load_sample)
transcript = st.text_area("Meeting transcript", height=260, key="transcript", on_change=clear_results,
                          placeholder="Ayesha: I'll finish the frontend changes by Thursday.")
api_key = os.getenv("GROQ_API_KEY", "").strip()
if not api_key or api_key == "your_api_key_here":
    st.info("To enable extraction, copy .env.example to .env, set GROQ_API_KEY, and restart the app. NLP analysis still works without a key.")

if st.button("Analyze Meeting", type="primary"):
    clear_results()
    try:
        if not transcript.strip():
            raise ValueError("Please enter a meeting transcript.")
        if len(transcript) > 50_000:
            raise ValueError("Please use a transcript of 50,000 characters or fewer for this MVP.")
        with st.spinner("Analyzing language and extracting meeting information…"):
            linguistic = analyze_transcript(transcript, get_nlp())
            st.session_state.linguistic = linguistic
            st.session_state.analyzed_text = transcript
            analysis, warnings = extract_meeting(transcript, linguistic, api_key)
            st.session_state.analysis = analysis
            st.session_state.warnings = warnings
    except (ValueError, RuntimeError, ExtractionError) as exc:
        st.error(str(exc))

analysis = st.session_state.get("analysis")
if analysis is not None:
    for warning in st.session_state.get("warnings", []):
        st.warning(warning)
    st.header("2. Analysis Overview")
    columns = st.columns(3)
    columns[0].metric("Action items", len(analysis.action_items))
    columns[1].metric("Decisions", len(analysis.decisions))
    columns[2].metric("Identified task owners", len({a.owner.strip().casefold() for a in analysis.action_items if a.owner}))
    actions = pd.DataFrame([a.model_dump() for a in analysis.action_items],
                           columns=["task", "owner", "deadline", "evidence"])
    decisions = pd.DataFrame([d.model_dump() for d in analysis.decisions],
                             columns=["decision", "evidence"])
    st.header("3. Action Items")
    if actions.empty:
        st.info("No verified action items found.")
    else:
        st.dataframe(actions.fillna({"owner": "Unassigned", "deadline": "Not specified"}), hide_index=True, width="stretch")
    st.header("4. Meeting Decisions")
    if decisions.empty:
        st.info("No verified confirmed decisions found.")
    else:
        st.dataframe(decisions, hide_index=True, width="stretch")
    st.caption("Evidence was matched to source text. This does not guarantee that the model interpreted it correctly.")

if "linguistic" in st.session_state:
    with st.expander("5. NLP Analysis"):
        linguistic = st.session_state.linguistic
        st.write("Actual spaCy output for the submitted transcript.")
        st.subheader("Sentences and speakers")
        st.dataframe(pd.DataFrame(linguistic["sentences"]), hide_index=True, width="stretch")
        st.subheader("Tokens, lemmas, POS tags, and dependencies")
        st.dataframe(pd.DataFrame(linguistic["tokens"]), hide_index=True, width="stretch")
        st.subheader("Named entities")
        if linguistic["entities"]:
            st.dataframe(pd.DataFrame(linguistic["entities"]), hide_index=True, width="stretch")
        else:
            st.write("No named entities detected.")

if analysis is not None:
    st.header("6. Export")
    left, right = st.columns(2)
    left.download_button("Download action items (CSV)", actions.to_csv(index=False).encode("utf-8-sig"),
                         "meetingmind_actions.csv", "text/csv")
    right.download_button("Download complete analysis (JSON)", analysis.model_dump_json(indent=2),
                          "meetingmind_analysis.json", "application/json")
