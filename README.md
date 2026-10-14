# MeetingMind — Intelligent Meeting Conversation Analyzer

A BTech NLP mini project that converts English written meeting conversations into
action items, responsible people, deadlines, confirmed decisions, and source evidence.
It uses pretrained models; no custom model training is performed.

## Windows PowerShell setup

Python 3.11+ is required. Run these commands from the project directory:

```powershell
Set-Location H:\NLP
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install -r requirements.txt
python -m spacy download en_core_web_sm
Copy-Item .env.example .env
notepad .env
```

Replace `your_api_key_here` with your Groq API key in `.env`, save, then run:

```powershell
python -m streamlit run app.py
```

Open the local URL printed by Streamlit (normally http://localhost:8501).
If PowerShell blocks activation, use the virtual environment interpreter directly:

```powershell
.\.venv\Scripts\python.exe -m streamlit run app.py
.\.venv\Scripts\python.exe -m pytest -q
```

With the environment activated, run all offline tests with:

```powershell
python -m pytest -q
```

The tests require the downloaded spaCy model but make no live Groq calls. Dependencies
are bounded by major version in `requirements.txt`. The API key is loaded from the
environment or `.env`; existing environment variables take precedence. Never commit
your key. `.env` is ignored. Restart the app after changing the key.

## Demonstration

1. Click **Load sample**, paste a transcript, or upload a UTF-8 `.txt` file.
2. Edit the transcript if needed, then click **Analyze Meeting**.
3. Review the task table, confirmed decisions, and exact supporting sentences.
4. Expand **NLP Analysis** to inspect linguistic processing.
5. Download action items as CSV or the complete analysis as JSON.

Uploading or loading a sample replaces the editable input. Input changes clear prior
results. Downloads and other reruns reuse the current result rather than calling Groq
again. Missing credentials still allow the linguistic analysis to be displayed.

The sample has four speakers, six clear assignments/commitments, two confirmed
decisions, several deadline expressions, and suggestions that must not be interpreted
as decisions. The statement about someone archiving release notes is deliberately
ambiguous; it should never be assigned to the speaker merely because they said it.

## NLP pipeline and architecture

**Transcript → spaCy linguistic analysis → Groq extraction → Pydantic validation →
source-evidence verification → Streamlit tables and exports**

- `nlp_utils.py`: loads `en_core_web_sm`; segments each speaker turn into sentences;
  performs tokenization, lemmatization, POS tagging, named entity recognition, and
  dependency parsing. Preserves source offsets and speaker labels without lowercasing,
  stop-word removal, or stemming. Speaker labels are parsed from `Name:` prefixes;
  unlabeled continuation lines inherit the preceding speaker. Labels are not treated
  as spaCy-recognized person entities.
- `extractor.py`: sends the original transcript, sentences with speakers, and candidate
  PERSON/DATE/TIME entities to Groq. The LLM identifies assignments and confirmed
  decisions, interprets unambiguous first-person ownership, and preserves deadlines.
  Other linguistic annotations are displayed for academic inspection.
- `schemas.py`: requires `ActionItem(task, owner, deadline, evidence)`,
  `Decision(decision, evidence)`, and `MeetingAnalysis(action_items, decisions)`.
  Owners and deadlines may be null; missing fields, wrong types, extra fields, and
  blank strings are rejected.
- `app.py`: input, cached spaCy model, per-session results, tables, errors, and exports.
  CSV has `task,owner,deadline,evidence` columns and blank cells for null values;
  JSON preserves null values and includes both tasks and decisions.

Unlike summarization, which condenses general discussion, structured information
extraction produces specific typed records with attribution and evidence. spaCy
performs linguistic analysis; the LLM performs task and decision extraction.

## Model and errors

The application consistently uses `openai/gpt-oss-20b` with strict JSON Schema output.
[Groq's structured-output documentation](https://console.groq.com/docs/structured-outputs)
lists strict-mode support for this model. Account access still depends on your API key
and permissions. There is no silent model switch or fake local extraction fallback.

The SDK's built-in retries are disabled. Connection failures, HTTP 408/409/429, and
server errors receive one retry after one second; each request has a 45-second timeout.
Invalid output, incomplete responses, denied access, and persistent failures produce
visible errors. Invalid evidence is discarded with a warning; verified items can
still be exported. Whole sentences must match, allowing whitespace differences only.

## Limitations and privacy

- English text only, up to 50,000 characters per analysis; no chunking. Large inputs
  may still exceed the provider's token or output limits and require shortening.
- Speaker format is `Name: utterance`. Ambiguous names, pronouns, implicit decisions,
  sarcasm, and unusual formatting can reduce extraction accuracy.
- Deadlines remain original phrases such as “by Friday”; no calendar normalization.
- Evidence matching proves the quote exists, not that ownership, deadlines, or intent
  were interpreted correctly. Review results before acting on them.
- The small spaCy model can miss or misclassify entities. Its candidates are hints.
- No transcript database or application file persistence. Transcripts and linguistic
  context are sent to Groq when extraction is requested and remain in session memory
  while using the app. Exports are downloaded by the user.
- No audio, transcription, microphone, diarization, meeting integrations, or custom
  model training is included.

## Deploy for a shareable link

Streamlit Community Cloud can host this Python app. The English spaCy model is included
in `requirements.txt` so the cloud build installs it automatically.

1. Push the project source to a GitHub repository. Exclude `.env`, `.venv`, and
   `.streamlit/secrets.toml` (already ignored).
2. Sign in at https://share.streamlit.io with GitHub and choose **Create app**.
3. Select the repository, its branch, and `app.py` as the entrypoint.
4. Under **Advanced settings**, choose Python 3.11 and add this secret, replacing
   the placeholder with your real key:

   ```toml
   GROQ_API_KEY = "your_api_key_here"
   ```

5. Deploy, open the generated `.streamlit.app` URL, and run the sample transcript.
   Set the app's sharing access so your professor can open it; test the link in a
   signed-out browser before the demonstration.

Streamlit exposes top-level secrets as environment variables, so the existing
`GROQ_API_KEY` environment lookup works without committing any credentials.

## Live verification checklist

Live requests are intentionally excluded from pytest. With a valid key, load the sample
and confirm that Ayesha owns her first-person commitment and direct UI assignment,
Sameer owns the testing tasks, original deadlines remain intact, PostgreSQL and REST
API choices appear as decisions, and vendor/dark-mode suggestions do not. Verify CSV
and JSON downloads against the tables. Without a key, account availability and live
extraction quality remain unverified; mocked tests do not establish live accuracy.
