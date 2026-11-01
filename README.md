# ContentCraft

**AI-Based Content Generation and Text Summarization System**

ContentCraft is a college-level Python and Streamlit project that generates content, summarizes
text and documents, rewrites passages and extracts keywords. The public version uses the
**Google Gemini API** through its OpenAI-compatible endpoint. A local Ollama provider remains
available for offline development.

The project is intentionally small and readable. It can run on a laptop for development and be
deployed publicly with Streamlit Community Cloud.

> **Privacy note:** in hosted mode, the text you enter and the prompts needed for the selected
> task are sent to Google Gemini. Do not upload confidential documents to a public deployment.

---

## 1. Features

| Module | What it does | Needs the AI provider? |
| --- | --- | --- |
| Content Generator | Blogs, articles, essays, emails, reports, social posts and study notes with five tones | Yes |
| Text Summarizer | Extractive TF-IDF summaries and abstractive Gemini summaries in short, medium or detailed lengths | Abstractive only |
| Document Summarizer | Reads PDF, DOCX and TXT files, chunks long text and creates a final summary plus key points | Abstractive only |
| Text Rewriter | Rewrites text as professional, academic, simple English, formal or concise | Yes |
| Keyword Extractor | TF-IDF keywords, scores and important sentences | No |
| Export Center | Copy results or download TXT, DOCX and PDF files | No |

Additional UI features:

- Sidebar navigation and a dashboard.
- AI provider status with clear setup instructions.
- Validation, loading indicators and honest error messages.
- No placeholder buttons and no hard-coded fake AI results.
- Results stay in the browser session; there is no database or login.

---

## 2. Technology Stack

- Python 3.11+ (tested with Python 3.13.14)
- Streamlit for the web interface
- Google Gemini `gemini-2.5-flash` through the OpenAI-compatible Gemini endpoint
- Optional Ollama with `qwen2.5:3b` for local development
- NLTK for sentence tokenization and stopwords
- scikit-learn for TF-IDF
- PyMuPDF for PDF text extraction and PDF export
- python-docx for DOCX extraction and export
- requests for the hosted and local model HTTP APIs
- pandas for displaying keyword tables

A Gemini API key is required for hosted AI features. A typical college demonstration can use the
free tier, subject to Google's current quota and terms. Extractive summarization and keyword
extraction work without any API key.

---

## 3. Project Structure

```text
contentcraft/
|-- app.py                          # Streamlit UI, dashboard and navigation
|-- requirements.txt
|-- README.md
|-- Dockerfile
|-- docker-compose.yml              # Optional local app + Ollama stack
|-- .dockerignore
|-- .streamlit/
|   |-- config.toml                 # Streamlit theme and 25 MB upload limit
|   `-- secrets.toml.example        # Gemini configuration template
|-- ai_engine/
|   |-- hosted_model.py             # Gemini/OpenAI-compatible API client
|   |-- local_model.py              # Optional Ollama client
|   |-- provider.py                 # Hosted/local provider selection
|   |-- model_client.py             # Shared model interface
|   |-- errors.py                   # Shared model exceptions
|   `-- prompts.py                  # Prompt templates and target-length helpers
|-- modules/
|   |-- content_generator.py        # Module 1
|   |-- summarizer.py               # Module 2 (extractive and abstractive)
|   |-- document_processor.py       # Module 3 (PDF, DOCX, TXT)
|   |-- text_rewriter.py            # Module 4
|   `-- keyword_extractor.py        # Module 5
|-- utils/
|   |-- text_cleaner.py             # Cleaning, sentence splitting, chunking
|   `-- export_manager.py           # TXT, DOCX and PDF builders
|-- assets/
|   `-- styles.css                  # Custom interface styling
`-- tests/
    |-- test_core.py                # Modules, exports and extraction
    |-- test_hosted_model.py        # Hosted client and provider selection
    `-- test_app.py                 # Streamlit UI smoke tests
```

---

## 4. Requirements

### Public deployment

- A GitHub account
- A Streamlit Community Cloud account
- A Google AI Studio account and Gemini API key
- No GPU or local AI installation is required

### Local development

- Python 3.11 or newer
- A Gemini API key, or optional Ollama for local mode
- 8 GB RAM recommended for the Streamlit app
- Ollama with `qwen2.5:3b` only if `AI_MODE=local`

The extractive summarizer and keyword extractor work on a low-end machine and do not need a model.

---

## 5. Quick Start

### Step 1: Open the project folder

```powershell
cd D:\projects\contentcraft
```

### Step 2: Create a virtual environment

Windows PowerShell:

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
```

macOS / Linux:

```bash
python3 -m venv .venv
source .venv/bin/activate
```

### Step 3: Install dependencies

```powershell
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

### Step 4: Create a Gemini API key

1. Open <https://aistudio.google.com/apikey>.
2. Create an API key for a Google AI Studio project.
3. Keep the key private. Do not paste it into source code, screenshots or GitHub.

### Step 5: Add the key locally

```powershell
Copy-Item .streamlit\secrets.toml.example .streamlit\secrets.toml
```

Edit `.streamlit/secrets.toml`:

```toml
AI_MODE = "hosted"
GEMINI_API_KEY = "your-gemini-api-key"
AI_BASE_URL = "https://generativelanguage.googleapis.com/v1beta/openai"
AI_MODEL = "gemini-2.5-flash"
AI_PROVIDER = "Google Gemini"
```

`.streamlit/secrets.toml` is already listed in `.gitignore`.

### Step 6: Optional NLTK data

ContentCraft tries to download the required NLTK data automatically on first use. If the machine
has no internet during the first run, it falls back to rule-based sentence splitting and a
built-in stopword list. To pre-download the data manually:

```powershell
python -m nltk.downloader punkt punkt_tab stopwords
```

### Step 7: Run the application

```powershell
streamlit run app.py
```

Open <http://localhost:8501>. The sidebar should show **Google Gemini** as the provider. If the key
is missing, the dashboard shows setup instructions and AI features stay unavailable; the app never
shows fake AI output.

---

## 6. Deploy to Streamlit Community Cloud

The repository is ready for Streamlit Community Cloud, but deployment requires your GitHub,
Streamlit and Google accounts. The API key must be entered in Streamlit's secret manager.

### Step 1: Push the project to GitHub

Create a public or private GitHub repository and push the project. Confirm that these files are
**not** committed:

- `.streamlit/secrets.toml`
- `.env`
- virtual environments and log files

The included `.gitignore` excludes them. `.streamlit/secrets.toml.example` contains placeholders
only and is safe to commit.

### Step 2: Create the app

1. Sign in at <https://share.streamlit.io>.
2. Choose **Create app**.
3. Select the GitHub repository, branch and `app.py`.
4. Use Python 3.12 if the platform asks for a version. The project also runs on Python 3.13.

### Step 3: Add the hosted API secrets

In **App settings > Secrets**, paste:

```toml
AI_MODE = "hosted"
GEMINI_API_KEY = "your-gemini-api-key"
AI_BASE_URL = "https://generativelanguage.googleapis.com/v1beta/openai"
AI_MODEL = "gemini-2.5-flash"
AI_PROVIDER = "Google Gemini"
```

Do not use `AI_API_KEY = "..."` in a committed file. `AI_API_KEY` is also accepted as a generic
key name, but `GEMINI_API_KEY` is clearer for this deployment.

### Step 4: Deploy and verify

1. Click **Deploy**.
2. Open the public app URL.
3. Check that the sidebar says the hosted model is ready.
4. Run a short content generation and a short abstractive summary.
5. Test one TXT, DOCX and PDF download.

### Updating the deployment

Push a new commit to the connected branch. Streamlit Community Cloud redeploys automatically.
If you change a secret, save it in App settings and reboot the app.

### Deployment limitations

- The public app is accessible to anyone with the URL. It has no login or per-user quota, so an
  unrestricted deployment can consume your Gemini quota. Add authentication and rate limiting
  before using it as a real public service.
- Streamlit Community Cloud may put inactive apps to sleep; the first visit can be slower.
- Gemini quotas, model names and data-handling terms are controlled by Google and can change.
- The app stores no history, database or uploaded files after the session ends.

---

## 7. Configuration

Settings are read from environment variables first, then from Streamlit secrets.

| Variable | Default | Purpose |
| --- | --- | --- |
| `AI_MODE` | `hosted` | `hosted`, `local`, or `auto`; `auto` uses hosted when a key exists |
| `GEMINI_API_KEY` | none | Gemini API key; required for hosted mode |
| `AI_API_KEY` | none | Generic hosted API key; used if `GEMINI_API_KEY` is absent |
| `AI_BASE_URL` | `https://generativelanguage.googleapis.com/v1beta/openai` | OpenAI-compatible API base |
| `AI_MODEL` | `gemini-2.5-flash` | Hosted model name |
| `AI_PROVIDER` | inferred from the endpoint | Friendly provider name shown in the UI |
| `OLLAMA_MODEL` | `qwen2.5:3b` | Local model name when `AI_MODE=local` |
| `OLLAMA_HOST` | `http://localhost:11434` | Local Ollama server URL |
| `OLLAMA_NUM_CTX` | `8192` | Context window requested from Ollama |

### Using another hosted provider

The hosted client uses the OpenAI-compatible chat completions format, so another compatible
provider can be selected without changing application code:

```toml
AI_MODE = "hosted"
AI_API_KEY = "your-provider-key"
AI_BASE_URL = "https://api.groq.com/openai/v1"
AI_MODEL = "llama-3.1-8b-instant"
AI_PROVIDER = "Groq"
```

Gemini remains the default and the configuration documented for this submission.

### Local Ollama mode

```powershell
$env:AI_MODE = "local"
$env:OLLAMA_MODEL = "qwen2.5:3b"
streamlit run app.py
```

Install Ollama from <https://ollama.com/download>, then run:

```powershell
ollama pull qwen2.5:3b
ollama list
```

Keep Ollama running while using local mode. Do not expose an unauthenticated Ollama port directly
to the public internet.

---

## 8. Using ContentCraft

### Dashboard

Shows the introduction, feature list, workflow, AI provider status and setup instructions.

### Content Generator

1. Enter a topic.
2. Choose content type, tone and target word count.
3. Optionally add special instructions.
4. Click **Generate content**.
5. Use **Regenerate** for a fresh draft with the same settings.
6. Copy or download the result.

The model treats the word count as a target, not an exact guarantee. The UI displays the actual
word count of every result.

### Text Summarizer

- **Extractive:** splits sentences, removes stopwords, scores sentences with TF-IDF and selects
  the highest-scoring sentences in their original order. It runs without the AI provider.
- **Abstractive:** sends the text to Gemini and asks for a new, concise summary. Long text is
  split into chunks, each chunk is summarized, and the summaries are combined.
- Choose **Short**, **Medium** or **Detailed**. The app displays original words, summary words and
  compression percentage.

### Document Summarizer

1. Upload a PDF, DOCX or TXT file (maximum 25 MB).
2. The app validates the file, extracts the text and shows the readable word count.
3. Choose a method and summary length.
4. Click **Summarize document** and watch the progress bar.
5. Review the final summary and important points, then copy or download.

PDF text extraction uses PyMuPDF. Image-only or scanned PDFs are rejected with a clear message
because OCR is not part of this project. DOCX table text is included.

### Text Rewriter

Paste up to 8,000 characters, choose a style and click **Rewrite text**. The prompt instructs the
model to keep the original facts and meaning. The result still needs human review.

### Keyword Extractor

Paste text and choose how many keywords to show. The module uses scikit-learn TF-IDF; it does not
call the AI provider. Results include a ranked keyword table, relative weights, raw TF-IDF scores,
a bar chart and the most important sentences.

### Downloads

Every result can be downloaded as:

- TXT - plain text with title and metadata
- DOCX - headings, bullet lists and basic formatting
- PDF - A4 pages with wrapped text and headings

PDF export uses the built-in Helvetica font (Latin-1). Non-Latin characters may be replaced in
PDF output; DOCX and TXT keep Unicode text.

---

## 9. How It Works

### Provider layer

`ai_engine/provider.py` chooses the model backend:

- `AI_MODE=hosted` uses `ai_engine/hosted_model.py` and the Gemini/OpenAI-compatible API.
- `AI_MODE=local` uses `ai_engine/local_model.py` and Ollama.
- `AI_MODE=auto` uses hosted mode when a key is configured and local mode otherwise.

The hosted client:

1. Reads the API key from the environment or Streamlit secrets.
2. Sends `POST /chat/completions` with the system and user messages.
3. Converts authentication, quota, missing-model, timeout and empty-response failures into clear
   messages.
4. Never substitutes sample output when the provider is unavailable.

The status panel reports that the hosted provider is configured. It does not send a test request
on every page load; the key and model are validated when the first AI action runs.

### Abstractive map-reduce summarization

1. Clean the text and split it into chunks of about 3,500 characters on paragraph/sentence
   boundaries.
2. Summarize each chunk with the configured model.
3. Combine the partial summaries. If they are still long, reduce them again.
4. Send the combined notes to the model for the final summary with an explicit target word count.
5. For document summaries, ask the model for up to six important points.

### Extractive summarization

Uses NLTK sentence tokenization, NLTK stopwords, scikit-learn TF-IDF sentence scoring and a small
position bonus for the opening sentence. Selected sentences are re-ordered to match the original.

### Keyword extraction

TF-IDF is calculated across sentences, so terms that are important in several sentences rank
higher. The top terms, weights and highest-scoring sentences are returned.

### Exports

`utils/export_manager.py` builds files in memory and returns bytes for Streamlit download buttons.
No uploaded document or generated result is written to disk by the application.

---

## 10. Testing

### Automated tests

Run from the project root:

```powershell
python -m unittest discover -s tests -v
```

The suite contains **40 tests** covering:

- text cleaning, sentence splitting, chunk limits and bullet parsing
- extractive summarization and validation
- abstractive single-pass and chunked map-reduce summarization using a deterministic fake model
- document summary key points
- TF-IDF keyword scoring and validation
- prompt wiring for generation and rewriting
- Ollama offline handling and mocked local HTTP generation
- hosted Gemini/OpenAI-compatible request construction with mocked HTTP
- hosted authentication and empty-response errors
- hosted/local provider selection and Gemini key fallback
- TXT, DOCX and PDF export validity
- TXT, DOCX and PDF extraction
- invalid, empty and unsupported uploads
- Streamlit page loading and the extractive summary and keyword UI flows

Test design: deterministic fake/mocked models are used for automated AI tests so the suite does
not depend on model wording or external quota. The local Ollama test was run separately. The hosted
HTTP client is covered with mocked responses.

### Test results in this environment

- Date: 2026-11-01
- Python 3.13.14, Streamlit 1.54.0, scikit-learn 1.8.0, NLTK 3.9.4, PyMuPDF 2.0,
  python-docx 1.2.0
- Result: `Ran 40 tests ... OK` (about 6 seconds)
- All six Streamlit pages loaded with the AppTest harness without exceptions.
- The hosted provider unit tests passed without making a real external request.

Live local Ollama checks completed earlier in development:

- Model status, content generation, regeneration, abstractive summary, document summary, rewrite
  and keyword extraction all produced real output.
- A 176-word TXT document produced an abstractive summary and six important points.
- Browser checks covered generation, downloads, summaries, document upload, rewriting, keyword
  tables and clipboard copy.

**Live Gemini check not yet run:** no Gemini API key was available in the development workspace.
After adding a key, run one short generation and one short abstractive summary to confirm the
provider, quota and model name for your account.

### Manual checklist after deployment

- Sidebar shows **Google Gemini** and a ready status.
- Content generation returns text and **Regenerate** creates a new version.
- Extractive and abstractive summaries both work.
- A PDF, DOCX or TXT upload produces a summary and important points.
- Rewrite and keyword extraction work.
- TXT, DOCX and PDF downloads open successfully.
- Removing the API key shows setup instructions instead of fake output.

---

## 11. Optional Docker Deployment

Docker is optional and was not tested in this environment because Docker is not installed.

### Local app + Ollama

```powershell
docker compose up --build -d
docker compose exec ollama ollama pull qwen2.5:3b
```

Open <http://localhost:8501>. The Compose file sets `AI_MODE=local` and reaches Ollama at
`http://ollama:11434`. Model files are stored in the `ollama_models` volume.

### Hosted mode container

```powershell
docker build -t contentcraft .
docker run --rm -p 8501:8501 -e AI_MODE=hosted -e GEMINI_API_KEY="your-key" contentcraft
```

For real deployments, provide secrets through the platform's secret manager instead of putting
keys in shell history or image layers.

---

## 12. Limitations and Academic Scope

This is intentionally a minimal one-day college project, not a production SaaS product.

- Gemini results can contain factual errors, repetition or awkward wording. Always review AI
  output.
- Word-count targets are approximate.
- Hosted mode sends user input to Google Gemini. It is not a private or offline mode.
- The app has no login, rate limiting or usage quota. A public URL can be used by anyone.
- Gemini free-tier quotas and model availability can change.
- Scanned PDFs are not supported because OCR is not included.
- Upload size is limited to 25 MB.
- NLTK stopwords and prompts are English-oriented.
- The app keeps no history, accounts, roles, database or payments.
- Each generation request waits for the model; output is not streamed token by token.
- Long documents require several model calls and can take a while.
- No web search or retrieval-augmented generation is used. The model only uses its training
  knowledge and the text you provide.
- PDF export supports Latin-1 characters only (built-in font limitation).
- The copy button requires localhost or HTTPS because of browser clipboard rules.

Not implemented on purpose: authentication, payments, user roles, Redis, Celery, PostgreSQL, RAG,
cloud vector databases and enterprise infrastructure.

---

## 13. Troubleshooting

### "GEMINI_API_KEY or AI_API_KEY is not configured"

- Add the key to Streamlit secrets or the environment.
- On Streamlit Community Cloud, use **App settings > Secrets** and reboot the app.
- Click **Refresh AI status** after restarting.

### "The hosted AI provider rejected the API key"

- Check that the key is active in Google AI Studio.
- Remove extra spaces or quotation marks when pasting the key.
- Confirm that the key is not restricted to an incompatible API or project.

### "The hosted AI provider rate limit was reached"

- Wait and try again.
- Check the Gemini quota for the project.
- Use a shorter document or wait for the quota window to reset.

### "Model was not found"

- Check `AI_MODEL`. The default is `gemini-2.5-flash`.
- Check `AI_BASE_URL`. The default Gemini OpenAI-compatible base is
  `https://generativelanguage.googleapis.com/v1beta/openai`.
- Model names and availability can change; use a model available to your account.

### Extractive tools work but AI tools fail

Extractive summaries and keyword extraction do not call Gemini. If they work while generation
fails, the issue is the API key, quota, model name or provider endpoint rather than Streamlit.

### NLTK LookupError

```powershell
python -m nltk.downloader punkt punkt_tab stopwords
```

The app also works without this data by using its fallbacks.

### PDF has no readable text

The PDF is probably scanned. ContentCraft does not include OCR; use a text-based PDF.

### Port 8501 is already in use

```powershell
streamlit run app.py --server.port 8502
```

### Copy button does nothing

Browsers restrict clipboard access on insecure origins. Use the public HTTPS deployment, open the
app through localhost, or use the TXT/DOCX/PDF download buttons.

### Local Ollama is offline

- Start the Ollama application, or run `ollama serve`.
- Confirm `AI_MODE=local`.
- Test the API:

```powershell
Invoke-WebRequest http://localhost:11434/api/tags -UseBasicParsing
```

---

## 14. Quick Command Reference

```powershell
# install
python -m pip install -r requirements.txt

# local configuration
Copy-Item .streamlit\secrets.toml.example .streamlit\secrets.toml

# run
streamlit run app.py

# tests
python -m unittest discover -s tests -v

# syntax check
python -m compileall app.py ai_engine modules utils

# optional local Ollama
ollama pull qwen2.5:3b
$env:AI_MODE = "local"
streamlit run app.py

# optional Docker local stack
docker compose up --build -d
docker compose exec ollama ollama pull qwen2.5:3b
```

---

## 15. Privacy and Data Flow

- Hosted mode sends the selected text, topic and task prompt to Google Gemini.
- Gemini returns the generated text to the Streamlit server, which displays it in your browser.
- Uploaded documents are parsed in memory. They are not saved to a database or application disk.
- Session results are held by Streamlit while the app is running and are cleared when the session
  or server ends.
- API keys are read from Streamlit secrets or environment variables and are never displayed in the
  interface, logs or exported files.
- The public app has no user accounts, so every visitor shares the same server-side API key and
  quota.

---

## 16. Academic Note

This project demonstrates:

- integration of a hosted large language model API with an OpenAI-compatible client
- provider abstraction with a local Ollama fallback
- prompt engineering for different content tasks
- classical NLP techniques (sentence splitting, stopwords, TF-IDF)
- document parsing and chunked map-reduce summarization
- modular Python design and a Streamlit interface
- public deployment with secret management
- in-memory file export and automated testing

All generated text must be reviewed by the user. The system is a writing aid, not a replacement for
research, fact-checking or the student's own work.
