# SEO NLP Analyzer

Analyzes any URL using Google Cloud Natural Language API and returns a full SEO report covering entities, sentiment, content classification, syntax quality, and entity-level sentiment.

**Two ways to use it:**
- `seo_dashboard.py` — Streamlit web dashboard (recommended)
- `seo_nlp_analyzer.py` — CLI tool for terminal use

---

## What it analyzes

| Section | What you learn |
|---|---|
| **Entities** | What topics/entities Google extracts, salience scores, Knowledge Graph links |
| **Sentiment** | Overall tone (positive/negative/neutral) and emotional intensity |
| **Categories** | What content category Google assigns to the page |
| **Syntax** | Passive voice %, lexical density, dominant nouns |
| **Entity Sentiment** | Per-entity tone — how each topic is talked about |

---

## Setup

### 1. Google Cloud credentials

1. Go to [Google Cloud Console](https://console.cloud.google.com/)
2. Create a project and enable the **Cloud Natural Language API**
3. Create a **Service Account** and download the JSON key file

### 2. Install dependencies

```bash
pip install -r requirements.txt
```

---

## Run locally

### Dashboard (Streamlit)

```bash
# Set up credentials
cp .streamlit/secrets.toml.example .streamlit/secrets.toml
# Fill in your service account values in secrets.toml

# Run
streamlit run seo_dashboard.py
```

Opens at `http://localhost:8501`

### CLI

```bash
export GOOGLE_APPLICATION_CREDENTIALS="/path/to/key.json"

# Analyze a URL
python seo_nlp_analyzer.py --url https://example.com/page

# Compare two URLs
python seo_nlp_analyzer.py --url https://yoursite.com --text "raw text"

# Export JSON
python seo_nlp_analyzer.py --url https://example.com --json > report.json

# Show more entities
python seo_nlp_analyzer.py --url https://example.com --top 30
```

---

## Deploy to Streamlit Community Cloud (free)

1. Push this repo to GitHub
2. Go to [share.streamlit.io](https://share.streamlit.io)
3. Click **New app** → select your repo → set main file to `seo_dashboard.py`
4. Open **Advanced settings → Secrets** and paste your credentials:

```toml
[gcp_service_account]
type = "service_account"
project_id = "your-project-id"
private_key_id = "your-key-id"
private_key = "-----BEGIN RSA PRIVATE KEY-----\n...\n-----END RSA PRIVATE KEY-----\n"
client_email = "your-sa@your-project.iam.gserviceaccount.com"
client_id = "your-client-id"
auth_uri = "https://accounts.google.com/o/oauth2/auth"
token_uri = "https://oauth2.googleapis.com/token"
auth_provider_x509_cert_url = "https://www.googleapis.com/oauth2/v1/certs"
client_x509_cert_url = "https://www.googleapis.com/robot/v1/metadata/x509/..."
```

5. Click **Deploy** — your app gets a public `yourapp.streamlit.app` URL

---

## Dashboard features

- **Single URL** — full analysis in 6 tabs
- **Two URLs** — side-by-side competitor comparison, same 6 tabs
- **Target keyword** — highlights the keyword in entity tables and tells you its salience rank
- **Color-coded sentiment** — green/yellow/red across all entity sentiment rows
- **Bar charts** — entity salience, content categories, top nouns

---

## SEO use cases

| Use case | How |
|---|---|
| Topical authority check | Compare your page vs top 3 competitors — see which entities they rank for that you're missing |
| Keyword entity validation | Check if your target keyword appears as a high-salience entity |
| Content gap analysis | Find entities competitors have that you don't |
| Category alignment | Confirm Google classifies your page in the right niche |
| Sentiment audit | Catch negative tone before publishing |
| Passive voice audit | Flag overly passive writing that hurts readability |
