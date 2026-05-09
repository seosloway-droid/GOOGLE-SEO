# AGENT.md — SEO Project Workflow Rules

This file defines how Claude should approach jobs in this project.
Read this at the start of every session before taking action.

---

## Project Overview

This is an SEO content analysis and improvement toolkit. The goal is to analyze
web content using NLP, identify SEO weaknesses, and rewrite content to fix them.
Output language is typically **Slovenian**, unless specified otherwise.

---

## Available Tools

| Script | Purpose | How to run |
|---|---|---|
| `seo_nlp_analyzer.py` | Fetches a URL, runs Google Cloud NLP analysis, produces an SEO report | `python seo_nlp_analyzer.py --url <URL>` |
| `seo_dashboard.py` | Streamlit web dashboard version of the analyzer | `streamlit run seo_dashboard.py` |
| `content_improver.py` | Takes an SEO report + original content, rewrites it using Claude AI | `python content_improver.py` (interactive) OR apply logic directly in chat |

---

## Standard Workflow

### Step 1 — Analyze
Run `seo_nlp_analyzer.py` on the target URL (or use `seo_dashboard.py` for a visual interface).
This produces an SEO analysis report covering: sentiment, entity density, verb usage, adjectives, readability, etc.

### Step 2 — Improve
Feed the SEO report + original content into `content_improver.py`.
Choose mode:
- **Full rewrite** — for content that needs a complete overhaul
- **Section by section** — for content that is mostly OK but needs targeted fixes
- **Rules first** — useful when you want to understand what rules apply before rewriting

### Step 3 — Save output
- Always save final improved content to the workspace folder (`/GOOGLE-SEO/`)
- Default output filename: `improved_content.md`
- If running multiple jobs, use descriptive names like `improved_content_<site>_<date>.md`

---

## Claude Behavior Rules

### Running scripts
- `content_improver.py` is **interactive** — it cannot be run via Claude's bash sandbox with user input.
- **Preferred approach:** Apply the content improvement logic directly in chat (Claude IS the AI behind the script).
- If the user wants to run scripts themselves, provide the exact terminal command.

### When user pastes SEO report + content
1. **FIRST — ask for page type** (see section below — this determines which metrics apply)
2. Ask for: **mode** (full / section / rules) and **language** (Slovenian / English)
3. Apply the same logic as `content_improver.py` — fix verbs, sentiment, adjectives, emotional language — using the correct targets for that page type
4. Save the output as a `.md` file to the workspace folder
5. Provide a direct link to the file

### ⚠️ Important: Google NLP context
Google's NLP documentation uses generic examples (movie reviews, Gettysburg Address, "Lawrence of Arabia").
It does NOT distinguish between content types. This means the raw NLP output has no built-in awareness
of whether it's analyzing a blog post or a product page — Claude must apply the correct targets manually
based on the page type declared by the user.

---

## Page Type — Ask Before Every Revision

**⛔ STOP — Before writing ANY analysis, recommendations, or rewrite:**
**Ask the user for page type FIRST. Do NOT proceed until the user answers.**
**Do not guess. Do not assume. Do not write anything else first.**

> "Kakšen tip strani je to?" (What type of page is this?)

Options to present:

| Page Type | When to use |
|---|---|
| **Blog** | Informational article, guide, how-to, news |
| **Service page** | Description of a service offering |
| **Location-based hub page** | City/region landing page (local SEO) |
| **Product page** | Single product on an e-commerce site |
| **E-commerce category page** | Product listing / category overview |
| **Affiliate page** | Review, comparison, "best of" content |

---

## NLP Targets by Page Type

Google NLP gives raw numbers — apply these benchmarks based on page type:

| Metric | Blog | Service page | Location hub | Product page | E-comm category | Affiliate |
|---|---|---|---|---|---|---|
| **Sentiment** | Nevtralen OK (0.0–0.3) | +0.2–0.4 | +0.2–0.4 | **+0.4 minimum** | +0.3–0.5 | +0.3–0.5 |
| **Lexical density** | 45–55% | 45–55% | 42–50% | **42–48%** | 40–48% | 45–55% |
| **Pasivni glas** | Do 20% OK | Do 15% | Do 15% | **Pod 10%** | Pod 10% | Do 15% |
| **Entitete** | Informacijske | Storitve, lokacije | Lokacije, storitve | **Produkti, blagovne znamke** | Kategorije, blagovne znamke | Produkti, primerjave |
| **Kategorija** | Informacijska | Commercial | Local/Commercial | **Commercial/Shopping** | Shopping | Commercial |
| **Glagoli** | Razlagalni | Akcijski | Lokacijski + akcijski | **Akcijski, koristni** | Akcijski | Primerjalni + akcijski |

### Key differences to enforce
- **Blog** — nevtralen ton je OK, fokus na informacijah, pasivni glas do 20% je sprejemljiv
- **Product page / E-commerce** — sentiment mora biti pozitiven (+0.4+), pasivni glas pod 10%, vsaka lastnost mora imeti "zakaj je to pomembno za kupca"
- **Affiliate** — primerjalni jezik, jasni CTA, sentiment pozitiven a ne pretirano promocijski
- **Location hub** — poudarek na lokalnih entitetah, storitve + lokacija skupaj, pozivni glas

---

### Content improvement requirements (from script)
- Fix verb deficiency — add action verbs that explain benefits and features
- Add emotional language — write for humans, not robots
- Add descriptive adjectives that differentiate products
- Every product feature must have a "why this matters to the user" explanation
- Target sentiment score: **+0.35 to +0.50**
- Keep all original product names, brands, and factual information
- Do NOT invent specifications or prices
- Write naturally, not like AI-generated content

### Language
- Default output language: **Slovenian**, unless user specifies English
- Always confirm language before starting a rewrite

---

## Key Dependencies

- `anthropic` Python SDK — requires `ANTHROPIC_API_KEY` env variable
- `google-cloud-language` — requires Google Cloud credentials
- `streamlit` — for dashboard only

---

## Notes

- This file should be updated whenever new scripts or workflow steps are added.
- If a new job type arises (e.g. keyword research, competitor analysis), add a section here.
