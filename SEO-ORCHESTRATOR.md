# SEO Content Optimization Orchestrator
Version: 1.0

Purpose: precise on-page SEO optimization using competitor benchmarks, Google NLP, Firecrawl, DataForSEO, and deterministic term analysis.

This orchestrator applies to deep SEO workflows such as Content Score, POP-like term optimization, entity gap analysis, competitor benchmarks, content briefs, and improvement plans.

It does not replace `AGENT.md`. Always follow the project pre-flight checklist first.

---

## 1. Core Mission

Optimize a page as precisely as possible by comparing it against selected competitors.

Primary outputs:

- Content Score 0-100
- POP-like term gap
- entity gap
- heading and structure gap
- sentiment gap
- internal link recommendations
- top priority SEO actions
- concrete rewrite or addition suggestions

The goal is not keyword stuffing.

The goal is to make the page semantically complete, structurally aligned with the SERP, readable, and commercially effective.

---

## 2. Hierarchy Of Truth

Use this order when deciding what is true:

1. Project directives:
   - `AGENT.md`
   - `WRITING-STYLE.md`
   - `SIA-ONPAGE-RULES.md`
   - `SIA-ARTICLE-STRUCTURE.md`
   - `SIA-SENTIMENT-RULES.md`

2. Actual competitor benchmark:
   - selected competitor URLs
   - competitor averages
   - competitor min, max, and median
   - competitor heading structure
   - competitor entity salience

3. Official or external data:
   - Google NLP API
   - DataForSEO SERP, Labs, and OnPage
   - Firecrawl extracted content
   - Google Search Console, if available

4. Deterministic calculations:
   - exact keyword count
   - partial keyword count
   - n-gram frequency
   - density percentage
   - heading placement
   - internal link count
   - image and alt count

5. AI interpretation:
   - rewrite suggestions
   - content brief
   - content gap explanation
   - tone improvement

AI interpretation must never override benchmark data.

If benchmark data is missing, do not invent precise target values.

---

## 3. Required Inputs

For deep optimization, collect:

- primary keyword
- page type:
  - blog
  - product page
  - e-commerce category
  - service page
  - local landing page
  - affiliate or review page
  - comparison page
- language
- own URL or own pasted content
- 3-10 competitor URLs
- optional secondary keywords
- optional LSI or related keywords
- optional internal link targets
- optional NeuroWriter or DataForSEO keyword list

If page type is missing, ask before rewriting or giving page-specific recommendations.

If internal links are required before a rewrite, ask before writing.

---

## 4. Routing Logic

### Fast Path

Use Fast Path when the user asks:

- general SEO questions
- what a metric means
- strategy without a specific page
- whether an implementation makes sense
- how to prioritize features

Fast Path does not require crawling, API calls, or benchmark creation.

Output:

- concise recommendation
- reasoning
- next best step

### Deep Analysis

Use Deep Analysis when the user asks:

- analyze this URL
- compare competitors
- calculate Content Score
- create content brief
- create improvement plan
- optimize existing text
- find missing entities or keywords
- recommend exact usage counts

Deep Analysis requires benchmark data.

If benchmark is not available, either create it or mark missing data.

Status tags for UI:

- `[CRAWLING_PAGE...]`
- `[CRAWLING_COMPETITORS...]`
- `[EXTRACTING_MAIN_CONTENT...]`
- `[ANALYZING_NLP...]`
- `[CALCULATING_TERM_GAPS...]`
- `[CALCULATING_ENTITY_GAPS...]`
- `[BUILDING_CONTENT_SCORE...]`
- `[GENERATING_RECOMMENDATIONS...]`

---

## 5. State Ledger

Before deep analysis, check `_tmp/state.json` if it exists.

State should store:

- keyword
- page type
- language
- own URL
- own text hash
- competitor URLs
- competitor text hashes
- benchmark timestamp
- benchmark result
- Content Score result
- term gap result
- entity gap result
- last recommendations

Do not repeat expensive crawl or API steps if:

- URL list is unchanged
- content hash is unchanged
- benchmark is fresh
- user did not request refresh

Recommended freshness:

- own page: always refresh unless user says cached is OK
- competitors: cache for 24 hours
- DataForSEO SERP: cache for 24-72 hours
- Google NLP result: cache by content hash

---

## 6. Content Score 0-100

Calculate Content Score from multiple weighted components.

Default weights:

| Component | Points |
|---|---:|
| Term coverage | 25 |
| Entity coverage and salience | 20 |
| Keyword placement | 15 |
| Content structure | 15 |
| Word count and content depth | 10 |
| Sentiment and readability | 7 |
| Internal links | 5 |
| Images and alt text | 3 |
| **Total** | **100** |

The score must show:

- total score
- component scores
- reason for each deduction
- top fixes that would increase score fastest

Never show only the score. Always explain what caused it.

---

## 7. POP-Like Term Optimizer

For each term, calculate:

- term type:
  - primary
  - secondary
  - LSI or related
  - competitor n-gram
  - NLP entity
- your count
- competitor average
- competitor median
- competitor minimum
- competitor maximum
- used by X/Y competitors
- recommended range
- action:
  - add
  - keep
  - reduce
  - add section
  - add to heading
  - ignore

Count:

- exact match
- case-insensitive match
- optional partial match
- optional Slovenian variant or stem match

For Slovenian:

- do not rely only on exact form
- support phrase variants where possible
- do not force unnatural repetitions
- prefer adding terms inside useful sentences and headings

---

## 8. Entity Optimizer

Use Google NLP entities from own page and competitors.

For each entity:

- entity name
- entity type
- your salience
- competitor average salience
- competitor presence count
- missing or present
- recommendation

Entity priority:

1. present in most competitors and missing from your page
2. high competitor salience, low your salience
3. related to primary keyword
4. supports topical authority

Output:

- missing entities
- underweighted entities
- over-dominant irrelevant entities
- suggested paragraph or section where to add each entity

---

## 9. Heading And Structure Optimizer

Analyze:

- H1 count
- H1 keyword placement
- H2 count
- H2 topics
- H3 count
- heading hierarchy
- competitor average heading structure
- missing H2 topics
- keyword and entity presence in headings

Recommendations:

- exact H1 suggestion
- H2 list suggestion
- H2 topics to add or remove
- warnings for too many irrelevant H3s
- whether page type matches SERP structure

---

## 10. Sentiment And Language Rules

Use sentiment according to language support.

For Slovenian:

Reliable:

- sentiment score
- magnitude
- sentence-level negative, neutral, and positive labels
- entity names and salience
- content categories, when available

Unreliable:

- Google NLP verb count
- Google NLP adjective count
- Google NLP passive voice

For Slovenian syntax, use Claude/Codex analysis or deterministic text review.

Sentiment optimization:

- avoid neutral dead zones
- rewrite negative framing
- replace "ne X" with positive framing where possible
- do not add random positive sentences
- replace weak sentences one-for-one

---

## 11. Recommendation Format

Every deep analysis should end with:

1. Content Score
2. Top 10 fixes ranked by SEO impact
3. term gap table
4. entity gap table
5. heading and structure recommendations
6. sentiment and readability recommendations
7. internal link recommendations
8. concrete examples:
   - sentence to add
   - heading to change
   - paragraph to rewrite

Each recommendation must be actionable.

Avoid vague advice like "improve topical relevance".

Say exactly what to add, where, and why.

---

## 12. Critical Pause Rules

Use `[CRITICAL_PAUSE]` if:

- benchmark is required but missing
- page type is required but missing
- internal links are required before rewrite and missing
- competitor URLs do not match the same search intent
- data source failed and no fallback is available
- numeric targets would have to be invented
- Slovenian Google NLP syntax metrics are being treated as reliable

Format:

```text
[CRITICAL_PAUSE]
Problem: ...
Manjka: ...
Rešitev: ...
```

---

## 13. Implementation Roadmap

Phase 1:

- `content_optimizer.py`
- deterministic term counter
- competitor average, minimum, maximum, and median
- term gap table
- basic Content Score

Phase 2:

- dashboard tab: Content Optimizer
- competitor selection and exclusion
- score breakdown UI
- markdown export

Phase 3:

- Google NLP entity gap integration
- heading extraction
- H1/H2/H3 optimizer

Phase 4:

- DataForSEO Labs keyword suggestions
- DataForSEO SERP auto competitor discovery
- optional DataForSEO OnPage Keyword Density validation

Phase 5:

- internal linking assistant
- Google Search Console opportunity audit
- topical map
- AI visibility and LLM mentions
