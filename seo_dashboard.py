"""
SEO NLP Analyzer — Streamlit Dashboard
Powered by Google Cloud Natural Language API
"""

import streamlit as st
import pandas as pd
import anthropic
from datetime import datetime
from urllib.request import urlopen, Request
from urllib.error import URLError
from html.parser import HTMLParser
from google.cloud import language_v1
from google.oauth2 import service_account

# ── Page config ───────────────────────────────────────────────────────────────

st.set_page_config(
    page_title="SEO NLP Analyzer",
    page_icon="🔍",
    layout="wide",
    initial_sidebar_state="collapsed",
)

st.markdown("""
<style>
    .block-container { padding-top: 1.5rem; }
    .stTabs [data-baseweb="tab-list"] { gap: 8px; }
    .stTabs [data-baseweb="tab"] { padding: 6px 16px; }
</style>
""", unsafe_allow_html=True)


# ── Google Cloud client ───────────────────────────────────────────────────────

@st.cache_resource
def get_client():
    if "gcp_service_account" in st.secrets:
        creds = service_account.Credentials.from_service_account_info(
            st.secrets["gcp_service_account"]
        )
        return language_v1.LanguageServiceClient(credentials=creds)
    return language_v1.LanguageServiceClient()


# ── Claude AI SEO Coach ───────────────────────────────────────────────────────

def get_anthropic_client():
    try:
        key = st.secrets["ANTHROPIC_API_KEY"]
        if key:
            return anthropic.Anthropic(api_key=key)
    except Exception:
        pass
    return None


def analyze_slovenian_syntax(text: str) -> dict:
    """Use Claude to accurately analyze POS for Slovenian text."""
    client = get_anthropic_client()
    if not client:
        return {}

    prompt = f"""Analyze this Slovenian text and count the following linguistic elements accurately.
Return ONLY a JSON object, no explanation.

Text to analyze:
\"\"\"
{text[:3000]}
\"\"\"

Count and return:
{{
  "verb_count": <number of verbs including all conjugated forms, infinitives, participles used as predicates>,
  "adjective_count": <number of adjectives and participles used attributively>,
  "adverb_count": <number of adverbs>,
  "passive_voice_count": <number of passive constructions like "je bil/bila/bilo", "se + verb">,
  "total_words": <total word count>,
  "top_verbs": [list of top 10 most meaningful verbs found],
  "top_adjectives": [list of top 10 most meaningful adjectives found],
  "passive_examples": [up to 3 example passive sentences],
  "notes": "<any important observations about the text quality>"
}}

Be precise. Count actual tokens, not semantic concepts."""

    try:
        response = client.messages.create(
            model="claude-sonnet-4-5",
            max_tokens=800,
            messages=[{"role": "user", "content": prompt}],
        )
        import json
        raw = response.content[0].text.strip()
        # Extract JSON from response
        if "```" in raw:
            raw = raw.split("```")[1].replace("json", "").strip()
        return json.loads(raw)
    except Exception:
        return {}


def generate_seo_report(data: dict, keyword: str, language: str, detail: str) -> str:
    client = get_anthropic_client()
    if not client:
        return ""

    s  = data["sentiment"]
    sx = data["syntax"]
    top_entities = data["entities"][:10]
    top_entity_sentiment = data["entity_sentiment"][:10]
    categories = data["categories"][:5]

    entity_lines = "\n".join(
        f"  - {e['name']} ({e['type']}) salience={e['salience']*100:.1f}% mentions={e['mentions']} KG={'yes' if e['wikipedia'] else 'no'}"
        for e in top_entities
    )
    ent_sent_lines = "\n".join(
        f"  - {e['name']}: score={e['score']:+.2f} magnitude={e['magnitude']:.2f}"
        for e in top_entity_sentiment
    )
    cat_lines = "\n".join(
        f"  - {c['category']} ({c['confidence']*100:.0f}%)"
        for c in categories
    ) or "  - No categories detected"

    detail_instruction = (
        "Give a SHORT report: 5 bullet points maximum, each one actionable. No long explanations."
        if detail == "Short (5 bullets)"
        else
        "Give a DETAILED report with: 1) Overall assessment, 2) What is working well, "
        "3) Top 3 problems with explanation + 2 concrete before/after examples for each, "
        "4) Priority action list ranked by impact, 5) Ideal target values for each metric."
    )

    lang_instruction = (
        "Write the entire report in Slovenian language."
        if language == "Slovenščina"
        else "Write the entire report in English."
    )

    cs = data.get("claude_syntax", {})
    is_slo = data.get("content_language", "English") == "Slovenščina"

    # Use Claude POS data for Slovenian if available
    verb_count = cs.get("verb_count", sx["verb_count"]) if is_slo and cs else sx["verb_count"]
    adj_count  = cs.get("adjective_count", sx["adjective_count"]) if is_slo and cs else sx["adjective_count"]
    passive_n  = cs.get("passive_voice_count", 0) if is_slo and cs else 0
    total_w    = cs.get("total_words", sx["total_tokens"]) if is_slo and cs else sx["total_tokens"]
    passive_pct = round(passive_n / total_w * 100, 1) if total_w and is_slo and cs else sx["passive_voice_pct"]
    top_verbs  = ", ".join(cs.get("top_verbs", [])[:8]) if is_slo and cs else "N/A"
    top_adjs   = ", ".join(cs.get("top_adjectives", [])[:8]) if is_slo and cs else "N/A"
    claude_note = cs.get("notes", "") if is_slo and cs else ""

    slo_warning = """
IMPORTANT LANGUAGE NOTE: This is Slovenian content.
- Verb and adjective counts are from Claude AI analysis (accurate for Slovenian)
- Google NLP API verb/adjective counts for Slovenian are UNRELIABLE due to morphology parsing issues
- Do NOT suggest adding more verbs/adjectives based purely on low counts — the text may already have enough
- Focus your advice on: sentiment, entity salience, entity sentiment, content categories
- For syntax advice, focus on: passive voice %, lexical density, and the actual top verbs/adjectives found
""" if is_slo else ""

    neg_sentences = [s for s in data["sentiment"].get("sentences", []) if s["score"] <= -0.25]
    neg_sent_lines = "\n".join(f'  - (score {s["score"]:+.2f}) "{s["text"][:100]}"' for s in neg_sentences[:5])

    prompt = f"""You are an expert SEO consultant. Analyze this NLP data and give specific, actionable SEO advice.
{slo_warning}
TARGET KEYWORD: {keyword if keyword else 'not specified'}
CONTENT LANGUAGE: {data.get('content_language', 'English')}

SENTIMENT:
  Score: {s['score']:+.3f} (range -1.0 to +1.0)
  Magnitude: {s['magnitude']:.2f} (emotional intensity)
  Sentences: {s['sentence_count']}
  Negative sentences to rewrite: {len(neg_sentences)}
{neg_sent_lines}

SYNTAX ({'Claude AI analysis' if is_slo else 'Google NLP API'}):
  Passive voice: {passive_pct:.1f}% ({passive_n} instances in {total_w} words)
  Lexical density: {sx['lexical_density']:.1%}
  Nouns (Google): {sx['noun_count']}
  Verbs ({'Claude' if is_slo else 'Google'}): {verb_count}
  Adjectives ({'Claude' if is_slo else 'Google'}): {adj_count}
  Top nouns: {', '.join(n for n, _ in sx['top_nouns'][:8])}
  Top verbs: {top_verbs}
  Top adjectives: {top_adjs}
{f'  Claude note: {claude_note}' if claude_note else ''}

TOP ENTITIES (what Google sees as main topics):
{entity_lines}

ENTITY SENTIMENT (how each topic is talked about):
{ent_sent_lines}

CONTENT CATEGORIES (what Google classifies this page as):
{cat_lines}

{detail_instruction}
{lang_instruction}

Be specific and direct. Reference actual numbers. Give concrete before/after examples in the content's language."""

    response = client.messages.create(
        model="claude-sonnet-4-5",
        max_tokens=1500 if detail == "Short (5 bullets)" else 3000,
        messages=[{"role": "user", "content": prompt}],
    )
    return response.content[0].text


def tab_ai_coach(data: dict, keyword: str):
    if get_anthropic_client() is None:
        st.warning("Claude AI API key not configured. Add ANTHROPIC_API_KEY to Streamlit Secrets.")
        return

    col1, col2 = st.columns(2)
    language = col1.radio("Language", ["English", "Slovenščina"], horizontal=True,
                          key="ai_language")
    detail   = col2.radio("Detail level", ["Short (5 bullets)", "Detailed report"],
                          horizontal=True, key="ai_detail")

    if st.button("🤖 Generate AI SEO Report", type="primary", use_container_width=True,
                 key="ai_generate_btn"):
        client = get_anthropic_client()
        if client is None:
            st.error("ANTHROPIC_API_KEY not found in Streamlit Secrets. Add it and reboot the app.")
        else:
            with st.spinner("Claude is analyzing your content..."):
                try:
                    report = generate_seo_report(data, keyword, language, detail)
                    st.session_state["ai_report"] = report
                except Exception as e:
                    st.error(f"Claude API error: {e}")
                    st.session_state["ai_report"] = ""

    if "ai_report" in st.session_state and st.session_state["ai_report"]:
        report = st.session_state["ai_report"]
        st.markdown("---")
        st.markdown(report)


# ── HTML text extractor ───────────────────────────────────────────────────────

class _TextExtractor(HTMLParser):
    SKIP_TAGS = {"script", "style", "nav", "footer", "header", "noscript", "meta"}

    def __init__(self):
        super().__init__()
        self._skip = 0
        self.chunks = []

    def handle_starttag(self, tag, attrs):
        if tag in self.SKIP_TAGS:
            self._skip += 1

    def handle_endtag(self, tag):
        if tag in self.SKIP_TAGS and self._skip:
            self._skip -= 1

    def handle_data(self, data):
        if not self._skip:
            s = data.strip()
            if s:
                self.chunks.append(s)


def fetch_url_text(url: str) -> str:
    req = Request(url, headers={"User-Agent": "Mozilla/5.0 (SEO-NLP-Analyzer)"})
    try:
        with urlopen(req, timeout=15) as r:
            html = r.read().decode("utf-8", errors="replace")
    except URLError as e:
        st.error(f"Could not fetch URL: {e}")
        return ""
    p = _TextExtractor()
    p.feed(html)
    return " ".join(p.chunks)


# ── NLP analysis ──────────────────────────────────────────────────────────────

def _parse_syntax_tokens(tokens) -> dict:
    total = passive = 0
    pos_counts: dict = {}
    lemma_counts: dict = {}

    for token in tokens:
        total += 1
        tag = language_v1.PartOfSpeech.Tag(token.part_of_speech.tag).name
        pos_counts[tag] = pos_counts.get(tag, 0) + 1
        if language_v1.PartOfSpeech.Voice(token.part_of_speech.voice).name == "PASSIVE":
            passive += 1
        if tag in ("NOUN", "PROPN"):
            lemma = token.lemma.lower()
            if len(lemma) > 2:
                lemma_counts[lemma] = lemma_counts.get(lemma, 0) + 1

    nouns = pos_counts.get("NOUN", 0) + pos_counts.get("PROPN", 0)
    verbs = pos_counts.get("VERB", 0)
    adjs  = pos_counts.get("ADJ", 0)
    advs  = pos_counts.get("ADV", 0)
    cw    = nouns + verbs + adjs + advs

    return {
        "total_tokens":      total,
        "passive_voice_pct": round(passive / total * 100, 1) if total else 0,
        "lexical_density":   round(cw / total, 3) if total else 0,
        "top_nouns": sorted(lemma_counts.items(), key=lambda x: x[1], reverse=True)[:15],
        "noun_count":      nouns,
        "verb_count":      verbs,
        "adjective_count": adjs,
        "adverb_count":    advs,
    }


@st.cache_data(show_spinner=False)
def run_analysis(text: str, content_language: str = "English") -> dict:
    client = get_client()
    is_slo = content_language == "Slovenščina"

    # Google NLP API has very limited Slovenian support:
    # - entity analysis: NOT supported for "sl"
    # - syntax: NOT supported for "sl"
    # Solution: always send as "en" to Google (works for entity/category detection
    # on any language text). Claude handles Slovenian-specific syntax analysis.
    document = {
        "content": text,
        "type_": language_v1.Document.Type.PLAIN_TEXT,
        "language": "en",
    }
    enc = language_v1.EncodingType.UTF8

    # Single annotateText call for both languages
    features = language_v1.AnnotateTextRequest.Features(
        extract_syntax=True,
        extract_entities=True,
        extract_document_sentiment=True,
        extract_entity_sentiment=True,
    )
    resp = client.annotate_text(
        request={"document": document, "features": features, "encoding_type": enc}
    )

    entities = sorted([{
        "name":      e.name,
        "type":      language_v1.Entity.Type(e.type_).name,
        "salience":  round(e.salience, 4),
        "mentions":  len(e.mentions),
        "wikipedia": e.metadata.get("wikipedia_url", ""),
    } for e in resp.entities], key=lambda x: x["salience"], reverse=True)

    sentences = [{
        "text":      s.text.content,
        "score":     round(s.sentiment.score, 3),
        "magnitude": round(s.sentiment.magnitude, 3),
    } for s in resp.sentences]

    sentiment = {
        "score":          round(resp.document_sentiment.score, 3),
        "magnitude":      round(resp.document_sentiment.magnitude, 3),
        "sentence_count": len(resp.sentences),
        "sentences":      sentences,
    }

    syntax = _parse_syntax_tokens(resp.tokens)

    entity_sentiment = sorted([{
        "name":      e.name,
        "type":      language_v1.Entity.Type(e.type_).name,
        "salience":  round(e.salience, 4),
        "score":     round(e.sentiment.score, 3),
        "magnitude": round(e.sentiment.magnitude, 3),
        "wikipedia": e.metadata.get("wikipedia_url", ""),
    } for e in resp.entities], key=lambda x: x["salience"], reverse=True)

    categories = []
    if len(text.split()) >= 20:
        try:
            cv = language_v1.ClassificationModelOptions.V2Model.ContentCategoriesVersion.V2
            cat_resp = client.classify_text(
                request={
                    "document": document,
                    "classification_model_options": {
                        "v2_model": {"content_categories_version": cv}
                    },
                }
            )
            categories = sorted([{
                "category":   c.name,
                "confidence": round(c.confidence, 3),
            } for c in cat_resp.categories], key=lambda x: x["confidence"], reverse=True)
        except Exception:
            # classifyText has limited Slovenian support — skip if fails
            categories = []

    # For Slovenian: replace unreliable Google POS with Claude analysis
    claude_syntax = {}
    if content_language == "Slovenščina":
        claude_syntax = analyze_slovenian_syntax(text)

    return {
        "entities":         entities,
        "sentiment":        sentiment,
        "syntax":           syntax,
        "entity_sentiment": entity_sentiment,
        "categories":       categories,
        "claude_syntax":    claude_syntax,
        "content_language": content_language,
    }


# ── Report export ─────────────────────────────────────────────────────────────

def build_markdown_report(data: dict, keyword: str, source: str, ai_report: str = "") -> str:
    s  = data["sentiment"]
    sx = data["syntax"]
    ts = datetime.now().strftime("%Y-%m-%d %H:%M")

    lines = []
    lines.append(f"# SEO NLP Analysis Report")
    lines.append(f"**Source:** {source}")
    lines.append(f"**Keyword:** {keyword if keyword else '—'}")
    lines.append(f"**Date:** {ts}")
    lines.append("")

    # ── Overview ──────────────────────────────────────────────────────────────
    lines.append("---")
    lines.append("## 📊 Overview")
    lines.append("")
    lines.append(f"| Metric | Value | Source |")
    lines.append(f"|---|---|---|")
    lines.append(f"| Sentiment Score | {s['score']:+.3f} | 🔵 Official Google NLP API |")
    lines.append(f"| Magnitude | {s['magnitude']:.3f} | 🔵 Official Google NLP API |")
    lines.append(f"| Sentences | {s['sentence_count']} | 🔵 Official Google NLP API |")
    lines.append(f"| Passive Voice | {sx['passive_voice_pct']:.1f}% | 🔵 API detection · 🟠 15% threshold = best practice |")
    lines.append(f"| Lexical Density | {sx['lexical_density']:.1%} | 🔵 API token counts · 🟠 40% threshold = best practice |")
    lines.append(f"| Nouns | {sx['noun_count']} | 🔵 Official Google NLP API |")
    lines.append(f"| Verbs | {sx['verb_count']} | 🔵 Official Google NLP API |")
    lines.append(f"| Adjectives | {sx['adjective_count']} | 🔵 Official Google NLP API |")
    lines.append(f"| Adverbs | {sx['adverb_count']} | 🔵 Official Google NLP API |")
    lines.append("")

    # ── Keyword check ─────────────────────────────────────────────────────────
    if keyword:
        kw_lower = keyword.lower()
        matches = [e for e in data["entities"] if kw_lower in e["name"].lower()]
        if matches:
            top = matches[0]
            sal = top["salience"] * 100
            kg  = "in Knowledge Graph ✓" if top["wikipedia"] else "not in Knowledge Graph"
            if sal >= 15:
                interp = f"✓ Excellent — Google clearly sees '{keyword}' as the main topic of this page."
            elif sal >= 8:
                interp = (
                    f"⚠ OK but too low. Target is 15%+. "
                    f"Add '{keyword}' to H1, H2 headings and opening paragraph to increase salience."
                )
            else:
                interp = (
                    f"⚠ Too low — Google does not see '{keyword}' as the main topic of this page. "
                    f"Add related entities: subtopics, materials, use cases, product types. "
                    f"Use '{keyword}' in H1, first 100 words, and at least 2–3 H2 headings."
                )
            lines.append(f"**Target keyword '{keyword}':** Salience {sal:.1f}% · Type: {top['type']} · {kg}")
            lines.append(f"→ {interp}")
        else:
            lines.append(f"**Target keyword '{keyword}':** ⚠ NOT detected as entity at all.")
            lines.append(
                f"→ Google cannot identify '{keyword}' as a topic. "
                f"Add it to H1, first 100 words, at least 2–3 H2 headings, and throughout body text."
            )
        lines.append("")

    # ── Categories ────────────────────────────────────────────────────────────
    lines.append("---")
    lines.append("## 📂 Content Categories")
    lines.append("*🔵 Official Google NLP API*")
    lines.append("")
    if data["categories"]:
        lines.append("| Category | Confidence |")
        lines.append("|---|---|")
        for c in data["categories"]:
            lines.append(f"| {c['category']} | {c['confidence']*100:.0f}% |")
    else:
        lines.append("*Not enough text to classify.*")
    lines.append("")

    # ── Entities ──────────────────────────────────────────────────────────────
    lines.append("---")
    lines.append("## 🏷 Entities (Top 20)")
    lines.append("*🔵 Official Google NLP API — salience, type, Knowledge Graph*")
    lines.append("")
    lines.append("| # | Entity | Type | Salience % | Mentions | KG |")
    lines.append("|---|---|---|---|---|---|")
    for i, e in enumerate(data["entities"][:20], 1):
        kg = "✓" if e["wikipedia"] else ""
        lines.append(f"| {i} | {e['name']} | {e['type']} | {e['salience']*100:.1f}% | {e['mentions']} | {kg} |")
    lines.append("")

    # ── Sentiment ─────────────────────────────────────────────────────────────
    lines.append("---")
    lines.append("## 😊 Sentiment Analysis")
    lines.append("*🔵 Official Google NLP API · 🟠 Thresholds = SEO best practice*")
    lines.append("")
    score = s["score"]
    mag   = s["magnitude"]
    if score >= 0.4:
        tone = "Clearly Positive ✓"
    elif score >= 0.1:
        tone = "Slightly Positive — below target for product pages"
    elif score >= -0.1:
        tone = f"Neutral — {'mixed signals (high magnitude)' if mag > 5 else 'calm/factual'}"
    else:
        tone = "Negative ✗ — rewrite needed"
    lines.append(f"- **Score:** {score:+.3f} → {tone}")
    lines.append(f"- **Magnitude:** {mag:.3f} → {'High emotional intensity' if mag > 8 else 'Moderate' if mag > 3 else 'Low intensity'}")
    lines.append(f"- **Sentences analyzed:** {s['sentence_count']}")
    lines.append("")

    # ── Syntax ────────────────────────────────────────────────────────────────
    lines.append("---")
    lines.append("## 🔤 Syntax Analysis")
    lines.append("*🔵 API token detection · 🟠 Thresholds = SEO/linguistic best practice · 🟣 Active voice = Google writing guidelines*")
    lines.append("")
    pv = sx["passive_voice_pct"]
    ld = sx["lexical_density"]
    lines.append(f"- **Passive voice:** {pv:.1f}% {'⚠ HIGH — rewrite to active voice' if pv > 15 else '✓ Good'}")
    lines.append(f"- **Lexical density:** {ld:.1%} {'⚠ LOW — add more specific details' if ld < 0.40 else '✓ Good'}")
    lines.append(f"- **Verb/Noun ratio:** {sx['verb_count']}/{sx['noun_count']} = {sx['verb_count']/max(sx['noun_count'],1)*100:.1f}%")
    lines.append("")
    if sx["top_nouns"]:
        lines.append("**Top implied topics (nouns):**")
        lines.append(", ".join(f"{n} ({c}x)" for n, c in sx["top_nouns"][:10]))
    lines.append("")

    # ── Sentence Sentiment ────────────────────────────────────────────────────
    sentences = data["sentiment"].get("sentences", [])
    if sentences:
        lines.append("---")
        lines.append("## 📝 Sentence-Level Sentiment")
        lines.append("*🔵 Official Google NLP API — sentiment score per sentence*")
        lines.append("")
        neg_sents = [s for s in sentences if s["score"] <= -0.25]
        pos_sents = [s for s in sentences if s["score"] >= 0.25]
        lines.append(f"- 🟢 Positive sentences: {len(pos_sents)}")
        lines.append(f"- 🟡 Neutral sentences: {len(sentences) - len(neg_sents) - len(pos_sents)}")
        lines.append(f"- 🔴 Negative sentences: {len(neg_sents)}")
        lines.append("")
        if neg_sents:
            lines.append("**⚠ Negative sentences to rewrite:**")
            lines.append("")
            for i, s in enumerate(sorted(neg_sents, key=lambda x: x["score"]), 1):
                lines.append(f"{i}. (score {s['score']:+.2f}) *{s['text'][:200]}*")
            lines.append("")
        lines.append("**All sentences:**")
        lines.append("")
        lines.append("| # | Score | Tone | Sentence |")
        lines.append("|---|---|---|---|")
        for i, s in enumerate(sentences, 1):
            tone = "🟢 Positive" if s["score"] >= 0.25 else "🔴 Negative" if s["score"] <= -0.25 else "🟡 Neutral"
            text = s["text"][:120].replace("|", "\\|")
            lines.append(f"| {i} | {s['score']:+.2f} | {tone} | {text} |")
        lines.append("")

    # ── Entity Sentiment ──────────────────────────────────────────────────────
    lines.append("---")
    lines.append("## 🎯 Entity Sentiment")
    lines.append("*🔵 Official Google NLP API · 🟠 ±0.25 threshold = SEO best practice · 🟣 E-E-A-T = Google Quality Rater Guidelines*")
    lines.append("")
    lines.append("| Entity | Type | Salience % | Score | Magnitude | Tone |")
    lines.append("|---|---|---|---|---|---|")
    for e in data["entity_sentiment"][:20]:
        tone = "Positive" if e["score"] >= 0.25 else "Negative" if e["score"] <= -0.25 else "Neutral"
        lines.append(f"| {e['name']} | {e['type']} | {e['salience']*100:.1f}% | {e['score']:+.2f} | {e['magnitude']:.2f} | {tone} |")
    lines.append("")

    # ── AI SEO Coach ──────────────────────────────────────────────────────────
    if ai_report:
        lines.append("---")
        lines.append("## 🤖 AI SEO Coach Report")
        lines.append("*Generated by Claude AI*")
        lines.append("")
        lines.append(ai_report)
        lines.append("")

    # ── Footer ────────────────────────────────────────────────────────────────
    lines.append("---")
    lines.append("*Generated by SEO NLP Analyzer · Powered by Google Cloud Natural Language API + Claude AI*")
    lines.append("")
    lines.append("**Legend:**")
    lines.append("- 🔵 Official Google data — directly from Google NLP API")
    lines.append("- 🟠 SEO best practice — industry standard, not officially confirmed by Google")
    lines.append("- 🟣 Google guidelines — from Google's Quality Rater Guidelines / Search documentation")

    return "\n".join(lines)


# ── UI helpers ────────────────────────────────────────────────────────────────

def _tone_label(score: float) -> str:
    if score >= 0.25:  return "Positive"
    if score <= -0.25: return "Negative"
    return "Neutral"

def _tone_icon(score: float) -> str:
    if score >= 0.25:  return "🟢"
    if score <= -0.25: return "🔴"
    return "🟡"

def _color_tone(val: str) -> str:
    return {"Positive": "color:green", "Negative": "color:red"}.get(val, "color:gray")

# Badge constants
OFFICIAL  = "🔵 **Official Google data**"
PRACTICE  = "🟠 **SEO best practice**"
GUIDELINE = "🟣 **Google guidelines**"

def _source_legend():
    st.caption(
        "🔵 Official Google data — directly from Google NLP API · "
        "🟠 SEO best practice — industry standard, not officially confirmed by Google · "
        "🟣 Google guidelines — from Google's public Quality Rater Guidelines / Search documentation"
    )


# ── Tab renderers ─────────────────────────────────────────────────────────────

def tab_overview(data: dict, keyword: str):
    s  = data["sentiment"]
    sx = data["syntax"]

    c1, c2, c3, c4, c5 = st.columns(5)
    c1.metric("Sentiment",       f"{s['score']:+.2f}",
              help="Čustveni ton besedila. Od −1.0 (zelo negativno) do +1.0 (zelo pozitivno). Za produktne strani cilj: +0.4 ali več. 🔵 Uradni Google podatek.")
    c2.metric("Magnitude",       f"{s['magnitude']:.2f}",
              help="Moč čustev v besedilu. Nizko (0–1) = suho/faktično. Visoko (5+) = čustveno angažirano. Nevtralen score + visoka magnitude = mešano besedilo. 🔵 Uradni Google podatek.")
    c3.metric("Sentences",       s["sentence_count"],
              help="Število stavkov ki jih je Google zaznal v besedilu. 🔵 Uradni Google podatek.")
    c4.metric("Passive Voice",   f"{sx['passive_voice_pct']:.1f}%",
              delta="Previsoko ⚠" if sx["passive_voice_pct"] > 15 else "OK ✓",
              delta_color="inverse",
              help="Pasivni glas: 'Bazen je bil postavljen' (slabo) vs 'Montažer postavi bazen' (dobro). Cilj: pod 15%. 🔵 Google zazna · 🟠 15% prag = SEO best practice")
    c5.metric("Lexical Density", f"{sx['lexical_density']:.1%}",
              help="Koliko % besed nosi pravi pomen (samostalniki, glagoli, pridevniki). Nizko = preveč splošnih besed. Cilj: 40%+. 🔵 Google šteje tokene · 🟠 40% prag = SEO best practice")

    _source_legend()
    st.divider()

    # ── Sentiment interpretation ──────────────────────────────────────────────
    score = s["score"]
    mag   = s["magnitude"]

    if score >= 0.4:
        st.success(f"**Sentiment {score:+.2f} — Clearly positive ✓**  Ideal for product and service pages.")
        st.caption(f"{OFFICIAL} — score calculated by Google NLP API · {PRACTICE} — +0.4 threshold for product pages is industry recommendation")
    elif score >= 0.1:
        col_a, col_b = st.columns([1, 2])
        col_a.warning(f"**Sentiment {score:+.2f} — Slightly positive**")
        col_b.markdown(f"""
**For a product/service page this is too low.** Target is +0.4 or higher.

**Why:** {score:+.2f} score + magnitude {mag:.1f} = mixed content.
Some parts of your page are positive, others negative — they cancel each other out.

**What to do:**
- Find sentences that express doubt, problems, or negatives and rewrite them positively
- Add more benefit-focused language: *"saves time"*, *"lasts 20 years"*, *"easy to maintain"*

**Example — Before:** *"Pool maintenance can be complex and time-consuming."*
**Example — After:** *"With our pools, maintenance takes less than 30 minutes a week."*
""")
        st.caption(f"{OFFICIAL} — score & magnitude from Google NLP API · {PRACTICE} — +0.4 target for product pages is SEO industry standard, not confirmed by Google")
    elif score >= -0.1:
        col_a, col_b = st.columns([1, 2])
        col_a.warning(f"**Sentiment {score:+.2f} — Neutral**")
        col_b.markdown(f"""
**Neutral + magnitude {mag:.1f} = mixed signals.**
{"Some parts positive, some negative — they cancel out. Most common problem on product pages." if mag > 5 else "Very calm, factual writing — fine for informational pages, but too dry for product pages."}

**Golden range for product pages:** Score +0.4 to +0.7, Magnitude 3–8

**What to do:**
- Add customer benefits after every feature statement
- Use emotionally positive words: *reliable, durable, effortless, beautiful, proven*

**Example — Before:** *"The pool is made of fiberglass."*
**Example — After:** *"The fiberglass construction ensures a smooth surface, easy cleaning, and a lifespan of over 30 years."*
""")
        st.caption(f"{OFFICIAL} — score & magnitude from Google NLP API · {PRACTICE} — golden range +0.4 to +0.7 is SEO best practice, not officially set by Google")
    else:
        st.error(f"**Sentiment {score:+.2f} — Negative ✗**  This will hurt CTR. Rewrite negative sentences.")
        st.caption(f"{OFFICIAL} — score from Google NLP API · {GUIDELINE} — Google's Quality Rater Guidelines state content should be helpful and positive for users")

    # ── Lexical density interpretation ────────────────────────────────────────
    ld = sx["lexical_density"]
    if ld < 0.40:
        col_a, col_b = st.columns([1, 2])
        col_a.warning(f"**Lexical density {ld:.1%} — Too low**")
        col_b.markdown(f"""
**Your content has too many filler words and not enough substance.**
Lexical density measures what % of your words carry real meaning (nouns, verbs, adjectives).
At {ld:.1%} you are just below the 40% threshold.

**What to do:** Replace vague phrases with specific details:
- Add dimensions, materials, weights, capacities
- Add specific product names, model numbers, technical specs
- Add concrete benefits with numbers: *"heats up 3x faster"*, *"saves €200/year"*

**Example — Before (fluffy):** *"Our pools are great and very high quality."*
**Example — After (dense):** *"Our fibreglass pools (4×8m, 1.5m depth) withstand temperatures from −20°C to +50°C and require no repainting for 25 years."*
""")
        st.caption(f"{OFFICIAL} — token counts (nouns/verbs/adjectives) from Google NLP API · {PRACTICE} — 40% threshold is linguistic/SEO best practice, not set by Google")
    else:
        st.success(f"**Lexical density {ld:.1%} — Good ✓**  Content is substantive and information-rich.")
        st.caption(f"{OFFICIAL} — token counts from Google NLP API · {PRACTICE} — 40% threshold is SEO best practice")

    # ── Keyword check ─────────────────────────────────────────────────────────
    if keyword:
        st.divider()
        kw_lower = keyword.lower()
        matches = [e for e in data["entities"] if kw_lower in e["name"].lower()]
        if matches:
            top = matches[0]
            sal = top["salience"] * 100
            kg  = "in Knowledge Graph ✓" if top["wikipedia"] else "not in Knowledge Graph"
            if sal >= 15:
                st.success(f"**'{keyword}'** — Salience **{sal:.1f}%** · Type: {top['type']} · {kg} — Excellent, Google clearly sees this as the main topic.")
            elif sal >= 8:
                st.warning(f"**'{keyword}'** — Salience **{sal:.1f}%** · Type: {top['type']} · {kg} — OK but could be stronger. Target is 15%+. Add keyword to H1, H2 headings and opening paragraph.")
            else:
                st.error(f"**'{keyword}'** — Salience only **{sal:.1f}%** · Type: {top['type']} · {kg} — Too low. Google does not see this as the main topic of your page. Add related entities: subtopics, materials, use cases, product types.")
        else:
            st.error(
                f"**'{keyword}'** not detected as an entity at all. "
                "This means Google cannot identify it as the main topic. "
                "Add it to H1, first 100 words, at least 2–3 H2 headings, and throughout the body text."
            )


def tab_entities(data: dict, keyword: str):
    df = pd.DataFrame(data["entities"])
    if df.empty:
        st.info("No entities found.")
        return

    df["salience %"] = (df["salience"] * 100).round(1)
    df["KG"]         = df["wikipedia"].apply(lambda x: "✓" if x else "")
    display          = df[["name", "type", "salience %", "mentions", "KG"]].copy()

    if keyword:
        kw_lower = keyword.lower()
        def _highlight(row):
            return (["background-color:#fff3cd"] * len(row)
                    if kw_lower in str(row["name"]).lower() else [""] * len(row))
        st.dataframe(display.style.apply(_highlight, axis=1),
                     use_container_width=True, hide_index=True)
    else:
        st.dataframe(display, use_container_width=True, hide_index=True)

    st.caption("Top 10 entities by salience")
    st.bar_chart(df.head(10).set_index("name")["salience %"])
    _source_legend()
    st.caption(f"{OFFICIAL} — entity names, types, salience scores, mention counts, Knowledge Graph links are all directly from Google NLP API · {PRACTICE} — 15% salience target for main keyword is SEO industry recommendation")


def tab_sentiment(data: dict):
    s     = data["sentiment"]
    score = s["score"]

    col1, col2 = st.columns(2)
    with col1:
        st.subheader(f"{_tone_icon(score)} {_tone_label(score)}")
        st.progress((score + 1) / 2,
                    text=f"Score: {score:+.3f}  (−1.0 negative → +1.0 positive)")
        st.progress(min(s["magnitude"] / 10, 1.0),
                    text=f"Magnitude: {s['magnitude']:.3f}  (emotional intensity)")
        st.caption(f"{s['sentence_count']} sentences analyzed")

    with col2:
        st.subheader("SEO interpretation")
        if score >= 0.25:
            st.success("Positive tone — ideal for product pages, reviews, and testimonials.")
        elif score <= -0.25:
            st.error("Negative tone detected — may hurt CTR. Consider softening your language.")
        else:
            st.info("Neutral tone — appropriate for informational and educational content.")

        mag = s["magnitude"]
        if mag < 1.0:
            st.info("Low emotional intensity — calm, factual writing.")
        elif mag < 5.0:
            st.success("Moderate emotional intensity — engaged, readable writing.")
        else:
            st.warning("High intensity — make sure the overall sentiment is positive.")


def tab_categories(data: dict):
    cats = data["categories"]
    if not cats:
        st.warning("Not enough text to classify (minimum 20 words required).")
        return

    df = pd.DataFrame(cats)
    df["confidence %"] = (df["confidence"] * 100).round(1)
    st.dataframe(df[["category", "confidence %"]], use_container_width=True, hide_index=True)
    st.bar_chart(df.set_index("category")["confidence %"])

    top = cats[0]["category"].split("/")[-1]
    st.info(
        f"Primary topic Google assigns: **{top}** · "
        "Confirm this matches your target keyword cluster for topical alignment."
    )
    _source_legend()
    st.caption(f"{OFFICIAL} — categories and confidence scores are directly from Google NLP API · {PRACTICE} — topical alignment recommendation is SEO best practice")


def tab_syntax(data: dict):
    sx  = data["syntax"]
    cs  = data.get("claude_syntax", {})
    is_slo = data.get("content_language", "English") == "Slovenščina"

    # For Slovenian: use Claude counts; for English: use Google counts
    verb_count = cs.get("verb_count", sx["verb_count"]) if is_slo and cs else sx["verb_count"]
    adj_count  = cs.get("adjective_count", sx["adjective_count"]) if is_slo and cs else sx["adjective_count"]
    adv_count  = cs.get("adverb_count", sx["adverb_count"]) if is_slo and cs else sx["adverb_count"]
    passive_n  = cs.get("passive_voice_count", 0) if is_slo and cs else 0
    total_w    = cs.get("total_words", sx["total_tokens"]) if is_slo and cs else sx["total_tokens"]
    passive_pct = round(passive_n / total_w * 100, 1) if total_w and is_slo and cs else sx["passive_voice_pct"]
    ld = sx["lexical_density"]

    if is_slo and cs:
        st.info("🤖 Verb, adjective and passive voice counts powered by **Claude AI** (more accurate for Slovenian)")

    col1, col2 = st.columns(2)
    with col1:
        source_label = "🤖 Claude AI" if is_slo and cs else "🔵 Google NLP API"
        st.subheader(f"Token breakdown ({source_label})")
        token_df = pd.DataFrame([
            {"Part of Speech": "Nouns",      "Count": sx["noun_count"],  "Source": "🔵 Google"},
            {"Part of Speech": "Verbs",      "Count": verb_count,        "Source": source_label},
            {"Part of Speech": "Adjectives", "Count": adj_count,         "Source": source_label},
            {"Part of Speech": "Adverbs",    "Count": adv_count,         "Source": source_label},
        ])
        st.dataframe(token_df, use_container_width=True, hide_index=True)
        st.metric("Total words", total_w)

        if is_slo and cs and cs.get("top_verbs"):
            st.caption(f"Top verbs: {', '.join(cs['top_verbs'][:8])}")
        if is_slo and cs and cs.get("top_adjectives"):
            st.caption(f"Top adjectives: {', '.join(cs['top_adjectives'][:8])}")

    with col2:
        st.subheader("Quality signals")
        st.progress(min(passive_pct / 30, 1.0), text=f"Passive voice: {passive_pct:.1f}%")
        if passive_pct > 15:
            st.error(f"⚠ {passive_pct:.1f}% passive voice is high — rewrite to active voice.")
        else:
            st.success(f"✓ Passive voice at {passive_pct:.1f}% — good.")

        if is_slo and cs and cs.get("passive_examples"):
            with st.expander("Passive voice examples found"):
                for ex in cs["passive_examples"]:
                    st.warning(ex)

        st.progress(min(ld / 0.7, 1.0), text=f"Lexical density: {ld:.1%}")
        if ld < 0.40:
            st.warning("Low lexical density — add more specific nouns and concrete details.")
        else:
            st.success(f"✓ Lexical density at {ld:.1%} — content is substantive.")

        if is_slo and cs and cs.get("notes"):
            st.info(f"📝 Claude note: {cs['notes']}")

    if sx["top_nouns"]:
        st.subheader("Top implied topics (most frequent nouns)")
        noun_df = pd.DataFrame(sx["top_nouns"], columns=["noun", "count"])
        st.bar_chart(noun_df.set_index("noun")["count"])
        st.caption("🔵 Google NLP API — If your target keyword isn't in the top nouns, increase its usage in headings.")

    _source_legend()
    if is_slo:
        st.caption(
            f"🔵 Google NLP API: noun counts, lexical density · "
            f"🤖 Claude AI: verb counts, adjective counts, passive voice (more accurate for Slovenian) · "
            f"{PRACTICE} — passive voice 15% and lexical density 40% thresholds are SEO best practice · "
            f"{GUIDELINE} — Google recommends active voice"
        )
    else:
        st.caption(
            f"{OFFICIAL} — all token counts detected by Google NLP API · "
            f"{PRACTICE} — passive voice 15% and lexical density 40% thresholds are SEO best practice · "
            f"{GUIDELINE} — Google recommends active voice"
        )


def tab_sentence_sentiment(data: dict):
    sentences = data["sentiment"].get("sentences", [])
    if not sentences:
        st.info("No sentence data available.")
        return

    df = pd.DataFrame(sentences)
    df["tone"]  = df["score"].apply(_tone_label)
    df["icon"]  = df["score"].apply(_tone_icon)
    df["#"]     = range(1, len(df) + 1)
    display     = df[["#", "text", "score", "magnitude", "tone"]].copy()

    # Summary counts
    neg_count  = len(df[df["score"] <= -0.25])
    pos_count  = len(df[df["score"] >= 0.25])
    neu_count  = len(df) - neg_count - pos_count

    c1, c2, c3 = st.columns(3)
    c1.metric("🟢 Positive sentences", pos_count)
    c2.metric("🟡 Neutral sentences",  neu_count)
    c3.metric("🔴 Negative sentences", neg_count,
              delta="Rewrite these ⚠" if neg_count > 0 else None,
              delta_color="inverse")

    st.divider()

    # Filter
    filter_opt = st.radio(
        "Show",
        ["All", "🔴 Negative only", "🟢 Positive only"],
        horizontal=True,
        key="sent_filter",
    )
    if filter_opt == "🔴 Negative only":
        display = display[df["score"] <= -0.25]
    elif filter_opt == "🟢 Positive only":
        display = display[df["score"] >= 0.25]

    def _highlight_sent(row):
        score = row["score"]
        if score <= -0.25:
            return ["background-color:#3d1a1a"] * len(row)
        if score >= 0.25:
            return ["background-color:#1a3d1a"] * len(row)
        return [""] * len(row)

    st.dataframe(
        display.style.apply(_highlight_sent, axis=1),
        use_container_width=True,
        hide_index=True,
        column_config={
            "text":      st.column_config.TextColumn("Sentence", width="large"),
            "score":     st.column_config.NumberColumn("Score", format="%+.2f"),
            "magnitude": st.column_config.NumberColumn("Magnitude", format="%.2f"),
            "tone":      st.column_config.TextColumn("Tone"),
        }
    )

    if neg_count > 0:
        st.divider()
        st.subheader("🔴 Sentences to rewrite:")
        neg_sentences = df[df["score"] <= -0.25].sort_values("score")
        for _, row in neg_sentences.iterrows():
            st.error(f"**#{int(row['#'])} (score {row['score']:+.2f}):** {row['text']}")

    _source_legend()
    st.caption(f"{OFFICIAL} — sentence-level sentiment scores directly from Google NLP API")


def tab_entity_sentiment(data: dict):
    df = pd.DataFrame(data["entity_sentiment"])
    if df.empty:
        st.info("No entity sentiment data found.")
        return

    df["salience %"] = (df["salience"] * 100).round(1)
    df["tone"]       = df["score"].apply(_tone_label)
    df["KG"]         = df["wikipedia"].apply(lambda x: "✓" if x else "")
    display          = df[["name", "type", "salience %", "score", "magnitude", "tone", "KG"]]

    st.dataframe(
        display.style.map(_color_tone, subset=["tone"]),
        use_container_width=True,
        hide_index=True,
    )

    negative = [e for e in data["entity_sentiment"] if e["score"] <= -0.25]
    positive = [e for e in data["entity_sentiment"] if e["score"] >= 0.25]

    if negative:
        names = ", ".join(f"**{e['name']}**" for e in negative[:3])
        st.error(
            f"Negative sentiment around: {names}. "
            "If these are your brand or product, rewrite the surrounding context."
        )
    if positive:
        names = ", ".join(f"**{e['name']}**" for e in positive[:3])
        st.success(f"Positive sentiment around: {names} — good for E-E-A-T signals.")
    if not negative and not positive:
        st.info("All entities are neutral — fine for informational content.")
    _source_legend()
    st.caption(
        f"{OFFICIAL} — entity names, types, salience, sentiment scores and magnitudes are directly from Google NLP API · "
        f"{PRACTICE} — ±0.25 threshold for positive/negative classification is SEO best practice · "
        f"{GUIDELINE} — E-E-A-T signals referenced from Google's Quality Rater Guidelines"
    )


def render_analysis(data: dict, keyword: str = "", source: str = ""):
    t1, t2, t3, t4, t5, t6, t7, t8 = st.tabs([
        "📊 Overview",
        "🏷 Entities",
        "😊 Sentiment",
        "📝 Sentences",
        "📂 Categories",
        "🔤 Syntax",
        "🎯 Entity Sentiment",
        "🤖 AI SEO Coach",
    ])
    with t1: tab_overview(data, keyword)
    with t2: tab_entities(data, keyword)
    with t3: tab_sentiment(data)
    with t4: tab_sentence_sentiment(data)
    with t5: tab_categories(data)
    with t6: tab_syntax(data)
    with t7: tab_entity_sentiment(data)
    with t8: tab_ai_coach(data, keyword)

    # ── Download button ───────────────────────────────────────────────────────
    st.divider()
    ai_report = st.session_state.get("ai_report", "")
    md = build_markdown_report(data, keyword, source, ai_report)
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    slug = keyword.replace(" ", "_").lower() if keyword else "analysis"
    filename = f"analiza_{slug}_{ts}.md"

    st.download_button(
        label="📥 Download Full Analysis",
        data=md,
        file_name=filename,
        mime="text/markdown",
        use_container_width=True,
        help="Downloads complete analysis as Markdown file. Save it to your 'analize' folder and use with Claude Code.",
    )
    st.caption("💡 Save to your `analize/` folder → open Claude Code → type: *'check analizo in priporocaj kako naprej'*")


# ── Info page ─────────────────────────────────────────────────────────────────

def page_info():
    st.title("ℹ️ How to use SEO NLP Analyzer")
    st.caption("Read this before you start — it will make your results 10x more useful.")

    st.markdown("---")

    st.header("What does this tool do?")
    st.markdown("""
This tool uses **Google's own Natural Language API** — the same AI Google uses internally
to understand web pages — to analyze your content and tell you exactly how Google reads it.

You get 6 types of analysis in one place:

| Analysis | What it tells you |
|---|---|
| **📊 Overview** | Quick summary of all key signals at a glance |
| **🏷 Entities** | Which topics/people/places Google extracts from your page and how important each is |
| **😊 Sentiment** | Whether your content reads as positive, negative or neutral |
| **📂 Categories** | What content category Google assigns to your page |
| **🔤 Syntax** | Writing quality — passive voice %, vocabulary richness |
| **🎯 Entity Sentiment** | How your content talks about each specific topic |
""")

    st.markdown("---")
    st.header("The 3 ways to analyze content")

    col1, col2, col3 = st.columns(3)
    with col1:
        st.subheader("1. Single URL")
        st.markdown("""
Paste your page URL.
Get a full analysis of how Google reads that page.

**Best for:**
- Auditing existing pages
- Checking pages before publishing
- Understanding why a page ranks or doesn't rank
""")
    with col2:
        st.subheader("2. URL vs Competitor")
        st.markdown("""
Paste your URL + a competitor's URL.
See both side by side.

**Best for:**
- Finding which entities competitors cover that you don't
- Understanding why they outrank you
- Spotting content gaps
""")
    with col3:
        st.subheader("3. Paste Article Text")
        st.markdown("""
Paste raw text directly — no URL needed.

**Best for:**
- Analyzing content before it's published
- Drafts, Google Docs, Word files
- Content that's behind a login
""")

    st.markdown("---")
    st.header("How to read each section")

    with st.expander("🏷 Entities — the most important tab for SEO"):
        st.markdown("""
**Salience %** = how central this topic is to your page. Higher = more important to Google.

- Your **target keyword should be in the top 3–5 entities** with high salience
- Entities marked **KG ✓** are in Google's Knowledge Graph — these carry extra SEO weight
- If your keyword is missing or has <5% salience → add it to your H1, first paragraph, and use it more throughout the text
- Compare your entity list with a competitor's — whatever entities they have that you don't = content gap

**Entity types explained:**
- `PERSON` — real person (author, expert, public figure)
- `ORGANIZATION` — company, brand, institution
- `LOCATION` — place, country, city
- `WORK_OF_ART` — book, film, song, product name
- `OTHER` / `CONSUMER_GOOD` — general topics and products
""")

    with st.expander("😊 Sentiment — when it matters"):
        st.markdown("""
**Score** ranges from −1.0 (very negative) to +1.0 (very positive).
**Magnitude** = how emotional the content is overall.

| Score | Magnitude | Meaning |
|---|---|---|
| +0.5 or higher | Any | Clearly positive — good for product/review pages |
| −0.3 or lower | Any | Negative tone — can hurt CTR on SERP |
| ~0.0 | Low | Neutral — fine for informational content |
| ~0.0 | High | Mixed — controversial or balanced content |

**When to act:**
- Product pages, landing pages, about pages → should be **positive**
- News, Wikipedia-style articles → **neutral is fine**
- Negative score on a page you want to rank → rewrite the most negative sentences
""")

    with st.expander("📂 Categories — topical alignment"):
        st.markdown("""
Google automatically assigns content categories to every page it crawls.

- **Your page's category should match your target keyword's category**
- If you're targeting "running shoes" but Google categorizes your page as "Fashion/Clothing" instead of "Sports/Running" → your content isn't topically aligned
- Fix: add more content that clearly signals the right topic (specific product names, use cases, expert terminology)
""")

    with st.expander("🔤 Syntax — writing quality signals"):
        st.markdown("""
**Passive voice %** — should be below 15%
- Passive: *"The article was written by John"*
- Active: *"John wrote the article"*
- Active voice is clearer, easier to read, and Google's quality guidelines prefer it

**Lexical density** — should be above 40%
- Measures what % of your words are "content words" (nouns, verbs, adjectives)
- Low density = fluffy, filler-heavy content
- High density = substantive, information-rich content

**Top nouns** = the words that appear most as subject nouns
- These should match your target keywords
- If they don't → you're writing around the topic instead of about it
""")

    with st.expander("🎯 Entity Sentiment — brand and competitor signals"):
        st.markdown("""
Shows the sentiment expressed specifically **about each entity**, not the page overall.

**When to use this:**
- Check that sentiment around your **own brand** is positive
- Check sentiment around your **main product/service** — if negative, find and rewrite those sentences
- When analyzing a **competitor's page** — see if they talk negatively about topics you cover positively (competitive positioning)

**Score interpretation:**
- `+0.25` or higher → positive framing
- `−0.25` or lower → negative framing
- Between −0.25 and +0.25 → neutral
""")

    st.markdown("---")
    st.header("Recommended workflows")

    with st.expander("🔍 Workflow 1 — Why is my page not ranking?"):
        st.markdown("""
1. Enter your page URL + the top-ranking competitor for your keyword
2. Go to **Entities** tab — compare salience scores side by side
3. Find entities they have with high salience that you're missing
4. Add those topics to your content naturally
5. Re-analyze after updating
""")

    with st.expander("✍️ Workflow 2 — Optimize content before publishing"):
        st.markdown("""
1. Write your article draft in Google Docs / Word
2. Copy all the text and paste it in the **Paste Text** tab
3. Check **Entities** — is your target keyword in the top 5?
4. Check **Syntax** — is passive voice below 15%?
5. Check **Sentiment** — is the tone appropriate for the page type?
6. Fix issues → re-paste → re-analyze until green
""")

    with st.expander("🏆 Workflow 3 — Reverse engineer a top-ranking competitor"):
        st.markdown("""
1. Find the #1 ranking page for your target keyword
2. Enter that URL in **Your page URL** field
3. Analyze it — look at:
   - Which entities have the highest salience?
   - What category does Google assign?
   - What are the top nouns?
4. Use this as a blueprint for your own content
""")

    st.markdown("---")
    st.info(
        "**Tip:** The Target Keyword field highlights your keyword in the Entities table "
        "and tells you its exact salience score. Always fill this in."
    )

    # ── Glossary ──────────────────────────────────────────────────────────────
    st.markdown("---")
    st.header("📖 Slovar pojmov — v preprostem jeziku")
    st.caption("Vse tehnične besede razložene brez žargona.")

    with st.expander("🔵 NLP — Natural Language Processing"):
        st.markdown("""
**NLP = računalnik bere in razume besedilo tako kot človek.**

Google uporablja NLP da razume o čem je tvoja stran — ne samo išče ključne besede,
ampak razume kontekst, temo, čustveni ton, in odnose med besedami.

*Primer:* Google NLP razume da "Intex bazen 366cm" govori o bazenih — ne samo o "Intex" ali "366cm" posebej.
""")

    with st.expander("🔵 Parser / Parsiranje"):
        st.markdown("""
**Parser = program ki razčleni stavek na dele in vsakemu določi vlogo.**

Kot v šoli ko si razčlenjeval stavke:
- "Bazen" → samostalnik (Subject)
- "je" → glagol (Verb)
- "lep" → pridevnik (Adjective)

*Zakaj je to važno:* Google parser za slovenščino ni popoln — nekatere slovenske glagolske oblike
napačno prepozna kot samostalnike. Zato za slovensko vsebino uporabljamo Claude AI namesto Google.
""")

    with st.expander("🔵 Token / Tokenizacija"):
        st.markdown("""
**Token = ena beseda ali ločilo.**

Stavek *"Bazen je lep."* = 4 tokeni: **"Bazen"** + **"je"** + **"lep"** + **"."**

*Zakaj je to važno:* Vse metrike (lexical density, verb count...) se računajo glede na število tokenov.
Daljše besedilo = več tokenov = salience ključnih besed se porazdeli na več.
""")

    with st.expander("🏷 Salience — kaj pomeni %"):
        st.markdown("""
**Salience = kako centralna/pomembna je neka tema za celotno besedilo.**

- **0–5%** = tema je omenjena mimogrede, Google je ne šteje kot glavno temo
- **5–15%** = tema je prisotna, ampak ne dominantna
- **15%+** = Google jasno vidi to kot glavno temo strani ✓

*Primer:* Stran o bazenih kjer "bazeni" doseže samo 3% salience pomeni da Google stran
vidi kot stran o čem drugem (npr. vzdrževanju, znamkah) — ne o bazenih nasploh.

*🟠 SEO best practice — ciljna vrednost za glavno ključno besedo: 15%+*
""")

    with st.expander("😊 Sentiment Score in Magnitude — razlika"):
        st.markdown("""
**Score** = smer čustev: od −1.0 (zelo negativno) do +1.0 (zelo pozitivno)

**Magnitude** = moč čustev: od 0 (brez čustev) do ∞ (zelo čustveno)

**Kombinacije:**

| Score | Magnitude | Kaj pomeni |
|---|---|---|
| +0.5 | 5.0 | Jasno pozitivno — dobro za produktne strani ✓ |
| +0.1 | 8.0 | Mešano — nekateri deli pozitivni, drugi negativni, se izničijo ⚠ |
| 0.0 | 0.5 | Nevtralno — brez čustev, dobro za informativne strani |
| −0.4 | 3.0 | Negativno — slabo za vse tipe strani ✗ |

*🔵 Uradni Google podatek — izračunava Google NLP API*
""")

    with st.expander("😊 Entity Sentiment — sentiment PO ENTITETAH"):
        st.markdown("""
**Entity Sentiment = kako GOVORIŠ O vsaki posamezni temi — ne kako se bralec počuti.**

*Primer:*
- "Intex bazeni zagotavljajo vrhunsko kakovost" → Intex: score +0.8 ✓
- "Intex bazeni so dragi in težko dostopni" → Intex: score −0.6 ✗

**Zakaj je to važno:**
- Če je sentiment okoli tvoje blagovne znamke negativen → prepiši te stavke
- Če je sentiment okoli konkurenta negativen → to je tvoja prednost
- Vse vrednosti 0.00 = besedilo je preveč seznam-orientiran, brez opisnih stavkov

*🔵 Uradni Google podatek · 🟠 ±0.25 prag je SEO best practice*
""")

    with st.expander("🔤 Lexical Density — kaj je to"):
        st.markdown("""
**Lexical density = koliko % besed nosi pravi pomen.**

Vsebinske besede (content words): samostalniki, glagoli, pridevniki, prislovi
Funkcijske besede (function words): "je", "in", "a", "ker", "zelo"...

*Primer nizke gostote (fluffy):*
> "Naši bazeni so res zelo dobri in zelo kakovostni za vse."
→ Malo vsebinskih besed, veliko splošnih — Google to vidi kot tanko vsebino.

*Primer visoke gostote (substantive):*
> "Fiberstekleni bazen 4×8m vzdržuje temperaturo 2°C višje od PVC modelov."
→ Vsaka beseda nosi pomen — Google to vidi kot kakovostno vsebino.

**Cilj: 40%+** *(🟠 SEO best practice, ni uradni Google standard)*
""")

    with st.expander("🔤 Pasivni glas — zakaj je slab za SEO"):
        st.markdown("""
**Pasivni glas = dejanje brez jasnega subjekta.**

- Pasivno: *"Bazen **je bil postavljen** v 4 urah."* — kdo ga je postavil?
- Aktivno: *"Montažer **postavi** bazen v 4 urah."* — jasno, direktno

**Zakaj je pasivni glas problem:**
- Bralec mora večkrat prebrati stavek da razume
- Google Quality Rater Guidelines priporočajo aktiven, jasen slog
- Visok % pasivnega glasu = nižja berljivost = slabši dwell time

**Cilj: pod 15%** *(🟣 Google writing guidelines · 🟠 15% prag je best practice)*

*Opomba za slovenščino:* Claude AI je boljši pri zaznavanju slovenskega pasivnega glasu
kot Google NLP API, ker razume "je bil/bila" in "se + glagol" konstrukcije.
""")

    with st.expander("🏷 Knowledge Graph (KG ✓) — kaj pomeni"):
        st.markdown("""
**Knowledge Graph = Googlova baza znanja o resničnih entitetah.**

Ko vidiš **KG ✓** pri entiteti to pomeni da Google ve KAJ ta entiteta je —
ne samo besedo, ampak dejansko stvar v resničnem svetu.

*Primeri:*
- "Intex" KG ✓ → Google ve da je to blagovna znamka bazenov iz Amerike
- "bazeni" KG ✓ → Google ve da so to kopalni bazeni
- "xyz123" KG ✗ → Google samo vidi besedo, ne ve kaj je

**Zakaj je KG ✓ dobro za SEO:**
- Strani ki pokrivajo KG entitete z visoko salience so bolj verjetno videne kot avtoritativne
- KG entitete pomagajo pri featured snippets in knowledge panels
- E-E-A-T: Google lažje oceni avtoriteto strani ki pokriva prepoznane entitete

*🔵 Uradni Google podatek*
""")

    with st.expander("📂 Content Categories — kako Google klasificira"):
        st.markdown("""
**Content Categories = Google samodejno razvrsti vsako stran v kategorijo.**

Google ima hierarhijo kategorij:
```
/Shopping/Home & Garden
  └── /Shopping/Home & Garden/Pool & Spa
        └── /Shopping/Home & Garden/Pool & Spa/Swimming Pools
```

**Zakaj je to važno za SEO:**
- Tvoja kategorija mora ustrezati namenu iskanja (search intent) tvoje ključne besede
- Če ciljate "nakup bazena" ampak Google klasificira stran kot "vzdrževanje" → mismatch
- Fix: dodaj več vsebine ki jasno signalizira pravo kategorijo

**Confidence %** = kako prepričan je Google da tvoja stran spada v to kategorijo

*🔵 Uradni Google podatek*
""")


# ── Main navigation ───────────────────────────────────────────────────────────

page = st.sidebar.radio(
    "Navigation",
    ["🔍 Analyzer", "ℹ️ Info"],
    label_visibility="collapsed",
)

# Top navigation as buttons
col_nav1, col_nav2, col_nav_spacer = st.columns([1, 1, 8])
if col_nav1.button("🔍 Analyzer", use_container_width=True,
                    type="primary" if page == "🔍 Analyzer" else "secondary"):
    st.session_state["page"] = "🔍 Analyzer"
    st.rerun()
if col_nav2.button("ℹ️ Info", use_container_width=True,
                    type="primary" if page == "ℹ️ Info" else "secondary"):
    st.session_state["page"] = "ℹ️ Info"
    st.rerun()

if "page" not in st.session_state:
    st.session_state["page"] = "🔍 Analyzer"

current_page = st.session_state.get("page", "🔍 Analyzer")

st.markdown("---")

# ── Analyzer page ─────────────────────────────────────────────────────────────

if current_page == "🔍 Analyzer":
    st.title("🔍 SEO NLP Analyzer")
    st.caption("Powered by Google Cloud Natural Language API")

    input_mode = st.radio(
        "Input type",
        ["🌐 URL", "📋 Paste text"],
        horizontal=True,
        label_visibility="collapsed",
    )

    with st.form("analyze_form"):
        if input_mode == "🌐 URL":
            c1, c2, c3 = st.columns([3, 3, 2])
            url1    = c1.text_input("Your page URL",
                                     placeholder="https://yoursite.com/page")
            url2    = c2.text_input("Competitor URL (optional)",
                                     placeholder="https://competitor.com/page")
            keyword = c3.text_input("Target keyword (optional)",
                                     placeholder="e.g. bazeni")
            raw_text = ""
        else:
            keyword  = st.text_input("Target keyword (optional)",
                                      placeholder="e.g. bazeni")
            raw_text = st.text_area(
                "Paste your article or page content here",
                placeholder="Paste the full text of your article, blog post, or page...",
                height=250,
            )
            url1 = url2 = ""

        # Language selector
        cl1, cl2 = st.columns([1, 3])
        content_language = cl1.radio(
            "Content language",
            ["English", "Slovenščina"],
            horizontal=True,
            help="Slovenščina: verb/adjective counts use Claude AI (more accurate). English: uses Google NLP API.",
        )
        if content_language == "Slovenščina":
            cl2.info("🤖 Slovenščina mode: Google API za entitete/sentiment/kategorije · Claude AI za glagole/pridevnike/pasivni glas")

        submitted = st.form_submit_button("Analyze", type="primary",
                                           use_container_width=True)

    if submitted:
        # Clear previous report when new analysis starts
        st.session_state["ai_report"] = ""

        if input_mode == "🌐 URL":
            if not url1:
                st.error("Please enter at least one URL.")
                st.stop()

            spinner_msg = f"Fetching and analyzing {url1} {'+ Claude linguistic analysis' if content_language == 'Slovenščina' else ''} ..."
            with st.spinner(spinner_msg):
                text1 = fetch_url_text(url1)
                if text1:
                    if len(text1) > 100_000:
                        text1 = text1[:100_000]
                    st.session_state["results"] = {"url1": run_analysis(text1, content_language)}
                    st.session_state["url1_label"] = url1
                    st.session_state["keyword"] = keyword

            if url2:
                with st.spinner(f"Fetching and analyzing {url2} ..."):
                    text2 = fetch_url_text(url2)
                    if text2:
                        if len(text2) > 100_000:
                            text2 = text2[:100_000]
                        st.session_state["results"]["url2"] = run_analysis(text2, content_language)
                        st.session_state["url2_label"] = url2
        else:
            if not raw_text.strip():
                st.error("Please paste some text to analyze.")
                st.stop()
            text1 = raw_text.strip()
            if len(text1) > 100_000:
                text1 = text1[:100_000]
            spinner_msg = "Analyzing text + Claude linguistic analysis ..." if content_language == "Slovenščina" else "Analyzing text ..."
            with st.spinner(spinner_msg):
                st.session_state["results"] = {"url1": run_analysis(text1, content_language)}
                st.session_state["url1_label"] = "pasted text"
                st.session_state["keyword"] = keyword

    # Always render results from session_state (persists across re-renders)
    if "results" in st.session_state and st.session_state["results"]:
        results   = st.session_state["results"]
        keyword   = st.session_state.get("keyword", "")
        url1      = st.session_state.get("url1_label", "")
        url2      = st.session_state.get("url2_label", "")

        if not results:
            st.error("No results — check that the URLs are publicly accessible.")
            st.stop()

        if "url2" in results:
            st.divider()
            col_a, col_b = st.columns(2)
            with col_a:
                st.markdown(f"#### Your page\n`{url1[:60]}`")
                render_analysis(results["url1"], keyword, source=url1)
            with col_b:
                st.markdown(f"#### Competitor\n`{url2[:60]}`")
                render_analysis(results["url2"], keyword, source=url2)
        else:
            st.divider()
            render_analysis(results["url1"], keyword, source=url1)

# ── Info page ─────────────────────────────────────────────────────────────────

elif current_page == "ℹ️ Info":
    page_info()
