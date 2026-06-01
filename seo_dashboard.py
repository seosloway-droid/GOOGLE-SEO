"""
SEO NLP Analyzer — Streamlit Dashboard
Powered by Google Cloud Natural Language API
"""

import streamlit as st
import pandas as pd
import anthropic
import requests
import base64
import json
import re
from datetime import datetime
from pathlib import Path
from typing import Any
from urllib.request import urlopen, Request
from urllib.error import URLError
from html.parser import HTMLParser
from google.cloud import language_v1
from google.oauth2 import service_account
from content_optimizer import (
    OptimizerInput,
    PageText,
    build_audit_artifacts,
    build_content_brief_markdown,
    build_improvement_plan_markdown,
    extract_headings_from_html,
    extract_markdown_headings,
    lsi_limit_for_depth,
    optimize_content,
)
try:
    from firecrawl import Firecrawl as FirecrawlClient
    FIRECRAWL_AVAILABLE = True
except ImportError:
    FIRECRAWL_AVAILABLE = False

# ── Page config ───────────────────────────────────────────────────────────────

BASE_DIR = Path(__file__).resolve().parent
ANALIZE_DIR = BASE_DIR / "analize"

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


# ── NeuroWriter integration ───────────────────────────────────────────────────

def parse_nw_text(raw: str) -> dict:
    """Parse NeuroWriter text export format.
    Supports sections: TITLE TERMS, DESCRIPTION TERMS, H1 HEADERS TERMS,
    H2 HEADERS TERMS, BASIC TEXT TERMS, EXTENDED TEXT TERMS
    """
    import re as _re
    basic    = {}
    extended = {}
    title    = []
    h1       = []
    h2       = []

    current_section = None
    for line in raw.splitlines():
        line = line.strip()
        if not line:
            continue

        # Detect section headers
        line_up = line.upper()
        if "TITLE TERMS" in line_up:
            current_section = "title"
            continue
        elif "DESCRIPTION TERMS" in line_up:
            current_section = "desc"
            continue
        elif "H1" in line_up and "TERM" in line_up:
            current_section = "h1"
            continue
        elif "H2" in line_up and "TERM" in line_up:
            current_section = "h2"
            continue
        elif "BASIC TEXT" in line_up or "BASIC TERMS" in line_up:
            current_section = "basic"
            continue
        elif "EXTENDED TEXT" in line_up or "EXTENDED TERMS" in line_up:
            current_section = "extended"
            continue
        elif "======" in line:
            continue

        if current_section in ("title", "h1", "h2", "desc"):
            # Simple list of terms
            term = line.lower().strip()
            if term:
                if current_section == "title":
                    title.append(term)
                elif current_section == "h1":
                    h1.append(term)
                elif current_section == "h2":
                    h2.append(term)
                # desc terms ignored (not used in scoring)

        elif current_section in ("basic", "extended"):
            # Format: "term: 1-5x" or "term: 3x"
            match = _re.match(r"^(.+?):\s*(\d+)(?:-(\d+))?x?$", line)
            if match:
                term = match.group(1).strip().lower()
                mn   = int(match.group(2))
                mx   = int(match.group(3)) if match.group(3) else mn
                entry = {"min": mn, "max": mx}
                if current_section == "basic":
                    basic[term] = entry
                else:
                    extended[term] = entry

    return {"basic": basic, "extended": extended, "title": title, "h1": h1, "h2": h2}


def parse_nw_json(raw: str) -> dict:
    """Parse NeuroWriter export — auto-detects JSON or text format."""
    import json as _json
    raw = raw.strip()

    # Auto-detect: if starts with { it's JSON, otherwise text format
    if raw.startswith("{"):
        try:
            data  = _json.loads(raw)
            basic = {}
            for item in data.get("basic_terms", []):
                for term, vals in item.items():
                    basic[term.lower()] = {"min": vals["min"], "max": vals["max"]}
            extended = {}
            for item in data.get("extended_terms", []):
                for term, vals in item.items():
                    extended[term.lower()] = {"min": vals["min"], "max": vals["max"]}
            return {
                "basic":    basic,
                "extended": extended,
                "title":    [t.lower() for t in data.get("title_terms", [])],
                "h1":       [t.lower() for t in data.get("h1_terms", [])],
                "h2":       [t.lower() for t in data.get("h2_terms", [])],
            }
        except Exception:
            return {}
    else:
        # Text format
        result = parse_nw_text(raw)
        if result.get("basic") or result.get("title"):
            return result
        return {}


def score_content_nw(text: str, nw: dict) -> dict:
    """Score content against NeuroWriter terms."""
    import re
    text_lower = text.lower()

    def count_term(term):
        # Count non-overlapping occurrences (word boundary aware)
        pattern = re.escape(term)
        return len(re.findall(pattern, text_lower))

    results = {"basic": [], "extended": [], "basic_score": 0, "extended_score": 0}

    # Score basic terms (weighted more)
    covered_basic = 0
    for term, req in nw.get("basic", {}).items():
        cnt    = count_term(term)
        mn, mx = req["min"], req["max"]
        if cnt >= mn:
            status = "✅"
            covered_basic += 1
        elif cnt > 0:
            status = "⚠️"
            covered_basic += 0.5
        else:
            status = "❌"
        results["basic"].append({
            "term": term, "count": cnt,
            "min": mn, "max": mx, "status": status,
        })

    total_basic = len(nw.get("basic", {}))
    results["basic_score"] = round(covered_basic / total_basic * 100) if total_basic else 0

    # Score extended terms
    covered_ext = 0
    for term, req in nw.get("extended", {}).items():
        cnt    = count_term(term)
        mn, mx = req["min"], req["max"]
        if cnt >= mn:
            status = "✅"
            covered_ext += 1
        elif cnt > 0:
            status = "⚠️"
            covered_ext += 0.5
        else:
            status = "❌"
        results["extended"].append({
            "term": term, "count": cnt,
            "min": mn, "max": mx, "status": status,
        })

    total_ext = len(nw.get("extended", {}))
    results["extended_score"] = round(covered_ext / total_ext * 100) if total_ext else 0

    # Overall score (basic weighted 70%, extended 30%)
    results["overall_score"] = round(
        results["basic_score"] * 0.7 + results["extended_score"] * 0.3
    )
    return results


def tab_nw_score(text: str, nw: dict, key_prefix: str = "main"):
    """Render NeuroWriter content score tab."""
    if not nw:
        st.info("Prilepi NeuroWriter JSON v polje spodaj da vidiš score.")
        return
    if not text:
        st.info("Ni besedila za analizo.")
        return

    scores = score_content_nw(text, nw)
    overall = scores["overall_score"]
    basic_s = scores["basic_score"]
    ext_s   = scores["extended_score"]

    # ── Score header ──────────────────────────────────────────────────────────
    color = "🟢" if overall >= 75 else ("🟡" if overall >= 50 else "🔴")
    st.markdown(f"## {color} NW Score: **{overall}%**")
    c1, c2 = st.columns(2)
    c1.metric("Basic terms", f"{basic_s}%",
              help="Obvezne besede — min/max frekvenca")
    c2.metric("Extended terms", f"{ext_s}%",
              help="Dodatne besede — bonus points")
    st.progress(overall / 100)
    st.divider()

    # ── Heading terms check ───────────────────────────────────────────────────
    st.subheader("📑 Heading terms")
    for level, terms in [("Title", nw.get("title", [])),
                          ("H1", nw.get("h1", [])),
                          ("H2", nw.get("h2", []))]:
        if terms:
            covered = [t for t in terms if t in text.lower()]
            missing = [t for t in terms if t not in text.lower()]
            st.markdown(
                f"**{level}:** "
                + " ".join(f"`✅ {t}`" for t in covered)
                + " ".join(f"`❌ {t}`" for t in missing)
            )

    st.divider()

    # ── Basic terms detail ────────────────────────────────────────────────────
    col_f, _ = st.columns([1, 3])
    filter_opt = col_f.radio("Prikaži", ["Vse", "❌ Manjka", "⚠️ Premalo"],
                              horizontal=True, key=f"nw_filter_{key_prefix}")

    st.subheader("📋 Basic terms")
    basic_rows = scores["basic"]
    if filter_opt == "❌ Manjka":
        basic_rows = [r for r in basic_rows if r["status"] == "❌"]
    elif filter_opt == "⚠️ Premalo":
        basic_rows = [r for r in basic_rows if r["status"] == "⚠️"]

    df_basic = pd.DataFrame(basic_rows)
    if not df_basic.empty:
        def _color_status(val):
            return {"✅": "color:green", "⚠️": "color:orange", "❌": "color:red"}.get(val, "")
        st.dataframe(
            df_basic[["status", "term", "count", "min", "max"]].style.map(
                _color_status, subset=["status"]),
            use_container_width=True, hide_index=True,
            column_config={
                "status": "Status",
                "term":   st.column_config.TextColumn("Beseda"),
                "count":  st.column_config.NumberColumn("V besedilu"),
                "min":    st.column_config.NumberColumn("Min"),
                "max":    st.column_config.NumberColumn("Max"),
            }
        )

    # ── Extended terms ────────────────────────────────────────────────────────
    with st.expander("📋 Extended terms"):
        ext_rows = [r for r in scores["extended"] if r["status"] != "✅"]
        if ext_rows:
            df_ext = pd.DataFrame(ext_rows)
            st.dataframe(df_ext[["status", "term", "count", "min", "max"]],
                         use_container_width=True, hide_index=True)
        else:
            st.success("Vse extended terms pokrite! ✅")


def generate_improvement_plan(benchmark: dict, keyword: str, my_text: str,
                              language: str, nw: dict = None) -> str:
    """Generate specific improvement instructions for existing content."""
    client = get_anthropic_client()
    if not client:
        return ""

    n    = benchmark.get("n", 0)
    h    = benchmark.get("avg_headings", {})
    sd   = benchmark.get("avg_sentence_dist", {})
    ents = benchmark.get("top_entities", [])
    paa  = benchmark.get("paa", [])

    # NW missing terms — pass ALL terms, no truncation
    nw_section = ""
    if nw:
        scores        = score_content_nw(my_text, nw)
        missing_basic = [r["term"] for r in scores["basic"] if r["status"] == "❌"]
        low_basic     = [f"{r['term']} ({r['count']}x, needs {r['min']}–{r['max']})"
                         for r in scores["basic"] if r["status"] == "⚠️"]
        ok_basic      = [r["term"] for r in scores["basic"] if r["status"] == "✅"]
        missing_ext   = [r["term"] for r in scores["extended"] if r["status"] == "❌"]
        low_ext       = [f"{r['term']} ({r['count']}x, needs {r['min']}+)"
                         for r in scores["extended"] if r["status"] == "⚠️"]
        nw_section = f"""
NEUROWRITER SCORE: {scores['overall_score']}% (basic: {scores['basic_score']}%, extended: {scores['extended_score']}%)

BASIC TERMS — MISSING ({len(missing_basic)} terms, must add):
{', '.join(missing_basic) if missing_basic else 'none'}

BASIC TERMS — BELOW MINIMUM ({len(low_basic)} terms, add more):
{', '.join(low_basic) if low_basic else 'none'}

BASIC TERMS — OK ({len(ok_basic)} terms, already covered):
{', '.join(ok_basic) if ok_basic else 'none'}

EXTENDED TERMS — MISSING ({len(missing_ext)} terms):
{', '.join(missing_ext) if missing_ext else 'none'}

EXTENDED TERMS — BELOW MINIMUM:
{', '.join(low_ext) if low_ext else 'none'}

HEADING TERMS:
Title needed: {', '.join(nw.get('title', []))}
H1 needed: {', '.join(nw.get('h1', []))}
H2 needed: {', '.join(nw.get('h2', []))}
"""

    # My text analysis (quick metrics)
    my_words    = len(my_text.split())
    my_sents    = len([s for s in my_text.split('.') if s.strip()])

    # Entity gap
    my_text_lower  = my_text.lower()
    majority_ents  = [e for e in ents if e["present_in"] >= max(2, n // 2)]
    missing_ents   = [e["name"] for e in majority_ents
                      if e["name"].lower() not in my_text_lower]

    lang_instruction = "Write the entire plan in Slovenian language." \
        if language == "Slovenščina" else "Write the entire plan in English."

    prompt = f"""You are an expert SEO editor. Your job is to give SPECIFIC, ACTIONABLE instructions
to improve existing content so it matches competitor benchmarks.

KEYWORD: {keyword if keyword else "(not specified)"}
BASED ON: {n} competitor pages

═══ COMPETITOR BENCHMARKS ═══
Sentiment score avg:    {benchmark.get('avg_sentiment', 0):+.3f}
Magnitude avg:          {benchmark.get('avg_magnitude', 0):.1f}
Positive sentences:     {sd.get('positive_pct', 0):.0f}%
Negative sentences:     {sd.get('negative_pct', 0):.0f}%
Word count avg:         {benchmark.get('avg_word_count', 0):,}
Lexical density avg:    {benchmark.get('avg_lexical_density', 0):.1%}
Passive voice avg:      {benchmark.get('avg_passive_voice', 0):.1f}%
Verb count avg:         {benchmark.get('avg_verb_count', 0):.0f}
H2 headings avg:        {h.get('h2', 0):.1f}
Common H2 topics:       {', '.join(h.get('h2_texts', [])[:8])}
Top entities:           {', '.join(e['name'] for e in majority_ents[:10])}
PAA questions:          {'; '.join(paa[:6]) if paa else 'not available'}
{nw_section}
═══ EXISTING CONTENT ═══
Word count: {my_words}
Approximate sentences: {my_sents}
Missing entities: {', '.join(missing_ents[:10]) if missing_ents else 'none'}

--- CONTENT START ---
{my_text[:6000]}
--- CONTENT END ---

═══ YOUR TASK ═══

Analyze the gap between the existing content and competitor benchmarks.
Write a specific IMPROVEMENT PLAN with these sections:

1. **Kaj dodati** — specific sentences/paragraphs to add (with examples)
2. **Kaj spremeniti** — specific sentences to rewrite (show before → after)
3. **Kaj odstraniti** — what to cut
4. **Manjkajoče entitete** — which competitor topics are missing and where to add them
5. **Struktura naslovov** — H2 changes needed
6. **NeuroWriter besede** — which missing NW terms to add and suggested sentences
7. **Prioritetni vrstni red** — rank changes by SEO impact (1 = most important)

Be SPECIFIC — reference actual sentences from the content. Show exact rewrites.
Do not be vague. Every suggestion must be immediately actionable.
{lang_instruction}"""

    response = client.messages.create(
        model="claude-sonnet-4-5",
        max_tokens=4000,
        messages=[{"role": "user", "content": prompt}],
    )
    return response.content[0].text


def generate_content_brief(benchmark: dict, keyword: str, language: str) -> str:
    """Generate a content brief from competitor benchmark data using Claude."""
    client = get_anthropic_client()
    if not client:
        return ""

    n   = benchmark.get("n", 0)
    bm  = benchmark
    h   = bm.get("avg_headings", {})
    sd  = bm.get("avg_sentence_dist", {})
    ents = bm.get("top_entities", [])
    cats = bm.get("top_categories", [])
    paa  = bm.get("paa", [])

    # Top entities present in majority of pages
    majority_ents = [e for e in ents if e["present_in"] >= max(2, n // 2)]

    lang_instruction = "Write the entire brief in Slovenian language." \
        if language == "Slovenščina" else "Write the entire brief in English."

    prompt = f"""You are an expert SEO content strategist. Based on competitor analysis data below,
create a detailed CONTENT BRIEF that tells a writer exactly how to write content that will
compete with the top-ranking pages for this keyword.

KEYWORD: {keyword if keyword else "(not specified)"}
BASED ON: {n} competitor pages analyzed

═══ COMPETITOR AVERAGES ═══

SENTIMENT & TONE:
  Score: {bm.get('avg_sentiment', 0):+.3f} (range -1 to +1)
  Magnitude: {bm.get('avg_magnitude', 0):.1f}
  Positive sentences: {sd.get('positive_pct', 0):.0f}%
  Negative sentences: {sd.get('negative_pct', 0):.0f}%
  Neutral sentences: {sd.get('neutral_pct', 0):.0f}%

CONTENT METRICS:
  Word count: {bm.get('avg_word_count', 0):,}
  Sentences: {sd.get('avg_sentence_count', 0):.0f}
  Lexical density: {bm.get('avg_lexical_density', 0):.1%}
  Passive voice: {bm.get('avg_passive_voice', 0):.1f}%
  Verb count: {bm.get('avg_verb_count', 0):.0f}
  Adjective count: {bm.get('avg_adj_count', 0):.0f}
  Noun count: {bm.get('avg_noun_count', 0):.0f}
  Verb/noun ratio: {bm.get('avg_verb_noun_ratio', 0):.1f}%

KEYWORD SALIENCE:
  Target keyword avg salience: {bm.get('avg_kw_salience', 0):.1f}%
  Top 5 entities cover: {bm.get('avg_top5_concentration', 0):.0f}% of total salience

HEADING STRUCTURE:
  H1: {h.get('h1', 0):.0f} | H2: {h.get('h2', 0):.1f} | H3: {h.get('h3', 0):.1f} | H4: {h.get('h4', 0):.1f}
  Common H2 topics: {', '.join(h.get('h2_texts', [])[:10])}

TOP ENTITIES (must include in content):
{chr(10).join(f"  - {e['name']} ({e['type']}) — avg salience {e['avg_salience']:.1f}%, present in {e['present_in']}/{n} pages" for e in majority_ents[:12])}

CONTENT CATEGORIES:
{chr(10).join(f"  - {c['category']} ({c['avg_confidence']:.0f}% confidence)" for c in cats[:3])}

PEOPLE ALSO ASK:
{chr(10).join(f"  - {q}" for q in paa[:8]) if paa else "  (not available)"}

═══ YOUR TASK ═══

Create a practical CONTENT BRIEF with these exact sections:

1. **Povzetek naloge** — 2-3 stavki kaj je cilj tega besedila
2. **Ciljna dolžina in struktura** — točno koliko besed, stavkov, naslovov
3. **Ton in sentiment** — kakšen ton pisati, konkretni primeri stavkov
4. **Obvezne entitete** — kateri pojmi MORAJO biti v besedilu in kolikokrat
5. **Predlagana struktura naslovov** — H1, H2 predlogi z dejanskimi naslovi
6. **People Also Ask** — katera vprašanja mora besedilo odgovoriti
7. **Česar se izogibati** — kaj NE delati glede na analizo
8. **Ciljne NLP metrike** — konkretne številke ki jih mora dosegati besedilo

Be specific and actionable. Use actual numbers. Give example sentences where helpful.
{lang_instruction}"""

    response = client.messages.create(
        model="claude-sonnet-4-5",
        max_tokens=4000,
        messages=[{"role": "user", "content": prompt}],
    )
    return response.content[0].text


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
    is_slo = data.get("content_language", "English") in ("Slovenščina", "Italiano 🇮🇹", "Hrvatski 🇭🇷")

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
    neg_sent_lines = "\n".join(f'  - (score {s["score"]:+.2f}) "{s["text"][:300]}"' for s in neg_sentences[:5])

    prompt = f"""You are an expert SEO consultant. Analyze this NLP data and give specific, actionable SEO advice.

IMPORTANT: Sentence text may be truncated at 300 chars for display — this does NOT mean content is incomplete.
Do NOT flag truncated sentence display as "incomplete content". The full text was analyzed correctly.

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


def tab_ai_coach(data: dict, keyword: str, key_prefix: str = "main"):
    if get_anthropic_client() is None:
        st.warning("Claude AI API key not configured. Add ANTHROPIC_API_KEY to Streamlit Secrets.")
        return

    col1, col2 = st.columns(2)
    language = col1.radio("Language", ["English", "Slovenščina"], horizontal=True,
                          key=f"ai_language_{key_prefix}")
    detail   = col2.radio("Detail level", ["Short (5 bullets)", "Detailed report"],
                          horizontal=True, key=f"ai_detail_{key_prefix}")

    if st.button("🤖 Generate AI SEO Report", type="primary", use_container_width=True,
                 key=f"ai_generate_btn_{key_prefix}"):
        client = get_anthropic_client()
        if client is None:
            st.error("ANTHROPIC_API_KEY not found in Streamlit Secrets. Add it and reboot the app.")
        else:
            with st.spinner("Claude is analyzing your content..."):
                try:
                    report = generate_seo_report(data, keyword, language, detail)
                    st.session_state[f"ai_report_{key_prefix}"] = report
                except Exception as e:
                    st.error(f"Claude API error: {e}")
                    st.session_state[f"ai_report_{key_prefix}"] = ""

    report_key = f"ai_report_{key_prefix}"
    if report_key in st.session_state and st.session_state[report_key]:
        report = st.session_state[report_key]
        st.markdown("---")
        st.markdown(report)


# ── Content cleaner ───────────────────────────────────────────────────────────

def clean_content_with_claude(text: str) -> str:
    """Remove product listings, prices, pagination from scraped text.
    Returns the original editorial sentences — no paraphrasing.
    Used for e-commerce category pages where <main> contains both
    description text and product grid.
    """
    client = get_anthropic_client()
    if not client or not text.strip():
        return text

    if len(text.split()) < 50:
        return text

    # Send up to 60,000 characters (~15,000 words) to Claude
    # If text is longer, append the remainder unmodified after cleaning
    text_to_clean = text[:60000]
    text_remainder = text[60000:] if len(text) > 60000 else ""

    try:
        response = client.messages.create(
            model="claude-sonnet-4-5",
            max_tokens=8000,
            messages=[{"role": "user", "content": f"""You are a text filter. Your job is to REMOVE noise from scraped web page content.

REMOVE these elements completely:
- Product names with prices (e.g. "Intex Frame Pool 366x76cm — €89.99")
- Product listings and grids
- Buttons: "Add to cart", "Buy now", "Close", "Zatvori", "V košarico", "Dodaj", "Kupi"
- Pagination ("Prikazano 1-20 od 57", "Stran 1 2 3", "Sljedeća", "Naslednja", "Next")
- Filter/sort controls ("Filtriraj", "Razvrsti", "Prikaži", "Sortiraj", "Pokaži")
- Breadcrumbs ("Domov > Bazeni", "Početna > Bazeni")
- Star ratings and review counts ("★★★★☆", "(24 ocen)", "(5 reviews)")
- SKU codes, stock status ("Na zalogi", "Na stanju", "Šifra:", "SKU:")
- Image URLs and file paths (e.g. "https://...image.svg", "https://...icon.png")
- Short product teasers (product name + 1-line description + price = remove)
- Navigation labels ("navigation", "menu", "navbar", "sidebar")
- Cookie banners ("Kolačići", "Koristimo kolačiće", "We use cookies", "Cookie", "GDPR", "Prihvaćam", "Sprejemam")
- Email addresses (anything with @)
- Phone numbers (sequences of digits with +, spaces, dashes)
- Physical addresses (street names, postal codes, cities used as contact info)
- Company registration numbers, VAT numbers (OIB, MBO, PIB, matična)
- Social media links and share buttons ("Dijeli", "Podijeli", "Share", "Facebook", "Instagram")
- "Close" buttons, popup overlays, modal content
- Footer contact information (address, email, phone in footer context)
- Any line that is ONLY a URL or ONLY an email address

KEEP everything else EXACTLY as written:
- Category description paragraphs
- Informational text about the topic
- Buying guides and advice sections
- FAQ content
- Any editorial/informational sentences of 2+ lines
- Brand names mentioned in context (Intex, Bestway etc. when part of description)

IMPORTANT: Return the kept text WORD FOR WORD. Do not rewrite, summarize or paraphrase. Just delete the noise and return what remains.

TEXT TO CLEAN:
---
{text_to_clean}
---

Return only the cleaned text, nothing else."""}]
        )
        cleaned = response.content[0].text.strip()

        # Safety check: compare against the portion sent, not full text
        sent_words   = len(text_to_clean.split())
        cleaned_words = len(cleaned.split())

        # If cleaned is less than 5% of sent text — something went wrong, return original
        if cleaned_words < sent_words * 0.05:
            return text

        # Append unprocessed remainder (text beyond 60k chars) to preserve full content
        if text_remainder:
            return cleaned + "\n" + text_remainder
        return cleaned

    except Exception:
        return text


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


@st.cache_resource
def get_firecrawl():
    try:
        key = st.secrets["FIRECRAWL_API_KEY"]
        if key and FIRECRAWL_AVAILABLE:
            return FirecrawlClient(api_key=key)
    except Exception:
        pass
    return None


# ── DataForSEO ────────────────────────────────────────────────────────────────

def get_dfseo_auth() -> tuple:
    try:
        login    = st.secrets["DATAFORSEO_LOGIN"]
        password = st.secrets["DATAFORSEO_PASSWORD"]
        if login and password:
            return login, password
    except Exception:
        pass
    return None, None


@st.cache_data(ttl=3600, show_spinner=False)
def dfseo_serp(keyword: str, location_code: int = 2840, language_code: str = "en") -> dict:
    """Get top 10 Google SERP results + People Also Ask for a keyword."""
    login, password = get_dfseo_auth()
    if not login:
        return {}

    creds = base64.b64encode(f"{login}:{password}".encode()).decode()
    headers = {
        "Authorization": f"Basic {creds}",
        "Content-Type": "application/json",
    }
    payload = [{
        "keyword":       keyword,
        "location_code": location_code,
        "language_code": language_code,
        "depth":         10,
        "se_type":       "organic",
    }]

    try:
        resp = requests.post(
            "https://api.dataforseo.com/v3/serp/google/organic/live/advanced",
            headers=headers,
            json=payload,
            timeout=30,
        )
        data = resp.json()
        if data.get("status_code") != 20000:
            return {}

        task = data["tasks"][0]
        if task.get("status_code") != 20000:
            return {}

        items = task["result"][0].get("items", [])

        organic = []
        paa     = []
        for item in items:
            if item.get("type") == "organic":
                organic.append({
                    "url":        item.get("url", ""),
                    "title":      item.get("title", ""),
                    "position":   item.get("rank_absolute", 0),
                    "word_count": item.get("word_count", 0),
                })
            elif item.get("type") == "people_also_ask":
                for q in item.get("items", []):
                    if q.get("type") == "people_also_ask_element":
                        paa.append(q.get("title", ""))

        return {"organic": organic[:10], "paa": paa[:10]}

    except Exception as e:
        st.warning(f"DataForSEO error: {e}")
        return {}


@st.cache_data(ttl=86400, show_spinner=False)
def dfseo_onpage_headings(url: str) -> dict:
    """Get heading structure (H1-H4) for a URL using DataForSEO On-Page API."""
    login, password = get_dfseo_auth()
    if not login:
        return {}

    creds   = base64.b64encode(f"{login}:{password}".encode()).decode()
    headers = {
        "Authorization": f"Basic {creds}",
        "Content-Type": "application/json",
    }

    # Step 1: Create task
    try:
        resp = requests.post(
            "https://api.dataforseo.com/v3/on_page/task_post",
            headers=headers,
            json=[{"target": url, "max_crawl_pages": 1}],
            timeout=30,
        )
        data = resp.json()
        if data.get("status_code") != 20000:
            return {}
        task_id = data["tasks"][0].get("id")
        if not task_id:
            return {}
    except Exception:
        return {}

    # Step 2: Poll for results (max 6 attempts × 3s = 18s max per URL)
    import time
    for _ in range(6):
        time.sleep(3)
        try:
            resp = requests.get(
                f"https://api.dataforseo.com/v3/on_page/pages",
                headers=headers,
                json=[{"id": task_id, "limit": 1}],
                timeout=30,
            )
            data = resp.json()
            if data.get("status_code") != 20000:
                continue
            items = data["tasks"][0].get("result", [{}])[0].get("items", [])
            if not items:
                continue

            page = items[0]
            meta = page.get("meta", {})

            # Extract headings
            headings = {}
            for level in ["h1", "h2", "h3", "h4"]:
                tags = meta.get(level, [])
                if isinstance(tags, list):
                    headings[level] = [t for t in tags if t]
                elif isinstance(tags, str) and tags:
                    headings[level] = [tags]
                else:
                    headings[level] = []

            return {
                "url":      url,
                "headings": headings,
                "h1_count": len(headings.get("h1", [])),
                "h2_count": len(headings.get("h2", [])),
                "h3_count": len(headings.get("h3", [])),
                "h4_count": len(headings.get("h4", [])),
                "total_headings": sum(len(v) for v in headings.values()),
            }
        except Exception:
            continue

    return {}


def normalize_domain(value: str) -> str:
    value = (value or "").strip().lower()
    value = re.sub(r"^https?://", "", value)
    value = re.sub(r"^www\.", "", value)
    value = value.split("/")[0]
    return value


def normalize_space(value: str) -> str:
    return " ".join((value or "").strip().split())


def normalize_text(value: str) -> str:
    return normalize_space(value).casefold()


def domain_matches(candidate: str, target: str) -> bool:
    candidate_norm = normalize_domain(candidate)
    target_norm = normalize_domain(target)
    if not candidate_norm or not target_norm:
        return False
    return candidate_norm == target_norm or candidate_norm.endswith(f".{target_norm}")


def extract_llm_message_payload(items: list[dict]) -> tuple[str, list[dict]]:
    message_parts: list[str] = []
    annotations: list[dict] = []
    for item in items or []:
        if item.get("type") != "message":
            continue
        for section in item.get("sections", []) or []:
            if section.get("type") == "text" and section.get("text"):
                message_parts.append(section.get("text", ""))
            for annotation in section.get("annotations", []) or []:
                if annotation.get("url"):
                    annotations.append({
                        "title": annotation.get("title", ""),
                        "url": annotation.get("url", ""),
                        "domain": normalize_domain(annotation.get("url", "")),
                    })
    return "\n\n".join(part for part in message_parts if part).strip(), annotations


@st.cache_data(ttl=1800, show_spinner=False)
def dfseo_llm_response_live(
    platform: str,
    query: str,
    model_name: str,
    target_country_iso_code: str = "US",
    web_search: bool = True,
    max_output_tokens: int = 500,
) -> dict:
    login, password = get_dfseo_auth()
    if not login:
        return {"error": "Missing DataForSEO credentials."}
    if not normalize_space(query):
        return {"error": "Query is required."}

    platform_path = {
        "chat_gpt": "chat_gpt",
        "gemini": "gemini",
        "perplexity": "perplexity",
    }.get(platform)
    if not platform_path:
        return {"error": f"Unsupported platform: {platform}"}

    creds = base64.b64encode(f"{login}:{password}".encode()).decode()
    headers = {
        "Authorization": f"Basic {creds}",
        "Content-Type": "application/json",
    }

    payload_item: dict[str, Any] = {
        "user_prompt": normalize_space(query),
        "model_name": model_name,
        "max_output_tokens": int(max_output_tokens),
        "web_search": bool(web_search),
    }
    if platform in {"chat_gpt", "perplexity"} and target_country_iso_code:
        payload_item["web_search_country_iso_code"] = target_country_iso_code.upper()

    try:
        resp = requests.post(
            f"https://api.dataforseo.com/v3/ai_optimization/{platform_path}/llm_responses/live",
            headers=headers,
            json=[payload_item],
            timeout=130,
        )
        data = resp.json()
        if data.get("status_code") != 20000:
            return {"error": data.get("status_message", "Unknown DataForSEO error.")}
        task = data.get("tasks", [{}])[0]
        if task.get("status_code") != 20000:
            return {"error": task.get("status_message", "Task failed.")}
        response_block = (task.get("result") or [{}])[0]
        answer, annotations = extract_llm_message_payload(response_block.get("items", []) or [])
        return {
            "platform": platform,
            "query": normalize_space(query),
            "requested_model": model_name,
            "model_name": response_block.get("model_name", model_name),
            "datetime": response_block.get("datetime", ""),
            "web_search": response_block.get("web_search", web_search),
            "money_spent": response_block.get("money_spent", 0),
            "input_tokens": response_block.get("input_tokens", 0),
            "output_tokens": response_block.get("output_tokens", 0),
            "reasoning_tokens": response_block.get("reasoning_tokens", 0),
            "answer": answer,
            "annotations": annotations,
            "fan_out_queries": response_block.get("fan_out_queries", []) or [],
            "raw_items": response_block.get("items", []) or [],
        }
    except Exception as e:
        return {"error": str(e)}


@st.cache_data(ttl=1800, show_spinner=False)
def dfseo_llm_mentions_search(
    platform: str,
    location_code: int,
    language_code: str,
    target_domain: str = "",
    target_brand: str = "",
    min_ai_search_volume: int = 0,
    limit: int = 20,
) -> dict:
    login, password = get_dfseo_auth()
    if not login:
        return {"error": "Missing DataForSEO credentials."}

    domain = normalize_domain(target_domain)
    brand = normalize_space(target_brand)
    if not domain and not brand:
        return {"error": "Enter a domain, a brand keyword, or both."}

    creds = base64.b64encode(f"{login}:{password}".encode()).decode()
    headers = {
        "Authorization": f"Basic {creds}",
        "Content-Type": "application/json",
    }

    target = []
    if domain:
        target.append({
            "domain": domain,
            "search_filter": "include",
            "include_subdomains": True,
        })
    if brand:
        target.append({
            "keyword": brand,
            "search_filter": "include",
        })

    payload = [{
        "platform": platform,
        "location_code": location_code,
        "language_code": language_code,
        "target": target,
        "limit": limit,
        "offset": 0,
        "order_by": ["ai_search_volume,desc"],
    }]
    if min_ai_search_volume > 0:
        payload[0]["filters"] = [["ai_search_volume", ">", min_ai_search_volume]]

    try:
        resp = requests.post(
            "https://api.dataforseo.com/v3/ai_optimization/llm_mentions/search/live",
            headers=headers,
            json=payload,
            timeout=60,
        )
        data = resp.json()
        if data.get("status_code") != 20000:
            return {"error": data.get("status_message", "Unknown DataForSEO error.")}
        task = data.get("tasks", [{}])[0]
        if task.get("status_code") != 20000:
            return {"error": task.get("status_message", "Task failed.")}
        result = task.get("result", [{}])[0]
        items = result.get("items") or []
        return {
            "items": items if isinstance(items, list) else [],
            "total_count": result.get("total_count", 0),
            "current_offset": result.get("current_offset", 0),
        }
    except Exception as e:
        return {"error": str(e)}


@st.cache_data(ttl=86400, show_spinner=False)
def dfseo_llm_mentions_locations() -> list[dict[str, Any]]:
    login, password = get_dfseo_auth()
    if not login:
        return []

    creds = base64.b64encode(f"{login}:{password}".encode()).decode()
    headers = {
        "Authorization": f"Basic {creds}",
        "Content-Type": "application/json",
    }
    try:
        resp = requests.get(
            "https://api.dataforseo.com/v3/ai_optimization/llm_mentions/locations_and_languages",
            headers=headers,
            timeout=30,
        )
        data = resp.json()
        if data.get("status_code") != 20000:
            return []
        task = data.get("tasks", [{}])[0]
        if task.get("status_code") != 20000:
            return []
        result = task.get("result") or []
        return result if isinstance(result, list) else []
    except Exception:
        return []


def extract_text_from_html(html: str) -> str:
    """Extract main content from raw HTML source code.
    Uses deterministic parsing only — NO Claude AI involved.

    Priority order:
    1. <main> tag — most reliable, targets editorial content
    2. <article> tag — second best
    3. Full body with nav/footer/header removed

    All steps are deterministic (Python HTML parser), 100% reliable.
    """
    import re as _re

    # Try to extract <main> content first
    main_match = _re.search(r'<main[^>]*>(.*?)</main>', html, _re.DOTALL | _re.IGNORECASE)
    if main_match:
        html_to_parse = main_match.group(1)
        source_tag = "<main>"
    else:
        # Try <article>
        art_match = _re.search(r'<article[^>]*>(.*?)</article>', html, _re.DOTALL | _re.IGNORECASE)
        if art_match:
            html_to_parse = art_match.group(1)
            source_tag = "<article>"
        else:
            html_to_parse = html
            source_tag = "<body>"

    # Deterministic HTML tag stripper — removes nav/footer/header/script/style
    p = _TextExtractor()
    p.feed(html_to_parse)
    text = " ".join(p.chunks)

    # Store which tag was used (for display)
    if text:
        text = f"[Extracted from {source_tag}]\n\n" + text

    return text


@st.cache_data(ttl=86400, show_spinner=False)
def fetch_competitor_text_cached(url: str) -> str:
    """Cached version of fetch_url_text for competitor URLs.
    Results cached for 24h — same URL won't be re-scraped within the day.
    """
    return fetch_url_text(url, fresh=False)


def fetch_url_text(url: str, fresh: bool = False) -> str:
    """Fetch page text — uses Firecrawl if available, else fallback HTML scraper.
    fresh=True → max_age=0, bypasses cache (use for your own page).
    fresh=False → max_age=1 day cache (use for competitor pages).
    """
    fc = get_firecrawl()
    if fc:
        try:
            # Append a random query string to completely bypass Firecrawl's backend cache
            fetch_url = url
            if fresh:
                import time
                separator = "&" if "?" in fetch_url else "?"
                fetch_url = f"{fetch_url}{separator}nocache={int(time.time())}"

            result = fc.scrape(
                fetch_url,
                formats=["markdown"],
                only_main_content=True,
                max_age=0 if fresh else 86400000,
                exclude_tags=[
                    # Structure noise
                    "nav", "footer", "header", "aside",
                    # Cookie / GDPR banners
                    ".cookie-banner", "#cookie", ".cookie-consent",
                    "[class*='cookie']", "[id*='cookie']",
                    # Miclado Accessibility Tool (megabazeni.si + sites using acctoolbar.min.js)
                    "#mic-access-tool-box", "#mic-access-tool-general-button2",
                    ".accessibility-button", "[class*='mic-access']", "[id*='mic-access']",
                    # Generic accessibility widgets (UserWay, accessiBe, AudioEye, EqualWeb, etc.)
                    ".accessibility-widget", "#accessibility-widget",
                    "[class*='accessibility']", "[id*='accessibility']",
                    "[class*='userway']", "[id*='userway']",
                    "[class*='accessibe']", "[id*='accessibe']",
                    "[class*='audioeye']", "[class*='equalweb']",
                    # E-commerce listing noise (product grids, filters, pagination)
                    ".product-listing", ".product-grid", ".product-card",
                    ".filters", ".facets", "[class*='filter']",
                    ".breadcrumb", ".pagination", "[class*='pagination']",
                    # Navigation helpers
                    ".nav", "[class*='navbar']", "[class*='sidebar']",
                    # Popups / overlays
                    "[class*='modal']", "[class*='popup']", "[class*='overlay']",
                ],
            )
            text = getattr(result, "markdown", None) or ""
            if not text and isinstance(result, dict):
                text = result.get("markdown", "") or result.get("content", "")
            if text:
                # Post-process: remove product listings, keep only editorial text
                return clean_content_with_claude(text)

        except Exception as e:
            st.warning(f"Firecrawl failed ({e}), falling back to basic scraper.")

    # Fallback: basic HTML scraper
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
    is_slo = content_language in ("Slovenščina", "Italiano 🇮🇹", "Hrvatski 🇭🇷")

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
    } for e in resp.entities
      if e.name and e.name.strip()],  # filter empty entity names
    key=lambda x: x["salience"], reverse=True)

    sentences = [{
        "text":      s.text.content,
        "score":     round(s.sentiment.score, 3),
        "magnitude": round(s.sentiment.magnitude, 3),
    } for s in resp.sentences]

    raw_score = resp.document_sentiment.score
    raw_magnitude = resp.document_sentiment.magnitude

    sentiment = {
        "score":          round(raw_score, 3),
        "magnitude":      round(raw_magnitude, 3),
        "sentence_count": len(resp.sentences),
        "sentences":      sentences,
        "debug_raw":      f"raw score={raw_score}, magnitude={raw_magnitude}, sentences={len(resp.sentences)}",
    }

    syntax = _parse_syntax_tokens(resp.tokens)

    entity_sentiment = sorted([{
        "name":      e.name,
        "type":      language_v1.Entity.Type(e.type_).name,
        "salience":  round(e.salience, 4),
        "score":     round(e.sentiment.score, 3),
        "magnitude": round(e.sentiment.magnitude, 3),
        "wikipedia": e.metadata.get("wikipedia_url", ""),
    } for e in resp.entities
      if e.name and e.name.strip()],  # filter empty entity names
    key=lambda x: x["salience"], reverse=True)

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
    if content_language in ("Slovenščina", "Italiano 🇮🇹", "Hrvatski 🇭🇷"):
        claude_syntax = analyze_slovenian_syntax(text)

    return {
        "entities":         entities,
        "sentiment":        sentiment,
        "syntax":           syntax,
        "entity_sentiment": entity_sentiment,
        "categories":       categories,
        "claude_syntax":    claude_syntax,
        "content_language": content_language,
        "analyzed_at":      datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
    }


# ── Report export ─────────────────────────────────────────────────────────────

def build_markdown_report(data: dict, keyword: str, source: str, ai_report: str = "", benchmark: dict = None) -> str:
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
    # Use Claude syntax data for Slovenian/Italian if available
    cs       = data.get("claude_syntax", {})
    lang     = data.get("content_language", "English")
    use_claude = lang in ("Slovenščina", "Italiano 🇮🇹", "Hrvatski 🇭🇷") and cs
    verb_src = "🤖 Claude AI (zanesljivejše za slovenščino)" if use_claude else "🔵 Official Google NLP API"
    verb_val = cs.get("verb_count", sx["verb_count"]) if use_claude else sx["verb_count"]
    adj_val  = cs.get("adjective_count", sx["adjective_count"]) if use_claude else sx["adjective_count"]
    pv_val   = round(cs.get("passive_voice_count", 0) / max(cs.get("total_words", 1), 1) * 100, 1) \
               if use_claude else sx["passive_voice_pct"]

    lines.append(f"| Metric | Value | Source |")
    lines.append(f"|---|---|---|")
    lines.append(f"| Content language | {lang} | — |")
    lines.append(f"| Sentiment Score | {s['score']:+.3f} | 🔵 Official Google NLP API |")
    lines.append(f"| Magnitude | {s['magnitude']:.3f} | 🔵 Official Google NLP API |")
    lines.append(f"| Sentences | {s['sentence_count']} | 🔵 Official Google NLP API |")
    lines.append(f"| Passive Voice | {pv_val:.1f}% | {verb_src} · 🟠 15% threshold = best practice |")
    lines.append(f"| Lexical Density | {sx['lexical_density']:.1%} | 🔵 API token counts · 🟠 40% threshold = best practice |")
    lines.append(f"| Nouns | {sx['noun_count']} | 🔵 Official Google NLP API |")
    lines.append(f"| Verbs | {verb_val} | {verb_src} |")
    lines.append(f"| Adjectives | {adj_val} | {verb_src} |")
    lines.append(f"| Adverbs | {sx['adverb_count']} | 🔵 Official Google NLP API |")
    if use_claude and cs.get("notes"):
        lines.append(f"| Claude note | {cs['notes'][:100]} | 🤖 Claude AI |")
    lines.append("")

    # ── Reliability section ───────────────────────────────────────────────────
    lines.append("### ✅ Zanesljivost podatkov v tej analizi")
    lines.append("")
    if lang in ("Slovenščina", "Italiano 🇮🇹", "Hrvatski 🇭🇷"):
        lang_label = {"Slovenščina": "slovenščino", "Italiano 🇮🇹": "italijanščino", "Hrvatski 🇭🇷": "hrvaščino"}.get(lang, lang)
        lines.append(f"Vsebina je v **{lang_label}**. Google NLP API ima omejeno podporo za ta jezik.")
        lines.append("")
        lines.append("**🔵 100% zanesljivo (Google NLP API):**")
        lines.append(f"- Sentiment score: {s['score']:+.3f} ✅")
        lines.append(f"- Magnitude: {s['magnitude']:.2f} ✅")
        lines.append(f"- Kategorija: {data['categories'][0]['category'] if data.get('categories') else 'ni podatka'} ✅")
        lines.append(f"- Sentence-level sentiment (vsi stavki) ✅")
        lines.append(f"- Negativnih stavkov: {len([s2 for s2 in data['sentiment'].get('sentences',[]) if s2['score'] <= -0.25])} ✅")
        lines.append(f"- Entitete + salience scores ✅")
        lines.append(f"- Entity sentiment ✅")
        lines.append("")
        if use_claude:
            lines.append("**🤖 Claude AI (~85% zanesljivo):**")
            lines.append(f"- Verb count: {verb_val} ✅ (Google bi pokazal ~{sx['verb_count']} — napačno)")
            lines.append(f"- Adjective count: {adj_val} ✅")
            lines.append(f"- Passive voice: {pv_val:.1f}% ✅")
        else:
            lines.append("**⚠️ NEZANESLJIVO za to vsebino (Google API ne podpira dobro):**")
            lines.append(f"- Verb count: {sx['verb_count']} ⚠️ — verjetno prenizko (API napačno parsira morfologijo)")
            lines.append(f"- Adjective count: {sx['adjective_count']} ⚠️ — verjetno napačno")
            lines.append(f"- Passive voice %: {sx['passive_voice_pct']:.1f}% ⚠️ — odvisno od verb detection")
            lines.append("")
            lines.append("💡 *Ponovi analizo z 'Slovenščina' mode v appu za zanesljivejše sintaktične podatke.*")
        lines.append("")
        lines.append("**⚠️ Ignoriraj pri analizi:**")
        lines.append("- Entity TYPE klasifikacija (PERSON, ORGANIZATION za slovenska besedila je pogosto napačna)")
        lines.append("- Entity sentiment magnitude 0.00 — ne pomeni nujno brez sentimenta, API tega ne meri zanesljivo za slovenščino")
    else:
        lines.append("Vsebina je v **angleščini**. Vsi podatki so zanesljivi.")
        lines.append("")
        lines.append("**🔵 100% zanesljivo:**")
        lines.append("- Sentiment score + magnitude ✅")
        lines.append("- Kategorije ✅")
        lines.append("- Entitete + salience ✅")
        lines.append("- Entity sentiment ✅")
        lines.append("- Verb/adjective counts ✅")
        lines.append("- Passive voice % ✅")
        lines.append("- Lexical density ✅")
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
        lines.append("")

        # ── Category mismatch diagnostic ──────────────────────────────────────
        TECH_NOISE_CATS = [
            "/Computers & Electronics",
            "/Internet & Telecom",
            "/Science/Computer Science",
        ]
        top_cat = data["categories"][0]["category"]
        is_mismatched = any(top_cat.startswith(nc) for nc in TECH_NOISE_CATS)
        if is_mismatched:
            lines.append(
                f"> ⚠️ **Category mismatch detected:** Google classified this page as `{top_cat}` "
                f"— this is almost certainly caused by scraped noise, not the actual page content."
            )
            lines.append("> ")
            lines.append("> **Likely causes:**")
            lines.append("> 1. **Accessibility widget** content is leaking into the scraped text")
            lines.append(">    (CTRL+F2, tipkovnico, prikaz, kontrast, pisava... are Software UI terms)")
            lines.append("> 2. **Slovenian text sent to API as English** — `classifyText` does not support")
            lines.append(">    Slovenian, so it pattern-matches on whatever it recognises (UI widget terms)")
            lines.append("> ")
            lines.append("> **Fix:** Remove the accessibility widget from the crawled HTML before analysis.")
            lines.append("> The real category (from competitor benchmark) is likely `/Home & Garden/Home Swimming Pools`.")
            lines.append("")
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
    _verb = cs.get("verb_count", sx["verb_count"]) if (data.get("claude_syntax") and data.get("content_language") in ("Slovenščina", "Italiano 🇮🇹", "Hrvatski 🇭🇷")) else sx["verb_count"]
    lines.append(f"- **Verb/Noun ratio:** {_verb}/{sx['noun_count']} = {_verb/max(sx['noun_count'],1)*100:.1f}% ({'🤖 Claude' if _verb != sx['verb_count'] else '🔵 Google'})")
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
        lines.append("**All sentences:** *(prikaz okrajšan na 120 znakov — vsebina je popolna, ni odrezana)*")
        lines.append("")
        lines.append("| # | Score | Tone | Sentence (display truncated at 120 chars) |")
        lines.append("|---|---|---|---|")
        for i, s in enumerate(sentences, 1):
            tone = "🟢 Positive" if s["score"] >= 0.25 else "🔴 Negative" if s["score"] <= -0.25 else "🟡 Neutral"
            text = s["text"][:120].replace("|", "\\|")
            suffix = "..." if len(s["text"]) > 120 else ""
            lines.append(f"| {i} | {s['score']:+.2f} | {tone} | {text}{suffix} |")
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

    # ── Competitor Benchmark ──────────────────────────────────────────────────
    if benchmark and benchmark.get("n", 0) > 0:
        bm = benchmark
        n  = bm["n"]
        lines.append("---")
        lines.append("## 🏆 Competitor Benchmark")
        lines.append(f"*Based on {n} competitor page{'s' if n > 1 else ''} analyzed*")
        lines.append("")

        # Metrics comparison
        my_sal = 0.0
        if keyword:
            kw_lower = keyword.lower()
            my_matches = [e["salience"] for e in data["entities"] if kw_lower in e["name"].lower()]
            my_sal = round(my_matches[0] * 100, 1) if my_matches else 0.0

        def _diff(mine, avg, higher_better=True):
            d = mine - avg
            arrow = "▲" if d > 0 else ("▼" if d < 0 else "=")
            status = ("✓" if (d >= 0) == higher_better else "⚠") if d != 0 else "="
            return f"{arrow}{abs(d):.2f} {status}"

        lines.append("### 📊 Your page vs Competitor average")
        lines.append("")
        lines.append("| Metric | Your page | Competitor avg | Difference |")
        lines.append("|---|---|---|---|")
        lines.append(f"| Sentiment score | {data['sentiment']['score']:+.3f} | {bm['avg_sentiment']:+.3f} | {_diff(data['sentiment']['score'], bm['avg_sentiment'])} |")
        lines.append(f"| Magnitude | {data['sentiment']['magnitude']:.2f} | {bm['avg_magnitude']:.2f} | {_diff(data['sentiment']['magnitude'], bm['avg_magnitude'])} |")
        lines.append(f"| Lexical density | {data['syntax']['lexical_density']:.1%} | {bm['avg_lexical_density']:.1%} | {_diff(data['syntax']['lexical_density'], bm['avg_lexical_density'])} |")
        lines.append(f"| Passive voice | {data['syntax']['passive_voice_pct']:.1f}% | {bm['avg_passive_voice']:.1f}% | {_diff(data['syntax']['passive_voice_pct'], bm['avg_passive_voice'], higher_better=False)} |")
        if keyword:
            lines.append(f"| '{keyword}' salience | {my_sal:.1f}% | {bm['avg_kw_salience']:.1f}% | {_diff(my_sal, bm['avg_kw_salience'])} |")
        if bm.get("avg_word_count", 0) > 0:
            my_wc = len(bm.get("my_text", "").split()) if bm.get("my_text") else 0
            if my_wc:
                lines.append(f"| Word count | {my_wc:,} | {bm['avg_word_count']:,} | {_diff(my_wc, bm['avg_word_count'])} |")
        # Heading structure
        avg_h = bm.get("avg_headings", {})
        if avg_h and avg_h.get("total", 0) > 0:
            lines.append(f"| H1 count | — | {avg_h.get('h1', 0):.1f} | — |")
            lines.append(f"| H2 count | — | {avg_h.get('h2', 0):.1f} | — |")
            lines.append(f"| H3 count | — | {avg_h.get('h3', 0):.1f} | — |")
            lines.append(f"| Total headings | — | {avg_h.get('total', 0):.0f} | — |")
        lines.append("")

        # Heading structure detail
        if avg_h and avg_h.get("h2_texts"):
            lines.append("### 📑 Competitor heading structure")
            lines.append(f"- Avg H1: {avg_h.get('h1', 0):.1f} | Avg H2: {avg_h.get('h2', 0):.1f} | Avg H3: {avg_h.get('h3', 0):.1f}")
            lines.append("")
            lines.append("**Most common H2 topics from competitors:**")
            seen = set()
            for t in avg_h["h2_texts"]:
                if t.lower() not in seen:
                    seen.add(t.lower())
                    lines.append(f"- {t}")
                if len(seen) >= 15:
                    break
            lines.append("")

        # Top competitor entities — content gap
        if bm.get("top_entities"):
            my_entity_names = {e["name"].lower() for e in data["entities"]}
            lines.append("### 🏷 Top entities across competitors (content gap)")
            lines.append("*Entities your competitors rank highly for — add missing ones to close the gap*")
            lines.append("")
            lines.append("| Entity | Type | Avg salience % | Pages | On your page | KG |")
            lines.append("|---|---|---|---|---|---|")
            for e in bm["top_entities"][:20]:
                on_page = "✓" if e["name"].lower() in my_entity_names else "❌ Missing"
                kg      = "✓" if e.get("kg") else ""
                lines.append(f"| {e['name']} | {e['type']} | {e['avg_salience']:.1f}% | {e['present_in']}/{n} | {on_page} | {kg} |")
            lines.append("")

            missing = [e["name"] for e in bm["top_entities"] if e["name"].lower() not in my_entity_names]
            if missing:
                lines.append(f"**❌ Missing from your page:** {', '.join(missing[:8])}")
                lines.append("→ Add these topics to your content to close the content gap.")
                lines.append("")

        # Competitor content categories
        if bm.get("top_categories"):
            lines.append("### 📂 Competitor content categories")
            lines.append("")
            lines.append("| Category | Avg confidence | In how many pages |")
            lines.append("|---|---|---|")
            for c in bm["top_categories"]:
                lines.append(f"| {c['category']} | {c['avg_confidence']:.1f}% | {c['present_in']}/{n} |")
            lines.append("")

        # People Also Ask
        if bm.get("paa"):
            my_text_lower = bm.get("my_text_lower", "")
            lines.append("### ❓ People Also Ask")
            lines.append("*Questions Google shows for your keyword — cover these for long-tail + AI snippets*")
            lines.append("")
            for q in bm["paa"]:
                covered = any(word in my_text_lower for word in q.lower().split() if len(word) > 4) if my_text_lower else False
                icon = "✓" if covered else "❌"
                lines.append(f"- **{icon} {q}**")
            lines.append("")
            lines.append("→ Add an FAQ section answering the ❌ questions — improves long-tail ranking and AI Overview citations.")
            lines.append("")

    # ── Individual competitor analyses ───────────────────────────────────────
    individual = st.session_state.get("bench_individual", [])
    if individual:
        lines.append("---")
        lines.append("## 🔎 Posamična analiza konkurentov")
        lines.append(f"*{len(individual)} konkurentov analiziranih*")
        lines.append("")
        kw_lower_ind = keyword.lower() if keyword else ""
        for i, (url, cd) in enumerate(individual, 1):
            domain = url.replace("https://", "").replace("http://", "").split("/")[0]
            s_ind  = cd["sentiment"]
            sx_ind = cd["syntax"]
            kw_sal_ind = next(
                (e["salience"] * 100 for e in cd["entities"]
                 if kw_lower_ind and kw_lower_ind in e["name"].lower()), 0
            )
            lines.append(f"### #{i} {domain}")
            lines.append(f"**URL:** {url}")
            lines.append("")
            lines.append(f"| Metric | Value |")
            lines.append(f"|---|---|")
            lines.append(f"| Sentiment | {s_ind['score']:+.3f} |")
            lines.append(f"| Magnitude | {s_ind['magnitude']:.2f} |")
            lines.append(f"| Sentences | {s_ind['sentence_count']} |")
            lines.append(f"| Lexical density | {sx_ind['lexical_density']:.1%} |")
            lines.append(f"| Passive voice | {sx_ind['passive_voice_pct']:.1f}% |")
            lines.append(f"| Word count | {cd.get('word_count', 0):,} |")
            if keyword:
                lines.append(f"| '{keyword}' salience | {kw_sal_ind:.1f}% |")
            neg_ind = len([s2 for s2 in cd["sentiment"].get("sentences", []) if s2["score"] <= -0.25])
            lines.append(f"| Negative sentences | {neg_ind} |")
            lines.append("")
            # Top 10 entities
            lines.append("**Top entities:**")
            lines.append("")
            lines.append("| # | Entity | Type | Salience % | KG |")
            lines.append("|---|---|---|---|---|")
            for j, e in enumerate(cd["entities"][:10], 1):
                kg = "✓" if e["wikipedia"] else ""
                lines.append(f"| {j} | {e['name']} | {e['type']} | {e['salience']*100:.1f}% | {kg} |")
            lines.append("")
            # Categories
            if cd.get("categories"):
                cats = ", ".join(f"{c['category']} ({c['confidence']*100:.0f}%)" for c in cd["categories"][:3])
                lines.append(f"**Categories:** {cats}")
                lines.append("")
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

    # ── Report card ───────────────────────────────────────────────────────────
    score = s["score"]
    mag   = s["magnitude"]
    pv    = sx["passive_voice_pct"]
    ld    = sx["lexical_density"]

    def _row(label, value, status, what, action, source):
        col_l, col_v, col_s, col_a = st.columns([2, 1, 1, 4])
        col_l.markdown(f"**{label}**")
        col_v.markdown(f"`{value}`")
        col_s.markdown(status)
        col_a.caption(f"{what} → {action} · *{source}*")

    st.markdown("#### Metrike — kaj pomenijo in kaj storiti")
    st.caption("Vrednost · Status · Razlaga → Akcija")
    st.divider()

    # Sentiment
    if score >= 0.4:
        sent_status = "✅ OK"
        sent_action = "Ni akcije potrebne"
    elif score >= 0.1:
        sent_status = "⚠️ Nizko"
        sent_action = "Poišči negativne stavke (tab 📝 Sentences) in jih prepiši pozitivno"
    elif score >= -0.1:
        sent_status = "⚠️ Nevtralno"
        sent_action = "Za produktne strani dodaj koristi in pozitivne opise. Za blog je OK."
    else:
        sent_status = "❌ Negativno"
        sent_action = "Odpri tab 📝 Sentences → poišči rdeče stavke → prepiši jih"
    _row("😊 Sentiment score", f"{score:+.2f}",
         sent_status,
         "Čustveni ton (−1 negativno → +1 pozitivno)",
         sent_action,
         "🔵 Google NLP API")

    # Magnitude
    if score >= 0 and mag > 5:
        mag_status = "⚠️ Mešano"
        mag_action = "Visoka moč + nevtralen score = nekateri deli pozitivni, drugi negativni. Poišči negativne stavke."
    elif mag < 1:
        mag_status = "ℹ️ Suho"
        mag_action = "Besedilo je zelo faktično. Za blog OK, za produktne strani dodaj čustvene opise."
    else:
        mag_status = "✅ OK"
        mag_action = "Ni akcije potrebne"
    _row("💪 Magnitude", f"{mag:.2f}",
         mag_status,
         "Moč čustev (0 = suho, 5+ = čustveno)",
         mag_action,
         "🔵 Google NLP API")

    # Passive voice
    if pv > 20:
        pv_status = "❌ Previsoko"
        pv_action = "Odpri tab 🔤 Syntax → poišči pasivne stavke → prepiši v aktivno obliko"
    elif pv > 15:
        pv_status = "⚠️ Visoko"
        pv_action = "Zmanjšaj pasivni glas. Primer: 'je bil postavljen' → 'postavi'"
    else:
        pv_status = "✅ OK"
        pv_action = "Ni akcije potrebne"
    _row("📝 Pasivni glas", f"{pv:.1f}%",
         pv_status,
         "Pasivno pisanje (cilj: pod 15%)",
         pv_action,
         "🔵 Google NLP · 🟠 15% = best practice")

    # Lexical density
    if ld < 0.35:
        ld_status = "❌ Prenizko"
        ld_action = "Besedilo je preveč 'prazno'. Zamenjaj splošne fraze s konkretnimi podatki, dimenzijami, lastnostmi."
    elif ld < 0.40:
        ld_status = "⚠️ Nizko"
        ld_action = "Dodaj več konkretnih samostalnikov in opisov. Manj splošnih besed kot 'zelo', 'res', 'dober'."
    else:
        ld_status = "✅ OK"
        ld_action = "Ni akcije potrebne"
    _row("📚 Lexical density", f"{ld:.1%}",
         ld_status,
         "Vsebinska gostota (cilj: 40%+)",
         ld_action,
         "🔵 Google NLP · 🟠 40% = best practice")

    st.divider()
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

    # ── Top 5 salience concentration ─────────────────────────────────────────
    st.divider()
    entities = data["entities"]
    if entities:
        top5_sal  = sum(e["salience"] for e in entities[:5]) * 100
        top10_sal = sum(e["salience"] for e in entities[:10]) * 100
        total_ents = len(entities)

        col_t1, col_t2, col_t3 = st.columns(3)
        col_t1.metric("Top 5 entities cover",  f"{top5_sal:.0f}%",
                      help="% of total salience covered by top 5 entities. Target: 60%+")
        col_t2.metric("Top 10 entities cover", f"{top10_sal:.0f}%",
                      help="% of total salience covered by top 10 entities")
        col_t3.metric("Total entities found",  total_ents,
                      help="Lower number = more focused page. High number = content is dispersed")

        st.info(
            f"Top 5 cover **{top5_sal:.0f}%** · Top 10 cover **{top10_sal:.0f}%** · "
            f"Total entities: {total_ents} · "
            f"**Run competitor benchmark to see the real target for your niche ↓**"
        )
        st.caption(
            f"{OFFICIAL} — salience scores from Google NLP API · "
            f"🟠 No universal target exists — run Competitor Benchmark below to get the real average for your keyword"
        )

    # ── Keyword check ─────────────────────────────────────────────────────────
    if keyword:
        st.divider()
        kw_lower = keyword.lower()
        matches = [e for e in data["entities"] if kw_lower in e["name"].lower()]
        if matches:
            top  = matches[0]
            sal  = top["salience"] * 100
            kg   = "in Knowledge Graph ✓" if top["wikipedia"] else "not in Knowledge Graph"
            rank = next((i+1 for i, e in enumerate(data["entities"])
                         if kw_lower in e["name"].lower()), "?")
            if sal >= 15:
                st.success(f"**'{keyword}'** — Rank #{rank} · Salience **{sal:.1f}%** · Type: {top['type']} · {kg} — Excellent, Google clearly sees this as the main topic.")
            elif sal >= 8:
                st.warning(f"**'{keyword}'** — Rank #{rank} · Salience **{sal:.1f}%** · Type: {top['type']} · {kg} — OK but could be stronger. Target: 15%+.")
            else:
                st.error(f"**'{keyword}'** — Rank #{rank} · Salience only **{sal:.1f}%** · {kg} — Too low. Google does not see this as the main topic.")
        else:
            st.error(
                f"**'{keyword}'** not detected as an entity. "
                "Add it to H1, first 100 words, and at least 2–3 H2 headings."
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
            return (["background-color:#2d5a1b; color:#ffffff; font-weight:bold"] * len(row)
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

    # Debug info — temporary
    if s.get("debug_raw"):
        st.caption(f"🔧 Debug: {s['debug_raw']}")

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
    is_slo = data.get("content_language", "English") in ("Slovenščina", "Italiano 🇮🇹", "Hrvatski 🇭🇷")

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


def tab_sentence_sentiment(data: dict, key_prefix: str = "main"):
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
        key=f"sent_filter_{key_prefix}",
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


def compute_benchmark(results_list: list, keyword: str) -> dict:
    """Compute average metrics from a list of NLP analysis results."""
    if not results_list:
        return {}

    n = len(results_list)

    avg_sentiment  = round(sum(r["sentiment"]["score"]    for r in results_list) / n, 3)
    avg_magnitude  = round(sum(r["sentiment"]["magnitude"] for r in results_list) / n, 3)
    avg_lex        = round(sum(r["syntax"]["lexical_density"] for r in results_list) / n, 3)
    avg_passive    = round(sum(r["syntax"]["passive_voice_pct"] for r in results_list) / n, 1)

    # Average keyword salience
    kw_lower = keyword.lower() if keyword else ""
    kw_saliences = []
    for r in results_list:
        matches = [e["salience"] for e in r["entities"] if kw_lower in e["name"].lower()]
        kw_saliences.append(matches[0] if matches else 0.0)
    avg_kw_salience = round(sum(kw_saliences) / n * 100, 1)

    # Top entities across all competitors (by avg salience)
    entity_map: dict = {}
    for r in results_list:
        for e in r["entities"][:15]:
            name = e["name"].lower()
            if name not in entity_map:
                entity_map[name] = {"name": e["name"], "type": e["type"],
                                    "saliences": [], "kg": e["wikipedia"]}
            entity_map[name]["saliences"].append(e["salience"])
    top_entities = sorted(
        [{"name": v["name"], "type": v["type"],
          "avg_salience": round(sum(v["saliences"]) / len(v["saliences"]) * 100, 1),
          "present_in": len(v["saliences"]), "kg": v["kg"]}
         for v in entity_map.values()],
        key=lambda x: x["avg_salience"], reverse=True
    )[:20]

    # Top categories
    cat_map: dict = {}
    for r in results_list:
        for c in r["categories"]:
            cat_map[c["category"]] = cat_map.get(c["category"], [])
            cat_map[c["category"]].append(c["confidence"])
    top_categories = sorted(
        [{"category": k, "avg_confidence": round(sum(v) / len(v) * 100, 1),
          "present_in": len(v)}
         for k, v in cat_map.items()],
        key=lambda x: x["avg_confidence"], reverse=True
    )[:5]

    # Word count
    word_counts = [r.get("word_count", 0) for r in results_list if r.get("word_count", 0) > 0]
    avg_word_count = round(sum(word_counts) / len(word_counts)) if word_counts else 0

    # Heading structure averages
    heading_data = [r["headings"] for r in results_list if r.get("headings")]
    def _avg_heading(level):
        counts = [h.get(f"{level}_count", 0) for h in heading_data]
        return round(sum(counts) / len(counts), 1) if counts else 0

    avg_headings = {
        "h1": _avg_heading("h1"),
        "h2": _avg_heading("h2"),
        "h3": _avg_heading("h3"),
        "h4": _avg_heading("h4"),
        "total": round(sum(_avg_heading(l) for l in ["h1","h2","h3","h4"]), 1),
        # Collect all H2 texts for content gap
        "h2_texts": [
            text
            for h in heading_data
            for text in h.get("headings", {}).get("h2", [])
            if text
        ]
    }

    # Top 5 salience concentration
    top5_concentrations = []
    for r in results_list:
        if r["entities"]:
            top5_concentrations.append(
                sum(e["salience"] for e in r["entities"][:5]) * 100
            )
    avg_top5_concentration = round(sum(top5_concentrations) / len(top5_concentrations), 1) \
        if top5_concentrations else 0

    # Sentence sentiment distribution
    pos_ratios, neg_ratios, neu_ratios = [], [], []
    avg_sent_scores = []
    for r in results_list:
        sents = r["sentiment"].get("sentences", [])
        if sents:
            pos = len([s for s in sents if s["score"] >= 0.25])
            neg = len([s for s in sents if s["score"] <= -0.25])
            neu = len(sents) - pos - neg
            pos_ratios.append(round(pos / len(sents) * 100, 1))
            neg_ratios.append(round(neg / len(sents) * 100, 1))
            neu_ratios.append(round(neu / len(sents) * 100, 1))
            avg_sent_scores.extend([s["score"] for s in sents])

    avg_sentence_dist = {
        "positive_pct": round(sum(pos_ratios) / len(pos_ratios), 1) if pos_ratios else 0,
        "negative_pct": round(sum(neg_ratios) / len(neg_ratios), 1) if neg_ratios else 0,
        "neutral_pct":  round(sum(neu_ratios) / len(neu_ratios), 1) if neu_ratios else 0,
        "avg_sentence_count": round(
            sum(r["sentiment"]["sentence_count"] for r in results_list) / n, 0),
    }

    # Syntax averages
    avg_verb_count  = round(sum(r["syntax"]["verb_count"]      for r in results_list) / n, 1)
    avg_adj_count   = round(sum(r["syntax"]["adjective_count"] for r in results_list) / n, 1)
    avg_noun_count  = round(sum(r["syntax"]["noun_count"]      for r in results_list) / n, 1)
    avg_verb_noun_ratio = round(avg_verb_count / avg_noun_count * 100, 1) if avg_noun_count else 0

    # Entity type distribution
    type_counts: dict = {}
    for r in results_list:
        for e in r["entities"][:20]:
            t = e["type"]
            type_counts[t] = type_counts.get(t, 0) + 1
    top_entity_types = sorted(type_counts.items(), key=lambda x: x[1], reverse=True)[:5]

    return {
        "n": n,
        "avg_sentiment":          avg_sentiment,
        "avg_magnitude":          avg_magnitude,
        "avg_lexical_density":    avg_lex,
        "avg_passive_voice":      avg_passive,
        "avg_kw_salience":        avg_kw_salience,
        "avg_word_count":         avg_word_count,
        "avg_top5_concentration": avg_top5_concentration,
        "avg_headings":           avg_headings,
        "avg_sentence_dist":      avg_sentence_dist,
        "avg_verb_count":         avg_verb_count,
        "avg_adj_count":          avg_adj_count,
        "avg_noun_count":         avg_noun_count,
        "avg_verb_noun_ratio":    avg_verb_noun_ratio,
        "top_entity_types":       top_entity_types,
        "top_entities":           top_entities,
        "top_categories":         top_categories,
    }


def tab_benchmark(benchmark: dict, my_data: dict, keyword: str):
    if not benchmark:
        st.info("No benchmark data yet. Run competitor analysis below.")
        return

    n = benchmark["n"]
    st.caption(f"Based on {n} competitor page{'s' if n > 1 else ''} analyzed")

    # ── Metrics comparison ────────────────────────────────────────────────────
    st.subheader("📊 Your page vs Competitor average")

    my_s   = my_data["sentiment"]
    my_sx  = my_data["syntax"]

    def _delta(my_val, avg_val, higher_is_better=True):
        diff = round(my_val - avg_val, 3)
        if higher_is_better:
            return f"{diff:+.3f}", "normal" if diff >= 0 else "inverse"
        else:
            return f"{diff:+.3f}", "inverse" if diff >= 0 else "normal"

    c1, c2, c3, c4, c5 = st.columns(5)
    sd, sc = _delta(my_s["score"], benchmark["avg_sentiment"])
    c1.metric("Sentiment", f"{my_s['score']:+.2f}",
              delta=f"{sd} vs avg {benchmark['avg_sentiment']:+.2f}", delta_color=sc)

    md2, mc = _delta(my_s["magnitude"], benchmark["avg_magnitude"])
    c2.metric("Magnitude", f"{my_s['magnitude']:.2f}",
              delta=f"{md2} vs avg {benchmark['avg_magnitude']:.2f}", delta_color=mc)

    ld, lc = _delta(my_sx["lexical_density"], benchmark["avg_lexical_density"])
    c3.metric("Lexical Density", f"{my_sx['lexical_density']:.1%}",
              delta=f"{ld} vs avg {benchmark['avg_lexical_density']:.1%}", delta_color=lc)

    pv, pc = _delta(my_sx["passive_voice_pct"], benchmark["avg_passive_voice"],
                    higher_is_better=False)
    c4.metric("Passive Voice", f"{my_sx['passive_voice_pct']:.1f}%",
              delta=f"{pv}% vs avg {benchmark['avg_passive_voice']:.1f}%", delta_color=pc)

    if keyword:
        kw_lower = keyword.lower()
        my_matches = [e["salience"] for e in my_data["entities"]
                      if kw_lower in e["name"].lower()]
        my_sal = round(my_matches[0] * 100, 1) if my_matches else 0.0
        kd, kc = _delta(my_sal, benchmark["avg_kw_salience"])
        c5.metric(f"'{keyword}' salience", f"{my_sal:.1f}%",
                  delta=f"{kd}% vs avg {benchmark['avg_kw_salience']:.1f}%", delta_color=kc)

    # ── Top 5 concentration comparison ───────────────────────────────────────
    avg_top5 = benchmark.get("avg_top5_concentration", 0)
    if avg_top5 > 0:
        st.divider()
        st.subheader("🎯 Salience concentration — real benchmark")
        st.caption("This is the actual target for your keyword — based on competitor data, not a generic rule.")

        my_entities = my_data["entities"]
        my_top5 = sum(e["salience"] for e in my_entities[:5]) * 100 if my_entities else 0

        col_a, col_b = st.columns(2)
        td, tc = _delta(my_top5, avg_top5)
        col_a.metric("Your top 5 concentration", f"{my_top5:.0f}%",
                     delta=f"{td}% vs competitor avg {avg_top5:.0f}%", delta_color=tc)
        col_b.metric("Competitor average", f"{avg_top5:.0f}%",
                     help=f"Average top 5 salience concentration across {n} competitor pages")

        if my_top5 >= avg_top5:
            st.success(
                f"✓ Your top 5 entities cover **{my_top5:.0f}%** — "
                f"at or above competitor average ({avg_top5:.0f}%). "
                f"Google sees your page as clearly focused."
            )
        else:
            gap = avg_top5 - my_top5
            st.warning(
                f"⚠ Your top 5 cover **{my_top5:.0f}%** vs competitor avg **{avg_top5:.0f}%** "
                f"(gap: {gap:.0f}%). "
                f"Your content is more dispersed than competitors. "
                f"Strengthen your main topic or remove off-topic sections."
            )
        st.caption(
            f"{OFFICIAL} — salience scores from Google NLP API · "
            f"🟠 Concentration comparison is empirical (based on your actual competitors), not a universal rule"
        )

    # ── Heading structure ─────────────────────────────────────────────────────
    avg_h = benchmark.get("avg_headings", {})
    if avg_h and avg_h.get("total", 0) > 0:
        st.divider()
        st.subheader("📑 Heading structure")
        st.caption("Average heading structure across competitor pages · DataForSEO On-Page API")

        col_h1, col_h2, col_h3, col_h4, col_ht = st.columns(5)
        col_h1.metric("Avg H1", f"{avg_h['h1']:.1f}")
        col_h2.metric("Avg H2", f"{avg_h['h2']:.1f}")
        col_h3.metric("Avg H3", f"{avg_h['h3']:.1f}")
        col_h4.metric("Avg H4", f"{avg_h['h4']:.1f}")
        col_ht.metric("Total headings", f"{avg_h['total']:.0f}")

        st.caption(
            f"🟠 Target: match competitor averages — especially H2 count "
            f"(competitors use ~{avg_h['h2']:.0f} H2 headings)"
        )

        # Show most common H2 texts from competitors
        h2_texts = avg_h.get("h2_texts", [])
        if h2_texts:
            st.markdown("**Most common competitor H2 headings** — use these as inspiration:")
            # Show unique H2s, max 15
            seen = set()
            unique_h2s = []
            for t in h2_texts:
                tl = t.lower()
                if tl not in seen:
                    seen.add(tl)
                    unique_h2s.append(t)
            for h2 in unique_h2s[:15]:
                st.markdown(f"- *{h2}*")

    # ── Top competitor entities ───────────────────────────────────────────────
    st.divider()
    st.subheader("🏷 Top entities across competitors")
    st.caption("Entities your competitors rank highly for — these are your content targets")

    my_entity_names = {e["name"].lower() for e in my_data["entities"]}
    ent_df = pd.DataFrame(benchmark["top_entities"])
    ent_df["on your page"] = ent_df["name"].apply(
        lambda x: "✓" if x.lower() in my_entity_names else "❌ Missing"
    )
    ent_df["KG"] = ent_df["kg"].apply(lambda x: "✓" if x else "")
    st.dataframe(
        ent_df[["name", "type", "avg_salience", "present_in", "on your page", "KG"]],
        use_container_width=True, hide_index=True,
        column_config={
            "avg_salience": st.column_config.NumberColumn("Avg salience %", format="%.1f%%"),
            "present_in":   st.column_config.NumberColumn(f"In how many/{n} pages"),
        }
    )

    missing = [e for e in benchmark["top_entities"] if e["name"].lower() not in my_entity_names]
    if missing:
        names = ", ".join(f"**{e['name']}**" for e in missing[:5])
        st.warning(f"❌ Missing from your page: {names} — add these topics to close the content gap.")

    # ── Word count ────────────────────────────────────────────────────────────
    if benchmark.get("avg_word_count", 0) > 0:
        st.divider()
        my_wc = len(benchmark.get("my_text", "").split()) if benchmark.get("my_text") else 0
        st.subheader("📝 Word count")
        wc1, wc2 = st.columns(2)
        wc1.metric("Competitor average", f"{benchmark['avg_word_count']:,} words")
        if my_wc:
            diff = my_wc - benchmark["avg_word_count"]
            wc2.metric("Your page", f"{my_wc:,} words",
                       delta=f"{diff:+,} vs avg",
                       delta_color="normal" if diff >= 0 else "inverse")
            if diff < -200:
                st.warning(f"⚠ Your page has {abs(diff):,} fewer words than competitors. "
                           f"Consider expanding content to {benchmark['avg_word_count']:,}+ words.")

    # ── Categories ────────────────────────────────────────────────────────────
    if benchmark["top_categories"]:
        st.divider()
        st.subheader("📂 Competitor content categories")
        cat_df = pd.DataFrame(benchmark["top_categories"])
        st.dataframe(cat_df, use_container_width=True, hide_index=True,
                     column_config={
                         "avg_confidence": st.column_config.NumberColumn("Avg confidence %", format="%.1f%%"),
                         "present_in": st.column_config.NumberColumn(f"In how many/{n} pages"),
                     })

    # ── People Also Ask ───────────────────────────────────────────────────────
    paa = benchmark.get("paa", [])
    if paa:
        st.divider()
        st.subheader("❓ People Also Ask")
        st.caption("Questions Google shows for your keyword — cover these to target long-tail searches and AI snippets")
        my_text_lower = benchmark.get("my_text_lower", "")
        for q in paa:
            covered = any(word in my_text_lower for word in q.lower().split()
                         if len(word) > 4) if my_text_lower else False
            icon = "✓" if covered else "❌"
            st.markdown(f"**{icon} {q}**")
        st.info("💡 Add an FAQ section answering these questions — improves long-tail ranking and AI Overview citations.")

    _source_legend()
    st.caption(f"{OFFICIAL} — NLP data from Google NLP API · DataForSEO — SERP + PAA data · {PRACTICE} — benchmark methodology")

    # ── Individual competitor analysis ────────────────────────────────────────
    individual = st.session_state.get("bench_individual", [])
    if individual:
        st.divider()
        st.subheader("🔎 Posamična analiza konkurentov")
        st.caption("Klikni na konkurenta da vidiš njegovo celotno analizo — enako kot za tvojo stran")

        for i, (url, comp_data) in enumerate(individual):
            domain = url.replace("https://", "").replace("http://", "").split("/")[0]
            sent   = comp_data["sentiment"]["score"]
            wc     = comp_data.get("word_count", 0)
            kw_lower = keyword.lower() if keyword else ""
            kw_sal = next(
                (e["salience"] * 100 for e in comp_data["entities"]
                 if kw_lower and kw_lower in e["name"].lower()), 0
            )
            label = (
                f"#{i+1} **{domain}** · "
                f"Sentiment: {sent:+.2f} · "
                f"Salience '{keyword}': {kw_sal:.1f}% · "
                f"Words: {wc:,}"
            ) if keyword else (
                f"#{i+1} **{domain}** · Sentiment: {sent:+.2f} · Words: {wc:,}"
            )

            with st.expander(label):
                render_analysis(comp_data, keyword, source=url,
                                key_prefix=f"comp_{i}", is_competitor=True)

    # ── Content Brief Generator ───────────────────────────────────────────────
    st.divider()
    st.subheader("📝 Generate Content Brief")
    st.caption(
        "Claude prebere vse zgornje benchmark podatke in napiše točen brief "
        "— koliko besed, kateri naslovi, katere entitete, kakšen ton."
    )

    if get_anthropic_client() is None:
        st.warning("Add ANTHROPIC_API_KEY to Streamlit Secrets to enable content brief generation.")
    else:
        brief_lang = st.radio("Brief language", ["Slovenščina", "English"],
                              horizontal=True, key="brief_lang")

        if st.button("📝 Generate Content Brief", type="primary",
                     use_container_width=True, key="gen_brief_btn"):
            with st.spinner("Claude analyzes benchmark data and writes your content brief..."):
                try:
                    brief = generate_content_brief(benchmark, keyword, brief_lang)
                    st.session_state["content_brief"] = brief
                    st.session_state["content_brief_keyword"] = keyword
                except Exception as e:
                    st.error(f"Error: {e}")
                    st.session_state["content_brief"] = ""

    if st.session_state.get("content_brief"):
        brief = st.session_state["content_brief"]
        st.markdown("---")
        st.markdown(brief)

        # Download button
        kw   = st.session_state.get("content_brief_keyword", "content")
        ts   = datetime.now().strftime("%Y%m%d_%H%M%S")
        fname = f"content_brief_{kw.replace(' ', '_')}_{ts}.md"
        st.download_button(
            label="📥 Download Content Brief",
            data=brief,
            file_name=fname,
            mime="text/markdown",
            use_container_width=True,
        )


def _freshness_indicator(data: dict, bench_date: str = ""):
    """Show data freshness status for own page and competitor benchmark."""
    from datetime import datetime as _dt

    analyzed_at = data.get("analyzed_at", "")
    now = _dt.now()

    parts = []

    # Own page freshness
    if analyzed_at:
        try:
            t = _dt.strptime(analyzed_at, "%Y-%m-%d %H:%M:%S")
            mins = int((now - t).total_seconds() / 60)
            if mins < 5:
                parts.append(f"🟢 **Tvoja stran:** Pravkar analizirana ({mins}m nazaj) — sveži podatki")
            elif mins < 60:
                parts.append(f"🟢 **Tvoja stran:** Analizirana {mins}m nazaj — sveži podatki")
            else:
                hrs = mins // 60
                parts.append(f"🟡 **Tvoja stran:** Analizirana {hrs}h nazaj — podatki so morda zastareli, re-analiziraj")
        except Exception:
            parts.append(f"🟢 **Tvoja stran:** Analizirana ob {analyzed_at}")

    # Competitor benchmark freshness
    if bench_date:
        try:
            t = _dt.strptime(bench_date, "%d.%m.%Y %H:%M")
            mins = int((now - t).total_seconds() / 60)
            if mins < 60:
                parts.append(f"🟢 **Benchmark:** Zgrajen {mins}m nazaj — sveži podatki")
            elif mins < 1440:
                hrs = mins // 60
                parts.append(f"🟡 **Benchmark:** Zgrajen {hrs}h nazaj — Firecrawl cache (< 24h)")
            else:
                days = mins // 1440
                parts.append(f"🔴 **Benchmark:** Zgrajen {days}d nazaj — klikni 🔄 Refresh competitors")
        except Exception:
            parts.append(f"🟡 **Benchmark:** Zgrajen {bench_date}")
    else:
        parts.append("⚪ **Benchmark:** Ni podatkov — poženi competitor analizo spodaj")

    if parts:
        st.caption(" · ".join(parts))


def render_analysis(data: dict, keyword: str = "", source: str = "",
                    benchmark: dict = None, key_prefix: str = "main",
                    is_competitor: bool = False):

    # ── Freshness indicator ───────────────────────────────────────────────────
    bench_date = st.session_state.get("bench_date", "")
    _freshness_indicator(data, bench_date)

    # ── NeuroWriter JSON input (persists in session) ──────────────────────────
    with st.expander("🔧 NeuroWriter JSON (optional — paste to enable NW Score tab)"):
        nw_raw = st.text_area(
            "Prilepi NeuroWriter JSON export",
            value=st.session_state.get("nw_raw", ""),
            height=80,
            placeholder='JSON: {"basic_terms":[...]} ALI TEXT FORMAT:\nTITLE TERMS: =====\nterm\nBASIC TEXT TERMS: =====\nterm: 1-5x',
            key=f"nw_raw_input_{source[:20]}",
        )
        if nw_raw and nw_raw != st.session_state.get("nw_raw", ""):
            st.session_state["nw_raw"]    = nw_raw
            st.session_state["nw_parsed"] = parse_nw_json(nw_raw)
        if st.session_state.get("nw_parsed"):
            nw = st.session_state["nw_parsed"]
            st.success(
                f"✅ NW loaded — {len(nw['basic'])} basic terms · "
                f"{len(nw['extended'])} extended terms"
            )

    nw = st.session_state.get("nw_parsed", {})

    if is_competitor:
        # Competitor: show all analysis tabs, skip Benchmark/NW/AI Coach
        t1, t2, t3, t4, t5, t6, t7 = st.tabs([
            "📊 Overview",
            "🏷 Entities",
            "😊 Sentiment",
            "📝 Sentences",
            "📂 Categories",
            "🔤 Syntax",
            "🎯 Entity Sentiment",
        ])
        with t1: tab_overview(data, keyword)
        with t2: tab_entities(data, keyword)
        with t3: tab_sentiment(data)
        with t4: tab_sentence_sentiment(data, key_prefix=key_prefix)
        with t5: tab_categories(data)
        with t6: tab_syntax(data)
        with t7: tab_entity_sentiment(data)
    else:
        t1, t2, t3, t4, t5, t6, t7, t8, t9, t10 = st.tabs([
            "📊 Overview",
            "🏷 Entities",
            "😊 Sentiment",
            "📝 Sentences",
            "📂 Categories",
            "🔤 Syntax",
            "🎯 Entity Sentiment",
            "🏆 Benchmark",
            f"🎯 NW Score{' (' + str(score_content_nw(st.session_state.get('my_text',''), nw)['overall_score']) + '%)' if nw and st.session_state.get('my_text') else ''}",
            "🤖 AI SEO Coach",
        ])
        with t1: tab_overview(data, keyword)
        with t2: tab_entities(data, keyword)
        with t3: tab_sentiment(data)
        with t4: tab_sentence_sentiment(data, key_prefix=key_prefix)
        with t5: tab_categories(data)
        with t6: tab_syntax(data)
        with t7: tab_entity_sentiment(data)
        with t8:  tab_benchmark(benchmark or {}, data, keyword)
        with t9:  tab_nw_score(st.session_state.get("my_text", ""), nw, key_prefix=key_prefix)
        with t10: tab_ai_coach(data, keyword, key_prefix=key_prefix)

    # ── Download button — for main and url2, not competitor sub-analyses ────────
    if key_prefix in ("main", "url2") and not is_competitor:
        st.divider()
        ai_report_key = f"ai_report_{key_prefix}"
        ai_report = st.session_state.get(ai_report_key, "")
        bench = st.session_state.get("benchmark", None) if key_prefix == "main" else None
        md = build_markdown_report(data, keyword, source, ai_report, bench)
        ts = datetime.now().strftime("%Y%m%d_%H%M%S")
        slug = keyword.replace(" ", "_").lower() if keyword else "analysis"
        label_prefix = "tvoja" if key_prefix == "main" else "konkurent"
        filename = f"analiza_{label_prefix}_{slug}_{ts}.md"

        st.download_button(
            label=f"📥 Download — {'Tvoja analiza' if key_prefix == 'main' else 'Analiza konkurenta'}",
            data=md,
            file_name=filename,
            mime="text/markdown",
            use_container_width=True,
            key=f"dl_btn_{key_prefix}",
            help="Downloads analysis as Markdown file.",
        )
        if key_prefix == "main":
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


def parse_lines(raw: str) -> list[str]:
    return [line.strip() for line in raw.splitlines() if line.strip()]


def save_content_optimizer_audit(result: dict) -> list[Path]:
    ANALIZE_DIR.mkdir(exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    saved_paths = []
    for filename, content in build_audit_artifacts(result, timestamp).items():
        path = ANALIZE_DIR / filename
        path.write_text(content, encoding="utf-8")
        saved_paths.append(path)
    return saved_paths


def latest_saved_audits(limit: int = 8) -> list[Path]:
    if not ANALIZE_DIR.exists():
        return []
    patterns = ("analiza_*.json", "plan_izboljsav_*.md", "content_brief_2_0_*.md")
    files = []
    for pattern in patterns:
        files.extend(ANALIZE_DIR.rglob(pattern))
    return sorted(files, key=lambda path: path.stat().st_mtime, reverse=True)[:limit]


AI_VISIBILITY_DIR = ANALIZE_DIR / "AI_VISIBILITY"


def build_ai_visibility_bundle(
    discovery_inputs: dict[str, Any] | None,
    discovery_result: dict[str, Any] | None,
    model_inputs: dict[str, Any] | None,
    model_result: list[dict[str, Any]] | None,
) -> dict[str, Any]:
    return {
        "saved_at": datetime.now().isoformat(timespec="seconds"),
        "discovery_inputs": discovery_inputs or {},
        "discovery_result": discovery_result or {},
        "model_inputs": model_inputs or {},
        "model_result": model_result or [],
    }


def build_ai_visibility_markdown(bundle: dict[str, Any]) -> str:
    discovery_inputs = bundle.get("discovery_inputs") or {}
    discovery_result = bundle.get("discovery_result") or {}
    model_inputs = bundle.get("model_inputs") or {}
    model_result = bundle.get("model_result") or []

    domain = discovery_inputs.get("domain") or model_inputs.get("domain") or "n/a"
    brand = discovery_inputs.get("brand") or model_inputs.get("brand") or "n/a"
    discovery_platform = discovery_inputs.get("platform", "n/a")
    discovery_items = discovery_result.get("items", []) or []

    llm_platforms = sorted({
        item.get("platform", "")
        for item in model_result
        if isinstance(item, dict) and item.get("platform")
    })
    brand_hits = sum(
        1
        for item in model_result
        if isinstance(item, dict)
        and not item.get("error")
        and brand
        and normalize_text(brand) in normalize_text(item.get("answer", ""))
    )
    lines = [
        "# AI Visibility Run",
        "",
        f"- Saved at: {bundle.get('saved_at', '')}",
        f"- Domain: {domain}",
        f"- Brand: {brand}",
        f"- Discovery platform: {discovery_platform}",
        f"- Discovery rows: {len(discovery_items)}",
        f"- Model responses: {len(model_result)}",
        f"- LLMs tested: {', '.join(llm_platforms) if llm_platforms else 'n/a'}",
        f"- Brand hits across model responses: {brand_hits}",
        "",
    ]

    if discovery_items:
        lines.extend([
            "## Top Discovery Queries",
            "",
            "| Question | AI Search Volume | Model | Sources |",
            "|---|---:|---|---:|",
        ])
        for item in discovery_items[:10]:
            question = (item.get("question", "") or "").replace("\n", " ").strip()
            lines.append(
                f"| {question[:120]} | {item.get('ai_search_volume', 0) or 0} | "
                f"{item.get('model_name', '')} | {len(item.get('sources', []) or [])} |"
            )
        lines.append("")

    if model_result:
        lines.extend([
            "## Model Response Testing",
            "",
            "| Query | LLM | Model | Sources | Error |",
            "|---|---|---|---:|---|",
        ])
        for item in model_result[:20]:
            query = (item.get("query", "") or "").replace("\n", " ").strip()
            error = item.get("error", "")
            lines.append(
                f"| {query[:100]} | {item.get('platform', '')} | {item.get('model_name', item.get('requested_model', ''))} | "
                f"{len(item.get('annotations', []) or [])} | {error[:80]} |"
            )
        lines.append("")

    return "\n".join(lines).strip() + "\n"


def save_ai_visibility_run(bundle: dict[str, Any]) -> list[Path]:
    AI_VISIBILITY_DIR.mkdir(parents=True, exist_ok=True)
    domain = bundle.get("discovery_inputs", {}).get("domain") or bundle.get("model_inputs", {}).get("domain") or ""
    brand = bundle.get("discovery_inputs", {}).get("brand") or bundle.get("model_inputs", {}).get("brand") or ""
    base_label = normalize_text(brand or domain or "ai_visibility_run")
    slug = re.sub(r"[^\wÀ-ž]+", "_", base_label, flags=re.UNICODE).strip("_") or "ai_visibility_run"
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    json_path = AI_VISIBILITY_DIR / f"ai_visibility_{slug}_{timestamp}.json"
    md_path = AI_VISIBILITY_DIR / f"ai_visibility_{slug}_{timestamp}.md"
    json_path.write_text(json.dumps(bundle, ensure_ascii=False, indent=2), encoding="utf-8")
    md_path.write_text(build_ai_visibility_markdown(bundle), encoding="utf-8")
    return [json_path, md_path]


def latest_saved_ai_visibility_runs(limit: int = 20) -> list[Path]:
    if not AI_VISIBILITY_DIR.exists():
        return []
    files = list(AI_VISIBILITY_DIR.glob("ai_visibility_*.json"))
    return sorted(files, key=lambda path: path.stat().st_mtime, reverse=True)[:limit]


def load_saved_ai_visibility_run(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def summarize_saved_ai_visibility_run(path: Path) -> dict[str, Any]:
    try:
        data = load_saved_ai_visibility_run(path)
    except Exception as exc:
        return {"path": path, "label": f"{path.name} · unreadable", "error": str(exc)}

    discovery_inputs = data.get("discovery_inputs") or {}
    model_result = data.get("model_result") or []
    domain = discovery_inputs.get("domain") or (data.get("model_inputs") or {}).get("domain") or "n/a"
    brand = discovery_inputs.get("brand") or (data.get("model_inputs") or {}).get("brand") or "n/a"
    queries = len((data.get("discovery_result") or {}).get("items", []) or [])
    responses = len(model_result)
    modified = datetime.fromtimestamp(path.stat().st_mtime).strftime("%Y-%m-%d %H:%M")
    label = f"{brand} · {domain} · discovery {queries} · responses {responses} · {modified}"
    return {
        "path": path,
        "data": data,
        "domain": domain,
        "brand": brand,
        "queries": queries,
        "responses": responses,
        "modified": modified,
        "label": label,
    }


def delete_saved_ai_visibility_run(path: Path) -> list[Path]:
    deleted_paths: list[Path] = []
    if not path.exists():
        return deleted_paths

    related_paths = [path]
    name = path.name
    if name.startswith("ai_visibility_") and name.endswith(".json"):
        stem = name[:-len(".json")]
        related_paths.append(path.with_name(f"{stem}.md"))

    seen: set[Path] = set()
    for candidate in related_paths:
        if candidate in seen:
            continue
        seen.add(candidate)
        if candidate.exists():
            candidate.unlink()
            deleted_paths.append(candidate)
    return deleted_paths


def latest_saved_optimizer_analyses(limit: int = 20) -> list[Path]:
    if not ANALIZE_DIR.exists():
        return []
    files = list(ANALIZE_DIR.rglob("analiza_*.json"))
    return sorted(files, key=lambda path: path.stat().st_mtime, reverse=True)[:limit]


def load_saved_optimizer_analysis(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def summarize_saved_optimizer_analysis(path: Path) -> dict[str, Any]:
    try:
        data = load_saved_optimizer_analysis(path)
    except Exception as exc:
        return {
            "path": path,
            "label": f"{path.name} · unreadable",
            "error": str(exc),
        }

    keyword = data.get("primary_keyword", "unknown keyword")
    own_page = data.get("own_page") or {}
    content_score = data.get("content_score") or {}
    score = content_score.get("overall")
    competitor_count = len(data.get("competitors") or [])
    word_count = own_page.get("word_count", 0)
    page_type = data.get("page_type") or own_page.get("page_type") or "unknown"
    modified = datetime.fromtimestamp(path.stat().st_mtime).strftime("%Y-%m-%d %H:%M")
    score_label = f"{score}/100" if isinstance(score, (int, float)) else "n/a"
    label = (
        f"{keyword} · {score_label} · {competitor_count} competitors · "
        f"{word_count:,} words · {modified}"
    )
    return {
        "path": path,
        "data": data,
        "keyword": keyword,
        "score": score,
        "competitor_count": competitor_count,
        "word_count": word_count,
        "page_type": page_type,
        "modified": modified,
        "label": label,
    }


def delete_saved_optimizer_analysis(path: Path) -> list[Path]:
    deleted_paths: list[Path] = []
    if not path.exists():
        return deleted_paths

    name = path.name
    suffix = ""
    if name.startswith("analiza_") and name.endswith(".json"):
        suffix = name[len("analiza_"):-len(".json")]

    related_paths = [path]
    if suffix:
        related_paths.extend([
            path.with_name(f"plan_izboljsav_{suffix}.md"),
            path.with_name(f"content_brief_2_0_{suffix}.md"),
        ])

    seen: set[Path] = set()
    for candidate in related_paths:
        if candidate in seen:
            continue
        seen.add(candidate)
        if candidate.exists():
            candidate.unlink()
            deleted_paths.append(candidate)
    return deleted_paths


@st.cache_data(show_spinner=False)
def analyze_entities_only(text: str) -> list[dict]:
    if not text.strip():
        return []
    client = get_client()
    document = {
        "content": text[:100_000],
        "type_": language_v1.Document.Type.PLAIN_TEXT,
        "language": "en",
    }
    features = language_v1.AnnotateTextRequest.Features(extract_entities=True)
    resp = client.annotate_text(
        request={
            "document": document,
            "features": features,
            "encoding_type": language_v1.EncodingType.UTF8,
        }
    )
    return sorted([{
        "name": e.name,
        "type": language_v1.Entity.Type(e.type_).name,
        "salience": round(e.salience, 4),
        "mentions": len(e.mentions),
        "wikipedia": e.metadata.get("wikipedia_url", ""),
    } for e in resp.entities if e.name and e.name.strip()],
        key=lambda item: item["salience"],
        reverse=True,
    )


@st.cache_data(show_spinner=False)
def analyze_sentiment_readability(text: str, content_language: str = "English") -> dict:
    if not text.strip():
        return {"sentiment": {}, "readability": {}}

    client = get_client()
    document = {
        "content": text[:100_000],
        "type_": language_v1.Document.Type.PLAIN_TEXT,
        "language": "en",
    }
    features = language_v1.AnnotateTextRequest.Features(
        extract_document_sentiment=True,
        extract_syntax=True,
    )
    resp = client.annotate_text(
        request={
            "document": document,
            "features": features,
            "encoding_type": language_v1.EncodingType.UTF8,
        }
    )

    sentences = [{
        "text": s.text.content,
        "score": round(s.sentiment.score, 3),
        "magnitude": round(s.sentiment.magnitude, 3),
    } for s in resp.sentences]
    negative_sentences = [item for item in sentences if item["score"] < -0.15]
    sentence_count = len(sentences)
    syntax = _parse_syntax_tokens(resp.tokens)

    return {
        "sentiment": {
            "score": round(resp.document_sentiment.score, 3),
            "magnitude": round(resp.document_sentiment.magnitude, 3),
            "sentence_count": sentence_count,
            "negative_count": len(negative_sentences),
            "negative_pct": round(len(negative_sentences) / sentence_count * 100, 1)
            if sentence_count else 0,
            "negative_sentences": negative_sentences[:8],
        },
        "readability": {
            "lexical_density": syntax.get("lexical_density", 0),
            "passive_voice_pct": syntax.get("passive_voice_pct", 0),
            "total_tokens": syntax.get("total_tokens", 0),
            "source_note": (
                "Google syntax/POS is approximate for Slovenian, Italian, and Croatian."
                if content_language in ("Slovenščina", "Italiano 🇮🇹", "Hrvatski 🇭🇷")
                else "Google NLP syntax metrics."
            ),
        },
    }


class _ImageAltExtractor(HTMLParser):
    def __init__(self):
        super().__init__()
        self.image_count = 0
        self.images_with_alt = 0

    def handle_starttag(self, tag, attrs):
        if tag.lower() != "img":
            return
        self.image_count += 1
        attr_map = {name.lower(): value for name, value in attrs if name}
        alt = (attr_map.get("alt") or "").strip()
        if alt:
            self.images_with_alt += 1


@st.cache_data(ttl=86400, show_spinner=False)
def fetch_image_metrics(url: str) -> dict:
    req = Request(url, headers={"User-Agent": "Mozilla/5.0 (SEO-NLP-Analyzer)"})
    with urlopen(req, timeout=15) as r:
        html = r.read().decode("utf-8", errors="replace")

    parser = _ImageAltExtractor()
    parser.feed(html)
    image_count = parser.image_count
    images_with_alt = parser.images_with_alt
    missing_alt = max(0, image_count - images_with_alt)
    return {
        "image_count": image_count,
        "images_with_alt": images_with_alt,
        "missing_alt": missing_alt,
        "alt_coverage": round(images_with_alt / image_count, 3) if image_count else 0,
    }


@st.cache_data(ttl=86400, show_spinner=False)
def fetch_url_metadata(url: str) -> dict:
    req = Request(url, headers={"User-Agent": "Mozilla/5.0 (SEO-NLP-Analyzer)"})
    with urlopen(req, timeout=15) as r:
        html = r.read().decode("utf-8", errors="replace")

    page = extract_headings_from_html(html)
    image_count = getattr(page, "image_count", 0)
    images_with_alt = getattr(page, "images_with_alt", 0)
    missing_alt = max(0, image_count - images_with_alt)
    return {
        "title": page.title,
        "meta_description": page.meta_description,
        "canonical": page.canonical,
        "h1": page.h1,
        "h2": page.h2,
        "h3": page.h3,
        "h4": page.h4,
        "h5": page.h5,
        "h6": page.h6,
        "images": {
            "image_count": image_count,
            "images_with_alt": images_with_alt,
            "missing_alt": missing_alt,
            "alt_coverage": round(images_with_alt / image_count, 3) if image_count else 0,
        },
        "raw_html": html,
    }


def build_page_text(text: str, raw_source: str = "", **kwargs) -> PageText:
    """Create PageText in a way that stays compatible with older deploys.

    Some environments may still load a PageText class version that does not
    accept `raw_source` in __init__. In that case we attach it after init.
    """
    page = PageText(text=text, **kwargs)
    try:
        setattr(page, "raw_source", raw_source or text)
    except Exception:
        pass
    return page


def render_content_optimizer_result(result: dict):
    score = result["content_score"]
    word_bench = result["word_count_benchmark"]

    st.divider()
    st.subheader("Content Score")
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Score", f"{score['score']}/{score['max']}")
    c2.metric("Your words", f"{word_bench['your_word_count']:,}")
    c3.metric("Competitor median", f"{word_bench['competitor_stats']['median']:,.0f}")
    c4.metric("Competitor avg", f"{word_bench['competitor_stats']['avg']:,.0f}")

    exact_body = result.get("exact_match_body_edge", {})
    placement_zones = result.get("placement_zones", {})
    if exact_body.get("available"):
        st.markdown("**Keyword Edge Snapshot**")
        k1, k2, k3, k4, k5 = st.columns(5)
        k1.metric("Body exact count", exact_body.get("your_exact_count", 0))
        k2.metric("Competitor avg", exact_body.get("competitor_avg", 0))
        k3.metric("Standard deviation", exact_body.get("competitor_std_dev", 0))
        k4.metric("Edge target", exact_body.get("edge_target_count", 0))
        k5.metric("Over-opt threshold", exact_body.get("over_opt_threshold_count", 0))
        st.caption(
            f"Parity range: {exact_body.get('parity_min', 0)}-{exact_body.get('parity_max', 0)} · "
            f"Gap to edge: {exact_body.get('gap_to_edge', 0)}"
        )
    else:
        st.info("Keyword Edge Snapshot ni na voljo v tem rezultatu. Naredi novo optimizer analizo po zadnjem update-u.")

    if placement_zones.get("available"):
        visible_zones = [
            item for item in placement_zones.get("rows", [])
            if item["zone"] in {"Body content", "H1", "H2", "Lists", "Strong/Bold", "Image alt"}
        ]
        if visible_zones:
            st.markdown("**Placement Zone Snapshot**")
            st.dataframe(pd.DataFrame([{
                "Zone": item["zone"],
                "Your exact": item["your_exact_count"],
                "Your close": item["your_close_count"],
                "Competitor exact avg": item["competitor_exact_avg"],
                "Used by competitors": f"{item['used_by_competitors']}/{item['competitor_total']}",
            } for item in visible_zones]), use_container_width=True, hide_index=True)

    grouped_tunings = result.get("grouped_tunings", [])
    if grouped_tunings:
        st.subheader("Grouped Tunings")
        top_cols = st.columns(len(grouped_tunings))
        for col, section in zip(top_cols, grouped_tunings):
            col.metric(
                section["label"],
                f"{section['score']}/{section['max']}",
                delta=section["status"].title(),
            )
        tabs = st.tabs([section["label"] for section in grouped_tunings])
        for tab, section in zip(tabs, grouped_tunings):
            with tab:
                st.caption(f"Status: {section['status'].title()} · Score {section['score']}/{section['max']}")
                for item in section.get("highlights", []):
                    st.markdown(f"- {item}")

    roadmap = result.get("roadmap_report", {})
    fixes = result.get("top_fixes", [])
    suggestions = result.get("auto_optimize_suggestions", [])
    if roadmap or fixes or suggestions:
        st.subheader("Action Panel")
        action_tabs = st.tabs(["Roadmap", "Priority Fixes", "Auto-Optimize"])
        with action_tabs[0]:
            if roadmap:
                roadmap_sections = [
                    ("Top 5 Easy Wins", "easy_wins"),
                    ("Top 5 Medium Wins", "medium_wins"),
                    ("Top 3 Hard Wins", "hard_wins"),
                ]
                for label, key in roadmap_sections:
                    items = roadmap.get(key, [])
                    if not items:
                        continue
                    st.markdown(f"**{label}**")
                    st.dataframe(pd.DataFrame([{
                        "Action": item["title"],
                        "Where": item.get("where", ""),
                        "Current": item["current"],
                        "Target": item["target"],
                        "Gap": item["gap"],
                        "Impact": item["impact"].title(),
                        "Effort": item["effort"].title(),
                    } for item in items]), use_container_width=True, hide_index=True)
            else:
                st.info("No roadmap items detected.")
        with action_tabs[1]:
            if fixes:
                for i, fix in enumerate(fixes, 1):
                    st.markdown(f"{i}. **{fix['type'].replace('_', ' ').title()}** — {fix['message']}")
            else:
                st.success("No priority fixes detected.")
        with action_tabs[2]:
            if suggestions:
                st.caption("Surgical suggestions: small edits, new sentences, heading tweaks, and missing context.")
                st.dataframe(pd.DataFrame([{
                    "Priority": item["priority"],
                    "Type": item["type"].replace("_", " ").title(),
                    "Target": item["target"],
                    "Action": item["action"],
                    "Example": item["example"],
                    "Reason": item["reason"],
                } for item in suggestions]), use_container_width=True, hide_index=True)
            else:
                st.info("No auto-optimize suggestions available.")

    exact_body = result.get("exact_match_body_edge", {})
    if exact_body.get("available"):
        with st.expander("Exact Match Body Edge", expanded=False):
            e1, e2, e3, e4 = st.columns(4)
            e1.metric("Your exact count", exact_body.get("your_exact_count", 0))
            e2.metric("Competitor avg", exact_body.get("competitor_avg", 0))
            e3.metric("Std dev", exact_body.get("competitor_std_dev", 0))
            e4.metric("Edge target", exact_body.get("edge_target_count", 0))
            e5, e6, e7, e8 = st.columns(4)
            e5.metric("Parity range", f"{exact_body.get('parity_min', 0)}-{exact_body.get('parity_max', 0)}")
            e6.metric("Over-opt threshold", exact_body.get("over_opt_threshold_count", 0))
            e7.metric("Gap to parity", exact_body.get("gap_to_parity", 0))
            e8.metric("Gap to edge", exact_body.get("gap_to_edge", 0))
            status_map = {
                "below_parity": "Below parity",
                "within_parity": "Within parity",
                "competitive_edge": "Competitive edge",
                "possible_over_optimization": "Possible over-optimization",
            }
            st.caption(
                f"Status: {status_map.get(exact_body.get('status', ''), exact_body.get('status', ''))}. "
                "Formula: competitor average + 1.5 x standard deviation = edge target. "
                "Average + 2.0 x standard deviation = over-optimization warning."
            )
            competitor_rows = exact_body.get("competitor_rows", [])
            if competitor_rows:
                st.dataframe(pd.DataFrame([{
                    "Competitor": item["competitor"],
                    "Exact body count": item["exact_count"],
                } for item in competitor_rows]), use_container_width=True, hide_index=True)

    placement_zones = result.get("placement_zones", {})
    if placement_zones.get("available"):
        with st.expander("Placement Zones", expanded=False):
            st.caption(
                "Shows where the primary keyword appears across body content and supporting on-page zones."
            )
            st.dataframe(pd.DataFrame([{
                "Zone": item["zone"],
                "Your exact": item["your_exact_count"],
                "Your close": item["your_close_count"],
                "Competitor exact median": item["competitor_exact_median"],
                "Competitor exact avg": item["competitor_exact_avg"],
                "Competitor exact max": item["competitor_exact_max"],
                "Competitor close median": item["competitor_close_median"],
                "Competitor close avg": item["competitor_close_avg"],
                "Used by competitors": f"{item['used_by_competitors']}/{item['competitor_total']}",
            } for item in placement_zones.get("rows", [])]), use_container_width=True, hide_index=True)

    density_report = result.get("keyword_density_report", {})
    if density_report.get("available"):
        with st.expander("Raw vs Clean Keyword Density", expanded=False):
            st.caption(
                f"Raw word count: {density_report.get('your_raw_word_count', 0):,} vs "
                f"clean word count: {density_report.get('your_clean_word_count', 0):,}. "
                "Raw includes the broader page source; clean focuses on extracted content."
            )
            density_rows = density_report.get("rows", [])
            primary_density_row = next((item for item in density_rows if item.get("type") == "primary"), None)
            if primary_density_row:
                p1, p2, p3, p4 = st.columns(4)
                p1.metric("Primary clean %", primary_density_row["your_clean_density"])
                p2.metric("Competitor clean median %", primary_density_row["competitor_clean_density_stats"]["median"])
                p3.metric("Competitor clean avg %", primary_density_row["competitor_clean_density_stats"]["avg"])
                p4.metric(
                    "Vs clean median",
                    round(
                        primary_density_row["your_clean_density"]
                        - primary_density_row["competitor_clean_density_stats"]["median"],
                        3,
                    ),
                )
            density_df = pd.DataFrame([{
                "Term": item["term"],
                "Type": item["type"],
                "Your raw %": item["your_raw_density"],
                "Your clean %": item["your_clean_density"],
                "Raw-clean diff": item["your_density_gap"],
                "Competitor raw median %": item["competitor_raw_density_stats"]["median"],
                "Competitor raw avg %": item["competitor_raw_density_stats"]["avg"],
                "Competitor clean median %": item["competitor_clean_density_stats"]["median"],
                "Competitor clean avg %": item["competitor_clean_density_stats"]["avg"],
                "Vs clean median": round(item["your_clean_density"] - item["competitor_clean_density_stats"]["median"], 3),
                "Vs clean avg": round(item["your_clean_density"] - item["competitor_clean_density_stats"]["avg"], 3),
                "Your raw count": item["your_raw_count"],
                "Your clean count": item["your_clean_count"],
            } for item in density_rows])
            if not density_df.empty:
                density_df["_primary_sort"] = density_df["Type"].apply(lambda value: 0 if value == "primary" else 1)
                density_df = density_df.sort_values(
                    by=["_primary_sort", "Vs clean median", "Competitor clean avg %"],
                    ascending=[True, True, False],
                ).drop(columns=["_primary_sort"])
            st.dataframe(density_df, use_container_width=True, hide_index=True)
            if density_rows:
                density_options = {
                    f"{item['term']} ({item['type']})": item for item in density_rows
                }
                default_density_index = 0
                for index, item in enumerate(density_rows):
                    if item.get("type") == "primary":
                        default_density_index = index
                        break
                selected_density_key = st.selectbox(
                    "Show exact competitor breakdown for term",
                    list(density_options.keys()),
                    index=default_density_index,
                    key="content_optimizer_density_term",
                )
                selected_density = density_options[selected_density_key]
                competitor_rows = selected_density.get("competitor_rows", [])
                if competitor_rows:
                    competitor_rows = sorted(
                        competitor_rows,
                        key=lambda item: (item.get("clean_density", 0), item.get("clean_count", 0)),
                        reverse=True,
                    )
                    st.caption(
                        "Below are the actual competitors and their counts for the selected term, "
                        "so you can see who is pulling the average/median up."
                    )
                    st.dataframe(pd.DataFrame([{
                        "Competitor": item["competitor"],
                        "URL": item.get("url", "") or item.get("competitor", ""),
                        "Raw count": item["raw_count"],
                        "Clean count": item["clean_count"],
                        "Raw %": item["raw_density"],
                        "Clean %": item["clean_density"],
                        "Vs your clean %": (
                            f"{round(item['clean_density'] - selected_density['your_clean_density'], 3):+}"
                        ),
                        "Vs your clean count": (
                            f"{item['clean_count'] - selected_density['your_clean_count']:+}"
                        ),
                        "Raw words": item["raw_word_count"],
                        "Clean words": item["clean_word_count"],
                    } for item in competitor_rows]), use_container_width=True, hide_index=True)
                    best_competitor = max(competitor_rows, key=lambda item: item.get("clean_density", 0))
                    st.info(
                        f"For `{selected_density['term']}`, your clean density is "
                        f"{selected_density['your_clean_density']}. Best competitor clean density is "
                        f"{best_competitor.get('clean_density', 0)} on `{best_competitor.get('url', best_competitor.get('competitor', 'competitor'))}`."
                    )

    component_rows = []
    for name, data in score["components"].items():
        component_rows.append({
            "component": name.replace("_", " ").title(),
            "score": data["score"],
            "max": data["max"],
            "deductions": " | ".join(data["deductions"][:3]),
        })
    with st.expander("Component Score Breakdown", expanded=False):
        st.dataframe(pd.DataFrame(component_rows), use_container_width=True, hide_index=True)

    intent_data = result.get("intent_classifier", {})
    if intent_data:
        my_intent = intent_data.get("my_page", {})
        with st.expander("Page Type / Intent Classifier", expanded=bool(intent_data.get("mismatch"))):
            i1, i2, i3 = st.columns(3)
            i1.metric("Detected type", my_intent.get("detected_page_type", ""))
            i2.metric("Confidence", f"{my_intent.get('confidence', 0):.0%}")
            i3.metric("Effective type", intent_data.get("effective_page_type", ""))
            if intent_data.get("mismatch"):
                st.warning(
                    "Selected page type differs from detected intent. "
                    "If recommendations feel off, rerun with Auto-detect or adjust competitor selection."
                )
            signals = my_intent.get("signals", [])
            if signals:
                st.caption("Signals: " + " | ".join(signals[:4]))
            intent_counts = intent_data.get("competitor_intent_counts", {})
            if intent_counts:
                st.dataframe(pd.DataFrame([{
                    "Competitor page type": key,
                    "Count": value,
                } for key, value in sorted(intent_counts.items(), key=lambda item: item[1], reverse=True)]),
                    use_container_width=True,
                    hide_index=True,
                )
            competitors = intent_data.get("competitors", [])
            if competitors:
                st.markdown("**Competitor fit helper**")
                st.caption("This does not auto-exclude anything; use it as a quick intent sanity check.")
                st.dataframe(pd.DataFrame([{
                    "Fit": item.get("fit", ""),
                    "URL": item.get("url", "") or f"Competitor {index}",
                    "Detected type": item.get("detected_page_type", ""),
                    "Confidence": f"{item.get('confidence', 0):.0%}",
                    "Reason": item.get("reason", ""),
                } for index, item in enumerate(competitors, 1)]),
                    use_container_width=True,
                    hide_index=True,
                )

    meta_data = result.get("meta_optimizer", {})
    if meta_data and meta_data.get("available"):
        with st.expander("Metadata Deep Dive", expanded=False):
            m1, m2, m3 = st.columns(3)
            m1.metric("Meta score", f"{meta_data.get('score', 0)}/{meta_data.get('max', 10)}")
            m2.metric("Title length", f"{meta_data.get('title_length', 0)} chars")
            m3.metric("Description length", f"{meta_data.get('meta_description_length', 0)} chars")

            checks = meta_data.get("checks", [])
            if checks:
                st.dataframe(pd.DataFrame([{
                    "Check": item["check"].replace("_", " ").title(),
                    "Status": "OK" if item["ok"] else "Fix",
                    "Message": item["message"],
                } for item in checks]), use_container_width=True, hide_index=True)

            title_suggestions = meta_data.get("title_suggestions", [])
            description_suggestions = meta_data.get("meta_description_suggestions", [])
            if title_suggestions or description_suggestions:
                c_title, c_desc = st.columns(2)
                if title_suggestions:
                    c_title.markdown("**SEO title suggestions**")
                    for item in title_suggestions[:3]:
                        c_title.markdown(f"- {item}")
                if description_suggestions:
                    c_desc.markdown("**Meta description suggestions**")
                    for item in description_suggestions[:3]:
                        c_desc.markdown(f"- {item}")

            competitor_titles = meta_data.get("competitor_titles", [])
            if competitor_titles:
                st.dataframe(pd.DataFrame([{"Title": item, "Length": len(item)} for item in competitor_titles]),
                             use_container_width=True,
                             hide_index=True)

    auto_terms = result.get("auto_lsi_terms", [])
    if auto_terms:
        with st.expander("Auto-detected LSI Terms", expanded=False):
            st.caption(
                "Extracted from competitor texts and added to the term-gap calculation. "
                f"Limit: {result.get('auto_lsi_limit', len(auto_terms))} terms."
            )
            auto_df = pd.DataFrame([{
                "Term": item["term"],
                "Words": item["words"],
                "Total count": item["total_count"],
                "Used by competitors": item["competitor_presence"],
            } for item in auto_terms])
            st.dataframe(auto_df, use_container_width=True, hide_index=True)

    entity_data = result.get("entity_gap", {})
    if entity_data and entity_data.get("available"):
        with st.expander("Entity Deep Dive", expanded=False):
            e1, e2, e3 = st.columns(3)
            e1.metric("Entity score", f"{entity_data['score']}/20")
            e2.metric("Missing entities", len(entity_data.get("missing_entities", [])))
            e3.metric("Underweighted", len(entity_data.get("underweighted_entities", [])))

            missing_entities = entity_data.get("missing_entities", [])
            if missing_entities:
                st.markdown("**Missing competitor entities**")
                st.dataframe(pd.DataFrame([{
                    "Entity": item["name"],
                    "Type": item["type"],
                    "Competitor avg salience": f"{item['avg_salience'] * 100:.2f}%",
                    "Your salience": f"{item['your_salience'] * 100:.2f}%",
                    "Used by": f"{item['present_in']}/{item['competitor_total']}",
                } for item in missing_entities]), use_container_width=True, hide_index=True)

            underweighted = entity_data.get("underweighted_entities", [])
            if underweighted:
                st.markdown("**Underweighted entities**")
                st.dataframe(pd.DataFrame([{
                    "Entity": item["name"],
                    "Type": item["type"],
                    "Competitor avg salience": f"{item['avg_salience'] * 100:.2f}%",
                    "Your salience": f"{item['your_salience'] * 100:.2f}%",
                    "Used by": f"{item['present_in']}/{item['competitor_total']}",
                } for item in underweighted]), use_container_width=True, hide_index=True)

            st.dataframe(pd.DataFrame([{
                "Entity": item["name"],
                "Type": item["type"],
                "Avg salience": f"{item['avg_salience'] * 100:.2f}%",
                "Used by": f"{item['present_in']}/{item['competitor_total']}",
            } for item in entity_data.get("top_competitor_entities", [])]),
                use_container_width=True,
                hide_index=True,
            )

    sentiment_data = result.get("sentiment_readability", {})
    if sentiment_data and sentiment_data.get("available"):
        metrics = sentiment_data.get("metrics", {})
        with st.expander("Sentiment & Readability Deep Dive", expanded=False):
            s1, s2, s3, s4 = st.columns(4)
            s1.metric("Score", f"{sentiment_data['score']}/7")
            s2.metric(
                "Sentiment",
                f"{metrics.get('your_sentiment_score', 0):+.3f}",
                delta=f"avg {metrics.get('competitor_sentiment_score_avg', 0):+.3f}",
            )
            s3.metric(
                "Negative sentences",
                f"{metrics.get('your_negative_pct', 0):.1f}%",
                delta=f"avg {metrics.get('competitor_negative_pct_avg', 0):.1f}%",
                delta_color="inverse",
            )
            s4.metric(
                "Lexical density",
                f"{metrics.get('your_lexical_density', 0):.1%}",
                delta=f"avg {metrics.get('competitor_lexical_density_avg', 0):.1%}",
            )

            negatives = metrics.get("negative_sentences", [])
            if negatives:
                st.dataframe(pd.DataFrame([{
                    "Sentence": item["text"],
                    "Score": item["score"],
                    "Magnitude": item["magnitude"],
                } for item in negatives]), use_container_width=True, hide_index=True)

    image_data = result.get("images_alt", {})
    if image_data and image_data.get("available"):
        metrics = image_data.get("metrics", {})
        with st.expander("Images & Alt Coverage Deep Dive", expanded=False):
            i1, i2, i3, i4 = st.columns(4)
            i1.metric("Score", f"{image_data['score']}/3")
            i2.metric("Images", f"{metrics.get('your_image_count', 0):.0f}",
                      delta=f"median {metrics.get('competitor_image_count_median', 0):.0f}")
            i3.metric("With alt", f"{metrics.get('your_images_with_alt', 0):.0f}")
            i4.metric(
                "Alt coverage",
                f"{metrics.get('your_alt_coverage', 0):.0%}",
                delta=f"avg {metrics.get('competitor_alt_coverage_avg', 0):.0%}",
            )

    term_opportunities = result.get("term_opportunities", {})
    if term_opportunities:
        st.subheader("Term Opportunities")
        section_map = [
            ("Missing high-opportunity terms", "missing_high_opportunity"),
            ("Underused terms", "underused_terms"),
            ("Overused terms", "overused_terms"),
        ]
        for label, key in section_map:
            items = term_opportunities.get(key, [])
            if not items:
                continue
            st.markdown(f"**{label}**")
            st.dataframe(pd.DataFrame([{
                "Term": item["term"],
                "Type": item["type"],
                "Your count": item["your_count"],
                "Target": f"{item['recommended_min']}-{item['recommended_max']}",
                "Gap": item.get("gap", item.get("excess", 0)),
                "Competitor median": item["competitor_median"],
                "Competitor avg": item["competitor_avg"],
                "Used by": f"{item['used_by_competitors']}/{item['competitor_total']}",
            } for item in items]), use_container_width=True, hide_index=True)

    variations_matrix = result.get("keyword_variations_matrix", {})
    if variations_matrix.get("available"):
        st.subheader("Keyword Variations Matrix")
        groups = variations_matrix.get("groups", {})
        if groups:
            st.caption("Groups: " + " | ".join(f"{group}: {count}" for group, count in groups.items()))
        variation_rows = variations_matrix.get("rows", [])
        st.dataframe(pd.DataFrame([{
            "Variation": item["term"],
            "Group": item["group"],
            "Your count": item["your_count"],
            "Your partial": item["your_partial_count"],
            "Parity range": f"{item['parity_min']}-{item['parity_max']}",
            "Competitor avg": item["competitor_avg"],
            "Competitor median": item["competitor_median"],
            "Competitor max": item["competitor_max"],
            "Used by": f"{item['used_by_competitors']}/{item['competitor_total']}",
            "Status": item["status"],
        } for item in variation_rows]), use_container_width=True, hide_index=True)
        if variation_rows:
            variation_options = {f"{item['term']} [{item['group']}]": item for item in variation_rows}
            selected_variation_key = st.selectbox(
                "Inspect competitor usage for variation",
                options=list(variation_options.keys()),
                key="content_optimizer_variation_select",
            )
            selected_variation = variation_options[selected_variation_key]
            st.dataframe(pd.DataFrame([{
                "Competitor": item["competitor"],
                "Count": item["count"],
            } for item in selected_variation.get("competitor_rows", [])]), use_container_width=True, hide_index=True)

    match_words = result.get("match_words", [])
    if match_words:
        st.subheader("Match Words / Query Variants")
        st.caption("Competitor-derived phrase variants around the primary keyword. Missing items are the best candidates to test first.")
        st.dataframe(pd.DataFrame([{
            "Variant": item["term"],
            "Status": item["status"],
            "Your exact": item["your_exact_count"],
            "Your partial": item["your_partial_count"],
            "Used by": f"{item['competitor_presence']}/{item['competitor_total']}",
            "Primary token hits": item["primary_token_hits"],
            "Total competitor uses": item["total_count"],
        } for item in match_words]), use_container_width=True, hide_index=True)

    with st.expander("Full Term Gap Table", expanded=False):
        st.subheader("Term Gap")
        rows = []
        for row in result["term_gap"]:
            rows.append({
                "Action": row["action"],
                "Term": row["term"],
                "Type": row["type"],
                "Your count": row["your_count"],
                "Your density": f"{row['your_density']:.2f}%",
                "Competitor avg": row["competitor_count_stats"]["avg"],
                "Competitor median": row["competitor_count_stats"]["median"],
                "Range": f"{row['recommended_min']}-{row['recommended_max']}",
                "Add": row["add_count"],
                "Used by": f"{row['used_by_competitors']}/{row['competitor_total']}",
            })

        if rows:
            df_terms = pd.DataFrame(rows)
            action_order = {"add section": 0, "add": 1, "reduce": 2, "keep": 3, "ignore": 4}
            df_terms["_sort"] = df_terms["Action"].map(action_order).fillna(9)
            df_terms = df_terms.sort_values(["_sort", "Type", "Term"]).drop(columns=["_sort"])
            st.dataframe(df_terms, use_container_width=True, hide_index=True)
        else:
            st.info("No terms were supplied.")

    with st.expander("Headings & Placement Deep Dive", expanded=False):
        st.subheader("Keyword Placement")
        placement = score["placement"]
        p1, p2, p3, p4, p5, p6 = st.columns(6)
        p1.metric("Title", "OK" if placement["in_title"] else "Missing")
        p2.metric("H1", "OK" if placement["in_h1"] else "Missing")
        p3.metric("H2", "OK" if placement["in_h2"] else "Missing")
        p4.metric("First 100 words", "OK" if placement["in_first_100_words"] else "Missing")
        p5.metric("H1 count", placement["h1_count"])
        p6.metric("H2 count", placement["h2_count"])
        placement_df = pd.DataFrame([
            {"Check": "Title", "Status": "OK" if placement["in_title"] else "Missing"},
            {"Check": "H1", "Status": "OK" if placement["in_h1"] else "Missing"},
            {"Check": "H2", "Status": "OK" if placement["in_h2"] else "Missing"},
            {"Check": "H1 (close variation)", "Status": "OK" if placement.get("in_h1_close") else "Missing"},
            {"Check": "H2 (close variation)", "Status": "OK" if placement.get("in_h2_close") else "Missing"},
            {"Check": "First 100 words", "Status": "OK" if placement["in_first_100_words"] else "Missing"},
            {"Check": "H1 count", "Status": str(placement["h1_count"])},
            {"Check": "H2 count", "Status": str(placement["h2_count"])},
            {"Check": "H3 count", "Status": str(placement["h3_count"])},
            {"Check": "H4 count", "Status": str(placement["h4_count"])},
            {"Check": "H5 count", "Status": str(placement["h5_count"])},
            {"Check": "H6 count", "Status": str(placement["h6_count"])},
        ])
        st.dataframe(placement_df, use_container_width=True, hide_index=True)

        heading_data = result.get("heading_optimizer", {})
        if heading_data:
            st.subheader("Heading Optimizer")
            bench = heading_data.get("benchmark", {})
            your_counts = bench.get("your_counts", {})
            comp_stats = bench.get("competitor_stats", {})
            st.markdown("**Heading structure targets**")
            heading_rows = []
            guidance_map = {item["level"]: item for item in heading_data.get("level_guidance", [])}
            for level in ["h1", "h2", "h3", "h4", "h5", "h6"]:
                stats_for_level = comp_stats.get(level, {})
                guidance = guidance_map.get(level, {})
                your_count = your_counts.get(level, 0)
                median_count = stats_for_level.get("median", 0)
                avg_count = stats_for_level.get("avg", 0)
                heading_rows.append({
                    "Level": level.upper(),
                    "Your count": your_count,
                    "Competitor median": median_count,
                    "Competitor avg": avg_count,
                    "Competitor range": f"{stats_for_level.get('min', 0)}-{stats_for_level.get('max', 0)}",
                    "Vs median": f"{int(your_count - median_count):+d}" if isinstance(median_count, (int, float)) else "0",
                    "Vs avg": f"{(your_count - avg_count):+.1f}" if isinstance(avg_count, (int, float)) else "0.0",
                    "Status": guidance.get("status", "ok").title(),
                    "Action hint": guidance.get("message", "Within range."),
                })
            st.dataframe(pd.DataFrame(heading_rows), use_container_width=True, hide_index=True)

            heading_terms = heading_data.get("heading_terms", {})
            summary = heading_terms.get("summary", {})
            all_levels = heading_terms.get("all_levels", {})
            if all_levels:
                st.markdown("**All-heading opportunity snapshot**")
                a1, a2, a3, a4, a5 = st.columns(5)
                a1.metric("All headings", all_levels.get("count", 0))
                a2.metric("Primary exact", all_levels.get("primary_exact", 0))
                a3.metric("Primary close", all_levels.get("primary_close", 0))
                a4.metric("Lead primary", all_levels.get("lead_primary_close", 0))
                a5.metric("Questions", all_levels.get("questions", 0))
                st.markdown("**All-heading primary coverage**")
                st.dataframe(pd.DataFrame([{
                    "All headings": all_levels.get("count", 0),
                    "Primary exact": all_levels.get("primary_exact", 0),
                    "Primary close": all_levels.get("primary_close", 0),
                    "Lead primary": all_levels.get("lead_primary_close", 0),
                    "Questions": all_levels.get("questions", 0),
                }]), use_container_width=True, hide_index=True)
            if summary:
                st.markdown("**Heading keyword coverage**")
                coverage_rows = []
                for level in ["h1", "h2", "h3", "h4", "h5", "h6"]:
                    level_data = summary.get(level, {})
                    count = level_data.get("count", 0)
                    coverage_rows.append({
                        "Level": level.upper(),
                        "Headings": count,
                        "Primary exact": level_data.get("primary_exact", 0),
                        "Primary close": level_data.get("primary_close", 0),
                        "Lead primary": level_data.get("lead_primary_close", 0),
                        "Lead secondary": level_data.get("lead_secondary", 0),
                        "Lead LSI": level_data.get("lead_lsi", 0),
                        "Lead entity": level_data.get("lead_entity", 0),
                        "Questions": level_data.get("questions", 0),
                        "Sentence-like": level_data.get("sentence_like", 0),
                        "Primary coverage %": (
                            f"{(level_data.get('primary_close', 0) / count) * 100:.0f}%"
                            if count else "0%"
                        ),
                        "Lead coverage %": (
                            f"{(level_data.get('lead_primary_close', 0) / count) * 100:.0f}%"
                            if count else "0%"
                        ),
                    })
                st.dataframe(pd.DataFrame(coverage_rows), use_container_width=True, hide_index=True)

            details = heading_terms.get("details", [])
            if details:
                st.markdown("**Heading-by-heading analysis**")
                st.dataframe(pd.DataFrame([{
                    "Level": item["level"],
                    "Heading": item["heading"],
                    "Primary exact": "Yes" if item["primary_exact"] else "No",
                    "Primary close": "Yes" if item["primary_close"] else "No",
                    "Lead keyword": item["lead_term"] or "—",
                    "Lead type": item["lead_type"],
                    "Lead strength": item["lead_strength"],
                    "Shape": item["shape"],
                } for item in details]), use_container_width=True, hide_index=True)

                weak_headings = [
                    item for item in details
                    if not item.get("primary_close") and not item.get("lead_term")
                ]
                if weak_headings:
                    st.markdown("**Headings with weak keyword signal**")
                    st.dataframe(pd.DataFrame([{
                        "Level": item["level"],
                        "Heading": item["heading"],
                        "Issue": "No primary close variation and no lead keyword",
                    } for item in weak_headings[:12]]), use_container_width=True, hide_index=True)

            recs = heading_data.get("recommendations", [])
            if recs:
                st.markdown("**Heading recommendations**")
                for item in recs[:10]:
                    st.markdown(f"- {item['message']}")

            missing_h2 = heading_data.get("missing_h2_terms", [])
            if missing_h2:
                st.markdown("**Important terms missing from H2**")
                st.dataframe(pd.DataFrame([{
                    "Term": item["term"],
                    "Type": item["type"],
                    "Used by": f"{item['used_by_competitors']}/{item['competitor_total']}",
                } for item in missing_h2]), use_container_width=True, hide_index=True)

            common_h2 = heading_data.get("common_h2_topics", [])
            if common_h2:
                st.markdown("**Common competitor H2 topics**")
                st.dataframe(pd.DataFrame([{
                    "Topic": item["topic"],
                    "Occurrences": item["count"],
                    "Present in competitors": item["present_in"],
                } for item in common_h2]), use_container_width=True, hide_index=True)

        heading_terms_matrix = result.get("heading_terms_matrix", {})
        if heading_terms_matrix.get("available"):
            st.subheader("Competitor Heading Terms Matrix")
            groups = heading_terms_matrix.get("groups", {})
            if groups:
                st.caption("Groups: " + " | ".join(f"{group}: {count}" for group, count in groups.items()))
            matrix_rows = heading_terms_matrix.get("rows", [])
            st.dataframe(pd.DataFrame([{
                "Term": item["term"],
                "Type": item["type"].replace("_", " ").title(),
                "Group": item["group"],
                "Your heading count": item["your_heading_count"],
                "Competitor avg": item["competitor_avg_heading_count"],
                "Competitor median": item["competitor_median_heading_count"],
                "Competitor max": item["competitor_max_heading_count"],
                "Used by": f"{item['used_by_competitors']}/{item['competitor_total']}",
                "Your H1": item["your_h1"],
                "Your H2": item["your_h2"],
                "Your H3-H6": item["your_h3_h6"],
                "Status": item["status"],
            } for item in matrix_rows]), use_container_width=True, hide_index=True)
            if matrix_rows:
                matrix_options = {
                    f"{item['term']} [{item['type']}]": item for item in matrix_rows
                }
                selected_heading_term_key = st.selectbox(
                    "Inspect competitor heading usage for term",
                    options=list(matrix_options.keys()),
                    key="content_optimizer_heading_term_select",
                )
                selected_heading_term = matrix_options[selected_heading_term_key]
                st.dataframe(pd.DataFrame([{
                    "Competitor": item["competitor"],
                    "Total": item["total"],
                    "H1": item["h1"],
                    "H2": item["h2"],
                    "H3": item["h3"],
                    "H4": item["h4"],
                    "H5": item["h5"],
                    "H6": item["h6"],
                } for item in selected_heading_term.get("competitor_rows", [])]), use_container_width=True, hide_index=True)

            common_h2 = heading_data.get("common_h2_topics", [])
            if common_h2:
                st.markdown("**Common competitor H2 topics**")
                st.dataframe(pd.DataFrame([{
                    "Topic": item["topic"],
                    "Occurrences": item["count"],
                    "Present in competitors": item["present_in"],
                } for item in common_h2]), use_container_width=True, hide_index=True)

    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    slug = result["primary_keyword"].replace(" ", "_").lower() or "content_optimizer"
    st.download_button(
        "Download JSON report",
        data=json.dumps(result, ensure_ascii=False, indent=2),
        file_name=f"content_optimizer_{slug}_{ts}.json",
        mime="application/json",
        use_container_width=True,
        key="dl_content_optimizer_json",
    )
    st.download_button(
        "Download improvement plan (.md)",
        data=build_improvement_plan_markdown(result),
        file_name=f"plan_izboljsav_{slug}_{ts}.md",
        mime="text/markdown",
        use_container_width=True,
        key="dl_content_optimizer_plan_md",
    )
    st.download_button(
        "Download Content Brief 2.0 (.md)",
        data=build_content_brief_markdown(result),
        file_name=f"content_brief_2_0_{slug}_{ts}.md",
        mime="text/markdown",
        use_container_width=True,
        key="dl_content_optimizer_brief_md",
    )

    st.divider()
    if st.button("Save audit to analize/", type="primary", use_container_width=True):
        try:
            saved_paths = save_content_optimizer_audit(result)
            st.success("Audit saved.")
            for path in saved_paths:
                st.markdown(f"- [{path.name}]({path})")
        except Exception as e:
            st.error(f"Could not save audit: {e}")

    recent_audits = latest_saved_audits()
    if recent_audits:
        with st.expander("Recent saved audit files"):
            for path in recent_audits:
                st.markdown(f"- [{path.name}]({path})")


# ── Main navigation ───────────────────────────────────────────────────────────

nav_pages = ["🔍 Analyzer", "📝 Content Brief", "🎯 Content Optimizer", "🤖 AI Visibility", "ℹ️ Info"]
if "page" not in st.session_state:
    st.session_state["page"] = "🔍 Analyzer"

page = st.sidebar.radio(
    "Navigation",
    nav_pages,
    index=nav_pages.index(st.session_state["page"])
    if st.session_state["page"] in nav_pages else 0,
    label_visibility="collapsed",
)

st.markdown("## 🔍 SEO NLP Analyzer")
st.caption("Powered by Google Cloud Natural Language API + Claude AI")
st.markdown("")

# Top navigation as buttons
col_nav1, col_nav2, col_nav3, col_nav4, col_nav5, col_nav_spacer = st.columns([1, 1, 1.2, 1.2, 1, 5.2])
if col_nav1.button("🔍 Analyzer", use_container_width=True,
                    type="primary" if page == "🔍 Analyzer" else "secondary"):
    st.session_state["page"] = "🔍 Analyzer"
    st.rerun()
if col_nav2.button("📝 Content Brief", use_container_width=True,
                    type="primary" if page == "📝 Content Brief" else "secondary"):
    st.session_state["page"] = "📝 Content Brief"
    st.rerun()
if col_nav3.button("🎯 Optimizer", use_container_width=True,
                    type="primary" if page == "🎯 Content Optimizer" else "secondary"):
    st.session_state["page"] = "🎯 Content Optimizer"
    st.rerun()
if col_nav4.button("🤖 AI Visibility", use_container_width=True,
                    type="primary" if page == "🤖 AI Visibility" else "secondary"):
    st.session_state["page"] = "🤖 AI Visibility"
    st.rerun()
if col_nav5.button("ℹ️ Info", use_container_width=True,
                    type="primary" if page == "ℹ️ Info" else "secondary"):
    st.session_state["page"] = "ℹ️ Info"
    st.rerun()

if page != st.session_state["page"]:
    st.session_state["page"] = page

current_page = st.session_state.get("page", "🔍 Analyzer")

st.markdown("---")

# ── Analyzer page ─────────────────────────────────────────────────────────────

if current_page == "🔍 Analyzer":
    st.title("🔍 SEO NLP Analyzer")
    st.caption("Powered by Google Cloud Natural Language API")

    input_mode = st.radio(
        "Input type",
        ["🌐 URL", "📋 Paste text", "🖥️ Paste HTML source"],
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
            html_source = ""
        elif input_mode == "📋 Paste text":
            keyword  = st.text_input("Target keyword (optional)",
                                      placeholder="e.g. bazeni")
            raw_text = st.text_area(
                "Paste your article or page content here",
                placeholder="Paste the full text of your article, blog post, or page...",
                height=250,
            )
            url1 = url2 = ""
            html_source = ""
        else:  # HTML source
            keyword = st.text_input("Target keyword (optional)",
                                     placeholder="e.g. bazeni")
            html_source = st.text_area(
                "Paste HTML source code (Ctrl+U → copy all)",
                placeholder="<!DOCTYPE html><html>...",
                height=250,
            )
            st.caption("💡 V brskalniku: Ctrl+U (ali desni klik → View Page Source) → Ctrl+A → Ctrl+C → prilepi sem")
            raw_text = ""
            url1 = url2 = ""

        # Language selector
        cl1, cl2 = st.columns([1, 3])
        content_language = cl1.radio(
            "Content language",
            ["English", "Slovenščina", "Italiano 🇮🇹", "Hrvatski 🇭🇷"],
            horizontal=True,
            help="Slovenščina/Italiano/Hrvatski: verb/adjective counts use Claude AI. English: uses Google NLP API.",
        )
        if content_language in ("Slovenščina", "Italiano 🇮🇹", "Hrvatski 🇭🇷"):
            lang_labels = {"Slovenščina": "Slovenščina", "Italiano 🇮🇹": "Italiano", "Hrvatski 🇭🇷": "Hrvatski"}
            lang_label = lang_labels.get(content_language, content_language)
            cl2.info(f"🤖 {lang_label} mode: Google API za entitete/sentiment/kategorije · Claude AI za glagole/pridevnike/pasivni glas")

        submitted = st.form_submit_button("Analyze", type="primary",
                                           use_container_width=True)

    if submitted:
        # Clear previous reports when new analysis starts
        st.session_state["ai_report"] = ""
        st.session_state["ai_report_main"] = ""

        if input_mode == "🌐 URL":
            if not url1:
                st.error("Please enter at least one URL.")
                st.stop()

            spinner_msg = f"Fetching and analyzing {url1} {'+ Claude linguistic analysis' if content_language in ('Slovenščina', 'Italiano 🇮🇹') else ''} ..."
            with st.spinner(spinner_msg):
                text1 = fetch_url_text(url1, fresh=True)  # always fresh — your own page
                if text1:
                    if len(text1) > 100_000:
                        text1 = text1[:100_000]
                    st.session_state["results"] = {"url1": run_analysis(text1, content_language)}
                    st.session_state["url1_label"] = url1
                    st.session_state["keyword"] = keyword
                    st.session_state["my_text"] = text1

            if url2:
                with st.spinner(f"Fetching and analyzing {url2} ..."):
                    text2 = fetch_url_text(url2, fresh=True)  # fresh for direct competitor comparison
                    if text2:
                        if len(text2) > 100_000:
                            text2 = text2[:100_000]
                        st.session_state["results"]["url2"] = run_analysis(text2, content_language)
                        st.session_state["url2_label"] = url2
        elif input_mode == "📋 Paste text":
            if not raw_text.strip():
                st.error("Please paste some text to analyze.")
                st.stop()
            text1 = raw_text.strip()
            if len(text1) > 100_000:
                text1 = text1[:100_000]
            spinner_msg = "Analyzing text + Claude linguistic analysis ..." if content_language in ("Slovenščina", "Italiano 🇮🇹", "Hrvatski 🇭🇷") else "Analyzing text ..."
            with st.spinner(spinner_msg):
                st.session_state["results"] = {"url1": run_analysis(text1, content_language)}
                st.session_state["url1_label"] = "pasted text"
                st.session_state["keyword"] = keyword
                st.session_state["my_text"] = text1

        else:  # HTML source
            if not html_source.strip():
                st.error("Please paste HTML source code.")
                st.stop()
            with st.spinner("Extracting main content from HTML + Claude cleaning..."):
                text1 = extract_text_from_html(html_source)
            if not text1 or not text1.strip():
                st.error("No content extracted from HTML. Check that you pasted valid HTML.")
                st.stop()
            if len(text1) > 100_000:
                text1 = text1[:100_000]
            st.info(f"✓ Extracted {len(text1.split())} words from HTML source")
            spinner_msg = "Analyzing + Claude linguistic analysis ..." if content_language in ("Slovenščina", "Italiano 🇮🇹", "Hrvatski 🇭🇷") else "Analyzing ..."
            with st.spinner(spinner_msg):
                st.session_state["results"] = {"url1": run_analysis(text1, content_language)}
                st.session_state["url1_label"] = "HTML source"
                st.session_state["keyword"] = keyword
                st.session_state["my_text"] = text1

    # Always render results from session_state (persists across re-renders)
    if "results" in st.session_state and st.session_state["results"]:
        results   = st.session_state["results"]
        keyword   = st.session_state.get("keyword", "")
        url1      = st.session_state.get("url1_label", "")
        url2      = st.session_state.get("url2_label", "")
        benchmark = st.session_state.get("benchmark", {})

        if not results:
            st.error("No results — check that the URLs are publicly accessible.")
            st.stop()

        if "url2" in results:
            st.divider()
            col_a, col_b = st.columns(2)
            with col_a:
                st.markdown(f"#### Your page\n`{url1[:60]}`")
                render_analysis(results["url1"], keyword, source=url1, benchmark=benchmark, key_prefix="main")
            with col_b:
                st.markdown(f"#### Competitor\n`{url2[:60]}`")
                render_analysis(results["url2"], keyword, source=url2, benchmark=benchmark, key_prefix="url2")

            # ── Combined download ─────────────────────────────────────────────
            st.divider()
            ts   = datetime.now().strftime("%Y%m%d_%H%M%S")
            slug = keyword.replace(" ", "_").lower() if keyword else "primerjava"
            md1  = build_markdown_report(results["url1"], keyword, url1,
                                          st.session_state.get("ai_report_main", ""), benchmark)
            md2  = build_markdown_report(results["url2"], keyword, url2,
                                          st.session_state.get("ai_report_url2", ""), None)
            combined = (
                f"# SEO Primerjava: Tvoja stran vs Konkurent\n"
                f"**Keyword:** {keyword or '—'} · **Datum:** {datetime.now().strftime('%Y-%m-%d %H:%M')}\n\n"
                f"---\n\n"
                f"# TVOJA STRAN: {url1[:80]}\n\n"
                f"{md1}\n\n"
                f"---\n\n"
                f"# KONKURENT: {url2[:80]}\n\n"
                f"{md2}"
            )
            st.download_button(
                label="📥 Download — Skupna analiza (tvoja stran + konkurent)",
                data=combined,
                file_name=f"primerjava_{slug}_{ts}.md",
                mime="text/markdown",
                use_container_width=True,
                key="dl_combined",
                help="Ena datoteka z obema analizama — idealno za Claude Code primerjavo",
            )
            st.caption("💡 Prilepi v Claude Code → 'primerjaj obe strani in povej kaj moram popraviti'")
        else:
            st.divider()
            render_analysis(results["url1"], keyword, source=url1, benchmark=benchmark, key_prefix="main")

        # ── Competitor Benchmark section ──────────────────────────────────────
        st.divider()
        st.subheader("🏆 Competitor Benchmark")
        st.caption("Analyze top-ranking competitors to get real benchmarks for your niche.")

        dfseo_login, _ = get_dfseo_auth()
        fc_available    = get_firecrawl() is not None

        if not fc_available:
            st.warning("⚠ Add FIRECRAWL_API_KEY to Streamlit Secrets for better scraping.")

        # ── Show cached benchmark status ──────────────────────────────────────
        if st.session_state.get("benchmark") and st.session_state.get("bench_saved_urls"):
            saved_urls = st.session_state["bench_saved_urls"]
            bench_date = st.session_state.get("bench_date", "")
            st.success(
                f"✅ Benchmark active — {len(saved_urls)} competitor pages cached · {bench_date} · "
                f"Open **🏆 Benchmark** tab to see results."
            )
            col_use, col_refresh = st.columns([3, 1])
            col_use.caption("Benchmark is reused automatically when you re-analyze your text.")
            do_refresh = col_refresh.button("🔄 Refresh competitors", key="refresh_bench_btn",
                                             help="Force re-scrape all competitor pages")
            if do_refresh:
                st.session_state["benchmark"] = {}
                st.session_state["bench_saved_urls"] = []
                st.session_state["benchmark_status"] = ""
                fetch_competitor_text_cached.clear()
                st.rerun()
            st.divider()

        # Mode selector — OUTSIDE form so it re-renders immediately
        if dfseo_login:
            bench_mode = st.radio(
                "How to find competitors",
                ["🔍 Auto — top Google results (DataForSEO)",
                 "✏️ Manual — enter URLs"],
                horizontal=True, key="bench_mode"
            )
        else:
            bench_mode = "✏️ Manual — enter URLs"
            st.info("Add DATAFORSEO_LOGIN + DATAFORSEO_PASSWORD to Secrets for auto competitor detection.")

        if "Manual" in bench_mode:
            # Pre-populate with previously saved URLs
            saved_default = "\n".join(st.session_state.get("bench_saved_urls", []))
            comp_urls_raw = st.text_area(
                "Competitor URLs (one per line)",
                value=saved_default,
                placeholder="https://competitor1.com/page\nhttps://competitor2.com/page\nhttps://competitor3.com/page",
                height=120,
                key="bench_urls",
            )
        else:
            comp_urls_raw = ""
            st.caption(f"Will fetch top Google results for: **{keyword or '(enter keyword above first)'}**")

        col_bl, col_bn = st.columns(2)
        bench_lang = col_bl.radio("Language", ["English", "Slovenščina", "Italiano 🇮🇹", "Hrvatski 🇭🇷"],
                                  horizontal=True, key="bench_lang")
        bench_n    = col_bn.selectbox("How many competitors", [3, 5, 10], index=1,
                                       key="bench_n")

        run_bench = st.button(
            "🏆 Analyze Competitors & Build Benchmark",
            type="primary", use_container_width=True,
            key="run_bench_btn"
        )

        if run_bench:
            serp_data = {}

            # Get competitor URLs
            if "Auto" in bench_mode and keyword:
                with st.spinner(f"Fetching top {bench_n} Google results for '{keyword}'..."):
                    serp_data = dfseo_serp(keyword)
                    comp_urls = [r["url"] for r in serp_data.get("organic", [])[:bench_n]
                                 if r["url"]]
                if comp_urls:
                    st.info(f"Found {len(comp_urls)} competitor pages from Google SERP")
                else:
                    st.error("No SERP results — check DataForSEO credentials.")
                    st.stop()
            else:
                comp_urls = [u.strip() for u in comp_urls_raw.strip().splitlines()
                             if u.strip().startswith("http")]

            if not comp_urls:
                st.error("No competitor URLs found. Enter URLs manually or add a keyword.")
            else:
                bench_results = []
                serp_wc      = {r["url"]: r.get("word_count", 0)
                                for r in serp_data.get("organic", [])}
                urls_to_scrape = comp_urls[:bench_n]

                debug_log = []
                progress  = st.progress(0, text="Starting competitor analysis...")
                status    = st.empty()

                for i, curl in enumerate(urls_to_scrape):
                    pct = i / len(urls_to_scrape)
                    progress.progress(pct, text=f"Page {i+1}/{len(urls_to_scrape)}: {curl[:60]}")
                    status.caption("Scraping...")
                    try:
                        # Use cached version — won't re-scrape if already done today
                        ct = fetch_competitor_text_cached(curl)
                        if ct:
                            wc = len(ct.split())
                            debug_log.append(f"✓ Scraped {curl[:60]} — {wc} words")
                            status.caption(f"Analyzing {curl[:60]}...")
                            if len(ct) > 100_000:
                                ct = ct[:100_000]
                            res = run_analysis(ct, bench_lang)
                            res["word_count"] = wc if wc > 0 else serp_wc.get(curl, 0)
                            # Fetch heading structure via DataForSEO On-Page
                            if get_dfseo_auth()[0]:
                                status.caption(f"Getting headings for {curl[:50]}...")
                                heading_data = dfseo_onpage_headings(curl)
                                res["headings"] = heading_data
                                if heading_data:
                                    debug_log.append(f"  ↳ H2: {heading_data.get('h2_count', 0)} | H3: {heading_data.get('h3_count', 0)}")
                                else:
                                    debug_log.append(f"  ↳ ⚠ Headings timeout — On-Page API ni vrnil podatkov")
                            bench_results.append(res)
                            debug_log.append(
                                f"✓ Analyzed — sentiment {res['sentiment']['score']:+.2f} · "
                                f"H2: {res.get('headings', {}).get('h2_count', '?')}"
                            )
                        else:
                            debug_log.append(f"⚠ Empty response: {curl[:60]}")
                    except Exception as e:
                        debug_log.append(f"❌ Error on {curl[:60]}: {e}")

                progress.progress(1.0, text="Done!")
                status.empty()
                st.session_state["bench_debug"] = debug_log

                if bench_results:
                    bm = compute_benchmark(bench_results, keyword)
                    bm["paa"]           = serp_data.get("paa", [])
                    my_text             = st.session_state.get("my_text", "")
                    bm["my_text"]       = my_text
                    bm["my_text_lower"] = my_text.lower()
                    st.session_state["benchmark"]              = bm
                    st.session_state["bench_saved_urls"]       = urls_to_scrape
                    st.session_state["bench_date"]             = datetime.now().strftime("%d.%m.%Y %H:%M")
                    st.session_state["benchmark_status"]       = f"✅ Benchmark built from {len(bench_results)} competitor pages."
                    # Save individual competitor results for detailed view
                    st.session_state["bench_individual"]       = list(zip(urls_to_scrape, bench_results))
                    st.rerun()
                else:
                    st.session_state["benchmark_status"] = "❌ Could not analyze any competitor pages. Check that URLs are publicly accessible."

        # Always show benchmark status persistently
        if st.session_state.get("benchmark_status"):
            status_msg = st.session_state["benchmark_status"]
            if status_msg.startswith("✅"):
                st.success(status_msg + " → Open the **🏆 Benchmark** tab above.")
            else:
                st.error(status_msg)

        # Debug log — shows what happened during last benchmark run
        if st.session_state.get("bench_debug"):
            with st.expander("🔧 Debug log (last benchmark run)", expanded=True):
                for line in st.session_state["bench_debug"]:
                    st.text(line)

# ── Content Brief page ────────────────────────────────────────────────────────

elif current_page == "📝 Content Brief":
    st.title("📝 Content Brief Generator")
    st.caption("Analiziraj konkurente brez lastnega besedila → Claude napiše točen brief kako pisati")

    # Mode selector
    cb_mode_top = st.radio(
        "Kaj hočeš narediti?",
        ["✏️ Pišem novo besedilo od začetka",
         "🔧 Izboljšujem obstoječe besedilo"],
        horizontal=True, key="cb_mode_top"
    )

    if cb_mode_top == "✏️ Pišem novo besedilo od začetka":
        st.info(
            "**Kako deluje:** Vneseš keyword + competitor URLs → app analizira vsako stran → "
            "izračuna povprečja → Claude napiše content brief (kako pisati novo vsebino)"
        )
    else:
        st.info(
            "**Kako deluje:** Vneseš keyword + competitor URLs + svoje besedilo → "
            "Claude primerja z benchmarkom → napiše točno kaj dodati, spremeniti, odstraniti"
        )

    cb_dfseo_login, _ = get_dfseo_auth()

    # ── Inputs ────────────────────────────────────────────────────────────────
    cb_keyword = st.text_input("Target keyword", placeholder="e.g. bazeni, vibratorji...",
                                key="cb_keyword")

    if cb_dfseo_login:
        cb_mode = st.radio(
            "Kako najti konkurente",
            ["🔍 Auto — top Google rezultati (DataForSEO)", "✏️ Ročno — vnesi URLs"],
            horizontal=True, key="cb_mode"
        )
    else:
        cb_mode = "✏️ Ročno — vnesi URLs"

    if "Ročno" in cb_mode:
        saved = "\n".join(st.session_state.get("bench_saved_urls", []))
        cb_urls_raw = st.text_area(
            "Competitor URLs (ena na vrstico)",
            value=saved,
            placeholder="https://competitor1.com/page\nhttps://competitor2.com/page",
            height=150, key="cb_urls"
        )
    else:
        cb_urls_raw = ""
        st.caption(f"Poiskal bom top Google rezultate za: **{cb_keyword or '(vnesi keyword zgoraj)'}**")

    # Own text — only shown in "improve" mode
    cb_own_text = ""
    cb_nw_json  = ""
    if "Izboljšujem" in cb_mode_top:
        cb_own_text = st.text_area(
            "📄 Tvoje obstoječe besedilo",
            placeholder="Prilepi sem besedilo ki ga hočeš izboljšati...",
            height=200, key="cb_own_text"
        )
        cb_nw_json = st.text_area(
            "🔧 NeuroWriter JSON (opcijsko — za NW term coverage)",
            value=st.session_state.get("nw_raw", ""),
            placeholder='{"basic_terms":[...],"extended_terms":[...]}',
            height=80, key="cb_nw_json_imp"
        )

    col_cl, col_cn = st.columns(2)
    cb_lang = col_cl.radio("Jezik vsebine", ["English", "Slovenščina", "Italiano 🇮🇹", "Hrvatski 🇭🇷"],
                            horizontal=True, key="cb_lang")
    cb_n    = col_cn.selectbox("Število konkurentov", [3, 5, 10], index=1, key="cb_n")

    btn_label = ("🔧 Analiziraj & napiši plan izboljšav"
                 if "Izboljšujem" in cb_mode_top
                 else "🚀 Analiziraj konkurente & generiraj brief")
    run_brief = st.button(btn_label, type="primary",
                          use_container_width=True, key="run_brief_btn")

    if run_brief:
        if not cb_keyword:
            st.error("Vnesi keyword.")
            st.stop()

        serp_data = {}
        if "Auto" in cb_mode and cb_keyword:
            with st.spinner(f"Iščem top {cb_n} Google rezultatov za '{cb_keyword}'..."):
                serp_data = dfseo_serp(cb_keyword)
                cb_urls = [r["url"] for r in serp_data.get("organic", [])[:cb_n] if r["url"]]
            if not cb_urls:
                st.error("Ni SERP rezultatov — preveri DataForSEO credentials.")
                st.stop()
        else:
            cb_urls = [u.strip() for u in cb_urls_raw.strip().splitlines()
                       if u.strip().startswith("http")]

        if not cb_urls:
            st.error("Vnesi vsaj 1 competitor URL.")
            st.stop()

        cb_results  = []
        cb_debug    = []
        serp_wc     = {r["url"]: r.get("word_count", 0) for r in serp_data.get("organic", [])}
        progress    = st.progress(0, text="Začenjam analizo konkurentov...")
        status_ph   = st.empty()

        for i, curl in enumerate(cb_urls[:cb_n]):
            progress.progress(i / len(cb_urls),
                               text=f"Stran {i+1}/{len(cb_urls)}: {curl[:60]}")
            status_ph.caption("Scrapin...")
            try:
                ct = fetch_competitor_text_cached(curl)
                if ct:
                    wc = len(ct.split())
                    cb_debug.append(f"✓ {curl[:60]} — {wc} besed")
                    status_ph.caption(f"Analiziram {curl[:50]}...")
                    if len(ct) > 100_000:
                        ct = ct[:100_000]
                    res = run_analysis(ct, cb_lang if cb_lang == "Slovenščina" else "English")
                    res["word_count"] = wc
                    if cb_dfseo_login:
                        status_ph.caption(f"Pridobivam naslove za {curl[:50]}...")
                        res["headings"] = dfseo_onpage_headings(curl)
                    cb_results.append(res)
                    cb_debug.append(f"✓ Analizirano — sentiment {res['sentiment']['score']:+.2f}")
                else:
                    cb_debug.append(f"⚠ Prazno: {curl[:60]}")
            except Exception as e:
                cb_debug.append(f"❌ Napaka: {curl[:60]}: {e}")

        progress.progress(1.0, text="Analiza končana!")
        status_ph.empty()

        # Save own text for NW score display later
        if cb_own_text.strip():
            st.session_state["cb_own_text_val"] = cb_own_text

        if cb_results:
            bm = compute_benchmark(cb_results, cb_keyword)
            bm["paa"] = serp_data.get("paa", [])
            st.session_state["cb_benchmark"]   = bm
            st.session_state["cb_keyword_val"] = cb_keyword
            st.session_state["cb_lang_val"]    = cb_lang
            st.session_state["bench_saved_urls"] = cb_urls[:cb_n]

            with st.expander("🔧 Debug log", expanded=False):
                for line in cb_debug:
                    st.text(line)

            is_improve = "Izboljšujem" in cb_mode_top
            spinner_msg = ("Claude analizira tvoje besedilo in piše plan izboljšav..."
                           if is_improve else "Claude generira content brief...")
            with st.spinner(spinner_msg):
                try:
                    if is_improve and cb_own_text.strip():
                        nw_parsed = parse_nw_json(cb_nw_json) if cb_nw_json.strip() else {}
                        if nw_parsed:
                            st.session_state["nw_raw"]    = cb_nw_json
                            st.session_state["nw_parsed"] = nw_parsed
                        brief = generate_improvement_plan(
                            bm, cb_keyword, cb_own_text, cb_lang, nw_parsed
                        )
                    else:
                        brief = generate_content_brief(bm, cb_keyword, cb_lang)
                    st.session_state["cb_brief"]      = brief
                    st.session_state["cb_brief_mode"] = "improve" if is_improve else "new"
                except Exception as e:
                    st.error(f"Napaka: {e}")
                    st.session_state["cb_brief"] = ""

    # Show brief
    if st.session_state.get("cb_brief"):
        brief     = st.session_state["cb_brief"]
        kw        = st.session_state.get("cb_keyword_val", "content")
        mode_flag = st.session_state.get("cb_brief_mode", "new")
        st.markdown("---")
        if mode_flag == "improve":
            st.subheader("🔧 Plan izboljšav")
        else:
            st.subheader("📝 Content Brief")
        st.markdown(brief)
        ts    = datetime.now().strftime("%Y%m%d_%H%M%S")
        prefix = "plan_izboljsav" if mode_flag == "improve" else "content_brief"
        fname  = f"{prefix}_{kw.replace(' ', '_')}_{ts}.md"
        st.download_button(
            label="📥 Download",
            data=brief,
            file_name=fname,
            mime="text/markdown",
            use_container_width=True,
            key="dl_brief_btn"
        )

        # NW Score — samo po "izboljšujem" ko imamo besedilo in NW JSON
        if mode_flag == "improve":
            own_text  = st.session_state.get("cb_own_text_val", "")
            nw_parsed = st.session_state.get("nw_parsed", {})
            if own_text and nw_parsed:
                st.divider()
                st.subheader("🎯 NW Score tvojega besedila")
                st.caption("Coverage tvojega obstoječega besedila glede na NeuroWriter priporočila")
                tab_nw_score(own_text, nw_parsed)

# ── Content Optimizer page ────────────────────────────────────────────────────

elif current_page == "🎯 Content Optimizer":
    st.title("🎯 Content Optimizer")
    st.caption("POP-like term gap + basic Content Score with manual text or URL-based competitor benchmarks.")

    saved_optimizer_analyses = [summarize_saved_optimizer_analysis(path) for path in latest_saved_optimizer_analyses()]
    with st.expander("📚 Saved analyses", expanded=bool(saved_optimizer_analyses)):
        st.caption("Optimizer analize se zdaj lahko ponovno odprejo tudi po restartu appa.")
        if not saved_optimizer_analyses:
            st.info("Še ni shranjenih optimizer analiz.")
        else:
            saved_options = {
                item["label"]: item for item in saved_optimizer_analyses if not item.get("error")
            }
            if saved_options:
                selected_saved_label = st.selectbox(
                    "Open a previous analysis",
                    options=list(saved_options.keys()),
                    key="content_optimizer_saved_analysis_select",
                )
                selected_saved = saved_options[selected_saved_label]
                c_saved_1, c_saved_2, c_saved_3, c_saved_4 = st.columns(4)
                c_saved_1.metric("Keyword", selected_saved["keyword"])
                c_saved_2.metric("Score", selected_saved["score"] if selected_saved["score"] is not None else "n/a")
                c_saved_3.metric("Competitors", selected_saved["competitor_count"])
                c_saved_4.metric("Words", f"{selected_saved['word_count']:,}")
                st.caption(
                    f"Page type: {selected_saved['page_type']} · Saved: {selected_saved['modified']} · "
                    f"File: `{selected_saved['path'].relative_to(BASE_DIR)}`"
                )
                btn_open, btn_delete = st.columns(2)
                if btn_open.button("Open saved analysis", use_container_width=True, key="open_saved_optimizer_analysis"):
                    st.session_state["content_optimizer_result"] = selected_saved["data"]
                    st.session_state["content_optimizer_loaded_path"] = str(selected_saved["path"])
                    st.success("Saved analysis loaded below.")
                if btn_delete.button("Delete saved analysis", use_container_width=True, key="delete_saved_optimizer_analysis"):
                    try:
                        deleted_paths = delete_saved_optimizer_analysis(selected_saved["path"])
                        if str(selected_saved["path"]) == st.session_state.get("content_optimizer_loaded_path"):
                            st.session_state["content_optimizer_loaded_path"] = None
                            st.session_state["content_optimizer_result"] = None
                        if deleted_paths:
                            st.success(
                                "Deleted: " + ", ".join(path.name for path in deleted_paths)
                            )
                        else:
                            st.info("Selected analysis was already missing.")
                        st.rerun()
                    except Exception as e:
                        st.error(f"Could not delete saved analysis: {e}")
            broken_saved = [item for item in saved_optimizer_analyses if item.get("error")]
            if broken_saved:
                st.warning("Some saved analyses could not be read.")
                for item in broken_saved[:5]:
                    st.markdown(f"- `{item['path'].name}` — {item['error']}")

    with st.form("content_optimizer_form"):
        optimizer_input_mode = st.radio(
            "Input mode",
            ["Use own URL + competitor URLs", "Paste own content/HTML + competitor URLs"],
            horizontal=True,
            help="Use your live URL, or paste unpublished content/HTML and compare it against competitor URLs.",
        )
        c1, c2, c3 = st.columns([2, 1, 1])
        opt_keyword = c1.text_input("Primary keyword", placeholder="e.g. montažni bazeni")
        opt_language = c2.radio("Language", ["Slovenščina", "English", "Italiano 🇮🇹", "Hrvatski 🇭🇷"],
                                horizontal=False)
        opt_page_type = c3.selectbox(
            "Page type",
            ["Auto-detect", "blog", "product page", "e-commerce category", "service page",
             "local landing page", "affiliate/review page", "comparison page"],
        )

        c_terms1, c_terms2, c_terms3 = st.columns(3)
        secondary_raw = c_terms1.text_area(
            "Secondary keywords (one per line)",
            placeholder="bestway bazeni\nintex bazeni",
            height=120,
        )
        lsi_raw = c_terms2.text_area(
            "LSI / related terms (one per line)",
            placeholder="filtrirni sistem\njekleni okvir\npostavitev bazena",
            height=120,
        )
        entity_raw = c_terms3.text_area(
            "Entity terms (one per line)",
            placeholder="Bestway\nIntex\nPVC folija",
            height=120,
        )
        auto_lsi = st.checkbox(
            "Auto-detect LSI terms from competitor texts",
            value=True,
            help="Finds repeated 1-5 word phrases used by competitors but missing from your content.",
        )
        lsi_depth = st.radio(
            "Auto LSI depth",
            ["Auto by page type", "Focused", "Balanced", "Deep", "Custom"],
            horizontal=True,
            disabled=not auto_lsi,
            help=(
                "Auto adapts to page type and competitor count. Focused=10, "
                "Balanced=20, Deep=40, Custom lets you choose."
            ),
        )
        custom_lsi_limit = st.slider("Custom Auto LSI term limit", 5, 50, 20, 5,
                                     disabled=not auto_lsi or lsi_depth != "Custom")
        analyze_entities = st.checkbox(
            "Analyze Google NLP entities",
            value=False,
            help="Runs Google NLP entity extraction for your page and selected competitors.",
        )
        analyze_sentiment = st.checkbox(
            "Analyze sentiment & readability",
            value=True,
            help="Adds Google NLP sentiment, negative sentence ratio, and lexical density to Content Score.",
        )
        analyze_images = st.checkbox(
            "Analyze image/alt coverage (URL mode)",
            value=True,
            disabled=False,
            help="Counts images and non-empty alt text from the raw page HTML.",
        )

        with st.expander("Advanced overrides", expanded=False):
            st.caption("Use only if automatic HTML/URL extraction reads title, meta, or headings incorrectly.")
            my_title = st.text_input("Override title (optional)", placeholder="Montažni bazeni za vrt")
            my_meta_description = st.text_area(
                "Override meta description (optional)",
                placeholder="Kratek SEO opis strani...",
                height=70,
            )
            c_head1, c_head2, c_head3 = st.columns(3)
            my_h1_raw = c_head1.text_area("Override H1 headings (one per line)", height=70)
            my_h2_raw = c_head2.text_area("Override H2 headings (one per line)", height=70)
            my_h3_raw = c_head3.text_area("Override H3 headings (one per line)", height=70)
            c_head4, c_head5, c_head6 = st.columns(3)
            my_h4_raw = c_head4.text_area("Override H4 headings (one per line)", height=70)
            my_h5_raw = c_head5.text_area("Override H5 headings (one per line)", height=70)
            my_h6_raw = c_head6.text_area("Override H6 headings (one per line)", height=70)

        my_text = ""
        own_url = ""
        comp_urls_raw = ""
        if optimizer_input_mode == "Paste own content/HTML + competitor URLs":
            my_text = st.text_area(
                "Your content or raw HTML code",
                height=220,
                placeholder=(
                    "Paste your page text here...\n\n"
                    "Or paste raw HTML, e.g. <html><head><title>...</title>..."
                ),
                help=(
                    "Use plain readable content or raw HTML. If you paste raw HTML, "
                    "the optimizer can use HTML title, meta description, and headings."
                ),
            )

            comp_urls_raw = st.text_area(
                "Competitor URLs (one per line)",
                height=180,
                placeholder="https://competitor1.com/page\nhttps://competitor2.com/page\nhttps://competitor3.com/page",
            )
        else:
            own_url = st.text_input("Your page URL", placeholder="https://yoursite.com/page")
            comp_urls_raw = st.text_area(
                "Competitor URLs (one per line)",
                height=180,
                placeholder="https://competitor1.com/page\nhttps://competitor2.com/page\nhttps://competitor3.com/page",
            )

        submit_label = "Fetch URLs"
        run_optimizer = st.form_submit_button(submit_label, type="primary",
                                              use_container_width=True)

    if run_optimizer:
        if not opt_keyword.strip():
            st.error("Enter a primary keyword.")
            st.stop()
        st.session_state["content_optimizer_result"] = None
        st.session_state["content_optimizer_loaded_path"] = None

        if optimizer_input_mode == "Paste own content/HTML + competitor URLs":
            if not my_text.strip():
                st.error("Paste your content or raw HTML.")
                st.stop()
            comp_urls = [url for url in parse_lines(comp_urls_raw) if url.startswith("http")]
            if not comp_urls:
                st.error("Enter at least one competitor URL.")
                st.stop()

            pasted_page = extract_headings_from_html(my_text) if "<" in my_text and ">" in my_text else extract_markdown_headings(my_text)
            own_meta = {
                "title": my_title or pasted_page.title,
                "meta_description": my_meta_description or pasted_page.meta_description,
                "canonical": pasted_page.canonical,
                "raw_html": my_text if "<" in my_text and ">" in my_text else my_text,
                "h1": parse_lines(my_h1_raw) or pasted_page.h1,
                "h2": parse_lines(my_h2_raw) or pasted_page.h2,
                "h3": parse_lines(my_h3_raw) or pasted_page.h3,
                "h4": parse_lines(my_h4_raw) or pasted_page.h4,
                "h5": parse_lines(my_h5_raw) or pasted_page.h5,
                "h6": parse_lines(my_h6_raw) or pasted_page.h6,
                "images": {
                    "image_count": pasted_page.image_count,
                    "images_with_alt": pasted_page.images_with_alt,
                    "missing_alt": max(0, pasted_page.image_count - pasted_page.images_with_alt),
                    "alt_coverage": round(pasted_page.images_with_alt / pasted_page.image_count, 3)
                    if pasted_page.image_count else 0,
                },
            }

            scraped_competitors = []
            progress = st.progress(0, text="[CRAWLING_COMPETITORS...] 0/{0} (0%)".format(len(comp_urls)))
            for i, url in enumerate(comp_urls):
                start_pct = int(i / len(comp_urls) * 100)
                progress.progress(
                    i / len(comp_urls),
                    text=f"[CRAWLING_COMPETITORS...] {i+1}/{len(comp_urls)} ({start_pct}%) {url[:60]}",
                )
                text = fetch_competitor_text_cached(url)
                try:
                    meta = fetch_url_metadata(url) if text else {}
                except Exception as e:
                    meta = {"error": f"metadata failed: {e}"}
                page = extract_markdown_headings(text) if text else PageText(text="")
                scraped_competitors.append({
                    "url": url,
                    "text": text,
                    "word_count": len(text.split()) if text else 0,
                    "title": meta.get("title") or page.title,
                    "meta_description": meta.get("meta_description", ""),
                    "canonical": meta.get("canonical", ""),
                    "h1": meta.get("h1") or page.h1,
                    "h2": meta.get("h2") or page.h2,
                    "h3": meta.get("h3") or page.h3,
                    "h4": meta.get("h4") or page.h4,
                    "h5": meta.get("h5") or page.h5,
                    "h6": meta.get("h6") or page.h6,
                    "images": meta.get("images", {}),
                    "raw_html": meta.get("raw_html", ""),
                    "use": bool(text),
                    "error": "" if text else "empty or inaccessible",
                })
                done_pct = int((i + 1) / len(comp_urls) * 100)
                progress.progress(
                    (i + 1) / len(comp_urls),
                    text=f"[CRAWLING_COMPETITORS...] {i+1}/{len(comp_urls)} ({done_pct}%) done",
                )
            progress.progress(1.0, text="[CRAWLING_COMPETITORS...] done (100%)")

            st.session_state["content_optimizer_url_data"] = {
                "own_url": "pasted content/html",
                "own_text": my_text,
                "own_meta": own_meta,
                "competitors": scraped_competitors,
                "source_mode": "pasted_own_competitor_urls",
            }
            st.success("Competitor URLs fetched. Review competitor selection below, then calculate score.")
        else:
            if not own_url.strip():
                st.error("Enter your page URL.")
                st.stop()
            comp_urls = [url for url in parse_lines(comp_urls_raw) if url.startswith("http")]
            if not comp_urls:
                st.error("Enter at least one competitor URL.")
                st.stop()

            scraped_competitors = []
            page_progress = st.progress(0, text="[CRAWLING_PAGE...] 0/2 (0%)")
            page_progress.progress(0.25, text="[CRAWLING_PAGE...] 1/2 (25%) fetching content")
            own_text = fetch_url_text(own_url.strip(), fresh=True)
            page_progress.progress(0.5, text="[CRAWLING_PAGE...] 1/2 (50%) content fetched")
            try:
                page_progress.progress(0.75, text="[CRAWLING_PAGE...] 2/2 (75%) fetching metadata")
                own_meta = fetch_url_metadata(own_url.strip())
            except Exception as e:
                st.warning(f"Could not fetch your page metadata: {e}")
                own_meta = {}
            page_progress.progress(1.0, text="[CRAWLING_PAGE...] done (100%)")
            if not own_text:
                st.error("Could not fetch your page.")
                st.stop()

            progress = st.progress(0, text="[CRAWLING_COMPETITORS...] 0/{0} (0%)".format(len(comp_urls)))
            for i, url in enumerate(comp_urls):
                start_pct = int(i / len(comp_urls) * 100)
                progress.progress(
                    i / len(comp_urls),
                    text=f"[CRAWLING_COMPETITORS...] {i+1}/{len(comp_urls)} ({start_pct}%) {url[:60]}",
                )
                text = fetch_competitor_text_cached(url)
                try:
                    meta = fetch_url_metadata(url) if text else {}
                except Exception as e:
                    meta = {"error": f"metadata failed: {e}"}
                page = extract_markdown_headings(text) if text else PageText(text="")
                scraped_competitors.append({
                    "url": url,
                    "text": text,
                    "word_count": len(text.split()) if text else 0,
                    "title": meta.get("title") or page.title,
                    "meta_description": meta.get("meta_description", ""),
                    "canonical": meta.get("canonical", ""),
                    "h1": meta.get("h1") or page.h1,
                    "h2": meta.get("h2") or page.h2,
                    "h3": meta.get("h3") or page.h3,
                    "h4": meta.get("h4") or page.h4,
                    "h5": meta.get("h5") or page.h5,
                    "h6": meta.get("h6") or page.h6,
                    "images": meta.get("images", {}),
                    "raw_html": meta.get("raw_html", ""),
                    "use": bool(text),
                    "error": "" if text else "empty or inaccessible",
                })
                done_pct = int((i + 1) / len(comp_urls) * 100)
                progress.progress(
                    (i + 1) / len(comp_urls),
                    text=f"[CRAWLING_COMPETITORS...] {i+1}/{len(comp_urls)} ({done_pct}%) done",
                )
            progress.progress(1.0, text="[CRAWLING_COMPETITORS...] done (100%)")

            st.session_state["content_optimizer_url_data"] = {
                "own_url": own_url.strip(),
                "own_text": own_text,
                "own_meta": own_meta,
                "competitors": scraped_competitors,
            }
            st.success("URLs fetched. Review competitor selection below, then calculate score.")

    url_data = (
        st.session_state.get("content_optimizer_url_data")
        if optimizer_input_mode in ("Use own URL + competitor URLs", "Paste own content/HTML + competitor URLs")
        else None
    )
    if url_data:
        st.divider()
        st.subheader("Competitor Selection")
        source_label = "Your pasted content/HTML" if url_data.get("source_mode") == "pasted_own_competitor_urls" else f"Your page: {url_data['own_url']}"
        st.caption(f"{source_label} · {len(url_data['own_text'].split()):,} words")

        selected_competitors = []
        for i, comp in enumerate(url_data["competitors"]):
            cols = st.columns([0.7, 4, 1, 1, 1, 1, 1, 1, 1])
            default_use = comp.get("use", False)
            use_comp = cols[0].checkbox("Use", value=default_use,
                                        key=f"content_optimizer_use_comp_{i}")
            cols[1].markdown(f"`{comp['url']}`")
            cols[2].metric("Words", f"{comp['word_count']:,}")
            cols[3].metric("H1", len(comp.get("h1", [])))
            cols[4].metric("H2", len(comp.get("h2", [])))
            cols[5].metric("H3", len(comp.get("h3", [])))
            cols[6].metric("H4", len(comp.get("h4", [])))
            cols[7].metric("H5", len(comp.get("h5", [])))
            cols[8].metric("H6", len(comp.get("h6", [])))
            if comp.get("error"):
                st.warning(f"{comp['url']} → {comp['error']}")
            if use_comp and comp.get("text"):
                selected_competitors.append(comp)

        if st.button("Calculate Content Score From Selected URLs", type="primary",
                     use_container_width=True):
            if not opt_keyword.strip():
                st.error("Enter a primary keyword.")
                st.stop()
            if not selected_competitors:
                st.error("Select at least one valid competitor.")
                st.stop()

            auto_lsi_limit = (
                custom_lsi_limit if lsi_depth == "Custom"
                else lsi_limit_for_depth(lsi_depth, opt_page_type, len(selected_competitors))
            )

            auto_my_page = extract_markdown_headings(url_data["own_text"])
            own_meta = url_data.get("own_meta", {})
            my_page = build_page_text(
                url_data["own_text"],
                raw_source=own_meta.get("raw_html", "") or url_data["own_text"],
                title=my_title or own_meta.get("title", "") or auto_my_page.title,
                meta_description=my_meta_description or own_meta.get("meta_description", ""),
                canonical=own_meta.get("canonical", ""),
                h1=parse_lines(my_h1_raw) or own_meta.get("h1") or auto_my_page.h1,
                h2=parse_lines(my_h2_raw) or own_meta.get("h2") or auto_my_page.h2,
                h3=parse_lines(my_h3_raw) or own_meta.get("h3") or auto_my_page.h3,
                h4=parse_lines(my_h4_raw) or own_meta.get("h4") or auto_my_page.h4,
                h5=parse_lines(my_h5_raw) or own_meta.get("h5") or auto_my_page.h5,
                h6=parse_lines(my_h6_raw) or own_meta.get("h6") or auto_my_page.h6,
                url=url_data["own_url"],
            )
            competitors = [
                build_page_text(
                    comp["text"],
                    raw_source=comp.get("raw_html", "") or comp["text"],
                    title=comp.get("title", ""),
                    meta_description=comp.get("meta_description", ""),
                    canonical=comp.get("canonical", ""),
                    h1=comp.get("h1", []),
                    h2=comp.get("h2", []),
                    h3=comp.get("h3", []),
                    h4=comp.get("h4", []),
                    h5=comp.get("h5", []),
                    h6=comp.get("h6", []),
                    url=comp["url"],
                )
                for comp in selected_competitors
            ]
            my_entities = []
            competitor_entities = []
            my_sentiment = {}
            competitor_sentiments = []
            my_readability = {}
            competitor_readability = []
            my_images = {}
            competitor_images = []

            with st.spinner("[CALCULATING_TERM_GAPS...] [BUILDING_CONTENT_SCORE...]"):
                if analyze_entities:
                    st.caption("[ANALYZING_NLP...] Google NLP entities")
                    try:
                        my_entities = analyze_entities_only(url_data["own_text"])
                        competitor_entities = [
                            analyze_entities_only(comp["text"])
                            for comp in selected_competitors
                        ]
                    except Exception as e:
                        st.warning(f"Google NLP entity analysis failed: {e}")
                if analyze_sentiment:
                    st.caption("[ANALYZING_SENTIMENT...] Google NLP sentiment/readability")
                    try:
                        my_nlp = analyze_sentiment_readability(url_data["own_text"], opt_language)
                        comp_nlp = [
                            analyze_sentiment_readability(comp["text"], opt_language)
                            for comp in selected_competitors
                        ]
                        my_sentiment = my_nlp.get("sentiment", {})
                        my_readability = my_nlp.get("readability", {})
                        competitor_sentiments = [item.get("sentiment", {}) for item in comp_nlp]
                        competitor_readability = [item.get("readability", {}) for item in comp_nlp]
                    except Exception as e:
                        st.warning(f"Google NLP sentiment/readability failed: {e}")
                if analyze_images:
                    st.caption("[ANALYZING_IMAGES...] image/alt coverage")
                    try:
                        my_images = own_meta.get("images") or fetch_image_metrics(url_data["own_url"])
                        competitor_images = [
                            comp.get("images") or fetch_image_metrics(comp["url"])
                            for comp in selected_competitors
                        ]
                    except Exception as e:
                        st.warning(f"Image/alt analysis failed: {e}")
                result = optimize_content(OptimizerInput(
                    primary_keyword=opt_keyword,
                    my_page=my_page,
                    competitor_pages=competitors,
                    secondary_keywords=parse_lines(secondary_raw),
                    lsi_keywords=parse_lines(lsi_raw),
                    entity_terms=parse_lines(entity_raw),
                    auto_lsi=auto_lsi,
                    auto_lsi_limit=auto_lsi_limit,
                    my_entities=my_entities,
                    competitor_entities=competitor_entities,
                    my_sentiment=my_sentiment,
                    competitor_sentiments=competitor_sentiments,
                    my_readability=my_readability,
                    competitor_readability=competitor_readability,
                    my_images=my_images,
                    competitor_images=competitor_images,
                    language=opt_language,
                    page_type=opt_page_type,
                ))
            st.session_state["content_optimizer_result"] = result
            try:
                saved_paths = save_content_optimizer_audit(result)
                saved_json = next((path for path in saved_paths if path.suffix == ".json"), None)
                if saved_json:
                    st.session_state["content_optimizer_loaded_path"] = str(saved_json)
                    st.toast(f"Analysis auto-saved: {saved_json.name}", icon="💾")
            except Exception as e:
                st.warning(f"Analysis calculated, but auto-save failed: {e}")

    if st.session_state.get("content_optimizer_result"):
        loaded_path = st.session_state.get("content_optimizer_loaded_path")
        if loaded_path:
            st.info(f"Showing saved analysis: `{Path(loaded_path).name}`")
        render_content_optimizer_result(st.session_state["content_optimizer_result"])

elif current_page == "🤖 AI Visibility":
    st.title("🤖 AI Visibility")
    st.caption("Discover AI queries, mentions, cited sources, and competitor visibility from DataForSEO LLM Mentions.")

    dfseo_login, _ = get_dfseo_auth()
    if not dfseo_login:
        st.warning("Dodaj `DATAFORSEO_LOGIN` in `DATAFORSEO_PASSWORD` v secrets, da lahko uporabljaš AI Visibility.")
    else:
        llm_location_rows = dfseo_llm_mentions_locations()
        recent_ai_runs = [summarize_saved_ai_visibility_run(path) for path in latest_saved_ai_visibility_runs()]
        if recent_ai_runs:
            with st.expander("📚 Saved AI runs"):
                selected_saved_ai = st.selectbox(
                    "Open saved AI run",
                    options=recent_ai_runs,
                    format_func=lambda item: item.get("label", item["path"].name),
                    key="saved_ai_visibility_select",
                )
                btn_ai_open, btn_ai_delete = st.columns(2)
                if btn_ai_open.button("Open saved AI run", use_container_width=True, key="open_saved_ai_visibility_run"):
                    saved_data = selected_saved_ai.get("data") or load_saved_ai_visibility_run(selected_saved_ai["path"])
                    st.session_state["ai_visibility_result"] = saved_data.get("discovery_result", {})
                    st.session_state["ai_visibility_inputs"] = saved_data.get("discovery_inputs", {})
                    st.session_state["ai_model_visibility_result"] = saved_data.get("model_result", [])
                    st.session_state["ai_model_visibility_inputs"] = saved_data.get("model_inputs", {})
                    st.session_state["ai_visibility_loaded_path"] = str(selected_saved_ai["path"])
                    st.rerun()
                if btn_ai_delete.button("Delete saved AI run", use_container_width=True, key="delete_saved_ai_visibility_run"):
                    deleted_paths = delete_saved_ai_visibility_run(selected_saved_ai["path"])
                    loaded_path = st.session_state.get("ai_visibility_loaded_path")
                    if loaded_path == str(selected_saved_ai["path"]):
                        st.session_state.pop("ai_visibility_result", None)
                        st.session_state.pop("ai_visibility_inputs", None)
                        st.session_state.pop("ai_model_visibility_result", None)
                        st.session_state.pop("ai_model_visibility_inputs", None)
                        st.session_state.pop("ai_visibility_loaded_path", None)
                    if deleted_paths:
                        st.success("Saved AI run deleted.")
                    else:
                        st.info("Nothing to delete.")
                    st.rerun()

        with st.form("ai_visibility_form"):
            c_ai_1, c_ai_2, c_ai_3 = st.columns([1.2, 1, 1])
            ai_platform = c_ai_1.selectbox("Platform", ["google", "chat_gpt"], index=0)
            ai_target_mode = c_ai_2.selectbox("Target mode", ["Domain + brand", "Domain only", "Brand only"])
            ai_limit = c_ai_3.slider("Result limit", 5, 50, 20, 5)

            c_ai_4, c_ai_5 = st.columns(2)
            ai_domain = c_ai_4.text_input("Domain (optional)", placeholder="megabazeni.si")
            ai_brand = c_ai_5.text_input("Brand / keyword (optional)", placeholder="Megabazeni")

            platform_locations = []
            for row in llm_location_rows:
                supported_languages = []
                for lang in row.get("available_languages", []) or []:
                    if ai_platform in (lang.get("available_platforms", []) or []):
                        supported_languages.append(lang)
                if supported_languages:
                    platform_locations.append({
                        "location_code": row.get("location_code"),
                        "location_name": row.get("location_name", ""),
                        "available_languages": supported_languages,
                    })

            default_location_name = "Slovenia" if ai_platform == "google" else "United States"
            default_location_idx = next(
                (
                    idx for idx, row in enumerate(platform_locations)
                    if row.get("location_name") == default_location_name
                ),
                0,
            )
            selected_location = None
            ai_location_code = 2840
            ai_language_code = "en"

            c_ai_6, c_ai_7, c_ai_8 = st.columns(3)
            if platform_locations:
                selected_location = c_ai_6.selectbox(
                    "Location",
                    options=platform_locations,
                    index=default_location_idx if default_location_idx < len(platform_locations) else 0,
                    format_func=lambda item: item.get("location_name", f"Location {item.get('location_code', '')}"),
                )
                ai_location_code = int(selected_location.get("location_code", 2840) or 2840)

                available_languages = selected_location.get("available_languages", []) or []
                default_lang_code = "sl" if ai_platform == "google" and selected_location.get("location_name") == "Slovenia" else "en"
                default_lang_idx = next(
                    (
                        idx for idx, row in enumerate(available_languages)
                        if row.get("language_code") == default_lang_code
                    ),
                    0,
                )
                selected_language = c_ai_7.selectbox(
                    "Language",
                    options=available_languages,
                    index=default_lang_idx if default_lang_idx < len(available_languages) else 0,
                    format_func=lambda item: f"{item.get('language_name', item.get('language_code', ''))} ({item.get('language_code', '')})",
                )
                ai_language_code = (selected_language.get("language_code", "en") or "en").strip().lower()
                c_ai_7.caption(f"language_code = {ai_language_code}")
                c_ai_6.caption(f"location_code = {ai_location_code}")
            else:
                ai_location_code = c_ai_6.number_input("Location code", min_value=1, value=2840, step=1)
                ai_language_code = c_ai_7.text_input("Language code", value="en").strip().lower()

            ai_min_volume = c_ai_8.number_input("Min AI search volume", min_value=0, value=0, step=10)

            c_ai_9, c_ai_10 = st.columns(2)
            competitor_domains_raw = c_ai_9.text_area(
                "Competitor domains (one per line)",
                placeholder="competitor1.com\ncompetitor2.com",
                height=100,
            )
            competitor_brands_raw = c_ai_10.text_area(
                "Competitor brands (one per line)",
                placeholder="Competitor One\nCompetitor Two",
                height=100,
            )

            if ai_platform == "chat_gpt":
                st.info("ChatGPT LLM Mentions trenutno podpira samo United States + English. Uporabi `location_code=2840` in `language_code=en`.")

            run_ai_visibility = st.form_submit_button("Run AI Visibility Discovery", type="primary", use_container_width=True)

        if run_ai_visibility:
            if ai_target_mode == "Domain only":
                ai_brand = ""
            elif ai_target_mode == "Brand only":
                ai_domain = ""
            if not normalize_domain(ai_domain) and not normalize_space(ai_brand):
                st.error("Vnesi domeno, brand, ali oboje.")
            elif ai_platform == "chat_gpt" and (int(ai_location_code) != 2840 or ai_language_code.strip().lower() != "en"):
                st.error("Za `chat_gpt` LLM Mentions moraš uporabiti `location_code=2840` in `language_code=en`.")
            else:
                with st.spinner("[DISCOVERING_AI_QUERIES...] [CHECKING_MENTIONS...]"):
                    ai_result = dfseo_llm_mentions_search(
                        platform=ai_platform,
                        location_code=int(ai_location_code),
                        language_code=ai_language_code.strip().lower(),
                        target_domain=ai_domain,
                        target_brand=ai_brand,
                        min_ai_search_volume=int(ai_min_volume),
                        limit=int(ai_limit),
                    )
                st.session_state["ai_visibility_result"] = ai_result
                st.session_state["ai_visibility_inputs"] = {
                    "domain": ai_domain,
                    "brand": ai_brand,
                    "competitor_domains": parse_lines(competitor_domains_raw),
                    "competitor_brands": parse_lines(competitor_brands_raw),
                    "platform": ai_platform,
                    "location_code": int(ai_location_code),
                    "language_code": ai_language_code,
                    "location_name": (selected_location or {}).get("location_name", "") if 'selected_location' in locals() else "",
                }

        ai_result = st.session_state.get("ai_visibility_result")
        ai_inputs = st.session_state.get("ai_visibility_inputs", {})
        loaded_ai_path = st.session_state.get("ai_visibility_loaded_path")
        if loaded_ai_path:
            st.info(f"Showing saved AI run: `{Path(loaded_ai_path).name}`")
        if ai_result:
            if ai_result.get("error"):
                st.error(f"AI Visibility error: {ai_result['error']}")
            else:
                target_domain = normalize_domain(ai_inputs.get("domain", ""))
                target_brand = normalize_space(ai_inputs.get("brand", ""))
                competitor_domains = [normalize_domain(item) for item in ai_inputs.get("competitor_domains", []) if normalize_domain(item)]
                competitor_brands = [normalize_text(item) for item in ai_inputs.get("competitor_brands", []) if normalize_text(item)]
                discovery_items = ai_result.get("items") or []
                if not isinstance(discovery_items, list):
                    discovery_items = []

                rows = []
                your_domain_cited = 0
                your_brand_mentioned = 0
                competitor_hits = 0
                total_volume = 0
                for item in discovery_items:
                    question = item.get("question", "")
                    answer = item.get("answer", "") or ""
                    sources = item.get("sources", []) or []
                    search_results = item.get("search_results", []) or []
                    brand_entities = item.get("brand_entities", []) or []
                    fan_out_queries = item.get("fan_out_queries", []) or []
                    ai_search_volume = item.get("ai_search_volume", 0) or 0
                    source_domains = [normalize_domain(src.get("domain", "")) for src in sources if src.get("domain")]
                    result_domains = [normalize_domain(src.get("domain", "")) for src in search_results if src.get("domain")]
                    all_domains = [*source_domains, *result_domains]
                    brand_titles = [normalize_text(entity.get("title", "")) for entity in brand_entities if entity.get("title")]
                    answer_norm = normalize_text(answer)

                    domain_cited = bool(target_domain and any(domain_matches(domain, target_domain) for domain in all_domains))
                    brand_mentioned = bool(
                        target_brand and (
                            normalize_text(target_brand) in answer_norm
                            or any(normalize_text(target_brand) in title for title in brand_titles)
                        )
                    )
                    competitor_cited = any(any(domain_matches(domain, comp) for domain in all_domains) for comp in competitor_domains)
                    competitor_brand_mentioned = any(brand in answer_norm or any(brand in title for title in brand_titles) for brand in competitor_brands)

                    your_domain_cited += int(domain_cited)
                    your_brand_mentioned += int(brand_mentioned)
                    competitor_hits += int(competitor_cited or competitor_brand_mentioned)
                    total_volume += ai_search_volume

                    rows.append({
                        "Question": question,
                        "AI Search Volume": ai_search_volume,
                        "Platform": item.get("platform", ai_inputs.get("platform", "")),
                        "Model": item.get("model_name", ""),
                        "Domain cited": "Yes" if domain_cited else "No",
                        "Brand mentioned": "Yes" if brand_mentioned else "No",
                        "Competitor hit": "Yes" if competitor_cited or competitor_brand_mentioned else "No",
                        "Sources": len(sources),
                        "Fan-out queries": len(fan_out_queries),
                        "First seen": item.get("first_response_at", ""),
                        "Last seen": item.get("last_response_at", ""),
                        "_raw": item,
                    })

                c_vis_1, c_vis_2, c_vis_3, c_vis_4, c_vis_5 = st.columns(5)
                c_vis_1.metric("Queries found", len(rows))
                c_vis_2.metric("Your domain cited", your_domain_cited)
                c_vis_3.metric("Your brand mentioned", your_brand_mentioned)
                c_vis_4.metric("Competitor hits", competitor_hits)
                c_vis_5.metric("Avg AI search volume", round(total_volume / len(rows), 1) if rows else 0)

                if rows:
                    display_df = pd.DataFrame([{k: v for k, v in row.items() if k != "_raw"} for row in rows])
                    st.dataframe(display_df, use_container_width=True, hide_index=True)

                    row_options = {f"{row['Question'][:120]} [{row['AI Search Volume']}]": row for row in rows}
                    selected_query_key = st.selectbox(
                        "Inspect query details",
                        options=list(row_options.keys()),
                        key="ai_visibility_selected_query",
                    )
                    selected_row = row_options[selected_query_key]
                    raw = selected_row["_raw"]

                    st.markdown("**Question**")
                    st.write(raw.get("question", ""))
                    st.markdown("**Answer**")
                    st.markdown(raw.get("answer", ""))

                    sources = raw.get("sources", []) or []
                    if sources:
                        st.markdown("**Sources**")
                        st.dataframe(pd.DataFrame([{
                            "Title": src.get("title", ""),
                            "Domain": src.get("domain", ""),
                            "URL": src.get("url", ""),
                        } for src in sources]), use_container_width=True, hide_index=True)

                    brand_entities = raw.get("brand_entities", []) or []
                    if brand_entities:
                        st.markdown("**Brand entities**")
                        st.dataframe(pd.DataFrame([{
                            "Title": entity.get("title", ""),
                            "Category": entity.get("category", ""),
                        } for entity in brand_entities]), use_container_width=True, hide_index=True)

                    fan_out_queries = raw.get("fan_out_queries", []) or []
                    if fan_out_queries:
                        st.markdown("**Fan-out queries**")
                        fanout_rows = []
                        for query in fan_out_queries:
                            if isinstance(query, dict):
                                fanout_rows.append({"Query": query.get("query", "") or query.get("keyword", "") or str(query)})
                            else:
                                fanout_rows.append({"Query": str(query)})
                        st.dataframe(pd.DataFrame(fanout_rows), use_container_width=True, hide_index=True)

                    search_results = raw.get("search_results", []) or []
                    if search_results:
                        st.markdown("**Search results**")
                        st.dataframe(pd.DataFrame([{
                            "Title": item.get("title", ""),
                            "Domain": item.get("domain", ""),
                            "URL": item.get("url", ""),
                            "Position": item.get("position", ""),
                        } for item in search_results]), use_container_width=True, hide_index=True)
                else:
                    st.info("No AI visibility rows found for the selected filters.")

        st.divider()
        st.subheader("Model Response Testing")
        st.caption("Primerjaj, ali se tvoj brand ali domena pojavita v odgovorih ChatGPT, Gemini in Perplexity za iste queryje.")

        with st.form("ai_model_visibility_form"):
            c_model_1, c_model_2 = st.columns([1.5, 1])
            llm_queries_raw = c_model_1.text_area(
                "Queries (one per line)",
                placeholder="best pedicure salon in ljubljana\nwhere to get pedikura ljubljana\nis BrandName good",
                height=120,
            )
            llm_country_iso = c_model_2.text_input("Web search country ISO", value="US")

            c_model_3, c_model_4, c_model_5 = st.columns(3)
            run_chatgpt = c_model_3.checkbox("ChatGPT", value=True)
            run_gemini = c_model_4.checkbox("Gemini", value=True)
            run_perplexity = c_model_5.checkbox("Perplexity", value=True)

            c_model_6, c_model_7, c_model_8 = st.columns(3)
            chatgpt_model = c_model_6.text_input("ChatGPT model", value="gpt-4o")
            gemini_model = c_model_7.text_input("Gemini model", value="gemini-2.5-flash")
            perplexity_model = c_model_8.text_input("Perplexity model", value="sonar")

            c_model_9, c_model_10, c_model_11 = st.columns([1, 1, 1.2])
            llm_max_queries = c_model_9.slider("Max queries", 1, 10, 3, 1)
            llm_max_output = c_model_10.slider("Max output tokens", 200, 1200, 500, 100)
            llm_web_search = c_model_11.checkbox("Enable web search", value=True)

            run_llm_visibility = st.form_submit_button("Run Model Response Testing", type="primary", use_container_width=True)

        if run_llm_visibility:
            selected_platforms = []
            if run_chatgpt:
                selected_platforms.append(("chat_gpt", normalize_space(chatgpt_model) or "gpt-4o"))
            if run_gemini:
                selected_platforms.append(("gemini", normalize_space(gemini_model) or "gemini-2.5-flash"))
            if run_perplexity:
                selected_platforms.append(("perplexity", normalize_space(perplexity_model) or "sonar"))

            llm_queries = parse_lines(llm_queries_raw)[: int(llm_max_queries)]
            if not selected_platforms:
                st.error("Izberi vsaj en LLM.")
            elif not llm_queries:
                st.error("Dodaj vsaj en query.")
            else:
                llm_results = []
                with st.spinner("[TESTING_LLM_RESPONSES...] [COMPARING_MODELS...]"):
                    for query in llm_queries:
                        for platform_name, model_name in selected_platforms:
                            response = dfseo_llm_response_live(
                                platform=platform_name,
                                query=query,
                                model_name=model_name,
                                target_country_iso_code=llm_country_iso.strip().upper(),
                                web_search=llm_web_search,
                                max_output_tokens=int(llm_max_output),
                            )
                            llm_results.append(response)

                st.session_state["ai_model_visibility_result"] = llm_results
                st.session_state["ai_model_visibility_inputs"] = {
                    "queries": llm_queries,
                    "country_iso_code": llm_country_iso.strip().upper(),
                    "domain": ai_inputs.get("domain", "") or ai_domain,
                    "brand": ai_inputs.get("brand", "") or ai_brand,
                    "competitor_domains": ai_inputs.get("competitor_domains", []),
                    "competitor_brands": ai_inputs.get("competitor_brands", []),
                }

        llm_results = st.session_state.get("ai_model_visibility_result") or []
        llm_inputs = st.session_state.get("ai_model_visibility_inputs", {})
        if llm_results:
            target_domain = normalize_domain(llm_inputs.get("domain", ""))
            target_brand = normalize_space(llm_inputs.get("brand", ""))
            competitor_domains = [normalize_domain(item) for item in llm_inputs.get("competitor_domains", []) if normalize_domain(item)]
            competitor_brands = [normalize_text(item) for item in llm_inputs.get("competitor_brands", []) if normalize_text(item)]

            llm_rows = []
            platform_summary: dict[str, dict[str, int]] = {}
            brand_platform_hits: set[str] = set()
            domain_platform_hits: set[str] = set()
            competitor_platform_hits: set[str] = set()
            for item in llm_results:
                if item.get("error"):
                    llm_rows.append({
                        "Query": item.get("query", ""),
                        "LLM": item.get("platform", ""),
                        "Model": item.get("requested_model", ""),
                        "Brand mentioned": "Error",
                        "Domain cited": "Error",
                        "Competitor hit": "Error",
                        "Sources": 0,
                        "Money spent": 0,
                        "Status": item.get("error", "Error"),
                        "_raw": item,
                    })
                    continue

                answer_norm = normalize_text(item.get("answer", ""))
                annotations = item.get("annotations", []) or []
                source_domains = [normalize_domain(source.get("domain", "")) for source in annotations if source.get("domain")]
                brand_mentioned = bool(target_brand and normalize_text(target_brand) in answer_norm)
                domain_cited = bool(target_domain and any(domain_matches(domain, target_domain) for domain in source_domains))
                competitor_domain_hit = any(any(domain_matches(domain, comp) for domain in source_domains) for comp in competitor_domains)
                competitor_brand_hit = any(brand in answer_norm for brand in competitor_brands)
                competitor_hit = competitor_domain_hit or competitor_brand_hit

                platform_key = item.get("platform", "")
                if brand_mentioned:
                    brand_platform_hits.add(platform_key)
                if domain_cited:
                    domain_platform_hits.add(platform_key)
                if competitor_hit:
                    competitor_platform_hits.add(platform_key)
                platform_bucket = platform_summary.setdefault(platform_key, {
                    "responses": 0,
                    "brand_hits": 0,
                    "domain_hits": 0,
                    "competitor_hits": 0,
                })
                platform_bucket["responses"] += 1
                platform_bucket["brand_hits"] += int(brand_mentioned)
                platform_bucket["domain_hits"] += int(domain_cited)
                platform_bucket["competitor_hits"] += int(competitor_hit)

                llm_rows.append({
                    "Query": item.get("query", ""),
                    "LLM": platform_key,
                    "Model": item.get("model_name", item.get("requested_model", "")),
                    "Brand mentioned": "Yes" if brand_mentioned else "No",
                    "Domain cited": "Yes" if domain_cited else "No",
                    "Competitor hit": "Yes" if competitor_hit else "No",
                    "Sources": len(annotations),
                    "Money spent": round(float(item.get("money_spent", 0) or 0), 5),
                    "Status": "OK",
                    "_raw": item,
                })

            if platform_summary:
                total_platforms = len(platform_summary)
                strongest_platform = max(
                    platform_summary.items(),
                    key=lambda item: (
                        item[1]["brand_hits"],
                        item[1]["domain_hits"],
                        -item[1]["competitor_hits"],
                        item[1]["responses"],
                    ),
                )[0]
                weakest_platform = min(
                    platform_summary.items(),
                    key=lambda item: (
                        item[1]["brand_hits"],
                        item[1]["domain_hits"],
                        -item[1]["competitor_hits"],
                        item[1]["responses"],
                    ),
                )[0]

                st.markdown("**Multi-platform compare**")
                c_cmp_1, c_cmp_2, c_cmp_3, c_cmp_4, c_cmp_5 = st.columns(5)
                c_cmp_1.metric(
                    "Brand visibility",
                    f"{len(brand_platform_hits)}/{total_platforms} LLMs",
                    help="V koliko testiranih LLM-jih je bil brand omenjen vsaj enkrat.",
                )
                c_cmp_2.metric(
                    "Domain citations",
                    f"{len(domain_platform_hits)}/{total_platforms} LLMs",
                    help="V koliko testiranih LLM-jih je bila tvoja domena citirana vsaj enkrat.",
                )
                c_cmp_3.metric(
                    "Competitor presence",
                    f"{len(competitor_platform_hits)}/{total_platforms} LLMs",
                    help="V koliko testiranih LLM-jih se je pojavil vsaj en competitor.",
                )
                c_cmp_4.metric("Strongest platform", strongest_platform.replace("_", " ").title())
                c_cmp_5.metric("Weakest platform", weakest_platform.replace("_", " ").title())

                st.markdown("**LLM visibility summary**")
                summary_rows = []
                for platform_name, stats in platform_summary.items():
                    summary_rows.append({
                        "LLM": platform_name,
                        "Responses": stats["responses"],
                        "Brand mentions": stats["brand_hits"],
                        "Domain citations": stats["domain_hits"],
                        "Competitor hits": stats["competitor_hits"],
                    })
                st.dataframe(pd.DataFrame(summary_rows), use_container_width=True, hide_index=True)

            llm_display_df = pd.DataFrame([{k: v for k, v in row.items() if k != "_raw"} for row in llm_rows])
            st.dataframe(llm_display_df, use_container_width=True, hide_index=True)

            llm_options = {
                f"{row['LLM']} · {row['Query'][:90]}": row
                for row in llm_rows
            }
            selected_llm_key = st.selectbox(
                "Inspect LLM response",
                options=list(llm_options.keys()),
                key="ai_model_visibility_selected_row",
            )
            selected_llm_row = llm_options[selected_llm_key]["_raw"]

            if selected_llm_row.get("error"):
                st.error(selected_llm_row["error"])
            else:
                c_llm_a, c_llm_b, c_llm_c, c_llm_d = st.columns(4)
                c_llm_a.metric("LLM", selected_llm_row.get("platform", ""))
                c_llm_b.metric("Model", selected_llm_row.get("model_name", ""))
                c_llm_c.metric("Input tokens", selected_llm_row.get("input_tokens", 0))
                c_llm_d.metric("Output tokens", selected_llm_row.get("output_tokens", 0))

                st.markdown("**Query**")
                st.write(selected_llm_row.get("query", ""))
                st.markdown("**Answer**")
                st.markdown(selected_llm_row.get("answer", ""))

                annotations = selected_llm_row.get("annotations", []) or []
                if annotations:
                    st.markdown("**Cited sources**")
                    st.dataframe(pd.DataFrame([{
                        "Title": source.get("title", ""),
                        "Domain": source.get("domain", ""),
                        "URL": source.get("url", ""),
                    } for source in annotations]), use_container_width=True, hide_index=True)

                fan_out_queries = selected_llm_row.get("fan_out_queries", []) or []
                if fan_out_queries:
                    st.markdown("**Fan-out queries**")
                    fanout_rows = []
                    for query in fan_out_queries:
                        if isinstance(query, dict):
                            fanout_rows.append({"Query": query.get("query", "") or query.get("keyword", "") or str(query)})
                        else:
                            fanout_rows.append({"Query": str(query)})
                    st.dataframe(pd.DataFrame(fanout_rows), use_container_width=True, hide_index=True)

        current_ai_bundle = build_ai_visibility_bundle(
            discovery_inputs=st.session_state.get("ai_visibility_inputs"),
            discovery_result=st.session_state.get("ai_visibility_result"),
            model_inputs=st.session_state.get("ai_model_visibility_inputs"),
            model_result=st.session_state.get("ai_model_visibility_result"),
        )
        has_ai_content = bool(
            (current_ai_bundle.get("discovery_result") or {}).get("items")
            or (current_ai_bundle.get("model_result") or [])
        )
        if has_ai_content:
            ai_slug_source = (
                current_ai_bundle.get("discovery_inputs", {}).get("brand")
                or current_ai_bundle.get("discovery_inputs", {}).get("domain")
                or current_ai_bundle.get("model_inputs", {}).get("brand")
                or current_ai_bundle.get("model_inputs", {}).get("domain")
                or "ai_visibility"
            )
            ai_slug = re.sub(r"[^\wÀ-ž]+", "_", normalize_text(ai_slug_source), flags=re.UNICODE).strip("_") or "ai_visibility"
            ai_ts = datetime.now().strftime("%Y%m%d_%H%M%S")

            st.divider()
            st.markdown("**Save / Export AI run**")
            dl_ai_1, dl_ai_2 = st.columns(2)
            dl_ai_1.download_button(
                "Download AI run JSON",
                data=json.dumps(current_ai_bundle, ensure_ascii=False, indent=2),
                file_name=f"ai_visibility_{ai_slug}_{ai_ts}.json",
                mime="application/json",
                use_container_width=True,
                key="dl_ai_visibility_json",
            )
            dl_ai_2.download_button(
                "Download AI run Markdown",
                data=build_ai_visibility_markdown(current_ai_bundle),
                file_name=f"ai_visibility_{ai_slug}_{ai_ts}.md",
                mime="text/markdown",
                use_container_width=True,
                key="dl_ai_visibility_md",
            )
            if st.button("Save AI run to analize/AI_VISIBILITY", type="primary", use_container_width=True, key="save_ai_visibility_run"):
                try:
                    saved_paths = save_ai_visibility_run(current_ai_bundle)
                    saved_json = next((path for path in saved_paths if path.suffix == ".json"), None)
                    if saved_json:
                        st.session_state["ai_visibility_loaded_path"] = str(saved_json)
                    st.success("AI run saved.")
                    for path in saved_paths:
                        st.markdown(f"- [{path.name}]({path})")
                except Exception as e:
                    st.error(f"Could not save AI run: {e}")

# ── Info page ─────────────────────────────────────────────────────────────────

elif current_page == "ℹ️ Info":
    page_info()
