"""
SEO NLP Analyzer — Streamlit Dashboard
Powered by Google Cloud Natural Language API
"""

import streamlit as st
import pandas as pd
import anthropic
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
    if "ANTHROPIC_API_KEY" in st.secrets:
        return anthropic.Anthropic(api_key=st.secrets["ANTHROPIC_API_KEY"])
    return None


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

    prompt = f"""You are an expert SEO consultant. Analyze this NLP data from Google's Natural Language API and give specific, actionable SEO advice.

TARGET KEYWORD: {keyword if keyword else 'not specified'}

SENTIMENT:
  Score: {s['score']:+.3f} (range -1.0 to +1.0)
  Magnitude: {s['magnitude']:.2f} (emotional intensity)
  Sentences: {s['sentence_count']}

SYNTAX:
  Passive voice: {sx['passive_voice_pct']:.1f}%
  Lexical density: {sx['lexical_density']:.1%}
  Nouns: {sx['noun_count']}, Verbs: {sx['verb_count']}, Adjectives: {sx['adjective_count']}
  Top nouns: {', '.join(n for n, _ in sx['top_nouns'][:8])}

TOP ENTITIES (what Google sees as main topics):
{entity_lines}

ENTITY SENTIMENT (how each topic is talked about):
{ent_sent_lines}

CONTENT CATEGORIES (what Google classifies this page as):
{cat_lines}

{detail_instruction}
{lang_instruction}

Be specific and direct. Reference the actual numbers from the data. Give concrete examples where relevant."""

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
        with st.spinner("Claude is analyzing your content..."):
            try:
                report = generate_seo_report(data, keyword, language, detail)
                st.session_state["ai_report"] = report
            except Exception as e:
                st.session_state["ai_report"] = f"ERROR: {e}"

    if "ai_report" in st.session_state and st.session_state["ai_report"]:
        report = st.session_state["ai_report"]
        if report.startswith("ERROR:"):
            st.error(report)
        else:
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
def run_analysis(text: str) -> dict:
    client = get_client()
    document = {
        "content": text,
        "type_": language_v1.Document.Type.PLAIN_TEXT,
        "language": "en",
    }

    features = language_v1.AnnotateTextRequest.Features(
        extract_syntax=True,
        extract_entities=True,
        extract_document_sentiment=True,
        extract_entity_sentiment=True,
    )
    resp = client.annotate_text(
        request={
            "document": document,
            "features": features,
            "encoding_type": language_v1.EncodingType.UTF8,
        }
    )

    entities = sorted([{
        "name":      e.name,
        "type":      language_v1.Entity.Type(e.type_).name,
        "salience":  round(e.salience, 4),
        "mentions":  len(e.mentions),
        "wikipedia": e.metadata.get("wikipedia_url", ""),
    } for e in resp.entities], key=lambda x: x["salience"], reverse=True)

    sentiment = {
        "score":          round(resp.document_sentiment.score, 3),
        "magnitude":      round(resp.document_sentiment.magnitude, 3),
        "sentence_count": len(resp.sentences),
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

    return {
        "entities":         entities,
        "sentiment":        sentiment,
        "syntax":           syntax,
        "entity_sentiment": entity_sentiment,
        "categories":       categories,
    }


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


# ── Tab renderers ─────────────────────────────────────────────────────────────

def tab_overview(data: dict, keyword: str):
    s  = data["sentiment"]
    sx = data["syntax"]

    c1, c2, c3, c4, c5 = st.columns(5)
    c1.metric("Sentiment",       f"{s['score']:+.2f}",
              help="−1.0 = very negative · +1.0 = very positive")
    c2.metric("Magnitude",       f"{s['magnitude']:.2f}",
              help="Emotional intensity (0 = calm, 10+ = very emotional)")
    c3.metric("Sentences",       s["sentence_count"])
    c4.metric("Passive Voice",   f"{sx['passive_voice_pct']:.1f}%",
              delta="High ⚠" if sx["passive_voice_pct"] > 15 else "OK ✓",
              delta_color="inverse")
    c5.metric("Lexical Density", f"{sx['lexical_density']:.1%}",
              help="Content word ratio — higher means more substantive")

    st.divider()

    # ── Sentiment interpretation ──────────────────────────────────────────────
    score = s["score"]
    mag   = s["magnitude"]

    if score >= 0.4:
        st.success(f"**Sentiment {score:+.2f} — Clearly positive ✓**  Ideal for product and service pages.")
    elif score >= 0.1:
        col_a, col_b = st.columns([1, 2])
        col_a.warning(f"**Sentiment {score:+.2f} — Slightly positive**")
        col_b.markdown(f"""
**For a product/service page this is too low.** Target is +0.4 or higher.

**Why:** {score:+.2f} score + magnitude {mag:.1f} = mixed content.
Some parts of your page are positive, others negative — they cancel each other out.
Google sees this as an uncertain or unconfident page about your topic.

**What to do:**
- Find sentences that express doubt, problems, or negatives and rewrite them positively
- Add more benefit-focused language: *"saves time"*, *"lasts 20 years"*, *"easy to maintain"*

**Example — Before:** *"Pool maintenance can be complex and time-consuming."*
**Example — After:** *"With our pools, maintenance takes less than 30 minutes a week."*
""")
    elif score >= -0.1:
        col_a, col_b = st.columns([1, 2])
        col_a.warning(f"**Sentiment {score:+.2f} — Neutral**")
        col_b.markdown(f"""
**Neutral + magnitude {mag:.1f} = mixed signals.**
{"Some parts positive, some negative — they cancel out. This is the most common problem on product pages." if mag > 5 else "Very calm, factual writing — fine for informational pages, but too dry for product pages."}

**Golden range for product pages:** Score +0.4 to +0.7, Magnitude 3–8

**What to do:**
- Add customer benefits after every feature statement
- Use emotionally positive words: *reliable, durable, effortless, beautiful, proven*

**Example — Before:** *"The pool is made of fiberglass."*
**Example — After:** *"The fiberglass construction ensures a smooth surface, easy cleaning, and a lifespan of over 30 years."*
""")
    else:
        st.error(f"**Sentiment {score:+.2f} — Negative ✗**  This will hurt CTR. Rewrite negative sentences.")

    # ── Lexical density interpretation ────────────────────────────────────────
    ld = sx["lexical_density"]
    if ld < 0.40:
        col_a, col_b = st.columns([1, 2])
        col_a.warning(f"**Lexical density {ld:.1%} — Too low**")
        col_b.markdown(f"""
**Your content has too many filler words and not enough substance.**
Lexical density measures what % of your words carry real meaning (nouns, verbs, adjectives).
At {ld:.1%} you are just below the 40% threshold — Google sees this as thin content.

**What to do:** Replace vague phrases with specific details:
- Add dimensions, materials, weights, capacities
- Add specific product names, model numbers, technical specs
- Add concrete benefits with numbers: *"heats up 3x faster"*, *"saves €200/year"*

**Example — Before (fluffy):** *"Our pools are great and very high quality."*
**Example — After (dense):** *"Our fibreglass pools (4×8m, 1.5m depth) withstand temperatures from −20°C to +50°C and require no repainting for 25 years."*
""")
    else:
        st.success(f"**Lexical density {ld:.1%} — Good ✓**  Content is substantive and information-rich.")

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


def tab_syntax(data: dict):
    sx = data["syntax"]

    col1, col2 = st.columns(2)
    with col1:
        st.subheader("Token breakdown")
        token_df = pd.DataFrame([
            {"Part of Speech": "Nouns",      "Count": sx["noun_count"]},
            {"Part of Speech": "Verbs",      "Count": sx["verb_count"]},
            {"Part of Speech": "Adjectives", "Count": sx["adjective_count"]},
            {"Part of Speech": "Adverbs",    "Count": sx["adverb_count"]},
        ])
        st.dataframe(token_df, use_container_width=True, hide_index=True)
        st.metric("Total tokens", sx["total_tokens"])

    with col2:
        st.subheader("Quality signals")
        pv = sx["passive_voice_pct"]
        ld = sx["lexical_density"]

        st.progress(min(pv / 30, 1.0), text=f"Passive voice: {pv:.1f}%")
        if pv > 15:
            st.error(f"⚠ {pv:.1f}% passive voice is high — rewrite to active voice.")
        else:
            st.success(f"✓ Passive voice at {pv:.1f}% — good.")

        st.progress(min(ld / 0.7, 1.0), text=f"Lexical density: {ld:.1%}")
        if ld < 0.40:
            st.warning("Low lexical density — add more specific nouns and concrete details.")
        else:
            st.success(f"✓ Lexical density at {ld:.1%} — content is substantive.")

    if sx["top_nouns"]:
        st.subheader("Top implied topics (most frequent nouns)")
        noun_df = pd.DataFrame(sx["top_nouns"], columns=["noun", "count"])
        st.bar_chart(noun_df.set_index("noun")["count"])
        st.caption(
            "If your target keyword isn't in the top nouns, "
            "increase its usage in headings and body text."
        )


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


def render_analysis(data: dict, keyword: str = ""):
    t1, t2, t3, t4, t5, t6, t7 = st.tabs([
        "📊 Overview",
        "🏷 Entities",
        "😊 Sentiment",
        "📂 Categories",
        "🔤 Syntax",
        "🎯 Entity Sentiment",
        "🤖 AI SEO Coach",
    ])
    with t1: tab_overview(data, keyword)
    with t2: tab_entities(data, keyword)
    with t3: tab_sentiment(data)
    with t4: tab_categories(data)
    with t5: tab_syntax(data)
    with t6: tab_entity_sentiment(data)
    with t7: tab_ai_coach(data, keyword)


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
                                     placeholder="e.g. seo tools")
            raw_text = ""
        else:
            keyword  = st.text_input("Target keyword (optional)",
                                      placeholder="e.g. seo tools")
            raw_text = st.text_area(
                "Paste your article or page content here",
                placeholder="Paste the full text of your article, blog post, or page...",
                height=250,
            )
            url1 = url2 = ""

        submitted = st.form_submit_button("Analyze", type="primary",
                                           use_container_width=True)

    if submitted:
        results = {}

        if input_mode == "🌐 URL":
            if not url1:
                st.error("Please enter at least one URL.")
                st.stop()

            with st.spinner(f"Fetching and analyzing {url1} ..."):
                text1 = fetch_url_text(url1)
                if text1:
                    if len(text1) > 100_000:
                        text1 = text1[:100_000]
                    results["url1"] = run_analysis(text1)

            if url2:
                with st.spinner(f"Fetching and analyzing {url2} ..."):
                    text2 = fetch_url_text(url2)
                    if text2:
                        if len(text2) > 100_000:
                            text2 = text2[:100_000]
                        results["url2"] = run_analysis(text2)
        else:
            if not raw_text.strip():
                st.error("Please paste some text to analyze.")
                st.stop()
            text1 = raw_text.strip()
            if len(text1) > 100_000:
                text1 = text1[:100_000]
            with st.spinner("Analyzing text ..."):
                results["url1"] = run_analysis(text1)
            url1 = "pasted text"

        if not results:
            st.error("No results — check that the URLs are publicly accessible.")
            st.stop()

        if "url2" in results:
            st.divider()
            col_a, col_b = st.columns(2)
            with col_a:
                st.markdown(f"#### Your page\n`{url1[:60]}`")
                render_analysis(results["url1"], keyword)
            with col_b:
                st.markdown(f"#### Competitor\n`{url2[:60]}`")
                render_analysis(results["url2"], keyword)
        else:
            st.divider()
            render_analysis(results["url1"], keyword)

# ── Info page ─────────────────────────────────────────────────────────────────

elif current_page == "ℹ️ Info":
    page_info()
