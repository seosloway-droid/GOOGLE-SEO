"""
SEO NLP Analyzer — Streamlit Dashboard
Powered by Google Cloud Natural Language API
"""

import streamlit as st
import pandas as pd
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
        "total_tokens":    total,
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

    # Single call: entities + sentiment + syntax + entity sentiment
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

    # Separate call: content classification
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
        "entities":        entities,
        "sentiment":       sentiment,
        "syntax":          syntax,
        "entity_sentiment": entity_sentiment,
        "categories":      categories,
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
    c1.metric("Sentiment",     f"{s['score']:+.2f}",
              help="−1.0 = very negative · +1.0 = very positive")
    c2.metric("Magnitude",     f"{s['magnitude']:.2f}",
              help="Emotional intensity (0 = calm, 10+ = very emotional)")
    c3.metric("Sentences",     s["sentence_count"])
    c4.metric("Passive Voice", f"{sx['passive_voice_pct']:.1f}%",
              delta="High ⚠" if sx["passive_voice_pct"] > 15 else "OK ✓",
              delta_color="inverse")
    c5.metric("Lexical Density", f"{sx['lexical_density']:.1%}",
              help="Content word ratio — higher means more substantive")

    if keyword:
        st.divider()
        kw_lower = keyword.lower()
        matches = [e for e in data["entities"] if kw_lower in e["name"].lower()]
        if matches:
            top = matches[0]
            kg  = "in Knowledge Graph ✓" if top["wikipedia"] else "not in Knowledge Graph"
            st.success(
                f"**'{keyword}'** found as entity · "
                f"Salience: **{top['salience']*100:.1f}%** · "
                f"Type: **{top['type']}** · {kg}"
            )
        else:
            st.warning(
                f"**'{keyword}'** not detected as an entity. "
                "Add it to H1, first paragraph, and use it in contextually related sentences."
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
    icon  = _tone_icon(score)
    label = _tone_label(score)

    col1, col2 = st.columns(2)
    with col1:
        st.subheader(f"{icon} {label}")
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
    t1, t2, t3, t4, t5, t6 = st.tabs([
        "📊 Overview",
        "🏷 Entities",
        "😊 Sentiment",
        "📂 Categories",
        "🔤 Syntax",
        "🎯 Entity Sentiment",
    ])
    with t1: tab_overview(data, keyword)
    with t2: tab_entities(data, keyword)
    with t3: tab_sentiment(data)
    with t4: tab_categories(data)
    with t5: tab_syntax(data)
    with t6: tab_entity_sentiment(data)


# ── Main UI ───────────────────────────────────────────────────────────────────

st.title("🔍 SEO NLP Analyzer")
st.caption("Powered by Google Cloud Natural Language API")

with st.form("analyze_form"):
    c1, c2, c3 = st.columns([3, 3, 2])
    url1    = c1.text_input("Your page URL",
                             placeholder="https://yoursite.com/page")
    url2    = c2.text_input("Competitor URL (optional)",
                             placeholder="https://competitor.com/page")
    keyword = c3.text_input("Target keyword (optional)",
                             placeholder="e.g. seo tools")
    submitted = st.form_submit_button("Analyze", type="primary",
                                       use_container_width=True)

if submitted:
    if not url1:
        st.error("Please enter at least one URL.")
        st.stop()

    results = {}

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
