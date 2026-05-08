"""
SEO Content NLP Analyzer
Uses Google Cloud Natural Language API to analyze content for SEO insights.
"""

import argparse
import sys
import json
from urllib.request import urlopen, Request
from urllib.error import URLError
from html.parser import HTMLParser

from google.cloud import language_v1


# ── HTML text extractor ──────────────────────────────────────────────────────

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
            stripped = data.strip()
            if stripped:
                self.chunks.append(stripped)


def fetch_url_text(url: str) -> str:
    req = Request(url, headers={"User-Agent": "Mozilla/5.0 (SEO-NLP-Analyzer)"})
    try:
        with urlopen(req, timeout=15) as resp:
            html = resp.read().decode("utf-8", errors="replace")
    except URLError as e:
        sys.exit(f"[ERROR] Could not fetch URL: {e}")
    parser = _TextExtractor()
    parser.feed(html)
    return " ".join(parser.chunks)


# ── NLP calls ────────────────────────────────────────────────────────────────

def _parse_syntax_tokens(tokens) -> dict:
    total = 0
    passive = 0
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

    nouns      = pos_counts.get("NOUN", 0) + pos_counts.get("PROPN", 0)
    verbs      = pos_counts.get("VERB", 0)
    adjectives = pos_counts.get("ADJ", 0)
    adverbs    = pos_counts.get("ADV", 0)
    content_words = nouns + verbs + adjectives + adverbs

    return {
        "total_tokens": total,
        "passive_voice_pct": round(passive / total * 100, 1) if total else 0,
        "lexical_density": round(content_words / total, 3) if total else 0,
        "pos_counts": pos_counts,
        "top_nouns": sorted(lemma_counts.items(), key=lambda x: x[1], reverse=True)[:15],
        "noun_count": nouns,
        "verb_count": verbs,
        "adjective_count": adjectives,
        "adverb_count": adverbs,
    }


def annotate_text(text: str) -> tuple[list, dict, dict, list]:
    """Single API call replacing analyzeEntities + analyzeSentiment +
    analyzeSyntax + analyzeEntitySentiment. Returns (entities, sentiment, syntax, entity_sentiment)."""
    client = language_v1.LanguageServiceClient()
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
    response = client.annotate_text(
        request={"document": document, "features": features,
                 "encoding_type": language_v1.EncodingType.UTF8}
    )

    # entities
    entities = []
    for e in response.entities:
        entities.append({
            "name": e.name,
            "type": language_v1.Entity.Type(e.type_).name,
            "salience": round(e.salience, 4),
            "mentions": len(e.mentions),
            "wikipedia_url": e.metadata.get("wikipedia_url", ""),
        })
    entities.sort(key=lambda x: x["salience"], reverse=True)

    # sentiment
    sentiment = {
        "score": round(response.document_sentiment.score, 3),
        "magnitude": round(response.document_sentiment.magnitude, 3),
        "sentence_count": len(response.sentences),
    }

    # syntax (parsed from the tokens already returned)
    syntax = _parse_syntax_tokens(response.tokens)

    # entity sentiment (same entities, just different fields)
    entity_sentiment = []
    for e in response.entities:
        entity_sentiment.append({
            "name": e.name,
            "type": language_v1.Entity.Type(e.type_).name,
            "salience": round(e.salience, 4),
            "sentiment_score": round(e.sentiment.score, 3),
            "sentiment_magnitude": round(e.sentiment.magnitude, 3),
            "wikipedia_url": e.metadata.get("wikipedia_url", ""),
        })
    entity_sentiment.sort(key=lambda x: x["salience"], reverse=True)

    return entities, sentiment, syntax, entity_sentiment


def classify_content(text: str) -> list:
    # classifyText needs ≥20 tokens; check word count first
    if len(text.split()) < 20:
        return []
    client = language_v1.LanguageServiceClient()
    document = {
        "content": text,
        "type_": language_v1.Document.Type.PLAIN_TEXT,
        "language": "en",
    }
    content_categories_version = (
        language_v1.ClassificationModelOptions.V2Model.ContentCategoriesVersion.V2
    )
    response = client.classify_text(
        request={
            "document": document,
            "classification_model_options": {
                "v2_model": {"content_categories_version": content_categories_version}
            },
        }
    )
    results = []
    for cat in response.categories:
        results.append({
            "name": cat.name,
            "confidence": round(cat.confidence, 3),
        })
    results.sort(key=lambda x: x["confidence"], reverse=True)
    return results


# ── Report printer ───────────────────────────────────────────────────────────

def _bar(value: float, width: int = 30) -> str:
    filled = int(round(value * width))
    return "█" * filled + "░" * (width - filled)


def _sentiment_label(score: float) -> str:
    if score >= 0.25:
        return "POSITIVE ✓"
    if score <= -0.25:
        return "NEGATIVE ✗"
    return "NEUTRAL  ~"


def print_report(source: str, entities: list, sentiment: dict, categories: list,
                 syntax: dict, entity_sentiment: list,
                 top_n: int = 20, output_json: bool = False):

    if output_json:
        print(json.dumps({
            "source": source,
            "entities": entities[:top_n],
            "sentiment": sentiment,
            "categories": categories,
            "syntax": syntax,
            "entity_sentiment": entity_sentiment[:top_n],
        }, indent=2, ensure_ascii=False))
        return

    W = 70
    div = "─" * W

    print()
    print("═" * W)
    print(" SEO NLP ANALYZER REPORT".center(W))
    print("═" * W)
    print(f"  Source : {source[:W-10]}")
    print(div)

    # ── Sentiment ────────────────────────────────────────────────────────────
    print("\n  SENTIMENT ANALYSIS")
    print(div)
    score = sentiment["score"]
    magnitude = sentiment["magnitude"]
    label = _sentiment_label(score)
    norm_score = (score + 1) / 2          # map [-1,1] → [0,1] for the bar
    norm_mag = min(magnitude / 10, 1.0)   # cap magnitude bar at 10
    print(f"  Overall tone : {label}")
    print(f"  Score        : {score:+.3f}  {_bar(norm_score)}")
    print(f"  Magnitude    : {magnitude:.3f}   {_bar(norm_mag)}  (emotional intensity)")
    print(f"  Sentences    : {sentiment['sentence_count']}")
    print()
    print("  SEO tip:")
    if score >= 0.25:
        print("  → Positive tone is good for reviews, testimonials, and product pages.")
    elif score <= -0.25:
        print("  → Negative tone may hurt CTR. Consider softening language.")
    else:
        print("  → Neutral tone is fine for informational/educational content.")

    # ── Categories ───────────────────────────────────────────────────────────
    print()
    print("  CONTENT CLASSIFICATION  (what Google thinks this page is about)")
    print(div)
    if not categories:
        print("  [Not enough text to classify — add more content]")
    else:
        for cat in categories[:5]:
            bar = _bar(cat["confidence"])
            print(f"  {cat['confidence']:.0%}  {bar}  {cat['name']}")
        print()
        print("  SEO tip:")
        top_cat = categories[0]["name"].split("/")[-1] if categories else ""
        print(f"  → Primary topic detected: '{top_cat}'")
        print("  → Match this with your target keyword cluster to confirm topical alignment.")

    # ── Entities ─────────────────────────────────────────────────────────────
    print()
    print(f"  TOP {top_n} ENTITIES  (salience = how central to the page)")
    print(div)
    print(f"  {'#':<4} {'ENTITY':<28} {'TYPE':<16} {'SALIENCE':>8}  {'MENTIONS':>7}  WIKI")
    print(f"  {'─'*4} {'─'*28} {'─'*16} {'─'*8}  {'─'*7}  ────")
    for i, e in enumerate(entities[:top_n], 1):
        wiki = "✓" if e["wikipedia_url"] else ""
        sal_pct = f"{e['salience']*100:.1f}%"
        print(f"  {i:<4} {e['name'][:28]:<28} {e['type'][:16]:<16} {sal_pct:>8}  {e['mentions']:>7}  {wiki}")

    print()
    print("  SEO tips:")
    top_entities = [e for e in entities[:5]]
    if top_entities:
        names = ", ".join(f"'{e['name']}'" for e in top_entities[:3])
        print(f"  → Google sees {names} as the main topics.")
        print("  → High-salience entities with Wikipedia links are recognized knowledge")
        print("    graph entities — great for semantic SEO and featured snippets.")
        print("  → If your target keyword is missing or has low salience, add it to")
        print("    headings, first paragraph, and use it in contextually related sentences.")

    # ── Syntax ───────────────────────────────────────────────────────────────
    print()
    print("  SYNTAX ANALYSIS")
    print(div)

    passive_pct   = syntax["passive_voice_pct"]
    lex_density   = syntax["lexical_density"]
    total_tokens  = syntax["total_tokens"]
    top_nouns     = syntax["top_nouns"]

    # Passive voice bar (red zone > 15%)
    passive_flag  = "⚠" if passive_pct > 15 else "✓"
    passive_norm  = min(passive_pct / 30, 1.0)   # bar saturates at 30%
    lex_norm      = min(lex_density / 0.7, 1.0)  # bar saturates at 70%

    print(f"  Total tokens    : {total_tokens}")
    print(f"  Nouns           : {syntax['noun_count']}")
    print(f"  Verbs           : {syntax['verb_count']}")
    print(f"  Adjectives      : {syntax['adjective_count']}")
    print(f"  Adverbs         : {syntax['adverb_count']}")
    print()
    print(f"  Passive voice   : {passive_pct:.1f}%  {_bar(passive_norm)}  {passive_flag}")
    print(f"  Lexical density : {lex_density:.1%}  {_bar(lex_norm)}")
    print()
    print("  Top implied topics (most frequent nouns):")
    for noun, count in top_nouns[:10]:
        noun_bar = _bar(min(count / (top_nouns[0][1] if top_nouns else 1), 1.0), width=20)
        print(f"    {count:>3}x  {noun_bar}  {noun}")
    print()
    print("  SEO tips:")
    if passive_pct > 15:
        print(f"  ⚠ {passive_pct:.1f}% passive voice is high. Rewrite passive sentences to active")
        print("    voice — clearer writing improves dwell time and readability scores.")
    else:
        print(f"  ✓ Passive voice at {passive_pct:.1f}% — good. Keep writing in active voice.")
    if lex_density < 0.40:
        print("  ⚠ Low lexical density — content may be too fluffy. Add more specific")
        print("    nouns, facts, and concrete details to improve topical depth.")
    else:
        print(f"  ✓ Lexical density at {lex_density:.1%} — content is substantive.")
    if top_nouns:
        kw_list = ", ".join(f"'{n}'" for n, _ in top_nouns[:5])
        print(f"  → Dominant nouns: {kw_list}")
        print("    If these don't match your target keywords, adjust your keyword usage.")

    # ── Entity Sentiment ─────────────────────────────────────────────────────
    print()
    print(f"  ENTITY SENTIMENT  (how each topic is talked about)")
    print(div)
    print(f"  {'ENTITY':<28} {'TYPE':<14} {'SALIENCE':>8}  {'SCORE':>6}  {'MAG':>5}  TONE")
    print(f"  {'─'*28} {'─'*14} {'─'*8}  {'─'*6}  {'─'*5}  ────")
    for e in entity_sentiment[:top_n]:
        tone = _sentiment_label(e["sentiment_score"]).split()[0]
        sal_pct = f"{e['salience']*100:.1f}%"
        print(
            f"  {e['name'][:28]:<28} {e['type'][:14]:<14} {sal_pct:>8}"
            f"  {e['sentiment_score']:>+6.2f}  {e['sentiment_magnitude']:>5.2f}  {tone}"
        )

    print()
    print("  SEO tips:")
    negative = [e for e in entity_sentiment[:top_n] if e["sentiment_score"] <= -0.25]
    positive = [e for e in entity_sentiment[:top_n] if e["sentiment_score"] >= 0.25]
    if negative:
        neg_names = ", ".join(f"'{e['name']}'" for e in negative[:3])
        print(f"  ⚠ Negative sentiment around: {neg_names}")
        print("    If these are your own brand/product, rewrite surrounding sentences.")
        print("    If these are competitors, this framing may help your page stand out.")
    if positive:
        pos_names = ", ".join(f"'{e['name']}'" for e in positive[:3])
        print(f"  ✓ Positive sentiment around: {pos_names} — good for E-E-A-T signals.")
    if not negative and not positive:
        print("  → All entities are neutral — fine for informational content.")

    print()
    print("═" * W)
    print()


# ── CLI ──────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="SEO NLP Analyzer — powered by Google Cloud Natural Language API",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python seo_nlp_analyzer.py --url https://example.com/blog-post
  python seo_nlp_analyzer.py --text "Your page content here..."
  python seo_nlp_analyzer.py --url https://example.com --json
  python seo_nlp_analyzer.py --url https://example.com --top 30
        """,
    )
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--url",  help="URL of the page to analyze")
    group.add_argument("--text", help="Raw text to analyze")
    parser.add_argument("--top",  type=int, default=20, help="Number of top entities to show (default: 20)")
    parser.add_argument("--json", action="store_true", help="Output raw JSON instead of formatted report")

    args = parser.parse_args()

    if args.url:
        print(f"Fetching {args.url} ...")
        text = fetch_url_text(args.url)
        source = args.url
    else:
        text = args.text
        source = "raw text input"

    if not text.strip():
        sys.exit("[ERROR] No text content found.")

    # Truncate to ~100k chars (API limit is 1M bytes but keep it fast)
    if len(text) > 100_000:
        text = text[:100_000]
        print("[INFO] Text truncated to 100,000 characters.")

    print("Running NLP analysis (single API call) ...")
    entities, sentiment, syntax, entity_sent = annotate_text(text)

    print("Running content classification ...")
    categories = classify_content(text)

    print_report(source, entities, sentiment, categories, syntax, entity_sent,
                 top_n=args.top, output_json=args.json)


if __name__ == "__main__":
    main()
