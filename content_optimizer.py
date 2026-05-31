"""
POP-like content optimizer engine.

This module is intentionally API-free. It takes already extracted page text and
competitor text, then returns deterministic term-gap and Content Score data.

Later phases can feed it Firecrawl, Google NLP, or DataForSEO results.
"""

from __future__ import annotations

import argparse
import json
import math
import re
import statistics
from dataclasses import dataclass, field
from html.parser import HTMLParser
from pathlib import Path
from typing import Any


WORD_RE = re.compile(r"[\wÀ-ž]+", re.UNICODE)
DEFAULT_STOPWORDS = {
    "a", "ali", "and", "are", "as", "at", "be", "brez", "by", "da", "de", "do",
    "for", "from", "ga", "he", "i", "in", "is", "it", "je", "jih", "jo", "ki",
    "kot", "la", "lahko", "le", "na", "ne", "of", "on", "or", "pa", "po", "pri",
    "se", "so", "su", "the", "to", "v", "va", "van", "za", "z", "že",
}


@dataclass(frozen=True)
class TermSpec:
    term: str
    term_type: str


@dataclass
class PageText:
    text: str
    title: str = ""
    meta_description: str = ""
    canonical: str = ""
    image_count: int = 0
    images_with_alt: int = 0
    h1: list[str] = field(default_factory=list)
    h2: list[str] = field(default_factory=list)
    h3: list[str] = field(default_factory=list)
    h4: list[str] = field(default_factory=list)
    h5: list[str] = field(default_factory=list)
    h6: list[str] = field(default_factory=list)
    url: str = ""


@dataclass
class OptimizerInput:
    primary_keyword: str
    my_page: PageText
    competitor_pages: list[PageText]
    secondary_keywords: list[str] = field(default_factory=list)
    lsi_keywords: list[str] = field(default_factory=list)
    entity_terms: list[str] = field(default_factory=list)
    auto_lsi: bool = False
    auto_lsi_limit: int = 20
    my_entities: list[dict[str, Any]] = field(default_factory=list)
    competitor_entities: list[list[dict[str, Any]]] = field(default_factory=list)
    my_sentiment: dict[str, Any] = field(default_factory=dict)
    competitor_sentiments: list[dict[str, Any]] = field(default_factory=list)
    my_readability: dict[str, Any] = field(default_factory=dict)
    competitor_readability: list[dict[str, Any]] = field(default_factory=list)
    my_images: dict[str, Any] = field(default_factory=dict)
    competitor_images: list[dict[str, Any]] = field(default_factory=list)
    language: str = "sl"
    page_type: str = ""


def auto_lsi_limit_for_page_type(page_type: str, competitor_count: int = 0) -> int:
    """Return a practical default for automatic competitor-term extraction."""
    normalized = normalize_text(page_type)
    base_by_type = {
        "product page": 15,
        "service page": 18,
        "local landing page": 20,
        "e-commerce category": 30,
        "affiliate/review page": 35,
        "comparison page": 35,
        "blog": 35,
    }
    base = base_by_type.get(normalized, 20)
    if competitor_count >= 8:
        base += 10
    elif competitor_count >= 5:
        base += 5
    return max(5, min(50, base))


def lsi_limit_for_depth(depth: str, page_type: str = "", competitor_count: int = 0) -> int:
    normalized = normalize_text(depth)
    if normalized == "focused":
        return 10
    if normalized == "balanced":
        return 20
    if normalized == "deep":
        return 40
    return auto_lsi_limit_for_page_type(page_type, competitor_count)


def _score_signal(scores: dict[str, float], page_type: str, points: float, reason: str, reasons: dict[str, list[str]]) -> None:
    scores[page_type] = scores.get(page_type, 0.0) + points
    reasons.setdefault(page_type, []).append(reason)


def classify_page_intent(page: PageText, primary_keyword: str = "") -> dict[str, Any]:
    """Classify page type from URL, headings, and text signals."""
    title = normalize_text(page.title)
    headings = normalize_text(" ".join([*page.h1, *page.h2, *page.h3]))
    url = normalize_text(page.url)
    text = normalize_text(page.text[:12000])
    combined = " ".join([url, title, headings, text])
    words = word_count(page.text)
    h2_count = len(page.h2)
    scores = {
        "blog": 0.0,
        "product page": 0.0,
        "e-commerce category": 0.0,
        "service page": 0.0,
        "local landing page": 0.0,
        "affiliate/review page": 0.0,
        "comparison page": 0.0,
    }
    reasons: dict[str, list[str]] = {key: [] for key in scores}

    blog_terms = [
        "kako", "vodic", "vodič", "nasvet", "nasveti", "guide", "how to", "blog",
        "kaj je", "zakaj", "faq", "pogosta vprasanja", "pogosta vprašanja",
    ]
    product_terms = [
        "dodaj v kosarico", "dodaj v košarico", "add to cart", "sku", "izdelek",
        "product", "zaloga", "na zalogi", "cena", "€", "eur", "specifikacije",
    ]
    category_terms = [
        "sortiraj", "filtriraj", "filter", "kategorija", "category", "izdelki",
        "prikaz", "artikli", "ponudba", "kolekcija",
    ]
    service_terms = [
        "storitev", "najem", "montaza", "montaža", "servis", "svetovanje",
        "povprasevanje", "povpraševanje", "kontaktirajte", "termin", "service",
    ]
    local_terms = [
        "ljubljana", "maribor", "celje", "koper", "kranj", "slovenija",
        "v blizini", "v bližini", "lokacija", "naslov", "zemljevid",
    ]
    review_terms = [
        "ocena", "review", "mnenje", "test", "izkusnje", "izkušnje",
        "najboljsi", "najboljši", "top", "prednosti in slabosti",
    ]
    comparison_terms = [
        "primerjava", "vs", "ali", "razlika", "primerjamo", "comparison",
        "kateri je boljsi", "kateri je boljši",
    ]

    for term in blog_terms:
        if term in combined:
            _score_signal(scores, "blog", 1.0, f"Informational signal `{term}`.", reasons)
    for term in product_terms:
        if term in combined:
            _score_signal(scores, "product page", 1.2, f"Product/commercial signal `{term}`.", reasons)
    for term in category_terms:
        if term in combined:
            _score_signal(scores, "e-commerce category", 1.1, f"Category/listing signal `{term}`.", reasons)
    for term in service_terms:
        if term in combined:
            _score_signal(scores, "service page", 1.1, f"Service signal `{term}`.", reasons)
    for term in local_terms:
        if term in combined:
            _score_signal(scores, "local landing page", 1.0, f"Local signal `{term}`.", reasons)
    for term in review_terms:
        if term in combined:
            _score_signal(scores, "affiliate/review page", 1.1, f"Review signal `{term}`.", reasons)
    for term in comparison_terms:
        if term in title or term in headings or term in url:
            _score_signal(scores, "comparison page", 1.4, f"Comparison signal `{term}`.", reasons)
        elif term in text:
            _score_signal(scores, "comparison page", 0.5, f"Weak comparison signal `{term}`.", reasons)

    if "/blog" in url or "/nasvet" in url or "/guide" in url:
        _score_signal(scores, "blog", 2.0, "URL looks informational.", reasons)
    if "/product" in url or "/izdelek" in url or "/shop" in url:
        _score_signal(scores, "product page", 2.0, "URL looks like product/shop page.", reasons)
    if "/category" in url or "/kategorija" in url or "/collections" in url:
        _score_signal(scores, "e-commerce category", 2.0, "URL looks like category page.", reasons)
    if "/service" in url or "/storitev" in url:
        _score_signal(scores, "service page", 2.0, "URL looks like service page.", reasons)

    if words >= 900 and h2_count >= 3:
        _score_signal(scores, "blog", 1.0, "Long structured content.", reasons)
    if words < 700 and ("€" in page.text or "cena" in text):
        _score_signal(scores, "product page", 1.0, "Short commercial page with price signal.", reasons)
    if h2_count <= 2 and any(term in combined for term in category_terms):
        _score_signal(scores, "e-commerce category", 0.8, "Listing page with limited editorial headings.", reasons)

    if primary_keyword and any(term in normalize_text(primary_keyword) for term in ("ali", "vs", "primerjava")):
        _score_signal(scores, "comparison page", 1.0, "Primary keyword suggests comparison intent.", reasons)

    ranked = sorted(scores.items(), key=lambda item: item[1], reverse=True)
    detected, top_score = ranked[0]
    second_score = ranked[1][1] if len(ranked) > 1 else 0.0
    confidence = 0.35
    if top_score:
        confidence = min(0.95, max(0.45, (top_score - second_score + 1.0) / (top_score + 2.0)))

    if top_score == 0:
        detected = "blog" if words >= 800 else "e-commerce category"
        confidence = 0.35
        reasons.setdefault(detected, []).append("Fallback based on content length.")

    return {
        "detected_page_type": detected,
        "confidence": round(confidence, 2),
        "scores": {key: round(value, 2) for key, value in scores.items()},
        "signals": reasons.get(detected, [])[:6],
        "word_count": words,
        "heading_counts": heading_counts(page),
    }


def competitor_fit_recommendation(
    effective_page_type: str,
    competitor_intent: dict[str, Any],
) -> dict[str, str]:
    detected = competitor_intent.get("detected_page_type", "")
    confidence = float(competitor_intent.get("confidence", 0) or 0)
    if not effective_page_type or not detected:
        return {
            "fit": "review",
            "reason": "Page type signal is incomplete.",
        }
    if normalize_text(detected) == normalize_text(effective_page_type):
        return {
            "fit": "good fit",
            "reason": "Competitor intent matches the target page type.",
        }
    if confidence >= 0.7:
        return {
            "fit": "exclude?",
            "reason": (
                f"Detected as `{detected}` with high confidence, while target is "
                f"`{effective_page_type}`."
            ),
        }
    return {
        "fit": "review",
        "reason": (
            f"Detected as `{detected}`, but confidence is moderate. Keep only if SERP intent fits."
        ),
    }


def classify_intent_set(data: OptimizerInput) -> dict[str, Any]:
    my_intent = classify_page_intent(data.my_page, data.primary_keyword)
    selected = normalize_text(data.page_type)
    auto_selected = selected in ("", "auto-detect", "auto detect", "auto")
    effective_page_type = my_intent["detected_page_type"] if auto_selected else data.page_type

    competitor_intents = []
    for page in data.competitor_pages:
        intent = {**classify_page_intent(page, data.primary_keyword), "url": page.url}
        intent.update(competitor_fit_recommendation(effective_page_type, intent))
        competitor_intents.append(intent)

    intent_counts: dict[str, int] = {}
    fit_counts: dict[str, int] = {}
    for item in competitor_intents:
        detected = item["detected_page_type"]
        intent_counts[detected] = intent_counts.get(detected, 0) + 1
        fit = item["fit"]
        fit_counts[fit] = fit_counts.get(fit, 0) + 1

    mismatch = bool(selected and not auto_selected and selected != my_intent["detected_page_type"])
    competitor_majority = max(intent_counts.items(), key=lambda item: item[1])[0] if intent_counts else ""

    return {
        "selected_page_type": data.page_type,
        "effective_page_type": effective_page_type,
        "auto_selected": auto_selected,
        "mismatch": mismatch,
        "my_page": my_intent,
        "competitor_intent_counts": intent_counts,
        "competitor_fit_counts": fit_counts,
        "competitor_majority_page_type": competitor_majority,
        "competitors": competitor_intents,
    }


class HeadingExtractor(HTMLParser):
    """Small HTML parser for SEO metadata, title, H1-H6, and images."""

    def __init__(self) -> None:
        super().__init__()
        self.title = ""
        self.meta_description = ""
        self.canonical = ""
        self.image_count = 0
        self.images_with_alt = 0
        self.headings: dict[str, list[str]] = {
            "h1": [], "h2": [], "h3": [], "h4": [], "h5": [], "h6": []
        }
        self._capture: str | None = None
        self._chunks: list[str] = []

    def handle_starttag(self, tag: str, attrs: list[tuple[str, str | None]]) -> None:
        attr_map = {name.lower(): value or "" for name, value in attrs if name}
        if tag in ("title", "h1", "h2", "h3", "h4", "h5", "h6"):
            self._capture = tag
            self._chunks = []
        elif tag == "meta":
            name = attr_map.get("name", "").casefold()
            prop = attr_map.get("property", "").casefold()
            content = normalize_space(attr_map.get("content", ""))
            if content and (name == "description" or prop == "og:description"):
                if not self.meta_description or name == "description":
                    self.meta_description = content
        elif tag == "link":
            rel = attr_map.get("rel", "").casefold()
            href = normalize_space(attr_map.get("href", ""))
            if href and "canonical" in rel:
                self.canonical = href
        elif tag == "img":
            self.image_count += 1
            if normalize_space(attr_map.get("alt", "")):
                self.images_with_alt += 1

    def handle_endtag(self, tag: str) -> None:
        if tag == self._capture:
            value = normalize_space(" ".join(self._chunks))
            if value:
                if tag == "title":
                    self.title = value
                else:
                    self.headings[tag].append(value)
            self._capture = None
            self._chunks = []

    def handle_data(self, data: str) -> None:
        if self._capture:
            self._chunks.append(data)


def normalize_space(value: str) -> str:
    return re.sub(r"\s+", " ", value or "").strip()


def normalize_text(value: str) -> str:
    return normalize_space(value).casefold()


def tokenize(text: str) -> list[str]:
    return WORD_RE.findall(normalize_text(text))


def word_count(text: str) -> int:
    return len(tokenize(text))


def count_exact_phrase(text: str, term: str) -> int:
    if not term.strip():
        return 0
    pattern = r"(?<![\wÀ-ž])" + re.escape(normalize_text(term)) + r"(?![\wÀ-ž])"
    return len(re.findall(pattern, normalize_text(text), flags=re.UNICODE))


def count_partial_phrase(text: str, term: str) -> int:
    """Count ordered term words with up to 2 words between each term word."""
    terms = tokenize(term)
    if len(terms) <= 1:
        return count_exact_phrase(text, term)
    gap = r"(?:\W+[\wÀ-ž]+){0,2}\W+"
    pattern = r"(?<![\wÀ-ž])" + gap.join(re.escape(t) for t in terms) + r"(?![\wÀ-ž])"
    return len(re.findall(pattern, normalize_text(text), flags=re.UNICODE))


def density(count: int, words: int) -> float:
    if words <= 0:
        return 0.0
    return round(count / words * 100, 3)


def stats(values: list[float]) -> dict[str, float]:
    if not values:
        return {"avg": 0.0, "median": 0.0, "min": 0.0, "max": 0.0}
    return {
        "avg": round(sum(values) / len(values), 3),
        "median": round(statistics.median(values), 3),
        "min": round(min(values), 3),
        "max": round(max(values), 3),
    }


def recommended_range(counts: list[int]) -> tuple[int, int]:
    positive = [c for c in counts if c > 0]
    if not positive:
        return 0, 0
    low = max(1, math.floor(statistics.median(positive) * 0.8))
    high = max(low, math.ceil(max(statistics.median(positive), sum(positive) / len(positive)) * 1.2))
    return low, high


def classify_action(yours: int, low: int, high: int, used_by: int) -> str:
    if used_by == 0:
        return "ignore"
    if low == 0 and high == 0:
        return "ignore"
    if yours == 0:
        return "add section" if low >= 3 else "add"
    if yours < low:
        return "add"
    if yours > max(high, 0):
        return "reduce"
    return "keep"


def build_terms(data: OptimizerInput) -> list[TermSpec]:
    seen: set[str] = set()
    terms: list[TermSpec] = []

    def add(term: str, term_type: str) -> None:
        clean = normalize_space(term)
        key = clean.casefold()
        if clean and key not in seen:
            terms.append(TermSpec(clean, term_type))
            seen.add(key)

    add(data.primary_keyword, "primary")
    for term in data.secondary_keywords:
        add(term, "secondary")
    for term in data.lsi_keywords:
        add(term, "lsi")
    if data.auto_lsi:
        for item in extract_competitor_terms(
            data.competitor_pages,
            data.my_page.text,
            seed_terms=[data.primary_keyword, *data.secondary_keywords, *data.lsi_keywords],
            limit=data.auto_lsi_limit,
        ):
            add(item["term"], "auto_lsi")
    for term in data.entity_terms:
        add(term, "entity")
    return terms


def ngrams(tokens: list[str], size: int) -> list[str]:
    if size <= 0 or len(tokens) < size:
        return []
    return [" ".join(tokens[i:i + size]) for i in range(0, len(tokens) - size + 1)]


def useful_ngram(term: str, seed_terms: list[str]) -> bool:
    words = term.split()
    if not words:
        return False
    if words[0] in DEFAULT_STOPWORDS or words[-1] in DEFAULT_STOPWORDS:
        return False
    if all(word in DEFAULT_STOPWORDS for word in words):
        return False
    if len(words) == 1 and len(words[0]) < 4:
        return False
    seed_text = " ".join(seed_terms).casefold()
    if term in seed_text:
        return False
    return True


def extract_competitor_terms(
    competitor_pages: list[PageText],
    my_text: str = "",
    seed_terms: list[str] | None = None,
    *,
    min_ngram: int = 1,
    max_ngram: int = 5,
    min_competitor_presence: int = 2,
    limit: int = 20,
) -> list[dict[str, Any]]:
    """Extract candidate LSI terms from competitor texts.

    Terms are ranked by competitor presence first, then total frequency, then
    phrase length. Existing seed terms and terms already present in my_text are
    filtered out to keep suggestions focused on gaps.
    """
    if not competitor_pages:
        return []

    seed_terms = seed_terms or []
    min_presence = min(min_competitor_presence, len(competitor_pages))
    my_norm = normalize_text(my_text)
    term_map: dict[str, dict[str, Any]] = {}

    for index, page in enumerate(competitor_pages):
        tokens = tokenize(page.text)
        seen_on_page: set[str] = set()
        for size in range(min_ngram, max_ngram + 1):
            for term in ngrams(tokens, size):
                if not useful_ngram(term, seed_terms):
                    continue
                if count_exact_phrase(my_norm, term) > 0:
                    continue
                if term not in term_map:
                    term_map[term] = {
                        "term": term,
                        "words": size,
                        "total_count": 0,
                        "competitor_indexes": set(),
                    }
                term_map[term]["total_count"] += 1
                seen_on_page.add(term)
        for term in seen_on_page:
            term_map[term]["competitor_indexes"].add(index)

    candidates = []
    for item in term_map.values():
        presence = len(item["competitor_indexes"])
        if presence < min_presence:
            continue
        candidates.append({
            "term": item["term"],
            "words": item["words"],
            "total_count": item["total_count"],
            "competitor_presence": presence,
        })

    candidates.sort(
        key=lambda item: (
            item["competitor_presence"],
            item["total_count"],
            item["words"],
        ),
        reverse=True,
    )
    return candidates[:limit]


def extract_headings_from_html(html: str) -> PageText:
    parser = HeadingExtractor()
    parser.feed(html)
    text = re.sub(r"<[^>]+>", " ", html)
    return PageText(
        text=normalize_space(text),
        title=parser.title,
        meta_description=parser.meta_description,
        canonical=parser.canonical,
        image_count=parser.image_count,
        images_with_alt=parser.images_with_alt,
        h1=parser.headings["h1"],
        h2=parser.headings["h2"],
        h3=parser.headings["h3"],
        h4=parser.headings["h4"],
        h5=parser.headings["h5"],
        h6=parser.headings["h6"],
    )


def extract_markdown_headings(text: str) -> PageText:
    h1: list[str] = []
    h2: list[str] = []
    h3: list[str] = []
    h4: list[str] = []
    h5: list[str] = []
    h6: list[str] = []
    title = ""
    for line in text.splitlines():
        clean = line.strip()
        if clean.startswith("# ") and not clean.startswith("## "):
            value = clean[2:].strip()
            h1.append(value)
            if not title:
                title = value
        elif clean.startswith("## ") and not clean.startswith("### "):
            h2.append(clean[3:].strip())
        elif clean.startswith("### ") and not clean.startswith("#### "):
            h3.append(clean[4:].strip())
        elif clean.startswith("#### ") and not clean.startswith("##### "):
            h4.append(clean[5:].strip())
        elif clean.startswith("##### ") and not clean.startswith("###### "):
            h5.append(clean[6:].strip())
        elif clean.startswith("###### "):
            h6.append(clean[7:].strip())
    return PageText(text=text, title=title, h1=h1, h2=h2, h3=h3, h4=h4, h5=h5, h6=h6)


def keyword_in_any(keyword: str, values: list[str]) -> bool:
    return any(count_exact_phrase(value, keyword) > 0 for value in values)


def analyze_placement(page: PageText, primary_keyword: str) -> dict[str, Any]:
    first_100 = " ".join(tokenize(page.text)[:100])
    return {
        "in_title": count_exact_phrase(page.title, primary_keyword) > 0,
        "in_h1": keyword_in_any(primary_keyword, page.h1),
        "in_h2": keyword_in_any(primary_keyword, page.h2),
        "in_first_100_words": count_exact_phrase(first_100, primary_keyword) > 0,
        "h1_count": len(page.h1),
        "h2_count": len(page.h2),
        "h3_count": len(page.h3),
        "h4_count": len(page.h4),
        "h5_count": len(page.h5),
        "h6_count": len(page.h6),
    }


def term_gap(data: OptimizerInput) -> list[dict[str, Any]]:
    terms = build_terms(data)
    my_words = word_count(data.my_page.text)
    comp_words = [word_count(page.text) for page in data.competitor_pages]
    rows: list[dict[str, Any]] = []

    for spec in terms:
        my_exact = count_exact_phrase(data.my_page.text, spec.term)
        my_partial = count_partial_phrase(data.my_page.text, spec.term)
        comp_exact = [count_exact_phrase(page.text, spec.term) for page in data.competitor_pages]
        comp_partial = [count_partial_phrase(page.text, spec.term) for page in data.competitor_pages]
        comp_density = [
            density(count, words) for count, words in zip(comp_exact, comp_words) if words > 0
        ]
        low, high = recommended_range(comp_exact)
        used_by = sum(1 for count in comp_exact if count > 0)
        action = classify_action(my_exact, low, high, used_by)
        add_count = max(0, low - my_exact) if action in ("add", "add section") else 0

        rows.append({
            "term": spec.term,
            "type": spec.term_type,
            "your_count": my_exact,
            "your_partial_count": my_partial,
            "your_density": density(my_exact, my_words),
            "competitor_counts": comp_exact,
            "competitor_partial_counts": comp_partial,
            "competitor_count_stats": stats([float(c) for c in comp_exact]),
            "competitor_density_stats": stats(comp_density),
            "used_by_competitors": used_by,
            "competitor_total": len(data.competitor_pages),
            "recommended_min": low,
            "recommended_max": high,
            "add_count": add_count,
            "action": action,
        })
    return rows


def word_count_benchmark(data: OptimizerInput) -> dict[str, Any]:
    comp_counts = [word_count(page.text) for page in data.competitor_pages]
    return {
        "your_word_count": word_count(data.my_page.text),
        "competitor_word_counts": comp_counts,
        "competitor_stats": stats([float(c) for c in comp_counts]),
    }


def heading_counts(page: PageText) -> dict[str, int]:
    return {
        "h1": len(page.h1),
        "h2": len(page.h2),
        "h3": len(page.h3),
        "h4": len(page.h4),
        "h5": len(page.h5),
        "h6": len(page.h6),
    }


def heading_benchmark(data: OptimizerInput) -> dict[str, Any]:
    competitor_counts = [heading_counts(page) for page in data.competitor_pages]
    levels = ["h1", "h2", "h3", "h4", "h5", "h6"]
    competitor_stats = {
        level: stats([float(item[level]) for item in competitor_counts])
        for level in levels
    }
    return {
        "your_counts": heading_counts(data.my_page),
        "competitor_counts": competitor_counts,
        "competitor_stats": competitor_stats,
    }


def common_h2_topics(competitor_pages: list[PageText], limit: int = 12) -> list[dict[str, Any]]:
    topic_map: dict[str, dict[str, Any]] = {}
    for page in competitor_pages:
        seen_on_page: set[str] = set()
        for heading in page.h2:
            clean = normalize_space(heading)
            key = clean.casefold()
            if not clean:
                continue
            if key not in topic_map:
                topic_map[key] = {"topic": clean, "count": 0, "present_in": 0}
            topic_map[key]["count"] += 1
            seen_on_page.add(key)
        for key in seen_on_page:
            topic_map[key]["present_in"] += 1

    topics = sorted(
        topic_map.values(),
        key=lambda item: (item["present_in"], item["count"], item["topic"]),
        reverse=True,
    )
    return topics[:limit]


def heading_optimizer(data: OptimizerInput, rows: list[dict[str, Any]]) -> dict[str, Any]:
    placement = analyze_placement(data.my_page, data.primary_keyword)
    benchmark = heading_benchmark(data)
    h2_text = " ".join(data.my_page.h2)
    important_types = {"primary", "secondary", "lsi", "auto_lsi", "entity"}
    missing_h2_terms = []

    for row in rows:
        if row["type"] not in important_types:
            continue
        if row["used_by_competitors"] <= 0:
            continue
        if count_exact_phrase(h2_text, row["term"]) > 0:
            continue
        priority = 1 if row["type"] == "primary" else (2 if row["type"] == "secondary" else 3)
        missing_h2_terms.append({
            "term": row["term"],
            "type": row["type"],
            "used_by_competitors": row["used_by_competitors"],
            "competitor_total": row["competitor_total"],
            "priority": priority,
        })

    missing_h2_terms.sort(
        key=lambda item: (item["priority"], -item["used_by_competitors"], item["term"])
    )

    recommendations = []
    if placement["h1_count"] == 0:
        recommendations.append({
            "priority": 1,
            "type": "h1_missing",
            "message": f"Add exactly one H1 that includes `{data.primary_keyword}`.",
        })
    elif placement["h1_count"] > 1:
        recommendations.append({
            "priority": 1,
            "type": "h1_multiple",
            "message": f"Use exactly one H1; found {placement['h1_count']}.",
        })
    if not placement["in_h1"]:
        recommendations.append({
            "priority": 1,
            "type": "h1_keyword",
            "message": f"Rewrite the H1 so it includes `{data.primary_keyword}` naturally.",
        })
    if not placement["in_h2"]:
        recommendations.append({
            "priority": 2,
            "type": "h2_keyword",
            "message": f"Add `{data.primary_keyword}` or a close variant to at least one H2.",
        })

    h2_median = benchmark["competitor_stats"]["h2"]["median"]
    if h2_median:
        diff = placement["h2_count"] - h2_median
        if abs(diff) > 2:
            direction = "Add more" if diff < 0 else "Reduce"
            recommendations.append({
                "priority": 2,
                "type": "h2_count",
                "message": (
                    f"Your page has {placement['h2_count']} H2 headings; "
                    f"selected competitors use ~{h2_median:.0f}. {direction} H2 structure accordingly."
                ),
            })

    for item in missing_h2_terms[:8]:
        recommendations.append({
            "priority": item["priority"] + 1,
            "type": "h2_term",
            "message": (
                f"Consider using `{item['term']}` in an H2 "
                f"({item['used_by_competitors']}/{item['competitor_total']} competitors use it)."
            ),
        })

    if placement["h3_count"] > max(8, placement["h2_count"] * 3):
        recommendations.append({
            "priority": 4,
            "type": "h3_overuse",
            "message": "H3 usage looks heavy compared with H2 structure; consolidate minor subtopics.",
        })
    lower_heading_count = placement["h4_count"] + placement["h5_count"] + placement["h6_count"]
    if lower_heading_count > max(6, placement["h2_count"] + placement["h3_count"]):
        recommendations.append({
            "priority": 4,
            "type": "deep_heading_overuse",
            "message": "H4-H6 usage is high; check whether the page is over-fragmented.",
        })

    recommendations.sort(key=lambda item: item["priority"])
    return {
        "benchmark": benchmark,
        "common_h2_topics": common_h2_topics(data.competitor_pages),
        "missing_h2_terms": missing_h2_terms[:15],
        "recommendations": recommendations[:15],
    }


def entity_key(name: str) -> str:
    return normalize_text(name)


def entity_salience(entity: dict[str, Any]) -> float:
    value = entity.get("salience", entity.get("avg_salience", 0)) or 0
    try:
        return float(value)
    except (TypeError, ValueError):
        return 0.0


def entity_gap(data: OptimizerInput) -> dict[str, Any]:
    if not data.competitor_entities:
        return {
            "available": False,
            "score": 10,
            "deductions": ["Entity salience scoring requires Google NLP integration."],
            "top_competitor_entities": [],
            "missing_entities": [],
            "underweighted_entities": [],
            "shared_entities": [],
        }

    my_map: dict[str, dict[str, Any]] = {}
    for entity in data.my_entities:
        key = entity_key(entity.get("name", ""))
        if key:
            my_map[key] = entity

    comp_map: dict[str, dict[str, Any]] = {}
    for comp_index, entities in enumerate(data.competitor_entities):
        seen: set[str] = set()
        for entity in entities[:20]:
            name = normalize_space(entity.get("name", ""))
            key = entity_key(name)
            if not key:
                continue
            if key not in comp_map:
                comp_map[key] = {
                    "name": name,
                    "type": entity.get("type", ""),
                    "saliences": [],
                    "competitor_indexes": set(),
                }
            comp_map[key]["saliences"].append(entity_salience(entity))
            seen.add(key)
        for key in seen:
            comp_map[key]["competitor_indexes"].add(comp_index)

    competitor_total = len(data.competitor_entities)
    majority_threshold = 1 if competitor_total == 1 else max(2, competitor_total // 2)
    top_entities = []
    for item in comp_map.values():
        avg_salience = sum(item["saliences"]) / len(item["saliences"]) if item["saliences"] else 0
        present_in = len(item["competitor_indexes"])
        top_entities.append({
            "name": item["name"],
            "type": item["type"],
            "avg_salience": round(avg_salience, 4),
            "present_in": present_in,
            "competitor_total": competitor_total,
        })
    top_entities.sort(key=lambda item: (item["present_in"], item["avg_salience"]), reverse=True)

    missing = []
    underweighted = []
    shared = []
    priority_entities = [
        item for item in top_entities
        if item["present_in"] >= majority_threshold
    ][:15]

    earned = 0.0
    possible = 0.0
    deductions = []
    for item in priority_entities:
        possible += 1.0
        key = entity_key(item["name"])
        my_entity = my_map.get(key)
        my_salience = entity_salience(my_entity) if my_entity else 0.0
        row = {
            **item,
            "your_salience": round(my_salience, 4),
        }
        if not my_entity:
            missing.append(row)
            deductions.append(f"Missing competitor entity `{item['name']}`.")
        elif my_salience < item["avg_salience"] * 0.5:
            earned += 0.5
            underweighted.append(row)
            deductions.append(f"Entity `{item['name']}` salience is below competitor benchmark.")
        else:
            earned += 1.0
            shared.append(row)

    score = round((earned / possible) * 20) if possible else 10
    return {
        "available": True,
        "score": score,
        "deductions": deductions,
        "top_competitor_entities": top_entities[:20],
        "missing_entities": missing,
        "underweighted_entities": underweighted,
        "shared_entities": shared,
    }


def score_term_coverage(rows: list[dict[str, Any]]) -> tuple[int, list[str]]:
    if not rows:
        return 0, ["No terms supplied."]
    weights = {"primary": 3.0, "secondary": 1.5, "lsi": 1.0, "auto_lsi": 0.8, "entity": 1.0}
    earned = 0.0
    possible = 0.0
    deductions: list[str] = []
    for row in rows:
        weight = weights.get(row["type"], 1.0)
        possible += weight
        low = row["recommended_min"]
        high = row["recommended_max"]
        yours = row["your_count"]
        if row["used_by_competitors"] == 0:
            earned += weight
        elif low == 0:
            earned += weight
        elif low <= yours <= max(high, low):
            earned += weight
        elif yours > high:
            earned += weight * 0.75
            deductions.append(f"`{row['term']}` appears above competitor range.")
        elif yours > 0:
            earned += weight * 0.5
            deductions.append(f"`{row['term']}` is below competitor range.")
        else:
            deductions.append(f"`{row['term']}` is missing.")
    return round(earned / possible * 25), deductions


def score_keyword_placement(placement: dict[str, Any]) -> tuple[int, list[str]]:
    checks = [
        ("in_title", 4, "Primary keyword missing from title."),
        ("in_h1", 4, "Primary keyword missing from H1."),
        ("in_first_100_words", 4, "Primary keyword missing from first 100 words."),
        ("in_h2", 3, "Primary keyword missing from H2 headings."),
    ]
    score = 0
    deductions: list[str] = []
    for key, points, message in checks:
        if placement.get(key):
            score += points
        else:
            deductions.append(message)
    return score, deductions


def score_structure(data: OptimizerInput) -> tuple[int, list[str]]:
    placement = analyze_placement(data.my_page, data.primary_keyword)
    comp_h2 = [len(page.h2) for page in data.competitor_pages if page.h2]
    h2_target = round(statistics.median(comp_h2)) if comp_h2 else 5
    h2_count = placement["h2_count"]
    score = 15
    deductions: list[str] = []

    if placement["h1_count"] != 1:
        score -= 4
        deductions.append(f"Expected 1 H1, found {placement['h1_count']}.")
    if h2_count == 0:
        score -= 5
        deductions.append("No H2 headings detected.")
    elif abs(h2_count - h2_target) > 2:
        score -= 3
        deductions.append(f"H2 count ({h2_count}) is far from benchmark target (~{h2_target}).")
    if placement["h3_count"] > max(8, h2_count * 3):
        score -= 2
        deductions.append("H3 usage looks heavy compared with H2 structure.")
    return max(0, score), deductions


def score_word_count(data: OptimizerInput) -> tuple[int, list[str]]:
    bench = word_count_benchmark(data)
    your_count = bench["your_word_count"]
    comp_stats = bench["competitor_stats"]
    target = comp_stats["median"] or comp_stats["avg"]
    if not target:
        return 5, ["No competitor word-count benchmark available."]

    ratio = your_count / target if target else 0
    if 0.85 <= ratio <= 1.2:
        return 10, []
    if 0.65 <= ratio < 0.85 or 1.2 < ratio <= 1.5:
        return 7, [f"Word count {your_count} differs from competitor median {target:.0f}."]
    return 4, [f"Word count {your_count} is far from competitor median {target:.0f}."]


def _float_value(payload: dict[str, Any], key: str, default: float = 0.0) -> float:
    try:
        return float(payload.get(key, default) or default)
    except (TypeError, ValueError):
        return default


def _average_dict_value(items: list[dict[str, Any]], key: str) -> float:
    values = [_float_value(item, key) for item in items if key in item]
    return round(sum(values) / len(values), 3) if values else 0.0


def _median_dict_value(items: list[dict[str, Any]], key: str) -> float:
    values = [_float_value(item, key) for item in items if key in item]
    return round(statistics.median(values), 3) if values else 0.0


def sentiment_readability_gap(data: OptimizerInput) -> dict[str, Any]:
    if not data.my_sentiment and not data.my_readability:
        return {
            "available": False,
            "score": 3,
            "deductions": ["Sentiment/readability scoring requires Google NLP data."],
            "metrics": {},
        }

    my_sentiment = data.my_sentiment or {}
    comp_sentiments = data.competitor_sentiments or []
    my_readability = data.my_readability or {}
    comp_readability = data.competitor_readability or []

    your_score = _float_value(my_sentiment, "score")
    competitor_score = _average_dict_value(comp_sentiments, "score")
    your_magnitude = _float_value(my_sentiment, "magnitude")
    competitor_magnitude = _average_dict_value(comp_sentiments, "magnitude")
    your_negative_pct = _float_value(my_sentiment, "negative_pct")
    competitor_negative_pct = _average_dict_value(comp_sentiments, "negative_pct")
    your_lexical_density = _float_value(my_readability, "lexical_density")
    competitor_lexical_density = _average_dict_value(comp_readability, "lexical_density")

    score = 0
    deductions: list[str] = []

    if comp_sentiments:
        if your_score >= competitor_score - 0.05:
            score += 2
        elif your_score >= 0.15:
            score += 1
            deductions.append("Sentiment is positive, but below selected competitors.")
        else:
            deductions.append("Sentiment is below selected competitor benchmark.")

        if your_negative_pct <= max(competitor_negative_pct + 5, 12):
            score += 2
        else:
            deductions.append("Negative sentence ratio is higher than selected competitors.")

        if your_magnitude >= competitor_magnitude * 0.75 or your_magnitude >= 0.8:
            score += 1
        else:
            deductions.append("Emotional magnitude is low compared with selected competitors.")
    else:
        if your_score >= 0.25:
            score += 2
        elif your_score >= 0:
            score += 1
        else:
            deductions.append("Overall sentiment is negative.")
        if your_negative_pct <= 12:
            score += 2
        else:
            deductions.append("Too many sentences have negative sentiment.")
        if your_magnitude >= 0.8:
            score += 1
        else:
            deductions.append("Emotional magnitude is low.")

    if your_lexical_density:
        if competitor_lexical_density:
            lower = competitor_lexical_density * 0.8
            upper = competitor_lexical_density * 1.2
            if lower <= your_lexical_density <= upper:
                score += 2
            elif 0.35 <= your_lexical_density <= 0.65:
                score += 1
                deductions.append("Lexical density is acceptable, but outside competitor range.")
            else:
                deductions.append("Lexical density is far from selected competitor benchmark.")
        elif 0.35 <= your_lexical_density <= 0.65:
            score += 2
        else:
            deductions.append("Lexical density is outside the practical readability range.")
    else:
        deductions.append("Lexical density is not available.")

    return {
        "available": True,
        "score": min(7, score),
        "deductions": deductions,
        "metrics": {
            "your_sentiment_score": round(your_score, 3),
            "competitor_sentiment_score_avg": competitor_score,
            "your_magnitude": round(your_magnitude, 3),
            "competitor_magnitude_avg": competitor_magnitude,
            "your_negative_pct": round(your_negative_pct, 1),
            "competitor_negative_pct_avg": round(competitor_negative_pct, 1),
            "your_lexical_density": round(your_lexical_density, 3),
            "competitor_lexical_density_avg": competitor_lexical_density,
            "negative_sentences": my_sentiment.get("negative_sentences", []),
        },
    }


def image_alt_gap(data: OptimizerInput) -> dict[str, Any]:
    if not data.my_images:
        return {
            "available": False,
            "score": 0,
            "deductions": ["Image and alt scoring requires URL/HTML extraction."],
            "metrics": {},
        }

    your_images = int(_float_value(data.my_images, "image_count"))
    your_alt_coverage = _float_value(data.my_images, "alt_coverage")
    competitor_median_images = _median_dict_value(data.competitor_images or [], "image_count")
    competitor_alt_coverage = _average_dict_value(data.competitor_images or [], "alt_coverage")

    score = 0
    deductions: list[str] = []

    if your_images > 0:
        score += 1
    else:
        deductions.append("No images detected on the page.")

    if your_alt_coverage >= 0.8:
        score += 1
    elif your_alt_coverage >= 0.5:
        deductions.append("Some images are missing useful alt text.")
    else:
        deductions.append("Most images are missing useful alt text.")

    if competitor_median_images:
        if your_images >= max(1, math.floor(competitor_median_images * 0.5)):
            score += 1
        else:
            deductions.append("Image count is low compared with selected competitors.")
    elif your_images:
        score += 1

    return {
        "available": True,
        "score": min(3, score),
        "deductions": deductions,
        "metrics": {
            "your_image_count": your_images,
            "your_images_with_alt": int(_float_value(data.my_images, "images_with_alt")),
            "your_missing_alt": int(_float_value(data.my_images, "missing_alt")),
            "your_alt_coverage": round(your_alt_coverage, 3),
            "competitor_image_count_median": competitor_median_images,
            "competitor_alt_coverage_avg": competitor_alt_coverage,
        },
    }


def _char_len(value: str) -> int:
    return len(normalize_space(value))


def _title_variants(primary_keyword: str, page_type: str) -> list[str]:
    keyword = normalize_space(primary_keyword)
    if not keyword:
        return []
    normalized_type = normalize_text(page_type)
    if "product" in normalized_type:
        return [
            f"{keyword.capitalize()} - cena, zaloga in specifikacije",
            f"{keyword.capitalize()} za hitro postavitev doma",
            f"Kupi {keyword} za brezskrbno poletje",
        ]
    if "category" in normalized_type:
        return [
            f"{keyword.capitalize()} - izberi pravi model za svoj vrt",
            f"{keyword.capitalize()} za domaco osvezitev",
            f"Najboljsi {keyword} za poletje",
        ]
    if "comparison" in normalized_type:
        return [
            f"{keyword.capitalize()}: primerjava in nasvet za izbiro",
            f"{keyword.capitalize()} - kaj se bolj splaca?",
            f"Primerjava: {keyword}",
        ]
    return [
        f"{keyword.capitalize()}: vodic za pravo izbiro",
        f"Kako izbrati {keyword}",
        f"{keyword.capitalize()} - nasveti, prednosti in izbira",
    ]


def _meta_description_variants(primary_keyword: str, page_type: str) -> list[str]:
    keyword = normalize_space(primary_keyword)
    if not keyword:
        return []
    normalized_type = normalize_text(page_type)
    if "product" in normalized_type:
        return [
            f"Preverite {keyword}, kljucne lastnosti, prednosti in informacije za nakup. Izberite resitev, ki ustreza vasemu prostoru in nacinu uporabe.",
            f"{keyword.capitalize()} so prakticna izbira za hitro osvezitev doma. Oglejte si lastnosti, uporabo in pomembne podrobnosti pred nakupom.",
            f"Izberite {keyword} glede na prostor, uporabo in opremo. Preglejte prednosti, tehnicne podatke in nasvete za pravo odlocitev.",
        ]
    if "category" in normalized_type:
        return [
            f"Odkrijte {keyword} za razlicne potrebe, prostore in proračune. Primerjajte moznosti ter izberite resitev za brezskrbno poletje.",
            f"{keyword.capitalize()} za domaco osvezitev, druzenje in sproscanje. Preverite ponudbo, razlike med modeli in nasvete za izbiro.",
            f"Primerjajte {keyword}, opremo in pomembne lastnosti pred nakupom. Najdite model, ki najbolje ustreza vasemu vrtu.",
        ]
    return [
        f"Preberite, kako izbrati {keyword}, katere lastnosti so pomembne in na kaj morate biti pozorni pred odlocitvijo.",
        f"{keyword.capitalize()} so lahko odlicna izbira, ce poznate razlike, prednosti in kljucne pogoje uporabe. Preverite prakticne nasvete.",
        f"Vodic za {keyword}: primerjava moznosti, najpomembnejse lastnosti in nasveti, ki vam pomagajo izbrati pravo resitev.",
    ]


def meta_optimizer(data: OptimizerInput) -> dict[str, Any]:
    title = normalize_space(data.my_page.title)
    description = normalize_space(data.my_page.meta_description)
    competitor_titles = [normalize_space(page.title) for page in data.competitor_pages if page.title]
    competitor_descriptions = [
        normalize_space(page.meta_description)
        for page in data.competitor_pages
        if page.meta_description
    ]
    title_lengths = [_char_len(item) for item in competitor_titles]
    description_lengths = [_char_len(item) for item in competitor_descriptions]

    checks = []
    score = 0

    def add_check(name: str, ok: bool, message: str, points: int) -> None:
        nonlocal score
        if ok:
            score += points
        checks.append({"check": name, "ok": ok, "message": message, "points": points})

    add_check("title_exists", bool(title), "SEO title is missing.", 2)
    add_check(
        "title_length",
        35 <= _char_len(title) <= 65 if title else False,
        f"SEO title length is {_char_len(title)} chars; practical target is 35-65.",
        2,
    )
    add_check(
        "keyword_in_title",
        count_exact_phrase(title, data.primary_keyword) > 0 if title else False,
        "Primary keyword is missing from SEO title.",
        2,
    )
    add_check("description_exists", bool(description), "Meta description is missing.", 2)
    add_check(
        "description_length",
        80 <= _char_len(description) <= 160 if description else False,
        f"Meta description length is {_char_len(description)} chars; practical target is 80-160.",
        1,
    )
    add_check(
        "keyword_in_description",
        count_exact_phrase(description, data.primary_keyword) > 0 if description else False,
        "Primary keyword is missing from meta description.",
        1,
    )

    deductions = [item["message"] for item in checks if not item["ok"]]
    return {
        "available": bool(title or description or competitor_titles or competitor_descriptions),
        "score": score,
        "max": 10,
        "title": title,
        "meta_description": description,
        "title_length": _char_len(title),
        "meta_description_length": _char_len(description),
        "checks": checks,
        "deductions": deductions,
        "competitor_title_length_stats": stats([float(v) for v in title_lengths]),
        "competitor_description_length_stats": stats([float(v) for v in description_lengths]),
        "competitor_titles": competitor_titles[:10],
        "competitor_descriptions": competitor_descriptions[:10],
        "title_suggestions": _title_variants(data.primary_keyword, data.page_type),
        "meta_description_suggestions": _meta_description_variants(
            data.primary_keyword,
            data.page_type,
        ),
    }


def _term_sentence(term: str, primary_keyword: str, page_type: str) -> str:
    normalized_page_type = normalize_text(page_type)
    if "product" in normalized_page_type:
        return (
            f"Pri izbiri izdelka preverite tudi {term}, saj ta neposredno vpliva "
            f"na udobje uporabe in dolgoročno zadovoljstvo."
        )
    if "category" in normalized_page_type:
        return (
            f"V ponudbi naj bodo jasno razložene tudi možnosti, kot je {term}, "
            f"da obiskovalec hitreje najde pravo rešitev zase."
        )
    if "service" in normalized_page_type or "local" in normalized_page_type:
        return (
            f"Pri načrtovanju storitve je pomemben tudi {term}, ker obiskovalcu "
            f"pomaga razumeti, kaj lahko pričakuje v praksi."
        )
    if primary_keyword and normalize_text(term) != normalize_text(primary_keyword):
        return (
            f"{primary_keyword.capitalize()} so bolj uporabni, ko bralcu jasno razložite "
            f"tudi {term} in njegov vpliv na končno izbiro."
        )
    return f"Dodajte kratek odstavek, ki naravno razloži {term} skozi korist za bralca."


def _entity_sentence(entity: str, primary_keyword: str) -> str:
    if primary_keyword:
        return (
            f"Pri temi {primary_keyword} posebej izpostavite {entity}, ker ta pojem "
            f"pomaga Googlu in bralcu razumeti širši kontekst strani."
        )
    return f"Dodajte stavek ali kratek odstavek, ki jasno pojasni entiteto {entity}."


def _rewrite_negative_sentence(sentence: str, primary_keyword: str) -> str:
    clean = normalize_space(sentence)
    if not clean:
        return "Prepišite negativen stavek v bolj miren, rešitev-usmerjen okvir."
    if primary_keyword:
        return (
            f"Namesto negativnega poudarka uporabite rešitev-usmerjen stavek, npr. "
            f"`{primary_keyword.capitalize()} lahko izbiro poenostavijo, če vnaprej "
            f"preverite ključne lastnosti in pogoje uporabe.`"
        )
    return "Prepišite stavek tako, da najprej pokaže rešitev, nato omejitev."


def auto_optimize_suggestions(
    data: OptimizerInput,
    rows: list[dict[str, Any]],
    entity_data: dict[str, Any],
    sentiment_data: dict[str, Any],
    image_data: dict[str, Any],
    heading_data: dict[str, Any],
) -> list[dict[str, Any]]:
    suggestions: list[dict[str, Any]] = []
    placement = analyze_placement(data.my_page, data.primary_keyword)

    def add(
        priority: int,
        suggestion_type: str,
        target: str,
        action: str,
        example: str = "",
        reason: str = "",
    ) -> None:
        suggestions.append({
            "priority": priority,
            "type": suggestion_type,
            "target": target,
            "action": action,
            "example": example,
            "reason": reason,
        })

    if not placement["in_h1"] or placement["h1_count"] != 1:
        add(
            1,
            "rewrite_h1",
            "H1",
            f"Use exactly one H1 that includes `{data.primary_keyword}`.",
            f"{data.primary_keyword.capitalize()} za pravo izbiro",
            "Primary keyword placement is one of the fastest structural fixes.",
        )

    if not placement["in_h2"]:
        add(
            2,
            "add_h2_keyword",
            "H2",
            f"Add `{data.primary_keyword}` to one H2 or create a new H2 section.",
            f"Kako izbrati {data.primary_keyword}",
            "The primary keyword is missing from H2 headings.",
        )

    for row in rows:
        if row["action"] not in ("add", "add section"):
            continue
        if row["add_count"] <= 0:
            continue
        target = "New subsection" if row["action"] == "add section" else "Existing relevant paragraph"
        add(
            2 if row["type"] in ("primary", "secondary") else 3,
            "add_term",
            target,
            f"Add `{row['term']}` about {row['add_count']}x without keyword stuffing.",
            _term_sentence(row["term"], data.primary_keyword, data.page_type),
            (
                f"Your count is {row['your_count']}; selected competitors suggest "
                f"{row['recommended_min']}-{row['recommended_max']}."
            ),
        )

    for row in rows:
        if row["action"] != "reduce":
            continue
        add(
            3,
            "reduce_term",
            "Overused term",
            f"Reduce exact repetition of `{row['term']}` and use natural variants.",
            f"Replace one repeated `{row['term']}` occurrence with a clearer descriptive phrase.",
            (
                f"Your count is {row['your_count']}; selected competitors suggest "
                f"{row['recommended_min']}-{row['recommended_max']}."
            ),
        )

    for item in entity_data.get("missing_entities", [])[:5]:
        add(
            3,
            "add_entity_context",
            "Entity gap",
            f"Add contextual coverage for `{item['name']}`.",
            _entity_sentence(item["name"], data.primary_keyword),
            f"Entity appears in {item['present_in']}/{item['competitor_total']} selected competitors.",
        )

    for item in entity_data.get("underweighted_entities", [])[:4]:
        add(
            4,
            "strengthen_entity",
            "Entity salience",
            f"Make `{item['name']}` more central in a paragraph or heading.",
            _entity_sentence(item["name"], data.primary_keyword),
            "Entity is present, but salience is below competitor benchmark.",
        )

    for item in heading_data.get("missing_h2_terms", [])[:5]:
        add(
            item.get("priority", 3) + 1,
            "add_h2_term",
            "H2",
            f"Use `{item['term']}` in an H2 if it matches the page intent.",
            f"{item['term'].capitalize()}: kaj morate vedeti",
            f"{item['used_by_competitors']}/{item['competitor_total']} selected competitors use this term.",
        )

    negative_sentences = sentiment_data.get("metrics", {}).get("negative_sentences", [])
    for item in negative_sentences[:3]:
        original = item.get("text", "")
        add(
            4,
            "rewrite_negative_sentence",
            "Sentence sentiment",
            f"Rewrite negative sentence: `{normalize_space(original)}`",
            _rewrite_negative_sentence(original, data.primary_keyword),
            f"Sentence sentiment score: {item.get('score', 0):+.3f}.",
        )

    image_metrics = image_data.get("metrics", {})
    if image_data.get("available") and image_metrics.get("your_missing_alt", 0) > 0:
        add(
            5,
            "add_image_alt",
            "Images",
            f"Add descriptive alt text to {image_metrics.get('your_missing_alt')} image(s).",
            f"{data.primary_keyword} - prikaz izdelka ali uporabe na vrtu",
            "Missing alt text weakens image context and accessibility.",
        )

    seen: set[tuple[str, str, str]] = set()
    unique = []
    for item in sorted(suggestions, key=lambda x: (x["priority"], x["type"], x["target"])):
        key = (item["type"], item["target"], item["action"])
        if key in seen:
            continue
        unique.append(item)
        seen.add(key)
    return unique[:20]


def score_content(
    data: OptimizerInput,
    rows: list[dict[str, Any]],
    entity_data: dict[str, Any] | None = None,
    sentiment_data: dict[str, Any] | None = None,
    image_data: dict[str, Any] | None = None,
) -> dict[str, Any]:
    placement = analyze_placement(data.my_page, data.primary_keyword)
    term_score, term_deductions = score_term_coverage(rows)
    placement_score, placement_deductions = score_keyword_placement(placement)
    structure_score, structure_deductions = score_structure(data)
    word_score, word_deductions = score_word_count(data)

    entity_data = entity_data or entity_gap(data)
    sentiment_data = sentiment_data or sentiment_readability_gap(data)
    image_data = image_data or image_alt_gap(data)

    component_scores = {
        "term_coverage": {"score": term_score, "max": 25, "deductions": term_deductions},
        "entity_coverage": {
            "score": entity_data["score"],
            "max": 20,
            "deductions": entity_data["deductions"],
        },
        "keyword_placement": {
            "score": placement_score,
            "max": 15,
            "deductions": placement_deductions,
        },
        "content_structure": {
            "score": structure_score,
            "max": 15,
            "deductions": structure_deductions,
        },
        "word_count_depth": {"score": word_score, "max": 10, "deductions": word_deductions},
        "sentiment_readability": {
            "score": sentiment_data["score"],
            "max": 7,
            "deductions": sentiment_data["deductions"],
        },
        "internal_links": {
            "score": 5,
            "max": 5,
            "deductions": [],
            "note": "Skipped in this phase.",
        },
        "images_alt": {
            "score": image_data["score"],
            "max": 3,
            "deductions": image_data["deductions"],
        },
    }
    total = sum(v["score"] for v in component_scores.values())
    return {
        "score": int(total),
        "max": 100,
        "components": component_scores,
        "placement": placement,
    }


def top_fixes(
    rows: list[dict[str, Any]],
    content_score: dict[str, Any],
    heading_data: dict[str, Any] | None = None,
    meta_data: dict[str, Any] | None = None,
) -> list[dict[str, Any]]:
    fixes: list[dict[str, Any]] = []
    priority = {"primary": 1, "secondary": 2, "lsi": 3, "entity": 3}
    for row in rows:
        if row["action"] in ("add", "add section"):
            fixes.append({
                "priority": priority.get(row["type"], 4),
                "type": "term_gap",
                "message": (
                    f"Add `{row['term']}` {row['add_count']}x "
                    f"(your {row['your_count']}, target {row['recommended_min']}-{row['recommended_max']})."
                ),
            })
        elif row["action"] == "reduce":
            fixes.append({
                "priority": priority.get(row["type"], 4) + 1,
                "type": "term_overuse",
                "message": (
                    f"Reduce `{row['term']}` usage "
                    f"(your {row['your_count']}, target {row['recommended_min']}-{row['recommended_max']})."
                ),
            })

    for component, data in content_score["components"].items():
        for deduction in data["deductions"]:
            fixes.append({
                "priority": 2 if component in ("keyword_placement", "content_structure") else 5,
                "type": component,
                "message": deduction,
            })

    if heading_data:
        for item in heading_data.get("recommendations", [])[:5]:
            fixes.append({
                "priority": item["priority"],
                "type": f"heading_{item['type']}",
                "message": item["message"],
            })

    if meta_data and meta_data.get("available"):
        for deduction in meta_data.get("deductions", [])[:4]:
            fixes.append({
                "priority": 2,
                "type": "meta_optimizer",
                "message": deduction,
            })

    fixes.sort(key=lambda item: item["priority"])
    return fixes[:10]


def optimize_content(data: OptimizerInput) -> dict[str, Any]:
    intent = classify_intent_set(data)
    if intent.get("auto_selected"):
        data.page_type = intent["effective_page_type"]
    rows = term_gap(data)
    entities = entity_gap(data)
    sentiment = sentiment_readability_gap(data)
    images = image_alt_gap(data)
    meta = meta_optimizer(data)
    score = score_content(data, rows, entities, sentiment, images)
    headings = heading_optimizer(data, rows)
    suggestions = auto_optimize_suggestions(data, rows, entities, sentiment, images, headings)
    return {
        "primary_keyword": data.primary_keyword,
        "language": data.language,
        "page_type": data.page_type,
        "intent_classifier": intent,
        "auto_lsi_enabled": data.auto_lsi,
        "auto_lsi_limit": data.auto_lsi_limit,
        "auto_lsi_terms": extract_competitor_terms(
            data.competitor_pages,
            data.my_page.text,
            seed_terms=[data.primary_keyword, *data.secondary_keywords, *data.lsi_keywords],
            limit=data.auto_lsi_limit,
        ) if data.auto_lsi else [],
        "word_count_benchmark": word_count_benchmark(data),
        "content_score": score,
        "entity_gap": entities,
        "sentiment_readability": sentiment,
        "images_alt": images,
        "meta_optimizer": meta,
        "heading_optimizer": headings,
        "auto_optimize_suggestions": suggestions,
        "term_gap": rows,
        "top_fixes": top_fixes(rows, score, headings, meta),
    }


def build_improvement_plan_markdown(result: dict[str, Any]) -> str:
    """Build a practical Markdown improvement plan from optimizer output."""
    score = result.get("content_score", {})
    word_bench = result.get("word_count_benchmark", {})
    word_stats = word_bench.get("competitor_stats", {})
    lines: list[str] = []

    lines.append(f"# Plan izboljsav: {result.get('primary_keyword', '')}")
    lines.append("")
    lines.append("## Povzetek")
    lines.append("")
    lines.append(f"- Content Score: {score.get('score', 0)}/{score.get('max', 100)}")
    lines.append(f"- Page type: {result.get('page_type', '') or 'ni dolocen'}")
    lines.append(f"- Language: {result.get('language', '') or 'ni dolocen'}")
    lines.append(f"- Tvoja dolzina: {word_bench.get('your_word_count', 0)} besed")
    lines.append(f"- Competitor median: {word_stats.get('median', 0):.0f} besed")
    lines.append("")

    lines.append("## Score po komponentah")
    lines.append("")
    lines.append("| Komponenta | Score | Kaj manjka |")
    lines.append("|---|---:|---|")
    for name, data in score.get("components", {}).items():
        deductions = "; ".join(data.get("deductions", [])[:3]) or "OK"
        lines.append(
            f"| {name.replace('_', ' ').title()} | "
            f"{data.get('score', 0)}/{data.get('max', 0)} | {deductions} |"
        )
    lines.append("")

    top_fixes_data = result.get("top_fixes", [])
    if top_fixes_data:
        lines.append("## Top prioritetne akcije")
        lines.append("")
        for index, item in enumerate(top_fixes_data, 1):
            lines.append(f"{index}. **{item.get('type', '').replace('_', ' ').title()}**: {item.get('message', '')}")
        lines.append("")

    suggestions = result.get("auto_optimize_suggestions", [])
    if suggestions:
        lines.append("## Konkretni popravki")
        lines.append("")
        for item in suggestions:
            lines.append(f"### P{item.get('priority', '')} - {item.get('type', '').replace('_', ' ').title()}")
            lines.append("")
            lines.append(f"- Lokacija: {item.get('target', '')}")
            lines.append(f"- Akcija: {item.get('action', '')}")
            if item.get("example"):
                lines.append(f"- Primer: {item.get('example', '')}")
            if item.get("reason"):
                lines.append(f"- Zakaj: {item.get('reason', '')}")
            lines.append("")

    term_rows = result.get("term_gap", [])
    important_terms = [
        row for row in term_rows
        if row.get("action") in ("add", "add section", "reduce")
    ]
    if important_terms:
        lines.append("## Term Gap")
        lines.append("")
        lines.append("| Akcija | Termin | Tip | Tvoje | Cilj | Dodaj | Konkurenti |")
        lines.append("|---|---|---|---:|---:|---:|---|")
        for row in important_terms[:30]:
            lines.append(
                f"| {row.get('action', '')} | {row.get('term', '')} | {row.get('type', '')} | "
                f"{row.get('your_count', 0)} | {row.get('recommended_min', 0)}-{row.get('recommended_max', 0)} | "
                f"{row.get('add_count', 0)} | {row.get('used_by_competitors', 0)}/{row.get('competitor_total', 0)} |"
            )
        lines.append("")

    entity_gap_data = result.get("entity_gap", {})
    if entity_gap_data.get("available"):
        missing = entity_gap_data.get("missing_entities", [])
        underweighted = entity_gap_data.get("underweighted_entities", [])
        lines.append("## Entity Gap")
        lines.append("")
        lines.append(f"- Entity score: {entity_gap_data.get('score', 0)}/20")
        if missing:
            lines.append("- Manjkajoce entitete: " + ", ".join(item.get("name", "") for item in missing[:10]))
        if underweighted:
            lines.append("- Prenizka salience: " + ", ".join(item.get("name", "") for item in underweighted[:10]))
        lines.append("")

    heading_data = result.get("heading_optimizer", {})
    if heading_data:
        lines.append("## Heading Optimizer")
        lines.append("")
        recs = heading_data.get("recommendations", [])
        if recs:
            for item in recs[:10]:
                lines.append(f"- {item.get('message', '')}")
        missing_h2 = heading_data.get("missing_h2_terms", [])
        if missing_h2:
            lines.append("- Termini za H2: " + ", ".join(item.get("term", "") for item in missing_h2[:10]))
        lines.append("")

    meta_data = result.get("meta_optimizer", {})
    if meta_data.get("available"):
        lines.append("## Meta Title in Meta Description")
        lines.append("")
        lines.append(f"- Meta score: {meta_data.get('score', 0)}/{meta_data.get('max', 10)}")
        lines.append(f"- Title: {meta_data.get('title', '') or 'manjka'}")
        lines.append(f"- Title length: {meta_data.get('title_length', 0)} znakov")
        lines.append(f"- Meta description: {meta_data.get('meta_description', '') or 'manjka'}")
        lines.append(f"- Meta description length: {meta_data.get('meta_description_length', 0)} znakov")
        if meta_data.get("deductions"):
            lines.append("- Popravki: " + "; ".join(meta_data.get("deductions", [])[:5]))
        lines.append("")
        if meta_data.get("title_suggestions"):
            lines.append("Predlogi SEO title:")
            for item in meta_data.get("title_suggestions", [])[:3]:
                lines.append(f"- {item}")
            lines.append("")
        if meta_data.get("meta_description_suggestions"):
            lines.append("Predlogi meta description:")
            for item in meta_data.get("meta_description_suggestions", [])[:3]:
                lines.append(f"- {item}")
            lines.append("")

    sentiment = result.get("sentiment_readability", {})
    if sentiment.get("available"):
        metrics = sentiment.get("metrics", {})
        lines.append("## Sentiment in berljivost")
        lines.append("")
        lines.append(f"- Sentiment: {metrics.get('your_sentiment_score', 0):+.3f} (competitor avg {metrics.get('competitor_sentiment_score_avg', 0):+.3f})")
        lines.append(f"- Negative sentences: {metrics.get('your_negative_pct', 0):.1f}% (competitor avg {metrics.get('competitor_negative_pct_avg', 0):.1f}%)")
        lines.append(f"- Lexical density: {metrics.get('your_lexical_density', 0):.1%} (competitor avg {metrics.get('competitor_lexical_density_avg', 0):.1%})")
        negatives = metrics.get("negative_sentences", [])
        if negatives:
            lines.append("")
            lines.append("Negativni stavki za prepis:")
            for item in negatives[:5]:
                lines.append(f"- \"{normalize_space(item.get('text', ''))}\" (score {item.get('score', 0):+.3f})")
        lines.append("")

    images = result.get("images_alt", {})
    if images.get("available"):
        metrics = images.get("metrics", {})
        lines.append("## Slike in alt text")
        lines.append("")
        lines.append(f"- Slike: {metrics.get('your_image_count', 0):.0f}")
        lines.append(f"- Alt coverage: {metrics.get('your_alt_coverage', 0):.0%}")
        lines.append(f"- Manjka alt: {metrics.get('your_missing_alt', 0):.0f}")
        lines.append("")

    lines.append("## Opomba")
    lines.append("")
    lines.append("Ta plan temelji na izbranih konkurentih in podatkih, ki so bili na voljo ob izracunu. Ne dodajaj terminov mehansko; vsak popravek naj izboljsa razumevanje, strukturo ali uporabnost strani.")
    lines.append("")
    return "\n".join(lines)


def _schema_recommendation(page_type: str) -> str:
    normalized = normalize_text(page_type)
    if "product" in normalized:
        return "Product schema + BreadcrumbList"
    if "category" in normalized:
        return "CollectionPage schema + ItemList + BreadcrumbList"
    if "service" in normalized or "local" in normalized:
        return "Service schema + LocalBusiness + FAQPage, ce so dodana vprasanja"
    if "review" in normalized or "comparison" in normalized:
        return "Article schema + FAQPage + ItemList za primerjave"
    return "Article schema + FAQPage, ce brief vsebuje vprasanja"


def build_content_brief_markdown(result: dict[str, Any]) -> str:
    """Build a strict content brief from optimizer benchmark output."""
    primary_keyword = result.get("primary_keyword", "")
    page_type = result.get("page_type", "") or "ni dolocen"
    language = result.get("language", "") or "ni dolocen"
    word_bench = result.get("word_count_benchmark", {})
    word_stats = word_bench.get("competitor_stats", {})
    median_words = int(word_stats.get("median") or word_stats.get("avg") or word_bench.get("your_word_count", 0) or 0)
    target_low = max(300, int(median_words * 0.9)) if median_words else 0
    target_high = int(median_words * 1.15) if median_words else 0
    heading_data = result.get("heading_optimizer", {})
    heading_benchmark = heading_data.get("benchmark", {})
    h2_stats = heading_benchmark.get("competitor_stats", {}).get("h2", {})
    target_h2 = int(round(h2_stats.get("median") or h2_stats.get("avg") or 5))
    term_rows = result.get("term_gap", [])
    entity_data = result.get("entity_gap", {})
    sentiment = result.get("sentiment_readability", {})
    sentiment_metrics = sentiment.get("metrics", {})
    meta_data = result.get("meta_optimizer", {})

    required_terms = [
        row for row in term_rows
        if row.get("used_by_competitors", 0) > 0 and row.get("recommended_min", 0) > 0
    ]
    required_terms.sort(
        key=lambda row: (
            {"primary": 0, "secondary": 1, "entity": 2, "lsi": 3, "auto_lsi": 4}.get(row.get("type", ""), 9),
            -row.get("used_by_competitors", 0),
            row.get("term", ""),
        )
    )

    h2_topics = []
    for item in heading_data.get("common_h2_topics", []):
        topic = normalize_space(item.get("topic", ""))
        if topic and topic.casefold() not in {value.casefold() for value in h2_topics}:
            h2_topics.append(topic)
    for item in heading_data.get("missing_h2_terms", []):
        term = normalize_space(item.get("term", ""))
        candidate = f"{term.capitalize()}: kaj morate vedeti" if term else ""
        if candidate and candidate.casefold() not in {value.casefold() for value in h2_topics}:
            h2_topics.append(candidate)
    if primary_keyword and not h2_topics:
        h2_topics = [
            f"Kako izbrati {primary_keyword}",
            f"Najpomembnejse prednosti: {primary_keyword}",
            "Pogosta vprasanja",
        ]

    required_entities = []
    if entity_data.get("available"):
        for item in entity_data.get("missing_entities", []) + entity_data.get("underweighted_entities", []):
            name = normalize_space(item.get("name", ""))
            if name and name not in required_entities:
                required_entities.append(name)

    examples = [
        item.get("example", "")
        for item in result.get("auto_optimize_suggestions", [])
        if item.get("example")
    ]

    lines: list[str] = []
    lines.append(f"# Content Brief 2.0: {primary_keyword}")
    lines.append("")
    lines.append("## Osnovni podatki")
    lines.append("")
    lines.append(f"- Primary keyword: {primary_keyword}")
    lines.append(f"- Page type: {page_type}")
    lines.append(f"- Language: {language}")
    if target_low and target_high:
        lines.append(f"- Target word count: {target_low}-{target_high} besed")
    else:
        lines.append("- Target word count: benchmark ni na voljo")
    lines.append(f"- Target H2 count: približno {target_h2}")
    lines.append(f"- Schema recommendation: {_schema_recommendation(page_type)}")
    lines.append("")

    lines.append("## H1")
    lines.append("")
    if primary_keyword:
        lines.append(f"# {primary_keyword.capitalize()} za pravo izbiro")
    else:
        lines.append("# [H1 z glavno kljucno besedo]")
    lines.append("")

    lines.append("## SEO Title in Meta Description")
    lines.append("")
    title_suggestions = meta_data.get("title_suggestions", []) if meta_data else []
    description_suggestions = meta_data.get("meta_description_suggestions", []) if meta_data else []
    if title_suggestions:
        lines.append("SEO title predlogi:")
        for item in title_suggestions[:3]:
            lines.append(f"- {item}")
    else:
        lines.append(f"- SEO title naj vsebuje `{primary_keyword}` in ostane v okvirju 35-65 znakov.")
    lines.append("")
    if description_suggestions:
        lines.append("Meta description predlogi:")
        for item in description_suggestions[:3]:
            lines.append(f"- {item}")
    else:
        lines.append(f"- Meta description naj vsebuje `{primary_keyword}` in ostane v okvirju 80-160 znakov.")
    lines.append("")

    lines.append("## Predlagana H2 struktura")
    lines.append("")
    for index, topic in enumerate(h2_topics[:max(3, target_h2)], 1):
        lines.append(f"{index}. {topic}")
    lines.append("")

    lines.append("## Required Terms")
    lines.append("")
    if required_terms:
        lines.append("| Termin | Tip | Min | Max | Uporablja konkurentov | Opomba |")
        lines.append("|---|---|---:|---:|---|---|")
        for row in required_terms[:35]:
            note = "obvezno" if row.get("type") in ("primary", "secondary") else "podporno"
            lines.append(
                f"| {row.get('term', '')} | {row.get('type', '')} | "
                f"{row.get('recommended_min', 0)} | {row.get('recommended_max', 0)} | "
                f"{row.get('used_by_competitors', 0)}/{row.get('competitor_total', 0)} | {note} |"
            )
    else:
        lines.append("Benchmark za required terms ni na voljo.")
    lines.append("")

    lines.append("## Required Entities")
    lines.append("")
    if required_entities:
        for entity in required_entities[:20]:
            lines.append(f"- {entity}")
    else:
        lines.append("- Entity benchmark ni na voljo ali ni zaznal vrzeli.")
    lines.append("")

    lines.append("## NLP in ton")
    lines.append("")
    if sentiment.get("available"):
        lines.append(f"- Sentiment target: vsaj {sentiment_metrics.get('competitor_sentiment_score_avg', 0):+.3f} ali bolj pozitiven")
        lines.append(f"- Negative sentence target: najvec {sentiment_metrics.get('competitor_negative_pct_avg', 0):.1f}%")
        lines.append(f"- Lexical density target: okoli {sentiment_metrics.get('competitor_lexical_density_avg', 0):.1%}")
    else:
        lines.append("- Sentiment/readability benchmark ni na voljo.")
    lines.append("- Stil: benefit-first, naravno, brez suhega kataloskega nastevanja.")
    lines.append("- Primary keyword naj bo subjekt pomembnih stavkov, ne samo mehansko ponovljen.")
    lines.append("")

    lines.append("## Primeri stavkov za vključitev")
    lines.append("")
    if examples:
        for example in examples[:8]:
            lines.append(f"- {example}")
    elif primary_keyword:
        lines.append(f"- {primary_keyword.capitalize()} naj bralcu najprej pokažejo korist, nato tehnicne lastnosti.")
    else:
        lines.append("- Dodaj konkretne stavke, ki pokrijejo manjkajoce termine in entitete.")
    lines.append("")

    lines.append("## Checklist pred objavo")
    lines.append("")
    lines.append(f"- [ ] H1 vsebuje `{primary_keyword}`")
    lines.append(f"- [ ] Prvih 100 besed vsebuje `{primary_keyword}`")
    lines.append("- [ ] Vsaj en H2 vsebuje primary keyword ali zelo bliznjo varianto")
    lines.append("- [ ] Required terms so uporabljeni v naravnem kontekstu")
    lines.append("- [ ] Required entities so razlozene, ne samo omenjene")
    lines.append("- [ ] Negativni stavki so prepisani v resitev-usmerjen okvir")
    lines.append("- [ ] Slike imajo opisne alt tekste, ce je stran vizualna")
    lines.append("")

    lines.append("## Opomba")
    lines.append("")
    lines.append("Brief temelji na izbranih konkurentih v Content Optimizerju. Ce iz konkurencnega seta odstranis ali dodas URL-je, regeneriraj brief.")
    lines.append("")
    return "\n".join(lines)


def audit_slug(value: str) -> str:
    slug = re.sub(r"[^\wÀ-ž]+", "_", normalize_text(value), flags=re.UNICODE).strip("_")
    return slug or "content_optimizer"


def build_audit_artifacts(result: dict[str, Any], timestamp: str) -> dict[str, str]:
    """Return all saved-report artifacts for one Content Optimizer run."""
    slug = audit_slug(result.get("primary_keyword", "content_optimizer"))
    return {
        f"analiza_{slug}_{timestamp}.json": json.dumps(result, ensure_ascii=False, indent=2),
        f"plan_izboljsav_{slug}_{timestamp}.md": build_improvement_plan_markdown(result),
        f"content_brief_2_0_{slug}_{timestamp}.md": build_content_brief_markdown(result),
    }


def input_from_json(payload: dict[str, Any]) -> OptimizerInput:
    def page_from_value(value: Any) -> PageText:
        if isinstance(value, str):
            return extract_markdown_headings(value)
        if isinstance(value, dict):
            return PageText(
                text=value.get("text", ""),
                title=value.get("title", ""),
                meta_description=value.get("meta_description", ""),
                canonical=value.get("canonical", ""),
                image_count=int(value.get("image_count", 0) or 0),
                images_with_alt=int(value.get("images_with_alt", 0) or 0),
                h1=list(value.get("h1", [])),
                h2=list(value.get("h2", [])),
                h3=list(value.get("h3", [])),
                h4=list(value.get("h4", [])),
                h5=list(value.get("h5", [])),
                h6=list(value.get("h6", [])),
                url=value.get("url", ""),
            )
        raise TypeError("Page value must be a string or object.")

    return OptimizerInput(
        primary_keyword=payload["primary_keyword"],
        my_page=page_from_value(payload["my_text"] if "my_text" in payload else payload["my_page"]),
        competitor_pages=[
            page_from_value(item)
            for item in payload.get("competitor_texts", payload.get("competitor_pages", []))
        ],
        secondary_keywords=list(payload.get("secondary_keywords", [])),
        lsi_keywords=list(payload.get("lsi_keywords", [])),
        entity_terms=list(payload.get("entity_terms", [])),
        auto_lsi=bool(payload.get("auto_lsi", False)),
        auto_lsi_limit=int(payload.get("auto_lsi_limit", 20)),
        my_entities=list(payload.get("my_entities", [])),
        competitor_entities=list(payload.get("competitor_entities", [])),
        my_sentiment=dict(payload.get("my_sentiment", {})),
        competitor_sentiments=list(payload.get("competitor_sentiments", [])),
        my_readability=dict(payload.get("my_readability", {})),
        competitor_readability=list(payload.get("competitor_readability", [])),
        my_images=dict(payload.get("my_images", {})),
        competitor_images=list(payload.get("competitor_images", [])),
        language=payload.get("language", "sl"),
        page_type=payload.get("page_type", ""),
    )


def main() -> None:
    parser = argparse.ArgumentParser(description="Run POP-like content optimization from JSON input.")
    parser.add_argument("--input", required=True, help="Path to JSON input file.")
    parser.add_argument("--pretty", action="store_true", help="Pretty-print JSON output.")
    args = parser.parse_args()

    payload = json.loads(Path(args.input).read_text(encoding="utf-8"))
    result = optimize_content(input_from_json(payload))
    print(json.dumps(result, ensure_ascii=False, indent=2 if args.pretty else None))


if __name__ == "__main__":
    main()
