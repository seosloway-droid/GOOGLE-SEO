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
    raw_source: str = ""
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


def common_prefix_len(left: str, right: str) -> int:
    size = min(len(left), len(right))
    for index in range(size):
        if left[index] != right[index]:
            return index
    return size


def close_token_match(term_token: str, text_token: str) -> bool:
    left = normalize_text(term_token)
    right = normalize_text(text_token)
    if not left or not right:
        return False
    if left == right:
        return True
    prefix = common_prefix_len(left, right)
    shortest = min(len(left), len(right))
    minimum = max(4, shortest - 2)
    return prefix >= minimum


def contains_close_variation(text: str, term: str) -> bool:
    term_tokens = tokenize(term)
    text_tokens = tokenize(text)
    if not term_tokens or not text_tokens:
        return False
    return all(any(close_token_match(term_token, text_token) for text_token in text_tokens) for term_token in term_tokens)


def keyword_in_any_close(keyword: str, values: list[str]) -> bool:
    return any(contains_close_variation(value, keyword) for value in values)


def classify_heading_shape(heading: str) -> str:
    clean = normalize_space(heading)
    if not clean:
        return "phrase"
    normalized = normalize_text(clean)
    question_starts = (
        "zakaj", "kako", "kaj", "kateri", "katera", "katero", "kdaj",
        "kje", "ali", "koliko", "komu", "cigav", "čigav",
    )
    if clean.endswith("?") or any(normalized.startswith(f"{item} ") for item in question_starts):
        return "question"
    if len(tokenize(clean)) >= 5 or any(mark in clean for mark in (":", ".", "!", ",")):
        return "sentence-like"
    return "phrase"


def build_heading_term_specs(data: OptimizerInput) -> list[tuple[str, str]]:
    terms: list[tuple[str, str]] = [(data.primary_keyword, "primary")]
    terms.extend((term, "secondary") for term in data.secondary_keywords)
    terms.extend((term, "lsi") for term in data.lsi_keywords)
    terms.extend((term, "entity") for term in data.entity_terms)
    seen: set[tuple[str, str]] = set()
    ordered: list[tuple[str, str]] = []
    for term, term_type in terms:
        clean = normalize_space(term)
        key = (normalize_text(clean), term_type)
        if clean and key not in seen:
            ordered.append((clean, term_type))
            seen.add(key)
    return ordered


def heading_starts_with_term(heading: str, term: str) -> tuple[bool, bool]:
    heading_tokens = tokenize(heading)
    term_tokens = tokenize(term)
    if not heading_tokens or not term_tokens or len(heading_tokens) < len(term_tokens):
        return False, False

    head_slice = heading_tokens[:len(term_tokens)]
    exact = head_slice == term_tokens
    close = all(close_token_match(term_token, heading_token) for term_token, heading_token in zip(term_tokens, head_slice))
    return exact, close


def analyze_heading_terms(page: PageText, data: OptimizerInput) -> dict[str, Any]:
    levels = ["h1", "h2", "h3", "h4", "h5", "h6"]
    term_specs = build_heading_term_specs(data)
    summary: dict[str, dict[str, Any]] = {}
    details: list[dict[str, Any]] = []

    for level in levels:
        headings = getattr(page, level)
        summary[level] = {
            "count": len(headings),
            "primary_exact": 0,
            "primary_close": 0,
            "lead_primary_exact": 0,
            "lead_primary_close": 0,
            "lead_secondary": 0,
            "lead_lsi": 0,
            "lead_entity": 0,
            "questions": 0,
            "sentence_like": 0,
        }
        for heading in headings:
            clean = normalize_space(heading)
            shape = classify_heading_shape(clean)
            lead_term = ""
            lead_type = "none"
            lead_strength = "none"
            primary_exact = count_exact_phrase(clean, data.primary_keyword) > 0 if data.primary_keyword else False
            primary_close = contains_close_variation(clean, data.primary_keyword) if data.primary_keyword else False
            if primary_exact:
                summary[level]["primary_exact"] += 1
            if primary_close:
                summary[level]["primary_close"] += 1
            if shape == "question":
                summary[level]["questions"] += 1
            if shape == "sentence-like":
                summary[level]["sentence_like"] += 1

            for term, term_type in term_specs:
                starts_exact, starts_close = heading_starts_with_term(clean, term)
                if not starts_close:
                    continue
                lead_term = term
                lead_type = term_type
                lead_strength = "exact" if starts_exact else "close"
                if term_type == "primary":
                    if starts_exact:
                        summary[level]["lead_primary_exact"] += 1
                    summary[level]["lead_primary_close"] += 1
                elif term_type == "secondary":
                    summary[level]["lead_secondary"] += 1
                elif term_type == "lsi":
                    summary[level]["lead_lsi"] += 1
                elif term_type == "entity":
                    summary[level]["lead_entity"] += 1
                break

            details.append({
                "level": level.upper(),
                "heading": clean,
                "primary_exact": primary_exact,
                "primary_close": primary_close,
                "lead_term": lead_term,
                "lead_type": lead_type,
                "lead_strength": lead_strength,
                "shape": shape,
            })

    all_levels = {
        "count": sum(item["count"] for item in summary.values()),
        "primary_exact": sum(item["primary_exact"] for item in summary.values()),
        "primary_close": sum(item["primary_close"] for item in summary.values()),
        "lead_primary_exact": sum(item["lead_primary_exact"] for item in summary.values()),
        "lead_primary_close": sum(item["lead_primary_close"] for item in summary.values()),
        "lead_secondary": sum(item["lead_secondary"] for item in summary.values()),
        "lead_lsi": sum(item["lead_lsi"] for item in summary.values()),
        "lead_entity": sum(item["lead_entity"] for item in summary.values()),
        "questions": sum(item["questions"] for item in summary.values()),
        "sentence_like": sum(item["sentence_like"] for item in summary.values()),
    }
    return {"summary": summary, "all_levels": all_levels, "details": details}


def heading_level_guidance(benchmark: dict[str, Any]) -> list[dict[str, Any]]:
    levels = ["h1", "h2", "h3", "h4", "h5", "h6"]
    your_counts = benchmark.get("your_counts", {})
    comp_stats = benchmark.get("competitor_stats", {})
    guidance: list[dict[str, Any]] = []

    for level in levels:
        your_count = int(your_counts.get(level, 0))
        stats_for_level = comp_stats.get(level, {})
        median = int(round(stats_for_level.get("median", 0) or 0))
        low = int(round(stats_for_level.get("min", 0) or 0))
        high = int(round(stats_for_level.get("max", 0) or 0))
        status = "ok"
        if level == "h1":
            if your_count < 1:
                status = "low"
            elif your_count > 1:
                status = "high"
        elif median or your_count:
            if your_count < low:
                status = "low"
            elif your_count > high:
                status = "high"
        guidance.append({
            "level": level,
            "your_count": your_count,
            "competitor_median": median,
            "competitor_avg": round(stats_for_level.get("avg", 0) or 0, 1),
            "competitor_min": low,
            "competitor_max": high,
            "status": status,
        })
    return guidance


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
        raw_source=html,
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
    return PageText(text=text, raw_source=text, title=title, h1=h1, h2=h2, h3=h3, h4=h4, h5=h5, h6=h6)


def keyword_in_any(keyword: str, values: list[str]) -> bool:
    return any(count_exact_phrase(value, keyword) > 0 for value in values)


def analyze_placement(page: PageText, primary_keyword: str) -> dict[str, Any]:
    first_100 = " ".join(tokenize(page.text)[:100])
    return {
        "in_title": count_exact_phrase(page.title, primary_keyword) > 0,
        "in_h1": keyword_in_any(primary_keyword, page.h1),
        "in_h2": keyword_in_any(primary_keyword, page.h2),
        "in_h1_close": keyword_in_any_close(primary_keyword, page.h1),
        "in_h2_close": keyword_in_any_close(primary_keyword, page.h2),
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


def keyword_density_report(data: OptimizerInput) -> dict[str, Any]:
    terms = build_terms(data)
    own_clean_text = data.my_page.text
    own_raw_text = data.my_page.raw_source or data.my_page.text
    own_clean_words = word_count(own_clean_text)
    own_raw_words = word_count(own_raw_text)

    competitor_clean_words = [word_count(page.text) for page in data.competitor_pages]
    competitor_raw_words = [word_count(page.raw_source or page.text) for page in data.competitor_pages]

    rows: list[dict[str, Any]] = []
    for spec in terms:
        own_clean_count = count_exact_phrase(own_clean_text, spec.term)
        own_raw_count = count_exact_phrase(own_raw_text, spec.term)
        competitor_clean_counts = [count_exact_phrase(page.text, spec.term) for page in data.competitor_pages]
        competitor_raw_counts = [
            count_exact_phrase(page.raw_source or page.text, spec.term) for page in data.competitor_pages
        ]
        competitor_clean_density = [
            density(count, words) for count, words in zip(competitor_clean_counts, competitor_clean_words) if words > 0
        ]
        competitor_raw_density = [
            density(count, words) for count, words in zip(competitor_raw_counts, competitor_raw_words) if words > 0
        ]
        competitor_rows = []
        for index, page in enumerate(data.competitor_pages):
            competitor_rows.append({
                "competitor": page.url or f"Competitor {index + 1}",
                "clean_count": competitor_clean_counts[index],
                "raw_count": competitor_raw_counts[index],
                "clean_density": density(competitor_clean_counts[index], competitor_clean_words[index]),
                "raw_density": density(competitor_raw_counts[index], competitor_raw_words[index]),
                "clean_word_count": competitor_clean_words[index],
                "raw_word_count": competitor_raw_words[index],
            })
        rows.append({
            "term": spec.term,
            "type": spec.term_type,
            "your_clean_count": own_clean_count,
            "your_raw_count": own_raw_count,
            "your_clean_density": density(own_clean_count, own_clean_words),
            "your_raw_density": density(own_raw_count, own_raw_words),
            "your_density_gap": round(density(own_raw_count, own_raw_words) - density(own_clean_count, own_clean_words), 3),
            "competitor_clean_density_stats": stats(competitor_clean_density),
            "competitor_raw_density_stats": stats(competitor_raw_density),
            "competitor_clean_count_stats": stats([float(v) for v in competitor_clean_counts]),
            "competitor_raw_count_stats": stats([float(v) for v in competitor_raw_counts]),
            "competitor_rows": competitor_rows,
        })

    return {
        "available": bool(rows),
        "your_clean_word_count": own_clean_words,
        "your_raw_word_count": own_raw_words,
        "rows": rows,
    }


def build_term_opportunity_table(rows: list[dict[str, Any]]) -> dict[str, Any]:
    missing_terms: list[dict[str, Any]] = []
    underused_terms: list[dict[str, Any]] = []
    overused_terms: list[dict[str, Any]] = []

    type_priority = {"primary": 1, "secondary": 2, "lsi": 3, "auto_lsi": 3, "entity": 4}

    for row in rows:
        item = {
            "term": row["term"],
            "type": row["type"],
            "your_count": row["your_count"],
            "your_partial_count": row["your_partial_count"],
            "recommended_min": row["recommended_min"],
            "recommended_max": row["recommended_max"],
            "gap": max(0, row["recommended_min"] - row["your_count"]),
            "excess": max(0, row["your_count"] - row["recommended_max"]),
            "used_by_competitors": row["used_by_competitors"],
            "competitor_total": row["competitor_total"],
            "competitor_median": row["competitor_count_stats"]["median"],
            "competitor_avg": row["competitor_count_stats"]["avg"],
        }
        if row["action"] in ("add", "add section"):
            if row["your_count"] == 0:
                missing_terms.append(item)
            else:
                underused_terms.append(item)
        elif row["action"] == "reduce":
            overused_terms.append(item)

    def sort_key(item: dict[str, Any]) -> tuple[Any, ...]:
        return (
            type_priority.get(item["type"], 9),
            -item["used_by_competitors"],
            -item.get("gap", 0),
            item["term"],
        )

    missing_terms.sort(key=sort_key)
    underused_terms.sort(key=sort_key)
    overused_terms.sort(
        key=lambda item: (
            type_priority.get(item["type"], 9),
            -item.get("excess", 0),
            -item["used_by_competitors"],
            item["term"],
        )
    )

    return {
        "missing_high_opportunity": missing_terms[:15],
        "underused_terms": underused_terms[:15],
        "overused_terms": overused_terms[:15],
    }


def extract_match_words(data: OptimizerInput, limit: int = 15) -> list[dict[str, Any]]:
    primary_tokens = tokenize(data.primary_keyword)
    if not primary_tokens or not data.competitor_pages:
        return []

    my_text = data.my_page.text
    exact_primary = normalize_text(data.primary_keyword)
    variant_map: dict[str, dict[str, Any]] = {}

    for index, page in enumerate(data.competitor_pages):
        seen_terms: set[str] = set()
        page_tokens = tokenize(page.text)
        for size in range(1, min(4, len(primary_tokens) + 2) + 1):
            for term in ngrams(page_tokens, size):
                normalized_term = normalize_text(term)
                if not term or normalized_term == exact_primary:
                    continue
                term_tokens = tokenize(term)
                if not term_tokens:
                    continue
                primary_hits = 0
                for primary_token in primary_tokens:
                    if any(close_token_match(primary_token, token) for token in term_tokens):
                        primary_hits += 1
                if primary_hits == 0:
                    continue
                if len(primary_tokens) > 1 and primary_hits < max(1, math.ceil(len(primary_tokens) / 2)):
                    continue
                if not useful_ngram(term, [data.primary_keyword]):
                    continue

                if term not in variant_map:
                    variant_map[term] = {
                        "term": term,
                        "words": len(term_tokens),
                        "primary_token_hits": primary_hits,
                        "total_count": 0,
                        "competitor_indexes": set(),
                    }
                variant_map[term]["total_count"] += 1
                seen_terms.add(term)
        for term in seen_terms:
            variant_map[term]["competitor_indexes"].add(index)

    results = []
    min_presence = 1 if len(data.competitor_pages) == 1 else 2
    for item in variant_map.values():
        presence = len(item["competitor_indexes"])
        if presence < min_presence:
            continue
        your_exact = count_exact_phrase(my_text, item["term"])
        your_partial = count_partial_phrase(my_text, item["term"])
        results.append({
            "term": item["term"],
            "words": item["words"],
            "primary_token_hits": item["primary_token_hits"],
            "competitor_presence": presence,
            "competitor_total": len(data.competitor_pages),
            "total_count": item["total_count"],
            "your_exact_count": your_exact,
            "your_partial_count": your_partial,
            "status": "missing" if your_exact == 0 and your_partial == 0 else "present",
        })

    results.sort(
        key=lambda item: (
            item["status"] != "missing",
            -item["competitor_presence"],
            -item["primary_token_hits"],
            -item["total_count"],
            item["term"],
        )
    )
    return results[:limit]


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
    heading_terms = analyze_heading_terms(data.my_page, data)
    level_guidance = heading_level_guidance(benchmark)
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
    if not placement["in_h2_close"]:
        recommendations.append({
            "priority": 2,
            "type": "h2_keyword",
            "message": f"Add `{data.primary_keyword}` or a close variant to at least one H2.",
        })

    all_heading_summary = heading_terms.get("all_levels", {})
    h2_summary = heading_terms["summary"].get("h2", {})
    if all_heading_summary.get("lead_primary_close", 0) == 0:
        recommendations.append({
            "priority": 2,
            "type": "heading_lead_keyword",
            "message": f"Start at least one heading with `{data.primary_keyword}` or a close variation.",
        })
    if h2_summary.get("lead_primary_close", 0) == 0:
        recommendations.append({
            "priority": 2,
            "type": "h2_lead_keyword",
            "message": f"Start at least one H2 with `{data.primary_keyword}` or a close variation.",
        })
    if h2_summary.get("questions", 0) == 0:
        recommendations.append({
            "priority": 3,
            "type": "h2_question",
            "message": "Consider adding one question-style H2 to mirror search intent and FAQs.",
        })

    for item in level_guidance:
        level = item["level"]
        if item["status"] == "ok":
            continue
        if level == "h1":
            continue
        if item["competitor_median"] == 0 and item["your_count"] == 0:
            continue
        priority = 2 if level == "h2" else (3 if level == "h3" else 4)
        if item["status"] == "low":
            message = (
                f"Your page has {item['your_count']} {level.upper()} headings; "
                f"selected competitors are around median {item['competitor_median']} / avg {item['competitor_avg']} "
                f"(range {item['competitor_min']}-{item['competitor_max']}). Add more if the structure feels too thin."
            )
        else:
            message = (
                f"Your page has {item['your_count']} {level.upper()} headings; "
                f"selected competitors are around median {item['competitor_median']} / avg {item['competitor_avg']} "
                f"(range {item['competitor_min']}-{item['competitor_max']}). Reduce them if the structure feels fragmented."
            )
        recommendations.append({
            "priority": priority,
            "type": f"{level}_count",
            "message": message,
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
        "heading_terms": heading_terms,
        "level_guidance": level_guidance,
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

    def add_check(name: str, ok: bool, ok_message: str, fix_message: str, points: int) -> None:
        nonlocal score
        if ok:
            score += points
        checks.append({
            "check": name,
            "ok": ok,
            "message": ok_message if ok else fix_message,
            "points": points,
        })

    keyword_in_title = (
        count_exact_phrase(title, data.primary_keyword) > 0
        or count_partial_phrase(title, data.primary_keyword) > 0
        if title else False
    )
    keyword_in_description = (
        count_exact_phrase(description, data.primary_keyword) > 0
        or count_partial_phrase(description, data.primary_keyword) > 0
        if description else False
    )

    add_check("title_exists", bool(title), "SEO title found.", "SEO title is missing.", 2)
    add_check(
        "title_length",
        35 <= _char_len(title) <= 65 if title else False,
        f"SEO title length is {_char_len(title)} chars; within practical target 35-65.",
        f"SEO title length is {_char_len(title)} chars; practical target is 35-65.",
        2,
    )
    add_check(
        "keyword_in_title",
        keyword_in_title,
        "Primary keyword or close phrase found in SEO title.",
        "Primary keyword is missing from SEO title.",
        2,
    )
    add_check(
        "description_exists",
        bool(description),
        "Meta description found.",
        "Meta description is missing.",
        2,
    )
    add_check(
        "description_length",
        80 <= _char_len(description) <= 160 if description else False,
        f"Meta description length is {_char_len(description)} chars; within practical target 80-160.",
        f"Meta description length is {_char_len(description)} chars; practical target is 80-160.",
        1,
    )
    add_check(
        "keyword_in_description",
        keyword_in_description,
        "Primary keyword or close phrase found in meta description.",
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

    type_order = {
        "rewrite_h1": 0,
        "add_h2_keyword": 1,
        "add_h2_term": 2,
        "reduce_term": 5,
        "add_term": 6,
        "add_entity_context": 7,
        "strengthen_entity": 8,
        "rewrite_negative_sentence": 9,
        "add_image_alt": 10,
    }
    seen: set[tuple[str, str, str]] = set()
    unique = []
    for item in sorted(
        suggestions,
        key=lambda x: (
            type_order.get(x["type"], 99),
            x["priority"],
            x["target"],
            x["action"],
        ),
    ):
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


def roadmap_item(
    item_type: str,
    title: str,
    current: str,
    target: str,
    gap: str,
    impact: str,
    effort: str,
    why: str,
    source: str,
    priority: int,
) -> dict[str, Any]:
    return {
        "type": item_type,
        "title": title,
        "current": current,
        "target": target,
        "gap": gap,
        "impact": impact,
        "effort": effort,
        "why_it_matters": why,
        "source": source,
        "priority": priority,
    }


def build_roadmap_report(
    data: OptimizerInput,
    rows: list[dict[str, Any]],
    content_score: dict[str, Any],
    heading_data: dict[str, Any],
    meta_data: dict[str, Any],
    entity_data: dict[str, Any],
    sentiment_data: dict[str, Any],
    image_data: dict[str, Any],
) -> dict[str, Any]:
    items: list[dict[str, Any]] = []
    placement = content_score.get("placement", {})
    word_bench = word_count_benchmark(data)
    word_stats = word_bench.get("competitor_stats", {})
    your_words = word_bench.get("your_word_count", 0)
    target_words = int(round(word_stats.get("median", 0) or word_stats.get("avg", 0) or 0))
    meta_checks = {item.get("check"): bool(item.get("ok")) for item in meta_data.get("checks", [])}

    type_impact = {"primary": "high", "secondary": "medium", "lsi": "medium", "entity": "medium", "auto_lsi": "medium"}
    type_priority = {"primary": 1, "secondary": 2, "lsi": 3, "entity": 3, "auto_lsi": 3}

    for row in rows:
        if row["action"] not in ("add", "add section", "reduce"):
            continue
        current = str(row["your_count"])
        target = f"{row['recommended_min']}-{row['recommended_max']}"
        if row["action"] == "reduce":
            gap = f"-{max(0, row['your_count'] - row['recommended_max'])}"
            title = f"Reduce `{row['term']}` usage"
            why = f"`{row['term']}` is above the selected competitor range and may dilute focus."
            effort = "easy"
        else:
            gap = f"+{row['add_count']}"
            title = f"Add `{row['term']}`"
            why = (
                f"`{row['term']}` appears in {row['used_by_competitors']}/{row['competitor_total']} selected competitors "
                f"and is below the benchmark range."
            )
            effort = "easy" if row["add_count"] <= 2 else ("medium" if row["add_count"] <= 4 else "hard")
        items.append(roadmap_item(
            "term_gap",
            title,
            current,
            target,
            gap,
            type_impact.get(row["type"], "medium"),
            effort,
            why,
            row["type"],
            type_priority.get(row["type"], 4),
        ))

    if placement.get("h1_count", 0) == 0:
        items.append(roadmap_item(
            "heading_h1_missing",
            "Add one H1",
            "0 H1",
            "1 H1 with primary keyword",
            "+1 H1",
            "high",
            "easy",
            "H1 is the strongest heading signal and should carry the primary topic.",
            "heading",
            1,
        ))
    elif placement.get("h1_count", 0) > 1:
        items.append(roadmap_item(
            "heading_h1_multiple",
            "Consolidate H1 headings",
            f"{placement.get('h1_count', 0)} H1",
            "1 H1 with primary keyword",
            f"-{placement.get('h1_count', 0) - 1} H1",
            "high",
            "easy",
            "Multiple H1 headings weaken hierarchy and make topic focus less clear.",
            "heading",
            1,
        ))
    if not placement.get("in_h1_close"):
        items.append(roadmap_item(
            "heading_h1_keyword",
            "Put the primary keyword in H1",
            "Primary keyword missing in H1",
            "At least 1 H1 with exact or close variation",
            "+1 keyword placement",
            "high",
            "easy",
            "Primary keyword placement in H1 is one of the clearest on-page topic signals.",
            "heading",
            1,
        ))
    if not placement.get("in_h2_close"):
        items.append(roadmap_item(
            "heading_h2_keyword",
            "Add the primary keyword to H2",
            "0 H2 with primary or close variation",
            "At least 1 H2 with primary or close variation",
            "+1 H2",
            "high",
            "easy",
            "Primary keyword variants in H2 headings help reinforce topic coverage.",
            "heading",
            2,
        ))

    heading_terms = heading_data.get("heading_terms", {})
    h2_summary = heading_terms.get("summary", {}).get("h2", {})
    if h2_summary.get("lead_primary_close", 0) == 0:
        items.append(roadmap_item(
            "heading_h2_lead_keyword",
            "Start an H2 with the primary keyword",
            "0 H2 start with primary/close variation",
            "1+ H2 start with primary/close variation",
            "+1 lead H2",
            "high",
            "easy",
            "Lead keywords in headings make topical relevance more explicit and scannable.",
            "heading",
            2,
        ))
    if h2_summary.get("questions", 0) == 0:
        items.append(roadmap_item(
            "heading_h2_question",
            "Add a question-style H2",
            "0 question H2",
            "1+ question H2",
            "+1 question H2",
            "medium",
            "easy",
            "Question headings often map well to search intent, FAQs, and long-tail queries.",
            "heading",
            3,
        ))

    h2_median = int(round(heading_data.get("benchmark", {}).get("competitor_stats", {}).get("h2", {}).get("median", 0) or 0))
    if h2_median and abs(placement.get("h2_count", 0) - h2_median) > 2:
        diff = h2_median - placement.get("h2_count", 0)
        items.append(roadmap_item(
            "heading_h2_count",
            "Align H2 count with competitors",
            f"{placement.get('h2_count', 0)} H2",
            f"About {h2_median} H2",
            f"{diff:+d} H2",
            "medium",
            "medium",
            "Heading depth that is too thin or too fragmented makes the page less comparable to ranking competitors.",
            "heading",
            3,
        ))

    if target_words and not (0.85 <= (your_words / target_words if target_words else 0) <= 1.2):
        diff_words = target_words - your_words
        items.append(roadmap_item(
            "word_count",
            "Align word count with competitors",
            f"{your_words} words",
            f"About {target_words} words",
            f"{diff_words:+d} words",
            "medium",
            "hard" if abs(diff_words) >= 500 else "medium",
            "Large word-count gaps often mean the page is under-covering or over-stretching the topic.",
            "content_structure",
            4,
        ))

    if meta_data.get("available"):
        title_length = meta_data.get("title_length", 0)
        if not meta_checks.get("title_exists"):
            items.append(roadmap_item(
                "meta_title_missing",
                "Add an SEO title",
                "Missing title",
                "1 title with primary keyword",
                "+1 title",
                "high",
                "easy",
                "The title tag is one of the strongest on-page relevance signals.",
                "meta",
                1,
            ))
        elif not meta_checks.get("keyword_in_title"):
            items.append(roadmap_item(
                "meta_title_keyword",
                "Add the primary keyword to the SEO title",
                f"{title_length} chars without primary keyword",
                "Title containing the primary keyword",
                "+1 keyword placement",
                "high",
                "easy",
                "Primary keyword placement in the title helps Google and users identify the page focus immediately.",
                "meta",
                1,
            ))
        if not meta_checks.get("description_exists"):
            items.append(roadmap_item(
                "meta_description_missing",
                "Add a meta description",
                "Missing description",
                "1 description with primary keyword",
                "+1 description",
                "medium",
                "easy",
                "A useful description improves SERP clarity and gives more room for relevant variants.",
                "meta",
                2,
            ))

    missing_entities = entity_data.get("missing_entities", [])[:3]
    for item in missing_entities:
        items.append(roadmap_item(
            "entity_gap",
            f"Add entity `{item['name']}`",
            "Missing entity",
            f"Present with salience ~{item['avg_salience']:.3f}",
            "+1 entity mention",
            "medium",
            "hard" if item["present_in"] >= max(3, item["competitor_total"] - 1) else "medium",
            f"The entity appears in {item['present_in']}/{item['competitor_total']} selected competitors.",
            "entity",
            3,
        ))

    metrics = sentiment_data.get("metrics", {})
    if sentiment_data.get("available") and metrics.get("negative_sentences"):
        items.append(roadmap_item(
            "sentiment_negative_sentences",
            "Rewrite negative sentences",
            f"{len(metrics.get('negative_sentences', []))} negative sentences",
            "Reduce negative framing",
            f"-{len(metrics.get('negative_sentences', []))} negative sentences",
            "medium",
            "medium",
            "Too much negative framing can weaken helpfulness and conversion tone.",
            "sentiment",
            4,
        ))

    image_metrics = image_data.get("metrics", {})
    if image_data.get("available") and image_metrics.get("your_missing_alt", 0) > 0:
        items.append(roadmap_item(
            "images_alt",
            "Add missing alt text",
            f"{image_metrics.get('your_missing_alt', 0)} images without alt",
            "Descriptive alt text on key images",
            f"+{image_metrics.get('your_missing_alt', 0)} alt text",
            "low",
            "easy",
            "Alt text improves image context and fills small semantic gaps.",
            "image",
            5,
        ))

    items.sort(key=lambda item: (item["priority"], {"high": 1, "medium": 2, "low": 3}.get(item["impact"], 4), item["title"]))
    buckets = {"easy": [], "medium": [], "hard": []}
    limits = {"easy": 5, "medium": 5, "hard": 3}
    for item in items:
        bucket = item["effort"]
        if bucket not in buckets:
            bucket = "medium"
        if len(buckets[bucket]) < limits[bucket]:
            buckets[bucket].append(item)

    return {
        "items": items,
        "easy_wins": buckets["easy"],
        "medium_wins": buckets["medium"],
        "hard_wins": buckets["hard"],
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


def _section_status(score: float, maximum: float) -> str:
    if maximum <= 0:
        return "info"
    ratio = score / maximum
    if ratio >= 0.8:
        return "strong"
    if ratio >= 0.55:
        return "needs work"
    return "priority"


def build_grouped_tunings(
    data: OptimizerInput,
    content_score: dict[str, Any],
    meta_data: dict[str, Any],
    heading_data: dict[str, Any],
    entity_data: dict[str, Any],
    sentiment_data: dict[str, Any],
    image_data: dict[str, Any],
    opportunities: dict[str, Any],
    match_words: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    placement = content_score.get("placement", {})
    word_bench = word_count_benchmark(data)
    median_words = word_bench.get("competitor_stats", {}).get("median", 0)

    meta_score = meta_data.get("score", 0)
    meta_max = meta_data.get("max", 10)
    metadata_items = meta_data.get("deductions", [])[:4]

    heading_recs = [item.get("message", "") for item in heading_data.get("recommendations", [])[:4]]
    all_heading_summary = heading_data.get("heading_terms", {}).get("all_levels", {})
    level_guidance = heading_data.get("level_guidance", [])
    heading_items = []
    if placement.get("h1_count", 0) != 1:
        heading_items.append(f"H1 count is {placement.get('h1_count', 0)}; target is 1.")
    heading_items.extend(heading_recs)
    if all_heading_summary:
        heading_items.append(
            f"Lead primary coverage across all headings: {all_heading_summary.get('lead_primary_close', 0)}/{all_heading_summary.get('count', 0)}."
        )
    off_levels = [item for item in level_guidance if item.get("status") != "ok" and item.get("level") != "h1"]
    if off_levels:
        samples = ", ".join(
            f"{item['level'].upper()} {item['your_count']} vs median {item['competitor_median']} / avg {item['competitor_avg']}"
            for item in off_levels[:3]
        )
        heading_items.append(f"Heading count gaps: {samples}.")

    missing_terms = opportunities.get("missing_high_opportunity", [])
    underused_terms = opportunities.get("underused_terms", [])
    overused_terms = opportunities.get("overused_terms", [])
    missing_variants = [item for item in match_words if item.get("status") == "missing"]
    term_items = []
    if missing_terms:
        term_items.append("Missing terms: " + ", ".join(item["term"] for item in missing_terms[:4]))
    if underused_terms:
        term_items.append("Underused terms: " + ", ".join(item["term"] for item in underused_terms[:4]))
    if overused_terms:
        term_items.append("Overused terms: " + ", ".join(item["term"] for item in overused_terms[:4]))
    if missing_variants:
        term_items.append("Missing query variants: " + ", ".join(item["term"] for item in missing_variants[:4]))
    term_score = max(0, 10 - min(10, len(missing_terms) * 2 + len(underused_terms) + len(overused_terms)))

    entity_items = []
    if entity_data.get("missing_entities"):
        entity_items.append("Missing entities: " + ", ".join(item["name"] for item in entity_data.get("missing_entities", [])[:4]))
    if entity_data.get("underweighted_entities"):
        entity_items.append("Low-salience entities: " + ", ".join(item["name"] for item in entity_data.get("underweighted_entities", [])[:4]))

    sentiment_items = sentiment_data.get("deductions", [])[:4]
    image_items = image_data.get("deductions", [])[:4]

    structure_score = 10
    structure_items = []
    if not placement.get("in_first_100_words"):
        structure_score -= 3
        structure_items.append("Primary keyword is missing from the first 100 words.")
    if median_words:
        your_words = word_bench.get("your_word_count", 0)
        if not (0.85 <= (your_words / median_words if median_words else 0) <= 1.2):
            structure_score -= 3
            structure_items.append(f"Word count is {your_words} vs competitor median {median_words:.0f}.")
    if placement.get("h3_count", 0) > max(8, placement.get("h2_count", 0) * 3):
        structure_score -= 2
        structure_items.append("H3 usage is heavy compared with H2 structure.")
    structure_items = structure_items[:4]

    sections = [
        {
            "key": "metadata",
            "label": "Metadata",
            "score": meta_score,
            "max": meta_max,
            "status": _section_status(meta_score, meta_max),
            "highlights": metadata_items or ["Metadata is in a workable range."],
        },
        {
            "key": "headings",
            "label": "Headings",
            "score": max(0, 10 - min(10, len(heading_recs) * 2)),
            "max": 10,
            "status": _section_status(max(0, 10 - min(10, len(heading_recs) * 2)), 10),
            "highlights": heading_items[:5] or ["Heading structure is in a workable range."],
        },
        {
            "key": "terms",
            "label": "Terms",
            "score": term_score,
            "max": 10,
            "status": _section_status(term_score, 10),
            "highlights": term_items[:5] or ["Term coverage looks balanced against selected competitors."],
        },
        {
            "key": "entities",
            "label": "Entities",
            "score": entity_data.get("score", 0),
            "max": 20,
            "status": _section_status(entity_data.get("score", 0), 20),
            "highlights": entity_items[:5] or ["Entity coverage is in a workable range."],
        },
        {
            "key": "sentiment",
            "label": "Sentiment",
            "score": sentiment_data.get("score", 0),
            "max": 7,
            "status": _section_status(sentiment_data.get("score", 0), 7),
            "highlights": sentiment_items or ["Sentiment and readability are in a workable range."],
        },
        {
            "key": "images",
            "label": "Images",
            "score": image_data.get("score", 0),
            "max": 3,
            "status": _section_status(image_data.get("score", 0), 3),
            "highlights": image_items or ["Image and alt coverage are in a workable range."],
        },
        {
            "key": "structure",
            "label": "Structure",
            "score": max(0, structure_score),
            "max": 10,
            "status": _section_status(max(0, structure_score), 10),
            "highlights": structure_items or ["Structure and length are in a workable range."],
        },
    ]
    return sections


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
    density_report = keyword_density_report(data)
    opportunity_table = build_term_opportunity_table(rows)
    match_words = extract_match_words(data)
    suggestions = auto_optimize_suggestions(data, rows, entities, sentiment, images, headings)
    roadmap = build_roadmap_report(data, rows, score, headings, meta, entities, sentiment, images)
    grouped_tunings = build_grouped_tunings(
        data, score, meta, headings, entities, sentiment, images, opportunity_table, match_words
    )
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
        "keyword_density_report": density_report,
        "term_opportunities": opportunity_table,
        "content_score": score,
        "entity_gap": entities,
        "sentiment_readability": sentiment,
        "images_alt": images,
        "meta_optimizer": meta,
        "heading_optimizer": headings,
        "roadmap_report": roadmap,
        "grouped_tunings": grouped_tunings,
        "auto_optimize_suggestions": suggestions,
        "match_words": match_words,
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
    lines.append(f"- Competitor average: {word_stats.get('avg', 0):.0f} besed")
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

    roadmap = result.get("roadmap_report", {})
    if roadmap:
        lines.append("## Roadmap")
        lines.append("")
        for section_name, key in (("Easy Wins", "easy_wins"), ("Medium Wins", "medium_wins"), ("Hard Wins", "hard_wins")):
            rows_data = roadmap.get(key, [])
            if not rows_data:
                continue
            lines.append(f"### {section_name}")
            lines.append("")
            lines.append("| Action | Current | Target | Gap | Impact | Effort | Why |")
            lines.append("|---|---|---|---|---|---|---|")
            for item in rows_data:
                lines.append(
                    f"| {item.get('title', '')} | {item.get('current', '')} | {item.get('target', '')} | "
                    f"{item.get('gap', '')} | {item.get('impact', '')} | {item.get('effort', '')} | "
                    f"{item.get('why_it_matters', '')} |"
                )
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
