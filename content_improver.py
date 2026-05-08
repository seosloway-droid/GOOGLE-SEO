"""
SEO Content Improver
Takes an SEO analysis report + original content and rewrites the content
following the rules and recommendations from the report.

Usage:
    python content_improver.py
"""

import os
import sys
import anthropic


def get_client() -> anthropic.Anthropic:
    key = os.environ.get("ANTHROPIC_API_KEY", "")
    if not key:
        sys.exit("Error: ANTHROPIC_API_KEY environment variable not set.")
    return anthropic.Anthropic(api_key=key)


def collect_multiline(prompt: str) -> str:
    """Collect multiline input until user types END on its own line."""
    print(prompt)
    print("(When done, type END on a new line and press Enter)")
    print("-" * 60)
    lines = []
    while True:
        try:
            line = input()
        except EOFError:
            break
        if line.strip() == "END":
            break
        lines.append(line)
    return "\n".join(lines).strip()


def improve_content(report: str, original: str, language: str, mode: str) -> str:
    client = get_client()

    mode_instruction = {
        "full": (
            "Rewrite the ENTIRE content from scratch following all rules from the report. "
            "Keep the same topics and products but transform the writing completely. "
            "The output should be the full rewritten content, ready to publish."
        ),
        "section": (
            "Improve the content section by section. For each section of the original content, "
            "show: ORIGINAL → IMPROVED. Fix the top 3 most critical issues from the report first."
        ),
        "rules": (
            "Extract the top 10 most important rewriting rules from the report, "
            "then apply ONLY those rules to rewrite the content. "
            "Show the rules first, then the full rewritten content."
        ),
    }[mode]

    lang_instruction = (
        "Write the improved content in Slovenian language."
        if language == "sl"
        else "Write the improved content in English."
    )

    prompt = f"""You are an expert SEO copywriter. You have received a detailed SEO analysis report
and the original content that needs to be improved.

Your task: Rewrite the original content to fix ALL the problems identified in the SEO report.

═══════════════════════════════════════════════
SEO ANALYSIS REPORT (use this as your rules):
═══════════════════════════════════════════════
{report}

═══════════════════════════════════════════════
ORIGINAL CONTENT TO IMPROVE:
═══════════════════════════════════════════════
{original}

═══════════════════════════════════════════════
INSTRUCTIONS:
═══════════════════════════════════════════════
{mode_instruction}

{lang_instruction}

Critical requirements:
- Fix verb deficiency: add action verbs that explain benefits and features
- Add emotional language: write for humans, not robots
- Add descriptive adjectives that differentiate products
- Every product feature must have a "why this matters to the user" explanation
- Target sentiment score: +0.35 to +0.50
- Keep all original product names, brands, and factual information
- Do NOT invent specifications or prices you don't know
- Write naturally, not like AI-generated content
"""

    print("\n⏳ Claude is rewriting your content...\n")

    with client.messages.stream(
        model="claude-sonnet-4-5",
        max_tokens=4000,
        messages=[{"role": "user", "content": prompt}],
    ) as stream:
        full_response = ""
        for text in stream.text_stream:
            print(text, end="", flush=True)
            full_response += text

    return full_response


def save_output(content: str):
    filename = "improved_content.md"
    with open(filename, "w", encoding="utf-8") as f:
        f.write(content)
    print(f"\n\n✅ Saved to: {filename}")


def main():
    print("=" * 60)
    print("  SEO CONTENT IMPROVER — powered by Claude AI")
    print("=" * 60)
    print()

    # Language
    print("Language of output:")
    print("  1. Slovenščina")
    print("  2. English")
    lang_choice = input("Choose (1/2): ").strip()
    language = "sl" if lang_choice == "1" else "en"

    print()

    # Mode
    print("Improvement mode:")
    print("  1. Full rewrite — rewrite entire content from scratch")
    print("  2. Section by section — show Original → Improved for each section")
    print("  3. Rules first — extract rules then rewrite")
    mode_choice = input("Choose (1/2/3): ").strip()
    mode = {"1": "full", "2": "section", "3": "rules"}.get(mode_choice, "full")

    print()

    # Collect SEO report
    report = collect_multiline("\n📋 PASTE YOUR SEO ANALYSIS REPORT:")
    if not report:
        sys.exit("Error: No report provided.")

    print()

    # Collect original content
    original = collect_multiline("\n📄 PASTE YOUR ORIGINAL CONTENT TO IMPROVE:")
    if not original:
        sys.exit("Error: No content provided.")

    print()

    # Run improvement
    improved = improve_content(report, original, language, mode)

    # Save
    print()
    save_choice = input("\n💾 Save output to improved_content.md? (y/n): ").strip().lower()
    if save_choice == "y":
        save_output(improved)

    print("\n✅ Done!")


if __name__ == "__main__":
    main()
