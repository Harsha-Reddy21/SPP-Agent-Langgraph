"""
quality.py — Computes the quality score for a product profile based on Q&A completeness
"""

from models import QAPair, QualityScore, Category


CATEGORY_WEIGHTS: dict[str, float] = {
    Category.SOLUTION:  0.40,
    Category.USER:      0.25,
    Category.TECHNICAL: 0.25,
    Category.GENERAL:   0.10,
}

ANSWER_QUALITY_THRESHOLDS = {
    "excellent": 90.0,
    "good":      70.0,
    "fair":      50.0,
}


def _answer_has_content(answer) -> bool:
    """Return True if the answer is non-null and non-empty."""
    if answer is None:
        return False
    if isinstance(answer, str):
        return answer.strip() != ""
    if isinstance(answer, list):
        return len(answer) > 0
    return True


def compute_quality_score(qa_pairs: list[QAPair]) -> QualityScore:
    """
    Compute an overall quality score and per-category breakdown.

    Scoring logic
    ─────────────
    • Required fields carry full weight; optional fields carry 0.5× weight.
    • Each category is scored independently, then weighted together.
    • Long free-text answers (>20 words) get a small richness bonus (+5 pts).
    """

    # ── per-category buckets ─────────────────────────────────────────────────
    category_data: dict[str, dict] = {
        cat: {"required_answered": 0, "required_total": 0,
              "optional_answered": 0, "optional_total": 0,
              "richness_bonus": 0.0}
        for cat in CATEGORY_WEIGHTS
    }

    required_answered = 0
    required_total    = 0
    optional_answered = 0
    optional_total    = 0

    for qa in qa_pairs:
        if qa.skipped:
            continue

        cat   = qa.category
        has_a = _answer_has_content(qa.answer)

        if qa.required:
            category_data[cat]["required_total"] += 1
            required_total += 1
            if has_a:
                category_data[cat]["required_answered"] += 1
                required_answered += 1
        else:
            category_data[cat]["optional_total"] += 1
            optional_total += 1
            if has_a:
                category_data[cat]["optional_answered"] += 1
                optional_answered += 1

        # richness bonus for text answers with >20 words
        if has_a and isinstance(qa.answer, str) and len(qa.answer.split()) > 20:
            category_data[cat]["richness_bonus"] = min(
                category_data[cat]["richness_bonus"] + 2.0, 5.0
            )

    # ── category scores ──────────────────────────────────────────────────────
    category_scores: dict[str, float] = {}
    for cat, data in category_data.items():
        req_score  = (data["required_answered"] / data["required_total"] * 100
                      if data["required_total"] else 100.0)
        opt_score  = (data["optional_answered"] / data["optional_total"] * 100
                      if data["optional_total"] else 100.0)

        raw = req_score * 0.7 + opt_score * 0.3 + data["richness_bonus"]
        category_scores[cat] = round(min(raw, 100.0), 1)

    # ── overall score ────────────────────────────────────────────────────────
    weighted = sum(
        category_scores[cat] * weight
        for cat, weight in CATEGORY_WEIGHTS.items()
    )
    overall = round(min(weighted, 100.0), 1)

    # ── completion percentages ───────────────────────────────────────────────
    req_completion  = round(required_answered / required_total * 100, 1)   if required_total  else 100.0
    opt_completion  = round(optional_answered / optional_total * 100, 1)   if optional_total  else 100.0

    # ── grade ────────────────────────────────────────────────────────────────
    if overall >= ANSWER_QUALITY_THRESHOLDS["excellent"]:
        grade = "Excellent"
    elif overall >= ANSWER_QUALITY_THRESHOLDS["good"]:
        grade = "Good"
    elif overall >= ANSWER_QUALITY_THRESHOLDS["fair"]:
        grade = "Fair"
    else:
        grade = "Poor"

    # ── improvement suggestions ──────────────────────────────────────────────
    suggestions: list[str] = []
    for cat, score in category_scores.items():
        if score < 70:
            suggestions.append(
                f"Complete more {cat.capitalize()} section fields to improve your score."
            )
    if opt_completion < 50:
        suggestions.append(
            "Filling in optional fields significantly improves profile quality."
        )
    if not suggestions:
        suggestions.append("Great job! Your profile is well-filled. Review and submit.")

    return QualityScore(
        overall=overall,
        required_completion=req_completion,
        optional_completion=opt_completion,
        category_scores=category_scores,
        grade=grade,
        suggestions=suggestions,
    )
