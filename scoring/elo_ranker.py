"""
Elo-style ranking for drug hypotheses via pairwise LLM tournament evaluation.
"""
from __future__ import annotations

import json
import logging
import math
import os
from typing import Any

import config
from utils.llm import chat_completion, extract_json_block

logger = logging.getLogger(__name__)

DEBATE_SYSTEM = """You are an impartial scientific judge evaluating drug discovery hypotheses.
Your role is to determine which of two hypotheses is more promising based on:
1. Scientific rigor and mechanistic plausibility
2. Novelty relative to existing therapies
3. Experimental feasibility and translational potential
4. ADMET and drug-likeness considerations
5. Quality of supporting evidence

Be decisive. Always pick a winner. Return only JSON."""

DEBATE_PROMPT = """\
Compare these two drug discovery hypotheses and determine which is more scientifically promising.

## Hypothesis A
Title: {title_a}
Target: {target_a}
Mechanism: {mechanism_a}
Candidate Scaffold: {scaffold_a}
Novelty: {novelty_a}
Composite Score: {score_a:.3f}

## Hypothesis B
Title: {title_b}
Target: {target_b}
Mechanism: {mechanism_b}
Candidate Scaffold: {scaffold_b}
Novelty: {novelty_b}
Composite Score: {score_b:.3f}

Return a JSON object:
{{
  "winner": "A" or "B",
  "margin": "decisive" | "narrow",
  "rationale": "2-3 sentences explaining the decision",
  "key_differentiator": "the single most important factor in the decision"
}}
"""


# ---------------------------------------------------------------------------
# Elo math
# ---------------------------------------------------------------------------

def expected_score(rating_a: float, rating_b: float) -> float:
    return 1.0 / (1.0 + 10 ** ((rating_b - rating_a) / 400))


def update_elo(
    rating_a: float,
    rating_b: float,
    winner: str,  # "A" or "B"
    k: float = config.ELO_K_FACTOR,
) -> tuple[float, float]:
    ea = expected_score(rating_a, rating_b)
    eb = expected_score(rating_b, rating_a)

    sa = 1.0 if winner == "A" else 0.0
    sb = 1.0 - sa

    new_a = rating_a + k * (sa - ea)
    new_b = rating_b + k * (sb - eb)
    return round(new_a, 2), round(new_b, 2)


# ---------------------------------------------------------------------------
# Pairwise debate
# ---------------------------------------------------------------------------

def debate_pair(
    hyp_a: dict[str, Any],
    hyp_b: dict[str, Any],
) -> dict[str, Any]:
    """
    Run a pairwise LLM debate between two hypotheses.
    Returns a result dict with winner, rationale, and updated Elo ratings.
    """
    score_a = hyp_a.get("composite_score", 0.5)
    score_b = hyp_b.get("composite_score", 0.5)

    prompt = DEBATE_PROMPT.format(
        title_a=hyp_a.get("title", "A"),
        target_a=hyp_a.get("target", ""),
        mechanism_a=hyp_a.get("mechanism", ""),
        scaffold_a=hyp_a.get("candidate_scaffold", "N/A"),
        novelty_a=hyp_a.get("novelty_rationale", ""),
        score_a=score_a,
        title_b=hyp_b.get("title", "B"),
        target_b=hyp_b.get("target", ""),
        mechanism_b=hyp_b.get("mechanism", ""),
        scaffold_b=hyp_b.get("candidate_scaffold", "N/A"),
        novelty_b=hyp_b.get("novelty_rationale", ""),
        score_b=score_b,
    )

    messages = [
        {"role": "system", "content": DEBATE_SYSTEM},
        {"role": "user", "content": prompt},
    ]

    try:
        raw = chat_completion(messages, temperature=0.2, max_tokens=400)
        result = extract_json_block(raw)
        if not isinstance(result, dict):
            raise ValueError("Expected dict from debate")
        winner = result.get("winner", "A")
        if winner not in ("A", "B"):
            winner = "A"
    except Exception as exc:
        logger.error("Debate failed: %s. Defaulting to higher scorer.", exc)
        winner = "A" if score_a >= score_b else "B"
        result = {
            "winner": winner,
            "margin": "narrow",
            "rationale": "Automated fallback: higher composite score wins.",
            "key_differentiator": "composite_score",
        }

    # Update Elo
    elo_a = hyp_a.get("elo_rating", config.ELO_DEFAULT_RATING)
    elo_b = hyp_b.get("elo_rating", config.ELO_DEFAULT_RATING)
    new_elo_a, new_elo_b = update_elo(elo_a, elo_b, winner)

    return {
        "winner": winner,
        "margin": result.get("margin", "narrow"),
        "rationale": result.get("rationale", ""),
        "key_differentiator": result.get("key_differentiator", ""),
        "elo_a_before": elo_a,
        "elo_b_before": elo_b,
        "elo_a_after": new_elo_a,
        "elo_b_after": new_elo_b,
        "winner_id": hyp_a["id"] if winner == "A" else hyp_b["id"],
        "loser_id": hyp_b["id"] if winner == "A" else hyp_a["id"],
    }


# ---------------------------------------------------------------------------
# Tournament runner
# ---------------------------------------------------------------------------

def run_tournament(
    hypotheses: list[dict[str, Any]],
    rounds: int | None = None,
    progress_callback=None,
) -> list[dict[str, Any]]:
    """
    Run a round-robin Elo tournament across all hypotheses.
    Updates elo_rating, wins, losses in place. Returns sorted list.
    """
    n = len(hypotheses)
    if n < 2:
        return hypotheses

    # Default: cap at n matches to keep GROQ calls bounded.
    # Full round-robin is n*(n-1)/2 which grows fast (5 hyps = 10 calls).
    all_pairs = [(i, j) for i in range(n) for j in range(i + 1, n)]
    max_rounds = rounds if rounds is not None else n  # n matches by default
    pairs = all_pairs[:max_rounds]

    total = len(pairs)
    for idx, (i, j) in enumerate(pairs):
        hyp_a = hypotheses[i]
        hyp_b = hypotheses[j]

        if progress_callback:
            progress_callback(
                f"Tournament match {idx + 1}/{total}: "
                f'"{hyp_a.get("title","A")[:30]}" vs "{hyp_b.get("title","B")[:30]}"'
            )

        result = debate_pair(hyp_a, hyp_b)

        # Apply Elo updates
        hypotheses[i]["elo_rating"] = result["elo_a_after"]
        hypotheses[j]["elo_rating"] = result["elo_b_after"]
        hypotheses[i]["comparisons"] = hypotheses[i].get("comparisons", 0) + 1
        hypotheses[j]["comparisons"] = hypotheses[j].get("comparisons", 0) + 1

        if result["winner"] == "A":
            hypotheses[i]["wins"] = hypotheses[i].get("wins", 0) + 1
            hypotheses[j]["losses"] = hypotheses[j].get("losses", 0) + 1
        else:
            hypotheses[j]["wins"] = hypotheses[j].get("wins", 0) + 1
            hypotheses[i]["losses"] = hypotheses[i].get("losses", 0) + 1

        logger.info(
            "Match: %s vs %s → Winner: %s | Elo: %.0f → %.0f / %.0f → %.0f",
            hyp_a["id"], hyp_b["id"],
            result["winner_id"],
            result["elo_a_before"], result["elo_a_after"],
            result["elo_b_before"], result["elo_b_after"],
        )

    # Sort by Elo descending
    hypotheses.sort(key=lambda h: h.get("elo_rating", 0), reverse=True)
    return hypotheses


# ---------------------------------------------------------------------------
# Persistence
# ---------------------------------------------------------------------------

def load_elo_ratings() -> dict[str, float]:
    if os.path.exists(config.ELO_FILE):
        with open(config.ELO_FILE) as f:
            return json.load(f)
    return {}


def save_elo_ratings(ratings: dict[str, float]) -> None:
    with open(config.ELO_FILE, "w") as f:
        json.dump(ratings, f, indent=2)


def apply_saved_ratings(hypotheses: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Load persisted Elo ratings and apply them to a hypothesis list."""
    saved = load_elo_ratings()
    for h in hypotheses:
        if h["id"] in saved:
            h["elo_rating"] = saved[h["id"]]
    return hypotheses


def persist_ratings(hypotheses: list[dict[str, Any]]) -> None:
    ratings = {h["id"]: h.get("elo_rating", config.ELO_DEFAULT_RATING) for h in hypotheses}
    save_elo_ratings(ratings)
