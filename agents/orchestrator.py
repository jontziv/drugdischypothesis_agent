"""
Multi-Agent Orchestrator
Coordinates all agents in the drug discovery pipeline:
  1. IngestionAgent  → fetches literature + molecules
  2. VectorStore     → embeds and indexes content
  3. HypothesisAgent → generates RAG-grounded hypotheses
  4. ScoringAgent    → scores each hypothesis (RDKit + LLM)
  5. EloRanker       → runs tournament, ranks by Elo
  6. FeedbackLoop    → incorporates experimental feedback
"""
from __future__ import annotations

import json
import logging
import os
from datetime import datetime
from typing import Any, Callable

import config
from agents.ingestion_agent import ingest_knowledge
from agents.hypothesis_agent import generate_hypotheses, critique_hypothesis
from scoring.molecular_scorer import llm_score_hypothesis
from scoring.elo_ranker import run_tournament, apply_saved_ratings, persist_ratings
from vector_store.store import index_papers, index_molecules, get_collection_stats

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Hypothesis persistence
# ---------------------------------------------------------------------------

def load_hypotheses() -> list[dict[str, Any]]:
    if os.path.exists(config.HYPOTHESES_FILE):
        with open(config.HYPOTHESES_FILE) as f:
            return json.load(f)
    return []


def save_hypotheses(hypotheses: list[dict[str, Any]]) -> None:
    with open(config.HYPOTHESES_FILE, "w") as f:
        json.dump(hypotheses, f, indent=2)


def load_feedback_log() -> list[dict[str, Any]]:
    if os.path.exists(config.FEEDBACK_FILE):
        with open(config.FEEDBACK_FILE) as f:
            return json.load(f)
    return []


def save_feedback_log(log: list[dict[str, Any]]) -> None:
    with open(config.FEEDBACK_FILE, "w") as f:
        json.dump(log, f, indent=2)


# ---------------------------------------------------------------------------
# Main pipeline
# ---------------------------------------------------------------------------

def run_full_pipeline(
    disease: str,
    target: str,
    n_hypotheses: int = 5,
    run_tournament_flag: bool = True,
    progress_callback: Callable[[str], None] | None = None,
    pubmed_max: int = config.PUBMED_MAX_RESULTS,
    chembl_max: int = config.CHEMBL_MAX_RESULTS,
) -> list[dict[str, Any]]:
    """
    End-to-end drug discovery hypothesis pipeline.
    Returns a ranked list of scored hypotheses.
    """
    def _cb(msg: str) -> None:
        logger.info(msg)
        if progress_callback:
            progress_callback(msg)

    # -----------------------------------------------------------------------
    # Step 1: Knowledge ingestion
    # -----------------------------------------------------------------------
    _cb("Step 1/5: Ingesting biomedical knowledge...")
    knowledge = ingest_knowledge(
        disease_query=disease,
        target_name=target,
        pubmed_max=pubmed_max,
        chembl_max=chembl_max,
        progress_callback=_cb,
    )

    # -----------------------------------------------------------------------
    # Step 2: Index into vector store
    # -----------------------------------------------------------------------
    _cb("Step 2/5: Embedding and indexing knowledge...")
    new_papers = index_papers(knowledge["papers"])
    new_mols = index_molecules(knowledge["molecules"])
    stats = get_collection_stats()
    _cb(
        f"Vector store: {stats['papers']} papers, {stats['molecules']} molecules indexed "
        f"({new_papers} new papers, {new_mols} new molecules this run)."
    )

    # -----------------------------------------------------------------------
    # Step 3: Hypothesis generation
    # -----------------------------------------------------------------------
    _cb(f"Step 3/5: Generating {n_hypotheses} drug discovery hypotheses...")
    hypotheses = generate_hypotheses(disease=disease, target=target, n_hypotheses=n_hypotheses)
    _cb(f"Generated {len(hypotheses)} hypotheses.")

    if not hypotheses:
        _cb("No hypotheses generated. Check API connectivity and GROQ key.")
        return []

    # -----------------------------------------------------------------------
    # Step 4: In silico scoring
    # -----------------------------------------------------------------------
    _cb("Step 4/5: Running in silico scoring (RDKit + LLM)...")
    for i, hyp in enumerate(hypotheses):
        _cb(f"  Scoring hypothesis {i + 1}/{len(hypotheses)}: {hyp.get('title', '')[:50]}")
        scores = llm_score_hypothesis(hyp)
        hyp["rdkit_scores"] = scores.get("rdkit_scores", {})
        hyp["llm_scores"] = scores.get("llm_scores", {})
        hyp["composite_score"] = scores.get("composite_score", 0.5)

    # Restore any previously saved Elo ratings
    hypotheses = apply_saved_ratings(hypotheses)

    # -----------------------------------------------------------------------
    # Step 5: Tournament ranking
    # -----------------------------------------------------------------------
    if run_tournament_flag and len(hypotheses) >= 2:
        _cb("Step 5/5: Running Elo tournament...")
        hypotheses = run_tournament(hypotheses, progress_callback=_cb)
        persist_ratings(hypotheses)
    else:
        _cb("Step 5/5: Skipping tournament (fewer than 2 hypotheses or disabled).")
        hypotheses.sort(key=lambda h: h.get("composite_score", 0), reverse=True)

    # -----------------------------------------------------------------------
    # Persist and return
    # -----------------------------------------------------------------------
    existing = load_hypotheses()
    existing_ids = {h["id"] for h in existing}
    new_hyps = [h for h in hypotheses if h["id"] not in existing_ids]
    all_hypotheses = existing + new_hyps

    # Update Elo of existing hypotheses from latest tournament
    id_map = {h["id"]: h for h in hypotheses}
    for h in all_hypotheses:
        if h["id"] in id_map:
            h["elo_rating"] = id_map[h["id"]]["elo_rating"]

    all_hypotheses.sort(key=lambda h: h.get("elo_rating", 0), reverse=True)
    save_hypotheses(all_hypotheses)

    _cb(f"Pipeline complete. {len(hypotheses)} new hypotheses ranked and saved.")
    return hypotheses


# ---------------------------------------------------------------------------
# Feedback loop
# ---------------------------------------------------------------------------

def submit_experimental_feedback(
    hypothesis_id: str,
    outcome: str,  # "positive" | "negative" | "partial"
    notes: str,
    elo_adjustment: float = 0.0,
) -> bool:
    """
    Record experimental feedback for a hypothesis and adjust its Elo rating.
    """
    hypotheses = load_hypotheses()
    hyp = next((h for h in hypotheses if h["id"] == hypothesis_id), None)
    if hyp is None:
        logger.error("Hypothesis %s not found.", hypothesis_id)
        return False

    # Elo adjustment based on outcome
    if outcome == "positive":
        elo_delta = 50 + elo_adjustment
    elif outcome == "negative":
        elo_delta = -50 + elo_adjustment
    else:  # partial
        elo_delta = 10 + elo_adjustment

    hyp["elo_rating"] = round(hyp.get("elo_rating", config.ELO_DEFAULT_RATING) + elo_delta, 2)

    # Log feedback
    feedback_entry = {
        "hypothesis_id": hypothesis_id,
        "hypothesis_title": hyp.get("title", ""),
        "outcome": outcome,
        "notes": notes,
        "elo_delta": elo_delta,
        "elo_after": hyp["elo_rating"],
        "timestamp": datetime.utcnow().isoformat(),
    }

    log = load_feedback_log()
    log.append(feedback_entry)
    save_feedback_log(log)

    hypotheses.sort(key=lambda h: h.get("elo_rating", 0), reverse=True)
    save_hypotheses(hypotheses)
    persist_ratings(hypotheses)

    logger.info(
        "Feedback recorded for %s: %s (Elo %+.0f → %.0f)",
        hypothesis_id, outcome, elo_delta, hyp["elo_rating"],
    )
    return True


# ---------------------------------------------------------------------------
# Re-rank only (no new generation)
# ---------------------------------------------------------------------------

def rerank_existing(progress_callback=None) -> list[dict[str, Any]]:
    """Re-run tournament on all saved hypotheses without regenerating them."""
    hypotheses = load_hypotheses()
    if len(hypotheses) < 2:
        return hypotheses
    hypotheses = run_tournament(hypotheses, progress_callback=progress_callback)
    persist_ratings(hypotheses)
    save_hypotheses(hypotheses)
    return hypotheses
