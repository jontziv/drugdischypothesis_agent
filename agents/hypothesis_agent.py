"""
Hypothesis Generation Agent
Uses RAG over indexed literature and molecules to propose structured drug hypotheses.
"""
from __future__ import annotations

import json
import logging
import uuid
from datetime import datetime
from typing import Any

import config
from utils.llm import chat_completion, extract_json_block
from vector_store.store import retrieve_relevant_papers, retrieve_relevant_molecules

logger = logging.getLogger(__name__)

SYSTEM_PROMPT = """You are an expert computational drug discovery scientist with deep knowledge of:
- Medicinal chemistry and pharmacology
- Molecular biology and disease mechanisms
- Structure-activity relationships (SAR)
- ADMET properties and drug-likeness
- Clinical trial design and translational medicine

Your role is to generate scientifically rigorous, novel, and experimentally actionable drug discovery hypotheses.
Each hypothesis must be grounded in the provided evidence and mechanistically plausible.
"""

HYPOTHESIS_TEMPLATE = """\
Generate {n} distinct, novel drug discovery hypotheses for the following research context.

## Research Context
- **Disease/Condition**: {disease}
- **Primary Target of Interest**: {target}

## Retrieved Literature Evidence
{literature_context}

## Known Active Molecules (from ChEMBL)
{molecule_context}

## Instructions
For each hypothesis, provide a JSON object with these exact fields:
- "id": a unique short identifier (e.g., "HYP-001")
- "title": a concise hypothesis title (max 15 words)
- "target": the specific molecular target or pathway
- "mechanism": the proposed mechanism of action (2-3 sentences)
- "candidate_scaffold": a proposed chemical scaffold or molecule name/SMILES
- "novelty_rationale": why this is novel vs. existing approaches (2-3 sentences)
- "supporting_evidence": list of 2-4 key evidence points from the literature provided
- "proposed_experiment": the key experiment to validate this hypothesis
- "confidence_rationale": brief reasoning about confidence level
- "risks": main scientific or translational risks

Return a JSON array of {n} hypothesis objects. No prose outside the JSON block.
"""


def _format_literature_context(papers: list[dict]) -> str:
    if not papers:
        return "No relevant literature retrieved."
    lines = []
    for i, p in enumerate(papers[:6], 1):
        lines.append(
            f"[{i}] \"{p.get('title', 'N/A')}\" "
            f"({p.get('journal', 'N/A')}, {p.get('year', 'N/A')}) "
            f"[Relevance: {p.get('relevance_score', 0):.2f}]\n"
            f"    Excerpt: {p.get('document', '')[:300]}..."
        )
    return "\n\n".join(lines)


def _format_molecule_context(molecules: list[dict]) -> str:
    if not molecules:
        return "No relevant molecules retrieved."
    lines = []
    for m in molecules[:8]:
        name = m.get("molecule_name", m.get("chembl_id", "Unknown"))
        smiles = m.get("smiles", "N/A")
        act = m.get("activity_type", "")
        val = m.get("activity_value", "")
        units = m.get("activity_units", "nM") if val else ""
        lines.append(f"• {name} | SMILES: {smiles[:60]} | {act}: {val} {units}")
    return "\n".join(lines)


def generate_hypotheses(
    disease: str,
    target: str,
    n_hypotheses: int = 5,
) -> list[dict[str, Any]]:
    """
    Generate drug discovery hypotheses via RAG + Llama 3.3.
    Returns a list of structured hypothesis dicts.
    """
    query = f"{disease} {target} drug treatment mechanism"

    papers = retrieve_relevant_papers(query, n_results=8)
    molecules = retrieve_relevant_molecules(query, n_results=10)

    lit_context = _format_literature_context(papers)
    mol_context = _format_molecule_context(molecules)

    prompt = HYPOTHESIS_TEMPLATE.format(
        n=n_hypotheses,
        disease=disease,
        target=target,
        literature_context=lit_context,
        molecule_context=mol_context,
    )

    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": prompt},
    ]

    logger.info("Generating %d hypotheses for %s / %s...", n_hypotheses, disease, target)
    raw = chat_completion(messages, temperature=0.5, max_tokens=4096)

    try:
        hypotheses_raw = extract_json_block(raw)
        if not isinstance(hypotheses_raw, list):
            hypotheses_raw = [hypotheses_raw]
    except ValueError as exc:
        logger.error("Failed to parse hypotheses JSON: %s", exc)
        hypotheses_raw = []

    # Enrich with metadata
    timestamp = datetime.utcnow().isoformat()
    hypotheses: list[dict[str, Any]] = []
    for i, h in enumerate(hypotheses_raw):
        if not isinstance(h, dict):
            continue
        h.setdefault("id", f"HYP-{uuid.uuid4().hex[:6].upper()}")
        h["disease"] = disease
        h["target_query"] = target
        h["created_at"] = timestamp
        h["elo_rating"] = config.ELO_DEFAULT_RATING
        h["wins"] = 0
        h["losses"] = 0
        h["comparisons"] = 0
        h["supporting_papers"] = [
            {"title": p.get("title"), "url": p.get("url"), "year": p.get("year")}
            for p in papers[:4]
        ]
        h["source_molecules"] = [
            m.get("molecule_name", m.get("chembl_id")) for m in molecules[:4]
        ]
        hypotheses.append(h)

    logger.info("Generated %d valid hypotheses.", len(hypotheses))
    return hypotheses


def critique_hypothesis(hypothesis: dict[str, Any]) -> str:
    """
    Generate a scientific critique of a single hypothesis.
    Used as part of the debate/tournament evaluation.
    """
    prompt = f"""\
Critically evaluate the following drug discovery hypothesis from the perspective of a skeptical expert reviewer.

## Hypothesis
Title: {hypothesis.get('title', '')}
Target: {hypothesis.get('target', '')}
Mechanism: {hypothesis.get('mechanism', '')}
Candidate Scaffold: {hypothesis.get('candidate_scaffold', '')}
Novelty Rationale: {hypothesis.get('novelty_rationale', '')}
Risks: {hypothesis.get('risks', '')}

Provide a structured critique covering:
1. Scientific plausibility (1-2 sentences)
2. Novelty assessment (1-2 sentences)
3. Experimental feasibility (1-2 sentences)
4. Major weaknesses or gaps (2-3 points)
5. Overall verdict: STRONG / MODERATE / WEAK

Keep your response concise and scientific. Do not use JSON format.
"""
    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": prompt},
    ]
    return chat_completion(messages, temperature=0.3, max_tokens=600)
