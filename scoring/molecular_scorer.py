"""
Molecular scoring using RDKit for physicochemical properties (Lipinski, ADMET proxies)
combined with LLM-based reasoning for binding likelihood and synthesizability.
"""
from __future__ import annotations

import logging
import re
from typing import Any

from utils.llm import chat_completion, extract_json_block

logger = logging.getLogger(__name__)

# Try importing RDKit; degrade gracefully if not installed
try:
    from rdkit import Chem
    from rdkit.Chem import Descriptors, QED, rdMolDescriptors
    from rdkit.Chem.Crippen import MolLogP
    RDKIT_AVAILABLE = True
except ImportError:
    RDKIT_AVAILABLE = False
    logger.warning("RDKit not available. Physicochemical scoring will be skipped.")


# ---------------------------------------------------------------------------
# RDKit physicochemical scoring
# ---------------------------------------------------------------------------

def _clean_smiles(smiles: str) -> str:
    """Strip LLM-added prefixes like 'SMILES:', 'Scaffold:', descriptive text, etc."""
    if not smiles:
        return ""
    # Remove common LLM prefix patterns
    smiles = re.sub(r"(?i)^(smiles|scaffold|structure|compound|molecule)\s*:\s*", "", smiles.strip())
    # Take only the first token if multiple space-separated values
    smiles = smiles.split()[0] if smiles.split() else smiles
    return smiles.strip()


def compute_rdkit_scores(smiles: str) -> dict[str, Any]:
    """
    Compute Lipinski Ro5 + QED + basic ADMET proxies from SMILES.
    Returns a dict of scores, or empty dict if RDKit unavailable / invalid SMILES.
    """
    smiles = _clean_smiles(smiles)
    if not RDKIT_AVAILABLE or not smiles:
        return {}

    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return {"valid_smiles": False}

    mw = Descriptors.ExactMolWt(mol)
    logp = MolLogP(mol)
    hbd = rdMolDescriptors.CalcNumHBD(mol)
    hba = rdMolDescriptors.CalcNumHBA(mol)
    tpsa = Descriptors.TPSA(mol)
    rot_bonds = rdMolDescriptors.CalcNumRotatableBonds(mol)
    rings = rdMolDescriptors.CalcNumRings(mol)
    qed_score = QED.qed(mol)
    n_atoms = mol.GetNumHeavyAtoms()

    # Lipinski Rule of Five
    ro5_violations = sum([
        mw > 500,
        logp > 5,
        hbd > 5,
        hba > 10,
    ])

    # Veber rules for oral bioavailability
    veber_ok = rot_bonds <= 10 and tpsa <= 140

    # Simple synthesizability proxy (penalize high MW and complexity)
    synth_score = max(0.0, 1.0 - (n_atoms / 80) - (rings / 10) - (ro5_violations * 0.15))

    return {
        "valid_smiles": True,
        "molecular_weight": round(mw, 2),
        "logp": round(logp, 2),
        "hbd": hbd,
        "hba": hba,
        "tpsa": round(tpsa, 2),
        "rotatable_bonds": rot_bonds,
        "num_rings": rings,
        "qed": round(qed_score, 3),
        "ro5_violations": ro5_violations,
        "lipinski_pass": ro5_violations <= 1,
        "veber_oral_ok": veber_ok,
        "synth_score": round(synth_score, 3),
        "drug_likeness_score": round(qed_score, 3),
    }


def physicochemical_summary(scores: dict[str, Any]) -> str:
    """Format RDKit scores as a human-readable summary string."""
    if not scores or not scores.get("valid_smiles", True):
        return "SMILES invalid or RDKit unavailable."
    return (
        f"MW={scores.get('molecular_weight', '?')} | "
        f"LogP={scores.get('logp', '?')} | "
        f"QED={scores.get('qed', '?')} | "
        f"Ro5 violations={scores.get('ro5_violations', '?')} | "
        f"Lipinski={'PASS' if scores.get('lipinski_pass') else 'FAIL'} | "
        f"TPSA={scores.get('tpsa', '?')} | "
        f"Synth={scores.get('synth_score', '?')}"
    )


# ---------------------------------------------------------------------------
# LLM-based in silico scoring
# ---------------------------------------------------------------------------

SCORING_SYSTEM = """You are a computational drug discovery expert specializing in in silico evaluation.
Score hypotheses objectively using your knowledge of pharmacology, medicinal chemistry, and translational medicine.
Always return valid JSON."""

SCORING_PROMPT = """\
Evaluate the following drug discovery hypothesis and return a JSON scoring object.

## Hypothesis
Title: {title}
Target: {target}
Mechanism: {mechanism}
Candidate Scaffold: {scaffold}
Novelty Rationale: {novelty}
Risks: {risks}

## Physicochemical Data (RDKit)
{physchem}

## Scoring Instructions
Rate each dimension 0.0–1.0. Return ONLY a JSON object with these fields:
- "binding_likelihood": probability the scaffold binds the target (0-1)
- "admet_score": predicted ADMET suitability (0-1, based on physchem + your knowledge)
- "novelty_score": degree of novelty vs. known drugs/approaches (0-1)
- "feasibility_score": experimental feasibility to test in lab (0-1)
- "mechanistic_plausibility": biological plausibility of the mechanism (0-1)
- "overall_score": weighted composite (0-1)
- "score_rationale": one sentence explaining the overall_score
"""


def llm_score_hypothesis(hypothesis: dict[str, Any]) -> dict[str, Any]:
    """Use Llama 3.3 to score a hypothesis across multiple dimensions."""
    smiles = _clean_smiles(hypothesis.get("candidate_scaffold", ""))
    rdkit_scores = compute_rdkit_scores(smiles)
    physchem_str = physicochemical_summary(rdkit_scores)

    prompt = SCORING_PROMPT.format(
        title=hypothesis.get("title", ""),
        target=hypothesis.get("target", ""),
        mechanism=hypothesis.get("mechanism", ""),
        scaffold=hypothesis.get("candidate_scaffold", "N/A"),
        novelty=hypothesis.get("novelty_rationale", ""),
        risks=hypothesis.get("risks", ""),
        physchem=physchem_str,
    )

    messages = [
        {"role": "system", "content": SCORING_SYSTEM},
        {"role": "user", "content": prompt},
    ]

    try:
        raw = chat_completion(messages, temperature=0.2, max_tokens=512)
        llm_scores = extract_json_block(raw)
        if not isinstance(llm_scores, dict):
            raise ValueError("Expected dict")
    except Exception as exc:
        logger.error("LLM scoring failed: %s", exc)
        llm_scores = {
            "binding_likelihood": 0.5,
            "admet_score": 0.5,
            "novelty_score": 0.5,
            "feasibility_score": 0.5,
            "mechanistic_plausibility": 0.5,
            "overall_score": 0.5,
            "score_rationale": "Scoring failed; defaults assigned.",
        }

    return {
        "rdkit_scores": rdkit_scores,
        "llm_scores": llm_scores,
        "composite_score": round(
            0.4 * float(llm_scores.get("overall_score", 0.5))
            + 0.2 * float(llm_scores.get("binding_likelihood", 0.5))
            + 0.2 * float(llm_scores.get("admet_score", 0.5))
            + 0.1 * float(llm_scores.get("novelty_score", 0.5))
            + 0.1 * float(llm_scores.get("feasibility_score", 0.5)),
            4,
        ),
    }
