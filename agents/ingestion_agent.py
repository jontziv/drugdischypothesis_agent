"""
Knowledge Ingestion Agent
Fetches biomedical literature from PubMed and molecule data from ChEMBL.
"""
from __future__ import annotations

import re
import time
import logging
from typing import Any

import requests
from Bio import Entrez
from tenacity import retry, stop_after_attempt, wait_exponential

import config

logger = logging.getLogger(__name__)

Entrez.email = config.PUBMED_EMAIL


# ---------------------------------------------------------------------------
# PubMed helpers
# ---------------------------------------------------------------------------

@retry(stop=stop_after_attempt(3), wait=wait_exponential(multiplier=1, min=2, max=10))
def search_pubmed(query: str, max_results: int = config.PUBMED_MAX_RESULTS) -> list[dict[str, Any]]:
    """Search PubMed and return a list of article dicts."""
    papers: list[dict[str, Any]] = []
    try:
        handle = Entrez.esearch(db="pubmed", term=query, retmax=max_results, usehistory="y")
        record = Entrez.read(handle)
        handle.close()

        ids = record["IdList"]
        if not ids:
            logger.info("No PubMed results for query: %s", query)
            return papers

        handle = Entrez.efetch(db="pubmed", id=",".join(ids), rettype="xml", retmode="xml")
        records = Entrez.read(handle)
        handle.close()

        for article in records.get("PubmedArticle", []):
            try:
                medline = article["MedlineCitation"]
                art = medline["Article"]

                title = str(art.get("ArticleTitle", ""))
                abstract_list = art.get("Abstract", {}).get("AbstractText", [])
                abstract = " ".join(str(a) for a in abstract_list) if abstract_list else ""

                authors_raw = art.get("AuthorList", [])
                authors = []
                for a in authors_raw:
                    last = a.get("LastName", "")
                    fore = a.get("ForeName", "")
                    if last:
                        authors.append(f"{last} {fore}".strip())

                journal = art.get("Journal", {}).get("Title", "")
                pub_date_dict = art.get("Journal", {}).get("JournalIssue", {}).get("PubDate", {})
                pub_year = str(pub_date_dict.get("Year", pub_date_dict.get("MedlineDate", "")))

                pmid = str(medline["PMID"])
                mesh_terms = [
                    str(m["DescriptorName"])
                    for m in medline.get("MeshHeadingList", [])
                ]

                papers.append({
                    "id": f"pubmed_{pmid}",
                    "pmid": pmid,
                    "title": title,
                    "abstract": abstract,
                    "authors": authors,
                    "journal": journal,
                    "year": pub_year,
                    "mesh_terms": mesh_terms,
                    "source": "pubmed",
                    "url": f"https://pubmed.ncbi.nlm.nih.gov/{pmid}/",
                })
            except Exception as exc:
                logger.warning("Failed to parse article: %s", exc)
                continue

    except Exception as exc:
        logger.error("PubMed search failed: %s", exc)

    return papers


# ---------------------------------------------------------------------------
# ChEMBL helpers
# ---------------------------------------------------------------------------

CHEMBL_BASE = "https://www.ebi.ac.uk/chembl/api/data"


@retry(stop=stop_after_attempt(3), wait=wait_exponential(multiplier=1, min=2, max=10))
def search_chembl_targets(target_name: str, max_results: int = 10) -> list[dict[str, Any]]:
    """Search ChEMBL for targets matching a name."""
    url = f"{CHEMBL_BASE}/target.json"
    params = {
        "target_synonym__icontains": target_name,
        "limit": max_results,
    }
    resp = requests.get(url, params=params, timeout=15)
    resp.raise_for_status()
    data = resp.json()
    targets = []
    for t in data.get("targets", []):
        targets.append({
            "chembl_id": t.get("target_chembl_id", ""),
            "name": t.get("pref_name", ""),
            "type": t.get("target_type", ""),
            "organism": t.get("organism", ""),
            "source": "chembl_target",
        })
    return targets


@retry(stop=stop_after_attempt(3), wait=wait_exponential(multiplier=1, min=2, max=10))
def fetch_molecules_for_target(
    chembl_target_id: str,
    max_results: int = config.CHEMBL_MAX_RESULTS,
) -> list[dict[str, Any]]:
    """Fetch active compounds for a ChEMBL target."""
    url = f"{CHEMBL_BASE}/activity.json"
    params = {
        "target_chembl_id": chembl_target_id,
        "standard_type__in": "IC50,Ki,Kd,EC50",
        "limit": max_results,
        "order_by": "standard_value",
    }
    resp = requests.get(url, params=params, timeout=15)
    resp.raise_for_status()
    data = resp.json()

    molecules: list[dict[str, Any]] = []
    seen: set[str] = set()

    for act in data.get("activities", []):
        mol_id = act.get("molecule_chembl_id", "")
        if not mol_id or mol_id in seen:
            continue
        seen.add(mol_id)

        smiles = act.get("canonical_smiles", "") or ""
        if not smiles:
            continue

        molecules.append({
            "id": f"chembl_{mol_id}",
            "chembl_id": mol_id,
            "smiles": smiles,
            "target_id": chembl_target_id,
            "activity_type": act.get("standard_type", ""),
            "activity_value": act.get("standard_value", ""),
            "activity_units": act.get("standard_units", ""),
            "molecule_name": act.get("molecule_pref_name", mol_id),
            "source": "chembl",
        })

    return molecules


@retry(stop=stop_after_attempt(3), wait=wait_exponential(multiplier=1, min=2, max=10))
def search_chembl_molecules(query: str, max_results: int = 20) -> list[dict[str, Any]]:
    """Free-text search for molecules in ChEMBL."""
    url = f"{CHEMBL_BASE}/molecule.json"
    params = {
        "molecule_synonyms__molecule_synonym__icontains": query,
        "limit": max_results,
    }
    resp = requests.get(url, params=params, timeout=15)
    resp.raise_for_status()
    data = resp.json()

    molecules: list[dict[str, Any]] = []
    for mol in data.get("molecules", []):
        smiles = (mol.get("molecule_structures") or {}).get("canonical_smiles", "") or ""
        if not smiles:
            continue
        mol_id = mol.get("molecule_chembl_id", "")
        molecules.append({
            "id": f"chembl_{mol_id}",
            "chembl_id": mol_id,
            "smiles": smiles,
            "molecule_name": mol.get("pref_name", mol_id),
            "mol_weight": (mol.get("molecule_properties") or {}).get("full_mwt", ""),
            "alogp": (mol.get("molecule_properties") or {}).get("alogp", ""),
            "source": "chembl",
        })

    return molecules


# ---------------------------------------------------------------------------
# Query helpers
# ---------------------------------------------------------------------------

def _sanitize(text: str) -> str:
    """Strip characters that break PubMed / ChEMBL queries."""
    return re.sub(r"['\"\(\)\[\]]", "", text).strip()


def _extract_keywords(target: str) -> list[str]:
    """
    Pull the most searchable tokens from a target name.
    E.g. 'BACE 1 beta secretase' → ['BACE1', 'secretase']
         'CDK2 kinase'           → ['CDK2', 'kinase']
    """
    filler = {"beta", "alpha", "gamma", "delta", "the", "and", "or",
              "of", "in", "a", "an", "type", "like", "associated"}
    tokens = re.split(r"[\s\-_/]+", target.lower())
    # Merge a bare digit token onto the previous token (BACE + 1 → BACE1)
    merged: list[str] = []
    for tok in tokens:
        if merged and re.fullmatch(r"\d+", tok):
            merged[-1] = merged[-1] + tok
        else:
            merged.append(tok)

    keywords = [t for t in merged if t not in filler and len(t) >= 2]
    return keywords[:4] if keywords else [_sanitize(target)]


# ---------------------------------------------------------------------------
# High-level ingestion function
# ---------------------------------------------------------------------------

def ingest_knowledge(
    disease_query: str,
    target_name: str,
    pubmed_max: int = config.PUBMED_MAX_RESULTS,
    chembl_max: int = config.CHEMBL_MAX_RESULTS,
    progress_callback=None,
) -> dict[str, list[dict]]:
    """
    Top-level ingestion: pulls literature and molecules for a disease/target.
    Returns { "papers": [...], "molecules": [...], "targets": [...] }
    """
    results: dict[str, list] = {"papers": [], "molecules": [], "targets": []}
    disease_clean = _sanitize(disease_query)
    target_clean = _sanitize(target_name)
    keywords = _extract_keywords(target_name)
    primary_kw = keywords[0]  # e.g. "BACE1"

    # ── PubMed ──────────────────────────────────────────────────────────────
    if progress_callback:
        progress_callback("Searching PubMed for relevant literature...")

    # Strategy 1: combined disease + primary target keyword
    pubmed_query = (
        f'"{disease_clean}"[Title/Abstract] AND "{primary_kw}"[Title/Abstract]'
    )
    papers = search_pubmed(pubmed_query, max_results=pubmed_max)

    # Strategy 2: fall back to disease + "drug" if combined returns nothing
    if not papers:
        logger.info("Combined PubMed query returned 0; trying disease-only fallback.")
        fallback_q = f'"{disease_clean}"[Title/Abstract] AND drug[Title/Abstract]'
        papers = search_pubmed(fallback_q, max_results=pubmed_max)

    # Strategy 3: plain text search as last resort
    if not papers:
        logger.info("Disease-only fallback returned 0; trying plain text search.")
        plain_q = f"{disease_clean} {primary_kw} drug treatment"
        papers = search_pubmed(plain_q, max_results=pubmed_max)

    results["papers"] = papers
    logger.info("Fetched %d papers from PubMed.", len(papers))
    if progress_callback:
        progress_callback(f"Found {len(papers)} papers. Searching ChEMBL targets...")

    # ── ChEMBL targets ───────────────────────────────────────────────────────
    # Try each extracted keyword until we get results
    targets: list[dict] = []
    for kw in keywords:
        targets = search_chembl_targets(kw, max_results=5)
        if targets:
            logger.info("ChEMBL targets found with keyword '%s'.", kw)
            break
        # Also try preferred name search
        try:
            targets = _search_chembl_targets_by_pref(kw, max_results=5)
            if targets:
                logger.info("ChEMBL targets found via pref_name for '%s'.", kw)
                break
        except Exception:
            pass

    results["targets"] = targets
    logger.info("Found %d ChEMBL targets.", len(targets))
    if progress_callback:
        progress_callback(f"Found {len(targets)} ChEMBL targets. Fetching molecules...")

    # ── ChEMBL molecules ─────────────────────────────────────────────────────
    molecules: list[dict] = []
    seen_ids: set[str] = set()

    for t in targets[:2]:
        t_id = t["chembl_id"]
        if t_id:
            mols = fetch_molecules_for_target(t_id, max_results=chembl_max // 2)
            for m in mols:
                if m["id"] not in seen_ids:
                    molecules.append(m)
                    seen_ids.add(m["id"])
            time.sleep(0.5)

    # Free-text molecule search — use first keyword (shorter = more results)
    for search_term in [primary_kw, disease_clean.split()[0]]:
        extra_mols = search_chembl_molecules(search_term, max_results=15)
        for m in extra_mols:
            if m["id"] not in seen_ids:
                molecules.append(m)
                seen_ids.add(m["id"])
        if molecules:
            break

    results["molecules"] = molecules
    logger.info("Fetched %d molecules from ChEMBL.", len(molecules))

    if progress_callback:
        progress_callback(
            f"Ingestion complete: {len(papers)} papers, {len(molecules)} molecules."
        )

    return results


@retry(stop=stop_after_attempt(3), wait=wait_exponential(multiplier=1, min=2, max=10))
def _search_chembl_targets_by_pref(target_name: str, max_results: int = 5) -> list[dict[str, Any]]:
    """Search ChEMBL targets by preferred name (separate from synonym search)."""
    url = f"{CHEMBL_BASE}/target.json"
    params = {"pref_name__icontains": target_name, "limit": max_results}
    resp = requests.get(url, params=params, timeout=15)
    resp.raise_for_status()
    data = resp.json()
    return [
        {
            "chembl_id": t.get("target_chembl_id", ""),
            "name": t.get("pref_name", ""),
            "type": t.get("target_type", ""),
            "organism": t.get("organism", ""),
            "source": "chembl_target",
        }
        for t in data.get("targets", [])
    ]
