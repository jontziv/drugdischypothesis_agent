"""
ChromaDB vector store for biomedical literature and molecules.
Uses sentence-transformers for local embeddings.
"""
from __future__ import annotations

import logging
from typing import Any

import chromadb
from chromadb.config import Settings
from sentence_transformers import SentenceTransformer

import config

logger = logging.getLogger(__name__)

_embedding_model: SentenceTransformer | None = None


def get_embedding_model() -> SentenceTransformer:
    global _embedding_model
    if _embedding_model is None:
        logger.info("Loading embedding model: %s", config.EMBEDDING_MODEL)
        _embedding_model = SentenceTransformer(config.EMBEDDING_MODEL)
    return _embedding_model


def _embed(texts: list[str]) -> list[list[float]]:
    model = get_embedding_model()
    return model.encode(texts, show_progress_bar=False).tolist()


# ---------------------------------------------------------------------------
# ChromaDB client (singleton)
# ---------------------------------------------------------------------------

_chroma_client: chromadb.PersistentClient | None = None


def get_chroma_client() -> chromadb.PersistentClient:
    global _chroma_client
    if _chroma_client is None:
        _chroma_client = chromadb.PersistentClient(path=config.CHROMA_PERSIST_DIR)
    return _chroma_client


def get_literature_collection() -> chromadb.Collection:
    client = get_chroma_client()
    return client.get_or_create_collection(
        name=config.LITERATURE_COLLECTION,
        metadata={"hnsw:space": "cosine"},
    )


def get_molecule_collection() -> chromadb.Collection:
    client = get_chroma_client()
    return client.get_or_create_collection(
        name=config.MOLECULE_COLLECTION,
        metadata={"hnsw:space": "cosine"},
    )


# ---------------------------------------------------------------------------
# Indexing
# ---------------------------------------------------------------------------

def index_papers(papers: list[dict[str, Any]]) -> int:
    """Embed and store papers. Returns count of newly added documents."""
    if not papers:
        return 0

    collection = get_literature_collection()
    existing = set(collection.get()["ids"])

    new_papers = [p for p in papers if p["id"] not in existing]
    if not new_papers:
        logger.info("All %d papers already indexed.", len(papers))
        return 0

    texts = [
        f"Title: {p['title']}\nAbstract: {p['abstract']}\nMeSH: {', '.join(p.get('mesh_terms', []))}"
        for p in new_papers
    ]
    embeddings = _embed(texts)
    ids = [p["id"] for p in new_papers]
    metadatas = [
        {
            "title": p.get("title", "")[:500],
            "pmid": p.get("pmid", ""),
            "journal": p.get("journal", "")[:200],
            "year": p.get("year", ""),
            "url": p.get("url", ""),
            "source": p.get("source", "pubmed"),
        }
        for p in new_papers
    ]
    documents = [t[:2000] for t in texts]

    collection.add(
        ids=ids,
        embeddings=embeddings,
        metadatas=metadatas,
        documents=documents,
    )
    logger.info("Indexed %d new papers.", len(new_papers))
    return len(new_papers)


def index_molecules(molecules: list[dict[str, Any]]) -> int:
    """Embed and store molecules by SMILES + metadata. Returns count of newly added."""
    if not molecules:
        return 0

    collection = get_molecule_collection()
    existing = set(collection.get()["ids"])

    new_mols = [m for m in molecules if m["id"] not in existing]
    if not new_mols:
        logger.info("All %d molecules already indexed.", len(molecules))
        return 0

    texts = [
        (
            f"Molecule: {m.get('molecule_name', m['id'])}\n"
            f"SMILES: {m.get('smiles', '')}\n"
            f"Target: {m.get('target_id', '')}\n"
            f"Activity: {m.get('activity_type', '')} {m.get('activity_value', '')} {m.get('activity_units', '')}"
        )
        for m in new_mols
    ]
    embeddings = _embed(texts)
    ids = [m["id"] for m in new_mols]
    metadatas = [
        {
            "molecule_name": str(m.get("molecule_name", ""))[:200],
            "smiles": str(m.get("smiles", ""))[:500],
            "chembl_id": str(m.get("chembl_id", "")),
            "target_id": str(m.get("target_id", "")),
            "activity_type": str(m.get("activity_type", "")),
            "activity_value": str(m.get("activity_value", "")),
            "source": str(m.get("source", "chembl")),
        }
        for m in new_mols
    ]
    documents = [t[:1000] for t in texts]

    collection.add(
        ids=ids,
        embeddings=embeddings,
        metadatas=metadatas,
        documents=documents,
    )
    logger.info("Indexed %d new molecules.", len(new_mols))
    return len(new_mols)


# ---------------------------------------------------------------------------
# Retrieval
# ---------------------------------------------------------------------------

def retrieve_relevant_papers(query: str, n_results: int = 8) -> list[dict[str, Any]]:
    """Retrieve top-N papers most relevant to a query string."""
    collection = get_literature_collection()
    count = collection.count()
    if count == 0:
        return []

    n_results = min(n_results, count)
    query_embedding = _embed([query])[0]
    results = collection.query(
        query_embeddings=[query_embedding],
        n_results=n_results,
        include=["documents", "metadatas", "distances"],
    )

    papers = []
    for i, doc in enumerate(results["documents"][0]):
        meta = results["metadatas"][0][i]
        dist = results["distances"][0][i]
        papers.append({
            "document": doc,
            "title": meta.get("title", ""),
            "pmid": meta.get("pmid", ""),
            "journal": meta.get("journal", ""),
            "year": meta.get("year", ""),
            "url": meta.get("url", ""),
            "relevance_score": round(1 - dist, 4),
        })
    return papers


def retrieve_relevant_molecules(query: str, n_results: int = 10) -> list[dict[str, Any]]:
    """Retrieve top-N molecules most relevant to a query string."""
    collection = get_molecule_collection()
    count = collection.count()
    if count == 0:
        return []

    n_results = min(n_results, count)
    query_embedding = _embed([query])[0]
    results = collection.query(
        query_embeddings=[query_embedding],
        n_results=n_results,
        include=["documents", "metadatas", "distances"],
    )

    molecules = []
    for i, doc in enumerate(results["documents"][0]):
        meta = results["metadatas"][0][i]
        dist = results["distances"][0][i]
        molecules.append({
            "document": doc,
            "molecule_name": meta.get("molecule_name", ""),
            "smiles": meta.get("smiles", ""),
            "chembl_id": meta.get("chembl_id", ""),
            "target_id": meta.get("target_id", ""),
            "activity_type": meta.get("activity_type", ""),
            "activity_value": meta.get("activity_value", ""),
            "relevance_score": round(1 - dist, 4),
        })
    return molecules


def get_collection_stats() -> dict[str, int]:
    """Return counts for both collections."""
    return {
        "papers": get_literature_collection().count(),
        "molecules": get_molecule_collection().count(),
    }


def reset_collections() -> None:
    """Drop and recreate both collections (for fresh start)."""
    client = get_chroma_client()
    try:
        client.delete_collection(config.LITERATURE_COLLECTION)
    except Exception:
        pass
    try:
        client.delete_collection(config.MOLECULE_COLLECTION)
    except Exception:
        pass
    logger.info("Collections reset.")
