# 🧬 Drug Discovery Hypothesis Agent

An autonomous multi-agent system for biomedical hypothesis generation, scientific scoring,
and Elo-ranked tournament evaluation — powered by **Llama 3.3** via **GROQ**, **ChromaDB**, and **RDKit**.

---

## Architecture

```
┌─────────────────────────────────────────────────────────┐
│                   Streamlit Frontend                    │
└───────────────────────────┬─────────────────────────────┘
                            │
                  ┌─────────▼──────────┐
                  │    Orchestrator     │  (agents/orchestrator.py)
                  └──┬───┬───┬───┬─────┘
                     │   │   │   │
         ┌───────────┘   │   │   └─────────────────┐
         ▼               ▼   ▼                     ▼
  ┌──────────────┐  ┌────────────┐  ┌───────────────────────┐
  │  Ingestion   │  │ Hypothesis │  │  Scoring + Elo Ranker  │
  │  Agent       │  │ Agent      │  │  (RDKit + LLM + Elo)  │
  │ (PubMed +    │  │ (RAG +     │  └───────────────────────┘
  │  ChEMBL)     │  │  Llama3.3) │
  └──────┬───────┘  └────────────┘
         │
         ▼
  ┌──────────────┐
  │  ChromaDB    │
  │ Vector Store │
  │ (embeddings) │
  └──────────────┘
```

### Agents

| Agent | File | Role |
|-------|------|------|
| Ingestion Agent | `agents/ingestion_agent.py` | Fetches PubMed papers + ChEMBL molecules |
| Vector Store | `vector_store/store.py` | Embeds + indexes via ChromaDB |
| Hypothesis Agent | `agents/hypothesis_agent.py` | RAG-grounded hypothesis generation via Llama 3.3 |
| Molecular Scorer | `scoring/molecular_scorer.py` | RDKit physicochemical + LLM multi-dim scoring |
| Elo Ranker | `scoring/elo_ranker.py` | Pairwise LLM debate + Elo tournament |
| Orchestrator | `agents/orchestrator.py` | Pipeline coordination + feedback loop |

---

## Setup

### 1. Prerequisites

- Python 3.10+
- A [GROQ API key](https://console.groq.com) (free tier available)

### 2. Install dependencies

```bash
# Recommended: use a virtual environment
python -m venv .venv
source .venv/bin/activate      # macOS/Linux
# .venv\Scripts\activate       # Windows

pip install -r requirements.txt
```

> **RDKit note:** If `pip install rdkit` fails, try:
> ```bash
> conda install -c conda-forge rdkit
> ```
> Or install without RDKit — the app degrades gracefully to LLM-only scoring.

### 3. Configure your API key

```bash
cp .env.example .env
# Edit .env and add your GROQ API key:
# GROQ_API_KEY=gsk_...
```

### 4. Run the app

```bash
streamlit run app.py
```

The app will open at `http://localhost:8501`.

---

## Usage

### Running the Pipeline

1. Enter a **Disease/Condition** (e.g. `Alzheimer's disease`)
2. Enter a **Drug Target** (e.g. `BACE1 beta-secretase`)
3. Adjust sliders for hypothesis count, PubMed/ChEMBL limits
4. Click **🚀 Run Pipeline**

### Example Input Combinations

Copy any row directly into the sidebar fields:

| # | Disease / Condition | Drug Target / Pathway |
|---|--------------------|-----------------------|
| 1 | Alzheimer's disease | BACE1 beta-secretase |
| 2 | Parkinson's disease | LRRK2 kinase |
| 3 | Type 2 diabetes | GLP-1 receptor agonism |
| 4 | Non-small cell lung cancer | EGFR tyrosine kinase |
| 5 | Breast cancer | HER2 receptor |
| 6 | Chronic myeloid leukemia | BCR-ABL tyrosine kinase |
| 7 | Rheumatoid arthritis | JAK1/JAK3 signaling pathway |
| 8 | Major depressive disorder | serotonin reuptake transporter (SERT) |
| 9 | HIV/AIDS | HIV-1 reverse transcriptase |
| 10 | Pulmonary fibrosis | TGF-beta signaling pathway |

> **Tip:** Start with combinations 1–3 — they have the richest PubMed and ChEMBL coverage, so the pipeline runs faster and produces higher-quality hypotheses.

The pipeline will:
1. Fetch real papers from PubMed and molecules from ChEMBL
2. Embed and index them into ChromaDB
3. Generate structured hypotheses via RAG + Llama 3.3
4. Score each hypothesis (RDKit + LLM multi-dimension)
5. Run an Elo tournament with LLM-judged pairwise debates
6. Display ranked results with full evidence

### Submitting Feedback

In the **Feedback Loop** tab, select a hypothesis, record the experimental outcome
(positive / partial / negative), and optionally add notes. The Elo rating adjusts
automatically and the leaderboard re-sorts.

### Re-ranking

Click **♻️ Re-rank Existing** in the sidebar to run a fresh tournament on all
saved hypotheses without generating new ones.

---

## Data Persistence

All state is stored in `data/`:

```
data/
├── chroma_db/         # ChromaDB vector store (papers + molecules)
├── hypotheses.json    # All generated hypotheses with scores + Elo
├── elo_ratings.json   # Persisted Elo ratings
└── feedback_log.json  # Experimental feedback history
```

---

## Models Used

| Model | Purpose |
|-------|---------|
| `llama-3.3-70b-versatile` (GROQ) | Hypothesis generation, scoring, debate judging, critique |
| `all-MiniLM-L6-v2` (local) | Document and molecule embeddings |

---

## External APIs

| API | Auth | Rate Limit |
|-----|------|-----------|
| PubMed/NCBI Entrez | Email only (no key) | 3 req/s |
| ChEMBL REST API | None | Polite use |
| GROQ API | API key in `.env` | Free tier: 30 req/min |

---

## Notes

- First run downloads the embedding model (~90MB). Subsequent runs use the cached version.
- The pipeline takes 2–5 minutes for a full run depending on GROQ rate limits.
- If GROQ rate limits are hit, `tenacity` retries with exponential backoff automatically.