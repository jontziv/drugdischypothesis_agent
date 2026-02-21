"""
Drug Discovery Hypothesis Agent — Streamlit Frontend
"""
from __future__ import annotations

import json
import logging
import sys
import os
import threading
import time
from datetime import datetime
from typing import Any

import streamlit as st
import pandas as pd

# Ensure project root is on path
sys.path.insert(0, os.path.dirname(__file__))

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(name)s: %(message)s")

# ── Module-level shared state for cross-thread communication ─────────────────
# st.session_state is NOT thread-safe — background threads must NOT touch it.
# We use this plain dict as a shared buffer, then copy into session_state
# on the main thread once the pipeline finishes.
_pipeline_state: dict = {
    "log": [],
    "results": [],
    "running": False,
    "done": False,
}

# ── Page config ─────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Drug Discovery Hypothesis Agent",
    page_icon="🧬",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── Styles ──────────────────────────────────────────────────────────────────
st.markdown("""
<style>
    .main-header {
        background: linear-gradient(135deg, #0f2027, #203a43, #2c5364);
        color: white;
        padding: 2rem;
        border-radius: 12px;
        margin-bottom: 1.5rem;
        text-align: center;
    }
    .main-header h1 { font-size: 2.2rem; margin: 0; }
    .main-header p  { opacity: 0.85; margin: 0.5rem 0 0; font-size: 1rem; }

    .hyp-card {
        border: 1px solid #e0e0e0;
        border-radius: 10px;
        padding: 1.2rem;
        margin-bottom: 1rem;
        background: #fafafa;
    }
    .hyp-card.rank-1 { border-left: 5px solid #FFD700; }
    .hyp-card.rank-2 { border-left: 5px solid #C0C0C0; }
    .hyp-card.rank-3 { border-left: 5px solid #CD7F32; }
    .hyp-card.rank-other { border-left: 5px solid #4CAF50; }

    .score-badge {
        display: inline-block;
        background: #1976D2;
        color: white;
        border-radius: 20px;
        padding: 2px 10px;
        font-size: 0.78rem;
        font-weight: bold;
        margin-right: 5px;
    }
    .elo-badge {
        display: inline-block;
        background: #7B1FA2;
        color: white;
        border-radius: 20px;
        padding: 2px 10px;
        font-size: 0.78rem;
        font-weight: bold;
    }
    .status-pill {
        display: inline-block;
        background: #E8F5E9;
        color: #2E7D32;
        border-radius: 4px;
        padding: 2px 8px;
        font-size: 0.75rem;
        border: 1px solid #A5D6A7;
    }
    .evidence-box {
        background: #E3F2FD;
        border-left: 3px solid #1565C0;
        padding: 0.6rem 1rem;
        border-radius: 0 6px 6px 0;
        margin: 0.4rem 0;
        font-size: 0.88rem;
    }
    .metric-chip {
        background: #F3E5F5;
        border-radius: 6px;
        padding: 4px 10px;
        font-size: 0.8rem;
        color: #4A148C;
    }
    div[data-testid="stSidebarNav"] { display: none; }
</style>
""", unsafe_allow_html=True)


# ── Session state init ───────────────────────────────────────────────────────
def _init_state():
    defaults = {
        "hypotheses": [],
        "pipeline_running": False,
        "pipeline_log": [],
        "pipeline_done": False,
        "selected_hyp_id": None,
        "feedback_submitted": False,
    }
    for k, v in defaults.items():
        if k not in st.session_state:
            st.session_state[k] = v

_init_state()


# ── Lazy imports ─────────────────────────────────────────────────────────────
@st.cache_resource(show_spinner=False)
def _get_orchestrator():
    from agents.orchestrator import (
        run_full_pipeline,
        submit_experimental_feedback,
        load_hypotheses,
        rerank_existing,
    )
    return run_full_pipeline, submit_experimental_feedback, load_hypotheses, rerank_existing


@st.cache_resource(show_spinner=False)
def _get_store_stats():
    from vector_store.store import get_collection_stats
    return get_collection_stats


@st.cache_resource(show_spinner=False)
def _get_critique():
    from agents.hypothesis_agent import critique_hypothesis
    return critique_hypothesis


# ── Helpers ─────────────────────────────────────────────────────────────────

def _score_bar(value: float, label: str) -> str:
    pct = int(value * 100)
    color = "#4CAF50" if pct >= 70 else "#FF9800" if pct >= 45 else "#F44336"
    return (
        f'<div style="margin:2px 0"><span style="font-size:0.78rem;color:#555">{label}</span>'
        f'<div style="background:#eee;border-radius:4px;height:8px;margin-top:2px">'
        f'<div style="background:{color};width:{pct}%;height:8px;border-radius:4px"></div></div>'
        f'<span style="font-size:0.7rem;color:#888">{pct}%</span></div>'
    )


def _rank_css(rank: int) -> str:
    return {1: "rank-1", 2: "rank-2", 3: "rank-3"}.get(rank, "rank-other")


def _rank_medal(rank: int) -> str:
    return {1: "🥇", 2: "🥈", 3: "🥉"}.get(rank, f"#{rank}")


# ── Sidebar ─────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("## 🧬 Drug Discovery Agent")
    st.markdown("---")

    st.markdown("### New Pipeline Run")
    disease = st.text_input(
        "Disease / Condition",
        placeholder="e.g. Alzheimer's disease",
        help="The disease or condition to focus on.",
    )
    target = st.text_input(
        "Drug Target / Pathway",
        placeholder="e.g. BACE1 beta-secretase",
        help="Primary molecular target or pathway of interest.",
    )
    n_hyp = st.slider("Hypotheses to generate", min_value=2, max_value=6, value=3,
                      help="Fewer = faster. Each hypothesis needs one GROQ scoring call.")
    run_tournament = st.checkbox("Run Elo tournament", value=True)
    pubmed_max = st.slider("Max PubMed papers", 5, 30, 10,
                           help="Fewer papers = faster ingestion.")
    chembl_max = st.slider("Max ChEMBL molecules", 5, 40, 15)

    run_btn = st.button(
        "🚀 Run Pipeline",
        disabled=st.session_state.pipeline_running or not disease or not target,
        use_container_width=True,
        type="primary",
    )

    st.markdown("---")
    st.markdown("### Existing Hypotheses")
    rerank_btn = st.button("♻️ Re-rank Existing", use_container_width=True)
    clear_btn = st.button("🗑️ Clear All Hypotheses", use_container_width=True)

    st.markdown("---")
    try:
        get_stats = _get_store_stats()
        stats = get_stats()
        st.markdown("### Vector Store")
        col1, col2 = st.columns(2)
        col1.metric("Papers", stats["papers"])
        col2.metric("Molecules", stats["molecules"])
    except Exception:
        st.caption("Vector store not yet initialised.")

    st.markdown("---")
    st.caption("Powered by Llama 3.3 · GROQ · ChromaDB · RDKit")


# ── Main header ─────────────────────────────────────────────────────────────
st.markdown("""
<div class="main-header">
    <h1>🧬 Drug Discovery Hypothesis Agent</h1>
    <p>Autonomous multi-agent system for biomedical hypothesis generation, scoring, and Elo-ranked tournament evaluation</p>
</div>
""", unsafe_allow_html=True)

tabs = st.tabs(["🏆 Ranked Hypotheses", "📊 Analytics", "🔬 Feedback Loop", "📋 Pipeline Log"])


# ── Pipeline runner ──────────────────────────────────────────────────────────
if run_btn and disease and target and not st.session_state.pipeline_running:
    st.session_state.pipeline_running = True
    st.session_state.pipeline_done = False
    st.session_state.hypotheses = []
    st.session_state.pipeline_log = []

    # Reset shared state for this run
    _pipeline_state["log"] = []
    _pipeline_state["results"] = []
    _pipeline_state["running"] = True
    _pipeline_state["done"] = False

    # Capture sidebar values before entering thread (closures over local vars)
    _disease = disease
    _target = target
    _n_hyp = n_hyp
    _run_tournament = run_tournament
    _pubmed_max = pubmed_max
    _chembl_max = chembl_max

    def _run():
        """Runs entirely in a background thread — no st.session_state access here."""
        run_full_pipeline, _, _, _ = _get_orchestrator()

        def _log(msg: str):
            _pipeline_state["log"].append(f"[{datetime.now().strftime('%H:%M:%S')}] {msg}")

        try:
            results = run_full_pipeline(
                disease=_disease,
                target=_target,
                n_hypotheses=_n_hyp,
                run_tournament_flag=_run_tournament,
                progress_callback=_log,
                pubmed_max=_pubmed_max,
                chembl_max=_chembl_max,
            )
            _pipeline_state["results"] = results
        except Exception as exc:
            _pipeline_state["log"].append(f"[ERROR] {exc}")
            logging.exception("Pipeline error")
        finally:
            _pipeline_state["running"] = False
            _pipeline_state["done"] = True

    thread = threading.Thread(target=_run, daemon=True)
    thread.start()

    # Poll on main thread — read from _pipeline_state, NOT session_state
    log_ph = st.empty()
    with st.spinner("Pipeline running... check the log below for live progress."):
        while _pipeline_state["running"]:
            lines = _pipeline_state["log"][-15:]
            log_ph.code("\n".join(lines) if lines else "Starting...", language="")
            time.sleep(1.5)

    thread.join(timeout=300)
    log_ph.empty()

    # Now safely copy results into session_state on main thread
    st.session_state.hypotheses = _pipeline_state["results"]
    st.session_state.pipeline_log = _pipeline_state["log"]
    st.session_state.pipeline_running = False
    st.session_state.pipeline_done = True
    st.rerun()


if rerank_btn:
    with st.spinner("Re-ranking hypotheses with Elo tournament..."):
        _, _, _, rerank_existing = _get_orchestrator()
        results = rerank_existing(progress_callback=None)
        st.session_state.hypotheses = results
    st.success(f"Re-ranked {len(results)} hypotheses.")
    st.rerun()


if clear_btn:
    import config as _cfg
    for fpath in [_cfg.HYPOTHESES_FILE, _cfg.ELO_FILE, _cfg.FEEDBACK_FILE]:
        if os.path.exists(fpath):
            os.remove(fpath)
    st.session_state.hypotheses = []
    st.session_state.pipeline_log = []
    st.success("All hypotheses cleared.")
    st.rerun()


# ── Load persisted hypotheses if session is empty ────────────────────────────
if not st.session_state.hypotheses:
    try:
        _, _, load_hypotheses, _ = _get_orchestrator()
        st.session_state.hypotheses = load_hypotheses()
    except Exception:
        pass


# ── TAB 1: Ranked Hypotheses ─────────────────────────────────────────────────
with tabs[0]:
    hypotheses = st.session_state.hypotheses

    if st.session_state.pipeline_running:
        st.info("Pipeline is running...")
        lines = _pipeline_state["log"][-10:]
        st.code("\n".join(lines) if lines else "Starting...", language="")

    elif not hypotheses:
        st.markdown("""
        <div style="text-align:center;padding:3rem;color:#888">
            <h3>No hypotheses yet</h3>
            <p>Enter a disease and target in the sidebar and click <strong>Run Pipeline</strong>.</p>
        </div>
        """, unsafe_allow_html=True)

    else:
        st.markdown(f"**{len(hypotheses)} hypothesis/es** ranked by Elo rating. "
                    f"Updated: {datetime.now().strftime('%Y-%m-%d %H:%M')}")

        col_f1, col_f2 = st.columns([3, 1])
        with col_f1:
            search_term = st.text_input("Filter hypotheses", placeholder="Search by title, target, scaffold...")
        with col_f2:
            show_top = st.selectbox("Show top", [5, 10, 20, 999], index=1)

        filtered = [
            h for h in hypotheses
            if not search_term or search_term.lower() in json.dumps(h).lower()
        ][:show_top]

        for rank, hyp in enumerate(filtered, 1):
            elo = hyp.get("elo_rating", 1500)
            composite = hyp.get("composite_score", 0.0)
            llm_scores = hyp.get("llm_scores", {})
            rdkit = hyp.get("rdkit_scores", {})

            with st.container():
                st.markdown(f"""
                <div class="hyp-card {_rank_css(rank)}">
                    <div style="display:flex;justify-content:space-between;align-items:flex-start">
                        <div>
                            <span style="font-size:1.3rem">{_rank_medal(rank)}</span>
                            <strong style="font-size:1.05rem"> {hyp.get('title','Untitled')}</strong>
                            <span class="status-pill" style="margin-left:8px">{hyp.get('id','')}</span>
                        </div>
                        <div>
                            <span class="score-badge">Score: {composite:.2f}</span>
                            <span class="elo-badge">Elo: {elo:.0f}</span>
                        </div>
                    </div>
                    <div style="margin-top:0.5rem;color:#555;font-size:0.88rem">
                        <strong>Target:</strong> {hyp.get('target','')} &nbsp;|&nbsp;
                        <strong>Disease:</strong> {hyp.get('disease','')} &nbsp;|&nbsp;
                        <strong>W/L:</strong> {hyp.get('wins',0)}/{hyp.get('losses',0)} ({hyp.get('comparisons',0)} matches)
                    </div>
                </div>
                """, unsafe_allow_html=True)

                with st.expander(f"View full hypothesis — {hyp.get('title','')[:60]}"):
                    col_main, col_scores = st.columns([3, 2])

                    with col_main:
                        st.markdown(f"**Mechanism of Action**\n\n{hyp.get('mechanism','N/A')}")
                        st.markdown(f"**Candidate Scaffold**\n\n`{hyp.get('candidate_scaffold','N/A')}`")
                        st.markdown(f"**Novelty Rationale**\n\n{hyp.get('novelty_rationale','N/A')}")
                        st.markdown(f"**Proposed Experiment**\n\n{hyp.get('proposed_experiment','N/A')}")
                        st.markdown(f"**Risks**\n\n{hyp.get('risks','N/A')}")

                        evidence = hyp.get("supporting_evidence", [])
                        if evidence:
                            st.markdown("**Supporting Evidence**")
                            for ev in evidence:
                                st.markdown(f'<div class="evidence-box">• {ev}</div>', unsafe_allow_html=True)

                        papers = hyp.get("supporting_papers", [])
                        if papers:
                            st.markdown("**Source Literature**")
                            for p in papers:
                                url = p.get("url", "")
                                title_p = p.get("title", "Unknown")
                                year = p.get("year", "")
                                if url:
                                    st.markdown(f"- [{title_p} ({year})]({url})")
                                else:
                                    st.markdown(f"- {title_p} ({year})")

                    with col_scores:
                        st.markdown("**In Silico Scores**")
                        for field, label in [
                            ("binding_likelihood", "Binding Likelihood"),
                            ("admet_score", "ADMET Score"),
                            ("novelty_score", "Novelty"),
                            ("feasibility_score", "Feasibility"),
                            ("mechanistic_plausibility", "Mechanism Plausibility"),
                            ("overall_score", "Overall LLM Score"),
                        ]:
                            val = float(llm_scores.get(field, 0))
                            st.markdown(_score_bar(val, label), unsafe_allow_html=True)

                        rationale = llm_scores.get("score_rationale", "")
                        if rationale:
                            st.caption(f"*{rationale}*")

                        if rdkit and rdkit.get("valid_smiles"):
                            st.markdown("**Physicochemical (RDKit)**")
                            st.dataframe(pd.DataFrame({
                                "Property": ["MW", "LogP", "QED", "TPSA", "HBD", "HBA",
                                             "Ro5 Violations", "Lipinski", "Synth Score"],
                                "Value": [
                                    rdkit.get("molecular_weight", "N/A"),
                                    rdkit.get("logp", "N/A"),
                                    rdkit.get("qed", "N/A"),
                                    rdkit.get("tpsa", "N/A"),
                                    rdkit.get("hbd", "N/A"),
                                    rdkit.get("hba", "N/A"),
                                    rdkit.get("ro5_violations", "N/A"),
                                    "✅ Pass" if rdkit.get("lipinski_pass") else "❌ Fail",
                                    rdkit.get("synth_score", "N/A"),
                                ],
                            }), use_container_width=True, hide_index=True)
                        elif rdkit and not rdkit.get("valid_smiles", True):
                            st.warning("SMILES string invalid — RDKit scoring skipped.")

                        if st.button("🔬 Generate Critique", key=f"critique_{hyp['id']}"):
                            with st.spinner("Generating scientific critique..."):
                                critique_text = _get_critique()(hyp)
                            st.markdown("**Scientific Critique**")
                            st.markdown(critique_text)


# ── TAB 2: Analytics ─────────────────────────────────────────────────────────
with tabs[1]:
    hypotheses = st.session_state.hypotheses
    if not hypotheses:
        st.info("Run the pipeline first to see analytics.")
    else:
        st.subheader("Elo Rating Distribution")
        df = pd.DataFrame([
            {
                "ID": h.get("id", ""),
                "Title": (h.get("title", "")[:45] + "...") if len(h.get("title", "")) > 45 else h.get("title", ""),
                "Disease": h.get("disease", ""),
                "Target": h.get("target", ""),
                "Elo": h.get("elo_rating", 1500),
                "Composite Score": h.get("composite_score", 0),
                "Binding": h.get("llm_scores", {}).get("binding_likelihood", 0),
                "ADMET": h.get("llm_scores", {}).get("admet_score", 0),
                "Novelty": h.get("llm_scores", {}).get("novelty_score", 0),
                "Feasibility": h.get("llm_scores", {}).get("feasibility_score", 0),
                "Wins": h.get("wins", 0),
                "Losses": h.get("losses", 0),
                "Comparisons": h.get("comparisons", 0),
            }
            for h in hypotheses
        ])

        col_c1, col_c2 = st.columns(2)
        with col_c1:
            st.markdown("**Elo Leaderboard**")
            st.bar_chart(df.set_index("Title")["Elo"], color="#7B1FA2")
        with col_c2:
            st.markdown("**Multi-Dimension Scores**")
            score_df = df[["Title", "Binding", "ADMET", "Novelty", "Feasibility"]].set_index("Title")
            try:
                st.dataframe(score_df.style.background_gradient(cmap="YlGn", axis=None), width="stretch")
            except Exception:
                st.dataframe(score_df, width="stretch")

        st.markdown("---")
        st.subheader("Full Hypothesis Table")
        display_cols = ["ID", "Title", "Disease", "Elo", "Composite Score", "Wins", "Losses", "Comparisons"]
        try:
            st.dataframe(
                df[display_cols].style.background_gradient(subset=["Elo", "Composite Score"], cmap="RdYlGn"),
                width="stretch",
                hide_index=True,
            )
        except Exception:
            st.dataframe(df[display_cols], width="stretch", hide_index=True)

        col_dl1, col_dl2 = st.columns(2)
        with col_dl1:
            st.download_button(
                "📥 Export CSV",
                data=df.to_csv(index=False).encode(),
                file_name=f"drug_hypotheses_{datetime.now().strftime('%Y%m%d_%H%M')}.csv",
                mime="text/csv",
            )
        with col_dl2:
            st.download_button(
                "📥 Export JSON",
                data=json.dumps(hypotheses, indent=2).encode(),
                file_name=f"drug_hypotheses_{datetime.now().strftime('%Y%m%d_%H%M')}.json",
                mime="application/json",
            )


# ── TAB 3: Feedback Loop ─────────────────────────────────────────────────────
with tabs[2]:
    hypotheses = st.session_state.hypotheses
    st.subheader("Submit Experimental Feedback")
    st.markdown(
        "Record real or simulated experimental outcomes. "
        "Feedback adjusts Elo ratings and re-sorts the leaderboard."
    )

    if not hypotheses:
        st.info("No hypotheses available. Run the pipeline first.")
    else:
        hyp_options = {
            f"{h.get('id','')} — {h.get('title','')[:60]}": h["id"]
            for h in hypotheses
        }
        selected_label = st.selectbox("Select hypothesis", list(hyp_options.keys()))
        selected_id = hyp_options[selected_label]

        col_fb1, col_fb2 = st.columns(2)
        with col_fb1:
            outcome = st.radio(
                "Experimental outcome",
                ["positive", "partial", "negative"],
                format_func=lambda x: {"positive": "✅ Positive", "partial": "⚠️ Partial", "negative": "❌ Negative"}[x],
            )
        with col_fb2:
            elo_adj = st.number_input("Manual Elo adjustment (±)", value=0.0, step=5.0)

        notes = st.text_area(
            "Experimental notes",
            placeholder="Describe the experiment, key findings, cell line, assay type, IC50 value...",
        )

        if st.button("📤 Submit Feedback", type="primary"):
            _, submit_feedback, _, _ = _get_orchestrator()
            ok = submit_feedback(
                hypothesis_id=selected_id,
                outcome=outcome,
                notes=notes,
                elo_adjustment=elo_adj,
            )
            if ok:
                st.success(f"Feedback recorded for {selected_id}. Elo updated.")
                _, _, load_hypotheses, _ = _get_orchestrator()
                st.session_state.hypotheses = load_hypotheses()
                st.rerun()
            else:
                st.error("Failed to record feedback. Check logs.")

        st.markdown("---")
        st.subheader("Feedback History")
        try:
            from agents.orchestrator import load_feedback_log
            log = load_feedback_log()
            if log:
                log_df = pd.DataFrame(log)[
                    ["timestamp", "hypothesis_id", "hypothesis_title", "outcome", "elo_delta", "elo_after", "notes"]
                ]
                log_df.columns = ["Timestamp", "ID", "Title", "Outcome", "Elo Δ", "Elo After", "Notes"]
                st.dataframe(log_df, use_container_width=True, hide_index=True)
            else:
                st.caption("No feedback recorded yet.")
        except Exception:
            st.caption("No feedback log available.")


# ── TAB 4: Pipeline Log ──────────────────────────────────────────────────────
with tabs[3]:
    st.subheader("Pipeline Log")
    log_lines = st.session_state.pipeline_log
    if log_lines:
        st.code("\n".join(log_lines), language="")
    elif st.session_state.pipeline_done:
        st.success("Pipeline completed successfully.")
    else:
        st.caption("No pipeline run yet this session.")
