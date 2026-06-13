# Week 2 Project — GraphRAG for Organizational Knowledge
## Track 2: Code-heavy (LangChain + LangGraph), Use Case 3

---

## Part 1: The RAG Framework

### Fictional Company: Veritask Sciences

Veritask Sciences is a digital-product and AI-implementation consultancy that builds
safety, pharmacovigilance, and regulatory-compliance data products for pharmaceutical
and biotech clients (GxP systems, SOC2/21 CFR Part 11 audit readiness, signal-detection
pipelines, regulatory submission tooling).

### The One-Liner

> My RAG app helps **Veritask program managers, tech leads, and execs** answer
> **cross-team "who/what/why" questions about projects, decisions, tools, and approvals**
> from **Veritask's internal knowledge base (MS Teams exports, project tracker, wiki,
> compliance docs — ~5 sources, ~30 entities)** in **a Streamlit comparison app**
> with **100% recall on graph-relationship questions vs. a vector-RAG baseline**.

### Framework Fields

| Field | Answer |
|---|---|
| **Use case** | PMs and tech leads at Veritask ask multi-hop questions ("Who worked on the SOC2 audit and what tools did they use? Who approved the pricing model change?") in a Streamlit demo app comparing GraphRAG vs vector RAG. |
| **Corpus** | ~5 mock sources: (1) Project tracker export (CSV/JSON — projects, status, owners), (2) MS Teams channel export (JSON — decision threads, approvals), (3) Internal wiki pages (Markdown — tool/skill descriptions, SOPs), (4) Compliance/audit docs (text — SOC2 audit report, pricing memo), (5) People directory (JSON — roles, skills). All mock, English, owned by "Veritask Ops". ~30 entities total across people, projects, skills, tools, documents, decisions. |
| **Ingestion + cleaning** | Each source loaded by a small Python loader into a common entity schema (id, type, name, attributes, text). Markdown/JSON boilerplate (headers, metadata blocks) stripped; Teams export flattened from thread JSON to plain decision/approval records. |
| **Ingestion + freshness** | One-time batch load for the demo (`graph_builder.py` runs at startup). In a real deployment: nightly refresh from project tracker API + Teams export, freshness SLA of 24h. |
| **Chunking + embedding** | Each graph node flattened into one short text chunk (name + type + attributes + linked-doc snippets), ~100–300 tokens — no further sub-chunking needed since nodes are already atomic. Embedding model: **all-MiniLM-L6-v2 (local, via sentence-transformers)**. Chosen because chunk size is small and uniform, so a lightweight 384-dim model gives good semantic separation at zero API cost, matching the small corpus size. |
| **Retrieve** | **Dual pipeline**: (a) Vector RAG baseline — FAISS, dense only, top-8; (b) GraphRAG — NetworkX multi-directed graph, entity-linking + 2-hop subgraph expansion (optionally relation-filtered: WORKED_ON, USED_TOOL, APPROVED, DECISION_FOR, AUTHORED, REFERENCES), with attached document text. |

**"I don't know" path**: If entity linking finds no seed nodes (LLM linker + deterministic
fallback both fail) and the relation-filtered traversal returns no edges even after
retry without the filter, GraphRAG responds "insufficient context in the knowledge
graph for this question" rather than guessing. Vector RAG similarly instructs the LLM
to say "not found in retrieved context" if the top-8 chunks don't address the question.

---

## Part 2: Architecture (adapted from Solution Kit)

```
                         ┌────────────────────────┐
                         │      User Question       │
                         └─────────────┬────────────┘
              ┌────────────────────────┴────────────────────────┐
              ▼                                                    ▼
   ┌─────────────────────┐                          ┌──────────────────────────┐
   │   Vector RAG (base)  │                          │   GraphRAG (LangGraph)    │
   │ 1. Flatten nodes      │                          │ 1. Build KG (NetworkX)    │
   │ 2. Embed (Nebius) +   │                          │ 2. Link Q -> entities      │
   │    FAISS store        │                          │    (LLM + fallback regex) │
   │ 3. Retrieve top-8      │                          │ 3. Expand 2-hop subgraph   │
   │ 4. LLM answer (Nebius) │                          │ 4. Attach related docs     │
   └─────────────────────┘                          │ 5. LLM answer (Nebius)     │
                                                       └──────────────────────────┘
              └────────────────────────┬────────────────────────┘
                                        ▼
                         ┌────────────────────────────┐
                         │ compare.py: recall vs gold   │
                         │ ui.py: Streamlit side-by-side│
                         └────────────────────────────┘
```

### LangGraph state machine (GraphRAG path)

Nodes: `link_entities` -> `expand_subgraph` -> `attach_documents` -> `generate_answer`
Conditional edge: if `link_entities` finds zero seeds -> route to `fallback_linker`
node (deterministic regex/topic map) -> `expand_subgraph`. If `expand_subgraph` with
relation filter returns 0 edges -> retry without filter (self-loop, max 1 retry) ->
`attach_documents`.

State object carries: `question`, `seed_entities`, `subgraph_nodes`, `subgraph_edges`,
`attached_docs`, `answer`, `retrieved_entity_ids` (for recall scoring).

---

## Tech Stack

| Component | Choice |
|---|---|
| Vector store | FAISS (CPU) |
| Graph library | NetworkX (MultiDiGraph) |
| Orchestration | LangChain (loaders/splitters/embeddings wrappers) + LangGraph (state machine for GraphRAG) |
| LLM (generation) | **Groq** — `llama-3.1-8b-instant` (temp 0, used for `GRAPH_ANSWER` and `VECTOR_ANSWER`) |
| Entity extraction (structured output) | **Groq** — same model by default, configurable via `GROQ_LINK_MODEL` (e.g. `llama-3.3-70b-versatile` for more reliable JSON output) |
| Embeddings | **Local, via sentence-transformers** — `all-MiniLM-L6-v2` (CPU, no API key needed; Groq has no embeddings endpoint) |
| UI | Streamlit (graph view + side-by-side comparison, per kit's `ui.py`) |
| Comparison/eval | `compare.py` — recall vs. gold entity sets over 10 benchmark questions |
| Language | Python 3.12, pip + requirements.txt |

> **Note on the handout's Nebius requirement**: the Week 2 handout asks both
> tracks to use Nebius Token Factory for at least one model call. This build
> uses Groq + local embeddings instead. If your cohort requires Nebius
> specifically, either request a Nebius API key or swap one model call (e.g.
> the embedding step) back to Nebius -- `llm.py` is structured so this is a
> small, isolated change. Flag this with your instructor either way.

> **Security note**: never commit API keys to the repo or paste them into
> chat tools. Use a `.env` file (gitignored) or your shell environment, and
> rotate any key that has been exposed.

---

## Mock Organization Dataset (≥30 nodes)

### People (8)
| id | name | role |
|---|---|---|
| p_nina | Nina Castellanos | Compliance Lead |
| p_arjun | Arjun Mehta | Senior ML Engineer |
| p_priya | Priya Subramaniam | Product Manager |
| p_leo | Leo Fischer | Data Scientist |
| p_sam | Samira El-Amin | CFO |
| p_devon | Devon Park | Security Engineer |
| p_grace | Grace Adeyemi | VP Engineering |
| p_oliver | Oliver Tan | Regulatory Affairs Specialist |

### Projects (5)
| id | name | status |
|---|---|---|
| proj_pv | Signal Detection Platform (Pharmacovigilance) | Active |
| proj_audit | SOC2 / Part 11 Compliance Audit Q1 2026 | Completed |
| proj_pricing | Usage-Based Pricing Redesign | Active |
| proj_subreg | Regulatory Submission Assistant | Active |
| proj_dq | Data Quality Remediation for Adverse Event Feeds | Completed |

### Skills (5)
ML Engineering, Regulatory Affairs (GxP/Part 11), Data Engineering, Security & Access
Control, Product Strategy

### Tools (6)
PyTorch, Snowflake, AWS Audit Manager, Okta, LangChain/LangGraph, Tableau

### Documents (5)
- Q1 2026 SOC2 Audit Final Report
- Pricing Strategy Memo (March 2026)
- Signal Detection Platform Architecture Spec
- Adverse Event Data Quality Runbook
- Information Security Policy v4

### Decisions (4)
- Adopt 3-tier usage-based pricing model
- Standardize on PyTorch for ML training pipelines
- Use AWS Audit Manager as primary cloud control attestation source
- Mandate MFA for all production system access

**Total: 8 + 5 + 5 + 6 + 5 + 4 = 33 nodes**, edges cover WORKED_ON, USED_TOOL, HAS_SKILL,
APPROVED, DECISION_FOR, AUTHORED, REFERENCES, OWNS.

---

## 10 Benchmark Queries (with gold entities for recall scoring)

1. **"Who worked on the SOC2 audit and what tools did they use?"**
   Gold: Nina Castellanos, Devon Park, AWS Audit Manager, Okta

2. **"What decisions were made about the pricing model and who approved them?"**
   Gold: Adopt 3-tier usage-based pricing model, Samira El-Amin, Priya Subramaniam

3. **"Which project is the Signal Detection Platform's architecture spec related to, and who authored it?"**
   Gold: Signal Detection Platform (Pharmacovigilance), Arjun Mehta, Leo Fischer

4. **"What skills does Oliver Tan have and which project uses those skills?"**
   Gold: Oliver Tan, Regulatory Affairs (GxP/Part 11), Regulatory Submission Assistant

5. **"Who approved the decision to mandate MFA for production systems?"**
   Gold: Grace Adeyemi, Mandate MFA for all production system access, Devon Park

6. **"What tools were standardized for ML training pipelines and who proposed that decision?"**
   Gold: PyTorch, Standardize on PyTorch for ML training pipelines, Arjun Mehta

7. **"Which documents reference the Data Quality Remediation project, and who worked on it?"**
   Gold: Adverse Event Data Quality Runbook, Data Quality Remediation for Adverse Event Feeds, Leo Fischer, Priya Subramaniam

8. **"Who is the VP Engineering and which decisions did they approve?"**
   Gold: Grace Adeyemi, Mandate MFA for all production system access, Standardize on PyTorch for ML training pipelines

9. **"What is the status of the Regulatory Submission Assistant project and who owns it?"**
   Gold: Regulatory Submission Assistant, Active, Priya Subramaniam, Oliver Tan

10. **"Which security tools are used across projects and who is the security engineer?"**
    Gold: Devon Park, Okta, AWS Audit Manager, Information Security Policy v4

> Queries 1, 2, 5, 6, 8, 10 are designed to need multi-hop / relationship traversal
> (GraphRAG should win). Queries 3, 4, 7, 9 mix entity attributes + one hop (both
> pipelines may do reasonably, useful for the "when each wins" analysis).

---

## Repo Layout (target)

```
GenAcademy-GraphRAG-for-Organizational-Knowledge/
├── data/
│   └── mock_data.py          # PEOPLE, PROJECTS, SKILLS, TOOLS, DOCUMENTS, DECISIONS, EDGES
├── lib/
│   └── __init__.py
├── graph_builder.py            # build_graph(), link_entities (LLM + fallback), expand_subgraph, subgraph_to_text
├── graph_rag.py                 # LangGraph state machine for GraphRAG
├── vector_rag.py                 # FAISS baseline pipeline
├── llm.py                          # Nebius Token Factory client wrappers (chat + embeddings)
├── questions.py                    # 10 benchmark questions + gold entity sets
├── compare.py                       # runs both pipelines, prints recall
├── ui.py                              # Streamlit comparison app
├── requirements.txt
├── .env.example
└── README.md
```

---

## Deliverables Checklist (per Week 2 handout)

- [ ] Knowledge graph with 20+ nodes (we have 33)
- [ ] GraphRAG vs vector RAG comparison on 10 queries (`compare.py`)
- [ ] Analysis of when each approach wins (in README / comparison doc)
- [ ] At least one Nebius Token Factory model call (we use it for embeddings + both LLM answer nodes + entity linker)
- [ ] Streamlit demo (`ui.py`)
- [ ] Project documentation (Google Doc): overview, datasets, prompts used, iterations, learnings
- [ ] Video demo (≤5 min)
- [ ] GitHub repo
