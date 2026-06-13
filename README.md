# GraphRAG for Organizational Knowledge — Veritask Sciences

Week 2 project (Track 2: LangChain + LangGraph). Compares a vector-RAG
baseline against a GraphRAG pipeline over a mock organizational knowledge
graph for **Veritask Sciences**, a fictional pharma-focused AI/digital-product
consultancy (pharmacovigilance, SOC2 / 21 CFR Part 11 compliance, regulatory
submissions, pricing).

See `PROJECT_PLAN.md` for the filled RAG framework, mock dataset design, the
10 benchmark questions with gold entity sets, and the LangGraph state machine
design.

## Setup

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
cp .env.example .env   # then add your GROQ_API_KEY
```

> Get a Groq API key from https://console.groq.com/keys. Never commit `.env`
> or paste real keys into chat -- rotate any key that has been exposed.
> Embeddings run locally via sentence-transformers (no API key needed); the
> first run will download the `all-MiniLM-L6-v2` model (~90MB).

## Run the comparison (CLI)

```bash
python compare.py
```

Prints, for each of the 10 benchmark questions, the answer and recall vs the
gold entity set for both pipelines, plus an overall summary.

## Run the Streamlit demo

```bash
streamlit run ui.py
```

Pick a benchmark question, click **Run both pipelines**, and compare the
answers side by side, recall vs gold entities, and the knowledge graph with
retrieved/traversed entities highlighted.

## Architecture

```
Question
   |
   +--> Vector RAG: flatten nodes -> embed (local MiniLM) -> FAISS -> top-8 -> LLM answer (Groq)
   |
   +--> GraphRAG (LangGraph):
          link_entities (Groq, structured JSON) --no seeds--> fallback_linker (regex/topic map)
              -> expand_subgraph (2-hop, optional relation filter, retry without filter if empty)
              -> attach_documents
              -> generate_answer (Groq, graph facts as primary truth)
```

## Mock dataset

33 nodes across People (8), Projects (5), Skills (5), Tools (6), Documents (5),
Decisions (4), with 63 edges covering WORKED_ON, USED_TOOL, HAS_SKILL,
REQUIRES_SKILL, AUTHORED, REFERENCES, DECISION_FOR, APPROVED, PROPOSED, OWNS.
Defined in `data/mock_data.py`.

## Files

| File | Purpose |
|---|---|
| `data/mock_data.py` | Mock org knowledge base (nodes + edges) |
| `graph_builder.py` | Build NetworkX graph, entity linking, subgraph expansion |
| `graph_rag.py` | LangGraph state machine for GraphRAG |
| `vector_rag.py` | FAISS baseline pipeline |
| `llm.py` | Nebius Token Factory chat + embedding wrappers |
| `questions.py` | 10 benchmark questions + gold entity sets |
| `compare.py` | CLI: runs both pipelines, prints recall |
| `ui.py` | Streamlit comparison app |

## When each approach wins

To be filled in after running `compare.py`: summarize which questions
GraphRAG wins on (typically multi-hop relationship questions: q1, q2, q5, q6,
q8, q10) vs where vector RAG is competitive (single-entity attribute lookups
with strong lexical overlap: q3, q4, q7, q9), based on actual recall numbers
from your run.
