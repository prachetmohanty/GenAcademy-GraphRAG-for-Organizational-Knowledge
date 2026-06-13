"""
Build the organizational knowledge graph and provide the helpers GraphRAG needs:
entity linking, k-hop subgraph expansion, and subgraph -> text serialization.

Two entity linkers are provided:
  - llm_link_entities: asks the LLM to extract entity names + relation types
    from the query, then fuzzy-matches names to node IDs. This is the
    primary linker.
  - heuristic_link_entities: regex over node names + a small topic map.
    Cheap, deterministic, used as a safety net when the LLM linker
    returns no seeds.
"""

from __future__ import annotations

import difflib
import re

import networkx as nx

from data.mock_data import ALL_NODES, EDGES
import llm

# Topic shortcuts for the deterministic fallback linker, e.g. mapping
# "audit" or "SOC2" mentions straight to the relevant project/document nodes.
TOPIC_SHORTCUTS = {
    "audit": ["proj_audit", "doc_soc2", "dec_audit_mgr"],
    "soc2": ["proj_audit", "doc_soc2", "dec_audit_mgr"],
    "part 11": ["proj_audit", "doc_soc2"],
    "pricing": ["proj_pricing", "doc_pricing_memo", "dec_pricing"],
    "mfa": ["dec_mfa", "doc_sec_policy", "tool_okta"],
    "security": ["skill_sec", "doc_sec_policy", "tool_okta", "p_devon"],
    "signal detection": ["proj_pv", "doc_pv_arch"],
    "pharmacovigilance": ["proj_pv", "doc_pv_arch"],
    "pytorch": ["tool_pytorch", "dec_pytorch", "p_arjun"],
    "ml training": ["tool_pytorch", "dec_pytorch"],
    "data quality": ["proj_dq", "doc_dq_runbook"],
    "submission": ["proj_subreg"],
    "regulatory": ["proj_subreg", "skill_reg"],
    "vp engineering": ["p_grace"],
    "vp of engineering": ["p_grace"],
}

RELATION_TYPES = [
    "WORKED_ON", "USED_TOOL", "HAS_SKILL", "REQUIRES_SKILL",
    "AUTHORED", "REFERENCES", "DECISION_FOR", "APPROVED", "PROPOSED", "OWNS",
]


def build_graph() -> nx.MultiDiGraph:
    """Build the organizational knowledge graph from mock_data."""
    g = nx.MultiDiGraph()
    for node in ALL_NODES:
        g.add_node(node["id"], **node)
    for src, dst, rel in EDGES:
        g.add_edge(src, dst, relation=rel)
    return g


def _node_label(g: nx.MultiDiGraph, node_id: str) -> str:
    return g.nodes[node_id].get("name", node_id)


# ---------------------------------------------------------------------------
# Entity linking
# ---------------------------------------------------------------------------

ENTITY_EXTRACTOR_SCHEMA = {
    "type": "object",
    "properties": {
        "entities": {
            "type": "array",
            "items": {"type": "string"},
            "description": "Names of people, projects, tools, skills, documents, "
                            "or decisions mentioned or implied in the question.",
        },
        "relations": {
            "type": "array",
            "items": {"type": "string", "enum": RELATION_TYPES},
            "description": "Relation types implied by the question, if any.",
        },
    },
    "required": ["entities", "relations"],
}

ENTITY_EXTRACTOR_PROMPT = """You are an entity extractor for an organizational \
knowledge graph about Veritask Sciences, a pharma-focused AI consultancy.

Given a user question, extract:
- entities: names of people, projects, tools, skills, documents, or decisions \
mentioned or clearly implied (use the names as written or your best guess at the \
canonical name).
- relations: any of {relations} that the question is asking about.

Return JSON only, matching the provided schema.

Question: {question}"""


def llm_link_entities(g: nx.MultiDiGraph, question: str, threshold: float = 0.6) -> tuple[list[str], list[str]]:
    """
    Use the LLM to extract entity names + relation types from the question,
    then fuzzy-match extracted names to graph node IDs.

    Returns (seed_node_ids, relation_filters).
    """
    prompt = ENTITY_EXTRACTOR_PROMPT.format(relations=", ".join(RELATION_TYPES), question=question)
    result = llm.structured_chat_completion(
        messages=[{"role": "user", "content": prompt}],
        schema=ENTITY_EXTRACTOR_SCHEMA,
    )

    extracted_names = result.get("entities", []) or []
    relations = result.get("relations", []) or []

    seeds: list[str] = []
    labels = {nid: _node_label(g, nid) for nid in g.nodes}
    for name in extracted_names:
        # exact (case-insensitive) match first
        match = next((nid for nid, label in labels.items() if label.lower() == name.lower()), None)
        if match is None:
            # fuzzy match against node names
            close = difflib.get_close_matches(name, list(labels.values()), n=1, cutoff=threshold)
            if close:
                match = next(nid for nid, label in labels.items() if label == close[0])
        if match and match not in seeds:
            seeds.append(match)

    return seeds, relations


def heuristic_link_entities(g: nx.MultiDiGraph, question: str) -> tuple[list[str], list[str]]:
    """
    Deterministic fallback linker: regex over node names in the question,
    plus a small topic-shortcut map (e.g. "audit" -> SOC2-related nodes).

    Returns (seed_node_ids, relation_filters=[]).
    """
    q_lower = question.lower()
    seeds: list[str] = []

    # Exact node-name mentions
    for nid, data in g.nodes(data=True):
        name = data.get("name", "")
        if name and name.lower() in q_lower:
            if nid not in seeds:
                seeds.append(nid)

    # Topic shortcuts
    for topic, node_ids in TOPIC_SHORTCUTS.items():
        if topic in q_lower:
            for nid in node_ids:
                if nid in g.nodes and nid not in seeds:
                    seeds.append(nid)

    return seeds, []


def link_entities(g: nx.MultiDiGraph, question: str) -> tuple[list[str], list[str]]:
    """
    Primary entry point: try the LLM linker first; if it returns no seeds,
    fall back to the deterministic heuristic linker.
    """
    seeds, relations = llm_link_entities(g, question)
    if not seeds:
        seeds, relations = heuristic_link_entities(g, question)
    return seeds, relations


# ---------------------------------------------------------------------------
# Subgraph expansion
# ---------------------------------------------------------------------------

def expand_subgraph(
    g: nx.MultiDiGraph,
    seed_ids: list[str],
    hops: int = 2,
    relation_filter: list[str] | None = None,
) -> nx.MultiDiGraph:
    """
    Perform a `hops`-hop breadth-first traversal from the seed nodes in both
    directions (treating the graph as undirected for traversal purposes,
    but preserving original edge direction/relation in the returned subgraph).

    If relation_filter is given, only traverse edges whose relation is in the
    filter. If that produces an empty subgraph (no edges beyond the seeds),
    the caller should retry with relation_filter=None.
    """
    seed_ids = [s for s in seed_ids if s in g.nodes]
    if not seed_ids:
        return nx.MultiDiGraph()

    visited = set(seed_ids)
    frontier = set(seed_ids)

    for _ in range(hops):
        next_frontier = set()
        for node in frontier:
            # outgoing edges
            for _, target, data in g.out_edges(node, data=True):
                if relation_filter and data.get("relation") not in relation_filter:
                    continue
                if target not in visited:
                    next_frontier.add(target)
            # incoming edges
            for source, _, data in g.in_edges(node, data=True):
                if relation_filter and data.get("relation") not in relation_filter:
                    continue
                if source not in visited:
                    next_frontier.add(source)
        if not next_frontier:
            break
        visited |= next_frontier
        frontier = next_frontier

    sub = g.subgraph(visited).copy()
    return sub


def subgraph_to_text(g: nx.MultiDiGraph, sub: nx.MultiDiGraph, seed_ids: list[str]) -> str:
    """
    Serialize a subgraph into a structured-fact text block for the LLM prompt:
    one line per node (name, type, attributes) and one line per edge
    (source -name-> relation -> target).
    """
    lines: list[str] = []

    lines.append("ENTITIES:")
    for nid, data in sub.nodes(data=True):
        marker = " (seed)" if nid in seed_ids else ""
        name = data.get("name", nid)
        ntype = data.get("type", "")
        lines.append(f"- [{ntype}] {name}{marker}")

    lines.append("")
    lines.append("RELATIONSHIPS:")
    seen_edges = set()
    for u, v, data in sub.edges(data=True):
        rel = data.get("relation", "RELATED_TO")
        u_name = _node_label(g, u)
        v_name = _node_label(g, v)
        key = (u_name, rel, v_name)
        if key in seen_edges:
            continue
        seen_edges.add(key)
        lines.append(f"- {u_name} --{rel}--> {v_name}")

    return "\n".join(lines)


def attach_documents(g: nx.MultiDiGraph, sub: nx.MultiDiGraph) -> str:
    """Return the full text of any Document-type nodes inside the subgraph."""
    lines: list[str] = []
    for nid, data in sub.nodes(data=True):
        if data.get("type") == "Document":
            lines.append(f"--- {data.get('name', nid)} ---")
            lines.append(data.get("text", ""))
    return "\n".join(lines)


def get_retrieved_entity_names(g: nx.MultiDiGraph, sub: nx.MultiDiGraph) -> set[str]:
    """Return the set of human-readable entity names present in a subgraph."""
    return {data.get("name", nid) for nid, data in sub.nodes(data=True)}
