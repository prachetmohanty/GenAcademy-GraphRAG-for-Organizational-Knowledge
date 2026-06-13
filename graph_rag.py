"""
LangGraph state machine for the GraphRAG pipeline.

Flow:
  link_entities -> [no seeds?] -> fallback_linker -> expand_subgraph
                 -> [seeds found] ------------------> expand_subgraph
  expand_subgraph -> [0 edges with relation filter?] -> retry without filter (max 1)
  expand_subgraph -> attach_documents -> generate_answer -> END
"""

from __future__ import annotations

from typing import TypedDict

import networkx as nx
from langgraph.graph import END, StateGraph

import graph_builder
import llm

GRAPH_ANSWER_PROMPT = """You are an internal knowledge assistant for Veritask \
Sciences, a pharma-focused AI consultancy. Treat the GRAPH FACTS below as the \
primary source of truth -- they are verified relationships from the organizational \
knowledge graph. Use the SUPPORTING DOCUMENTS only for additional prose detail. \
Name every relevant person, project, tool, document, or decision by its full name. \
If the graph facts and documents do not contain enough information to answer, say \
so explicitly -- do not guess.

GRAPH FACTS:
{graph_facts}

SUPPORTING DOCUMENTS:
{documents}

Question: {question}

Answer:"""


class GraphRAGState(TypedDict, total=False):
    question: str
    seed_entities: list[str]
    relation_filters: list[str]
    used_fallback: bool
    retried_without_filter: bool
    subgraph_nodes: list[str]
    subgraph_edges: int
    graph_facts: str
    documents: str
    answer: str
    retrieved_entity_ids: list[str]


def _link_entities_node(g: nx.MultiDiGraph):
    def _run(state: GraphRAGState) -> GraphRAGState:
        seeds, relations = graph_builder.llm_link_entities(g, state["question"])
        return {**state, "seed_entities": seeds, "relation_filters": relations, "used_fallback": False}
    return _run


def _fallback_linker_node(g: nx.MultiDiGraph):
    def _run(state: GraphRAGState) -> GraphRAGState:
        seeds, relations = graph_builder.heuristic_link_entities(g, state["question"])
        return {**state, "seed_entities": seeds, "relation_filters": relations, "used_fallback": True}
    return _run


def _expand_subgraph_node(g: nx.MultiDiGraph):
    def _run(state: GraphRAGState) -> GraphRAGState:
        seeds = state.get("seed_entities", [])
        relation_filter = state.get("relation_filters") or None
        sub = graph_builder.expand_subgraph(g, seeds, hops=2, relation_filter=relation_filter)

        retried = state.get("retried_without_filter", False)
        if relation_filter and sub.number_of_edges() == 0 and not retried:
            sub = graph_builder.expand_subgraph(g, seeds, hops=2, relation_filter=None)
            retried = True

        return {
            **state,
            "subgraph_nodes": list(sub.nodes),
            "subgraph_edges": sub.number_of_edges(),
            "retried_without_filter": retried,
        }
    return _run


def _attach_documents_node(g: nx.MultiDiGraph):
    def _run(state: GraphRAGState) -> GraphRAGState:
        node_ids = state.get("subgraph_nodes", [])
        sub = g.subgraph(node_ids).copy()
        seeds = state.get("seed_entities", [])
        graph_facts = graph_builder.subgraph_to_text(g, sub, seeds)
        documents = graph_builder.attach_documents(g, sub)
        retrieved_names = graph_builder.get_retrieved_entity_names(g, sub)
        return {
            **state,
            "graph_facts": graph_facts,
            "documents": documents or "(no supporting documents in subgraph)",
            "retrieved_entity_ids": sorted(retrieved_names),
        }
    return _run


def _generate_answer_node():
    def _run(state: GraphRAGState) -> GraphRAGState:
        if state.get("subgraph_edges", 0) == 0 and not state.get("subgraph_nodes"):
            return {
                **state,
                "answer": "I could not find this in our knowledge graph -- "
                          "no relevant entities were linked to this question.",
            }
        prompt = GRAPH_ANSWER_PROMPT.format(
            graph_facts=state.get("graph_facts", ""),
            documents=state.get("documents", ""),
            question=state["question"],
        )
        answer = llm.chat_completion(messages=[{"role": "user", "content": prompt}])
        return {**state, "answer": answer}
    return _run


def _route_after_link(state: GraphRAGState) -> str:
    return "fallback_linker" if not state.get("seed_entities") else "expand_subgraph"


def build_graph_rag_app(g: nx.MultiDiGraph):
    """Compile the LangGraph state machine bound to a given knowledge graph."""
    builder = StateGraph(GraphRAGState)

    builder.add_node("link_entities", _link_entities_node(g))
    builder.add_node("fallback_linker", _fallback_linker_node(g))
    builder.add_node("expand_subgraph", _expand_subgraph_node(g))
    builder.add_node("attach_documents", _attach_documents_node(g))
    builder.add_node("generate_answer", _generate_answer_node())

    builder.set_entry_point("link_entities")
    builder.add_conditional_edges(
        "link_entities",
        _route_after_link,
        {"fallback_linker": "fallback_linker", "expand_subgraph": "expand_subgraph"},
    )
    builder.add_edge("fallback_linker", "expand_subgraph")
    builder.add_edge("expand_subgraph", "attach_documents")
    builder.add_edge("attach_documents", "generate_answer")
    builder.add_edge("generate_answer", END)

    return builder.compile()


def answer_with_graph_rag(app, question: str) -> dict:
    """Run the compiled GraphRAG app for one question."""
    result = app.invoke({"question": question})
    return {
        "answer": result.get("answer", ""),
        "seed_entities": result.get("seed_entities", []),
        "used_fallback": result.get("used_fallback", False),
        "retrieved_entity_ids": result.get("retrieved_entity_ids", []),
        "subgraph_nodes": result.get("subgraph_nodes", []),
        "subgraph_edges": result.get("subgraph_edges", 0),
        "graph_facts": result.get("graph_facts", ""),
    }