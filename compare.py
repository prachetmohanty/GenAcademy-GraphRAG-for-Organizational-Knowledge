"""
Run both pipelines (Vector RAG baseline and GraphRAG) over the 10 benchmark
questions and print recall against the gold entity sets for each.

Usage:
    python compare.py
"""

from __future__ import annotations

import graph_builder
import graph_rag
import vector_rag
from questions import BENCHMARK_QUESTIONS


def _recall(retrieved_names: set[str], gold: list[str]) -> tuple[float, list[str], list[str]]:
    gold_set = set(gold)
    hits = [g for g in gold if g in retrieved_names]
    misses = [g for g in gold if g not in retrieved_names]
    recall = len(hits) / len(gold_set) if gold_set else 1.0
    return recall, hits, misses


def main() -> None:
    print("Building knowledge graph...")
    g = graph_builder.build_graph()
    print(f"Graph has {g.number_of_nodes()} nodes and {g.number_of_edges()} edges.\n")

    print("Building FAISS vector index (embedding all nodes locally)...")
    index = vector_rag.VectorIndex()
    index.build()
    print("Vector index ready.\n")

    print("Compiling GraphRAG LangGraph app...")
    app = graph_rag.build_graph_rag_app(g)
    print("Ready.\n")

    vector_recalls: list[float] = []
    graph_recalls: list[float] = []

    for item in BENCHMARK_QUESTIONS:
        qid = item["id"]
        question = item["question"]
        gold = item["gold_entities"]

        print("=" * 80)
        print(f"[{qid}] {question}")
        print(f"Gold entities: {gold}")

        # --- Vector RAG ---
        v_result = vector_rag.answer_with_vector_rag(index, question)
        v_names = {name for _, name, _ in v_result["retrieved"]}
        v_recall, v_hits, v_misses = _recall(v_names, gold)
        vector_recalls.append(v_recall)

        print(f"\nVector RAG recall: {v_recall:.0%}  (hits={v_hits}, misses={v_misses})")
        print(f"Vector RAG answer:\n{v_result['answer']}")

        # --- GraphRAG ---
        gr_result = graph_rag.answer_with_graph_rag(app, question)
        gr_names = set(gr_result["retrieved_entity_ids"])
        gr_recall, gr_hits, gr_misses = _recall(gr_names, gold)
        graph_recalls.append(gr_recall)

        print(f"\nGraphRAG recall: {gr_recall:.0%}  (hits={gr_hits}, misses={gr_misses})")
        print(f"GraphRAG seeds: {gr_result['seed_entities']} "
              f"(fallback used: {gr_result['used_fallback']})")
        print(f"GraphRAG answer:\n{gr_result['answer']}\n")

    print("=" * 80)
    print("SUMMARY")
    print(f"Vector RAG  avg recall: {sum(vector_recalls) / len(vector_recalls):.0%}")
    print(f"GraphRAG    avg recall: {sum(graph_recalls) / len(graph_recalls):.0%}")


if __name__ == "__main__":
    main()
