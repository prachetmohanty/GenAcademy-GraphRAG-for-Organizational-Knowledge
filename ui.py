"""
Streamlit app: GraphRAG vs Vector RAG side-by-side comparison over the
Veritask Sciences mock organizational knowledge graph.

Run with:
    streamlit run ui.py
"""

from __future__ import annotations

import matplotlib.pyplot as plt
import networkx as nx
import streamlit as st

import graph_builder
import graph_rag
import vector_rag
from questions import BENCHMARK_QUESTIONS

st.set_page_config(page_title="GraphRAG vs Vector RAG", layout="wide")

TYPE_COLORS = {
    "Person": "#4C8BF5",
    "Project": "#3CB371",
    "Skill": "#9B59B6",
    "Tool": "#E67E22",
    "Document": "#F1C40F",
    "Decision": "#E74C3C",
}


@st.cache_resource
def load_graph() -> nx.MultiDiGraph:
    return graph_builder.build_graph()


@st.cache_resource
def load_vector_index() -> vector_rag.VectorIndex:
    index = vector_rag.VectorIndex()
    with st.spinner("Embedding nodes locally (sentence-transformers)..."):
        index.build()
    return index


@st.cache_resource
def load_graph_rag_app(_g: nx.MultiDiGraph):
    return graph_rag.build_graph_rag_app(_g)


def recall(retrieved_names: set[str], gold: list[str]) -> tuple[float, list[str], list[str]]:
    gold_set = set(gold)
    hits = [g for g in gold if g in retrieved_names]
    misses = [g for g in gold if g not in retrieved_names]
    r = len(hits) / len(gold_set) if gold_set else 1.0
    return r, hits, misses


def draw_graph(g: nx.MultiDiGraph, highlight: set[str] | None = None) -> plt.Figure:
    fig, ax = plt.subplots(figsize=(12, 7))
    pos = nx.spring_layout(g, seed=42, k=0.6)

    node_colors = []
    node_sizes = []
    for nid, data in g.nodes(data=True):
        node_colors.append(TYPE_COLORS.get(data.get("type", ""), "#AAAAAA"))
        if highlight and nid in highlight:
            node_sizes.append(900)
        else:
            node_sizes.append(350)

    nx.draw_networkx_edges(g, pos, ax=ax, alpha=0.3, arrows=True, arrowsize=8)
    nx.draw_networkx_nodes(g, pos, ax=ax, node_color=node_colors, node_size=node_sizes,
                            edgecolors="black", linewidths=0.5)

    labels = {nid: data.get("name", nid) for nid, data in g.nodes(data=True)}
    nx.draw_networkx_labels(g, pos, labels=labels, ax=ax, font_size=7)

    ax.set_axis_off()
    return fig


def main() -> None:
    g = load_graph()

    st.title("GraphRAG vs Vector RAG")
    st.caption(
        "Both pipelines answer the same question about a mock organization "
        "(Veritask Sciences). GraphRAG traverses a knowledge graph; vector RAG "
        "retrieves the top-k most semantically similar chunks. The contrast "
        "shows when relationships matter."
    )

    # --- Sidebar ---
    with st.sidebar:
        st.header("Question")
        labels = [q["question"] for q in BENCHMARK_QUESTIONS]
        choice = st.selectbox("Pick a benchmark question", labels)
        item = next(q for q in BENCHMARK_QUESTIONS if q["question"] == choice)

        with st.expander("Why this question"):
            st.write(item["why"])
            st.markdown(f"**Gold entities ({len(item['gold_entities'])}):** "
                         + ", ".join(item["gold_entities"]))

        run = st.button("Run both pipelines", type="primary")

        st.divider()
        st.subheader("Graph legend")
        for ntype, color in TYPE_COLORS.items():
            st.markdown(
                f"<span style='background-color:{color}; color:white; "
                f"padding:2px 8px; border-radius:4px; font-size:0.8em'>{ntype}</span>",
                unsafe_allow_html=True,
            )
        st.caption(f"Graph has {g.number_of_nodes()} nodes and {g.number_of_edges()} edges.")

        highlight_choice = st.radio(
            "Highlight on the graph",
            ["Vector RAG retrieved", "GraphRAG traversed", "Union", "None"],
            index=2,
        )

    # --- Run pipelines ---
    if run:
        index = load_vector_index()
        app = load_graph_rag_app(g)

        with st.spinner("Running Vector RAG..."):
            v_result = vector_rag.answer_with_vector_rag(index, item["question"])
        with st.spinner("Running GraphRAG..."):
            gr_result = graph_rag.answer_with_graph_rag(app, item["question"])

        v_names = {name for _, name, _ in v_result["retrieved"]}
        gr_names = set(gr_result["retrieved_entity_ids"])

        v_recall, v_hits, v_misses = recall(v_names, item["gold_entities"])
        gr_recall, gr_hits, gr_misses = recall(gr_names, item["gold_entities"])

        st.session_state["v_result"] = v_result
        st.session_state["gr_result"] = gr_result
        st.session_state["v_names"] = v_names
        st.session_state["gr_names"] = gr_names
        st.session_state["v_recall"] = (v_recall, v_hits, v_misses)
        st.session_state["gr_recall"] = (gr_recall, gr_hits, gr_misses)

    # --- Graph visualization ---
    highlight_ids: set[str] = set()
    name_to_id = {data.get("name", nid): nid for nid, data in g.nodes(data=True)}

    if "v_names" in st.session_state:
        v_ids = {name_to_id[n] for n in st.session_state["v_names"] if n in name_to_id}
        gr_ids = {name_to_id[n] for n in st.session_state["gr_names"] if n in name_to_id}
        if highlight_choice == "Vector RAG retrieved":
            highlight_ids = v_ids
        elif highlight_choice == "GraphRAG traversed":
            highlight_ids = gr_ids
        elif highlight_choice == "Union":
            highlight_ids = v_ids | gr_ids

    fig = draw_graph(g, highlight=highlight_ids)
    st.pyplot(fig)

    st.divider()

    # --- Results ---
    if "v_result" in st.session_state:
        col1, col2 = st.columns(2)

        with col1:
            st.subheader("Vector RAG")
            v_recall, v_hits, v_misses = st.session_state["v_recall"]
            st.metric("Recall vs gold set", f"{v_recall:.0%}")
            if v_misses:
                st.caption(f"Missed: {', '.join(v_misses)}")
            st.markdown("**Answer**")
            st.write(st.session_state["v_result"]["answer"])
            with st.expander("Retrieved chunks"):
                for _, name, chunk in st.session_state["v_result"]["retrieved"]:
                    st.markdown(f"- **{name}**: {chunk}")

        with col2:
            st.subheader("GraphRAG")
            gr_recall, gr_hits, gr_misses = st.session_state["gr_recall"]
            st.metric("Recall vs gold set", f"{gr_recall:.0%}")
            if gr_misses:
                st.caption(f"Missed: {', '.join(gr_misses)}")
            st.markdown("**Answer**")
            st.write(st.session_state["gr_result"]["answer"])
            with st.expander("Traversed subgraph facts"):
                st.text(st.session_state["gr_result"]["graph_facts"])
            st.caption(
                f"Seeds: {', '.join(st.session_state['gr_result']['seed_entities']) or '(none)'} "
                f"| Fallback linker used: {st.session_state['gr_result']['used_fallback']}"
            )
    else:
        st.info("Pick a question and click **Run both pipelines** to compare results.")


if __name__ == "__main__":
    main()
