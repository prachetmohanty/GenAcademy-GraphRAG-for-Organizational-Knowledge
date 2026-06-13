"""
Vector RAG baseline: flatten every graph node into one text chunk, embed
locally with sentence-transformers, store in FAISS, retrieve top-k, and ask
the LLM (Groq) to answer from those chunks.
"""

from __future__ import annotations

import numpy as np

try:
    import faiss
except ImportError as exc:  # pragma: no cover
    raise ImportError(
        "faiss-cpu is required for vector_rag.py. Install with "
        "`pip install faiss-cpu`."
    ) from exc

from data.mock_data import ALL_NODES
import llm

VECTOR_ANSWER_PROMPT = """You are an internal knowledge assistant for Veritask \
Sciences, a pharma-focused AI consultancy. Answer the user's question using ONLY \
the retrieved context chunks below. Name every relevant person, project, tool, \
document, or decision by its full name. If the context does not contain enough \
information to answer, say so explicitly -- do not guess.

Retrieved context:
{context}

Question: {question}

Answer:"""


class VectorIndex:
    """A small in-memory FAISS index over flattened graph-node chunks."""

    def __init__(self):
        self.ids: list[str] = []
        self.chunks: list[str] = []
        self.names: list[str] = []
        self.index: faiss.Index | None = None

    def build(self) -> None:
        self.ids = [n["id"] for n in ALL_NODES]
        self.names = [n["name"] for n in ALL_NODES]
        self.chunks = [_flatten_node(n) for n in ALL_NODES]

        vectors = llm.embed_texts(self.chunks)
        arr = np.array(vectors, dtype="float32")
        faiss.normalize_L2(arr)

        dim = arr.shape[1]
        self.index = faiss.IndexFlatIP(dim)  # cosine similarity via normalized inner product
        self.index.add(arr)

    def search(self, query: str, top_k: int = 8) -> list[tuple[str, str, str]]:
        """Return a list of (node_id, name, chunk_text) for the top_k matches."""
        if self.index is None:
            raise RuntimeError("Call build() before search().")

        q_vec = np.array(llm.embed_texts([query]), dtype="float32")
        faiss.normalize_L2(q_vec)

        k = min(top_k, len(self.chunks))
        _, idxs = self.index.search(q_vec, k)
        return [(self.ids[i], self.names[i], self.chunks[i]) for i in idxs[0]]


def _flatten_node(node: dict) -> str:
    """Flatten a single graph node into one short text chunk for embedding."""
    return f"[{node.get('type', '')}] {node.get('name', '')}: {node.get('text', '')}"


def answer_with_vector_rag(index: VectorIndex, question: str, top_k: int = 8) -> dict:
    """
    Run the vector RAG pipeline for one question.

    Returns {"answer": str, "retrieved": [(id, name, chunk), ...]}.
    """
    retrieved = index.search(question, top_k=top_k)
    context = "\n\n".join(f"[{name}] {chunk}" for _, name, chunk in retrieved)

    prompt = VECTOR_ANSWER_PROMPT.format(context=context, question=question)
    answer = llm.chat_completion(messages=[{"role": "user", "content": prompt}])

    return {"answer": answer, "retrieved": retrieved}
