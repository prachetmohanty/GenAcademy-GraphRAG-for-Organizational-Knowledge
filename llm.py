"""
LLM and embedding wrappers.

Chat completions (VECTOR_ANSWER, GRAPH_ANSWER, ENTITY_EXTRACTOR) go to Groq's
OpenAI-compatible API. Embeddings (used only by the FAISS vector-RAG baseline)
run locally via sentence-transformers, since Groq does not serve an embeddings
endpoint.

Set GROQ_API_KEY in your environment (or a .env file) before running.
Never commit real API keys -- use .env (gitignored) or your shell environment.
"""

from __future__ import annotations

import json
import os

from dotenv import load_dotenv
 
load_dotenv() 

GROQ_BASE_URL = "https://api.groq.com/openai/v1"
GROQ_MODEL="llama-3.1-8b-instant"
# Defaults -- override via environment variables.
CHAT_MODEL = os.environ.get("GROQ_MODEL", "llama-3.1-8b-instant")
# Entity linking benefits from a larger model for reliable structured JSON output.
# Falls back to CHAT_MODEL if not set.
LINK_MODEL = os.environ.get("GROQ_LINK_MODEL", CHAT_MODEL)
# Local embedding model (no API key needed). Swap for a larger model if you
# have GPU available, e.g. "BAAI/bge-base-en-v1.5".
EMBED_MODEL = os.environ.get("EMBED_MODEL", "sentence-transformers/all-MiniLM-L6-v2")

_embedder = None  # lazily-loaded SentenceTransformer instance


def _client():
    from openai import OpenAI

    api_key = os.environ.get("GROQ_API_KEY")
    print(api_key)
    if not api_key:
        raise RuntimeError(
            "GROQ_API_KEY is not set. Export it or add it to a .env file "
            "before running the app. Get a key from https://console.groq.com/keys"
        )
    return OpenAI(base_url=GROQ_BASE_URL, api_key=api_key)


def chat_completion(messages: list[dict], model: str = CHAT_MODEL, temperature: float = 0.0) -> str:
    """Single-turn chat completion. Returns the assistant's text content."""
    client = _client()
    resp = client.chat.completions.create(
        model=model,
        messages=messages,
        temperature=temperature,
    )
    return resp.choices[0].message.content or ""


def structured_chat_completion(
    messages: list[dict],
    schema: dict,
    model: str = LINK_MODEL,
    temperature: float = 0.0,
) -> dict:
    """
    Chat completion that returns JSON matching `schema` (used by the entity
    linker). Groq supports `response_format={"type": "json_object"}` on most
    models but not full json_schema validation, so we add the schema to the
    prompt as an instruction and parse the result defensively.
    """
    client = _client()

    schema_hint = (
        "\n\nRespond with ONLY a JSON object matching this schema (no markdown, "
        f"no commentary):\n{json.dumps(schema)}"
    )
    augmented_messages = list(messages)
    if augmented_messages and augmented_messages[-1]["role"] == "user":
        augmented_messages[-1] = {
            **augmented_messages[-1],
            "content": augmented_messages[-1]["content"] + schema_hint,
        }

    try:
        resp = client.chat.completions.create(
            model=model,
            messages=augmented_messages,
            temperature=temperature,
            response_format={"type": "json_object"},
        )
        content = resp.choices[0].message.content or "{}"
    except Exception:
        # Some models/deployments reject response_format -- retry without it.
        resp = client.chat.completions.create(
            model=model,
            messages=augmented_messages,
            temperature=temperature,
        )
        content = resp.choices[0].message.content or "{}"

    content = content.strip()
    if content.startswith("```"):
        content = content.strip("`")
        if content.startswith("json"):
            content = content[4:]
    try:
        return json.loads(content)
    except json.JSONDecodeError:
        return {"entities": [], "relations": []}


def embed_texts(texts: list[str], model: str = EMBED_MODEL) -> list[list[float]]:
    """
    Embed a batch of texts locally via sentence-transformers (no API call,
    no API key required). Loads the model once and caches it.
    """
    global _embedder
    if _embedder is None:
        from sentence_transformers import SentenceTransformer

        _embedder = SentenceTransformer(model)

    vectors = _embedder.encode(texts, normalize_embeddings=False)
    return [v.tolist() for v in vectors]
