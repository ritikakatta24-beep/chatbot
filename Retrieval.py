"""
RETRIEVAL LAYER (Gemini embeddings version — lightweight, deploy-friendly)
============================================================================
Instead of loading a local embedding model (sentence-transformers + PyTorch,
which needs 500MB+ RAM and crashes Render's free tier), this version sends
text to Gemini's embedContent API and gets back embedding vectors. Same
RAG logic (cosine similarity, top-k, threshold) — just no local model.

Install:
    pip install numpy pandas

Needs:
    export GEMINI_API_KEY=your_key_here
"""

import os
import json
import numpy as np
import pandas as pd
import urllib.request
from dotenv import load_dotenv

load_dotenv()

EMBED_MODEL = "gemini-embedding-001"
EMBED_URL = f"https://generativelanguage.googleapis.com/v1beta/models/{EMBED_MODEL}:embedContent"


def _embed_one(text: str) -> list:
    """Call Gemini's embedContent endpoint for a single piece of text."""
    api_key = os.environ.get("GEMINI_API_KEY")
    if not api_key:
        raise RuntimeError("GEMINI_API_KEY not set — required for embeddings too, not just generation.")

    payload = json.dumps({
        "content": {"parts": [{"text": text}]}
    }).encode("utf-8")

    req = urllib.request.Request(
        EMBED_URL,
        data=payload,
        headers={"Content-Type": "application/json", "x-goog-api-key": api_key},
    )
    with urllib.request.urlopen(req) as resp:
        data = json.loads(resp.read())
    return data["embedding"]["values"]


def cosine_similarity(a, b):
    a, b = np.array(a), np.array(b)
    return float(np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b)))


class Retriever:
    def __init__(self, kb_path="knowledge_base.csv"):
        self.kb_path = kb_path
        self.df = pd.read_csv(kb_path)

        self.corpus = (self.df["topic"] + ". " + self.df["info"]).tolist()
        self.fact_embeddings = self._embed_all(self.corpus)

    def _embed_all(self, texts):
        """Embed a list of texts one at a time via the API.
        (Gemini's embedContent also supports batch mode — fine to upgrade
        to that later if you have many facts and want fewer API calls.)"""
        return [_embed_one(t) for t in texts]

    def add_fact(self, topic: str, info: str):
        """Add new data WITHOUT touching any other code — this is the
        'can I add data later' answer in action. Also works if you just
        open the CSV in Excel, add a row yourself, and save."""
        new_id = int(self.df["id"].max()) + 1 if len(self.df) else 1
        new_row = pd.DataFrame([{"id": new_id, "topic": topic, "info": info}])
        self.df = pd.concat([self.df, new_row], ignore_index=True)
        self.corpus = (self.df["topic"] + ". " + self.df["info"]).tolist()
        # Only embed the NEW fact and append — no need to re-call the API
        # for facts that haven't changed (this matters more now since each
        # embedding is a network call, not a free local computation).
        self.fact_embeddings.append(_embed_one(f"{topic}. {info}"))

    def save(self, kb_path=None):
        self.df.to_csv(kb_path or self.kb_path, index=False)

    def retrieve(self, query: str, top_k: int = 3, min_score: float = 0.35):
        """
        1. Convert the user's question into an embedding (via API call)
        2. Compare it to every fact's embedding (cosine similarity)
        3. Return the top_k closest facts, above a minimum confidence score
        """
        query_embedding = _embed_one(query)
        scores = [cosine_similarity(query_embedding, fe) for fe in self.fact_embeddings]

        records = self.df.to_dict("records")
        ranked = sorted(zip(records, scores), key=lambda x: x[1], reverse=True)
        results = [(fact, score) for fact, score in ranked[:top_k] if score >= min_score]
        return results


if __name__ == "__main__":
    r = Retriever("knowledge_base.csv")

    test_questions = [
        "where is the pps lab",
        "how many canteens are there",
        "DBMS lab location",
        "where can I eat",
        "where is the sports ground",
        "what is the weather today",
    ]

    for q in test_questions:
        print(f"\nQ: {q}")
        results = r.retrieve(q)
        if not results:
            print("  -> No relevant match found (would trigger 'I don't have that info')")
        for fact, score in results:
            print(f"  [{score:.2f}] {fact['topic']}: {fact['info']}")