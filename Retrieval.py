"""
RETRIEVAL LAYER (real embeddings version)
==========================================
Uses sentence-transformers to convert text into true semantic embeddings —
sentences with similar MEANING land close together in vector space, even
with completely different wording (e.g. "where can I eat" ~ "canteen").

Install first:
    pip install sentence-transformers numpy pandas

Model used: all-MiniLM-L6-v2
- Small (~80MB), fast, runs fine on CPU, great accuracy for this kind of task.
"""
# SentenceTransformed automaticalaly download pytorch and other dependencies when you install it, so you don't need to install them separately.
import pandas as pd
from sentence_transformers import SentenceTransformer, util


class Retriever:
    def __init__(self, kb_path="knowledge_base.csv"):
        self.kb_path = kb_path
        self.df = pd.read_csv(kb_path)

        # Load the embedding model once (this is a separate model from the LLM)
        self.model = SentenceTransformer("all-MiniLM-L6-v2") # here this all-MiniLM-L6-v2 is an embedding model which is nothing but neural network only

        self.corpus = (self.df["topic"] + ". " + self.df["info"]).tolist()
        self.fact_embeddings = self._embed(self.corpus)

    def _embed(self, texts):
        """Convert a list of texts into embedding vectors."""
        return self.model.encode(texts, convert_to_tensor=True)

    def add_fact(self, topic: str, info: str):
        """Add new data WITHOUT touching any other code — this is the
        'can I add data later' answer in action. Also works if you just
        open the CSV in Excel, add a row yourself, and save."""
        new_id = int(self.df["id"].max()) + 1 if len(self.df) else 1
        new_row = pd.DataFrame([{"id": new_id, "topic": topic, "info": info}])
        self.df = pd.concat([self.df, new_row], ignore_index=True)
        self.corpus = (self.df["topic"] + ". " + self.df["info"]).tolist()
        # Re-embed everything (fine for small/medium knowledge bases;
        # for thousands of facts, you'd only embed the new one and append)
        self.fact_embeddings = self._embed(self.corpus)

    def save(self, kb_path=None):
        self.df.to_csv(kb_path or self.kb_path, index=False)

    def retrieve(self, query: str, top_k: int = 3, min_score: float = 0.35):
        """
        1. Convert the user's question into an embedding
        2. Compare it to every fact's embedding (cosine similarity)
        3. Return the top_k closest facts, above a minimum confidence score

        Note: min_score is higher here (0.35) than a TF-IDF version would use
        because real embeddings give smoother, more meaningful similarity
        scores — tune this threshold based on testing with real questions.
        """
        query_embedding = self.model.encode(query, convert_to_tensor=True)
        scores = util.cos_sim(query_embedding, self.fact_embeddings)[0]
        scores = scores.cpu().numpy()

        # Pair each row (as a dict) with its score, sort best-first
        records = self.df.to_dict("records")
        ranked = sorted(zip(records, scores), key=lambda x: x[1], reverse=True)
        results = [(fact, float(score)) for fact, score in ranked[:top_k] if score >= min_score]
        return results


if __name__ == "__main__":
    r = Retriever("knowledge_base.csv")

    test_questions = [
        "where is the pps lab",
        "how many canteens are there",
        "DBMS lab location",
        "where can I eat",          # paraphrased — should still match canteen now
        "where is the sports ground",
        "what is the weather today",  # should find nothing relevant
    ]

    for q in test_questions:
        print(f"\nQ: {q}")
        results = r.retrieve(q)
        if not results:
            print("  -> No relevant match found (would trigger 'I don't have that info')")
        for fact, score in results:
            print(f"  [{score:.2f}] {fact['topic']}: {fact['info']}")