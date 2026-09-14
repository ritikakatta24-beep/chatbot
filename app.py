"""
MAIN API
========
This is the '3 fixed steps' backend we talked about:
  1. Take user's question
  2. Retrieve relevant facts (retrieval.py)
  3. Send to LLM with strict instructions, return the answer

This code NEVER changes when you add new data — it just reads whatever
is currently in knowledge_base.json at query time.

Run locally with:
    export GEMINI_API_KEY=your_key_here
    uvicorn app:app --reload
Then open http://127.0.0.1:8000/docs to test in the browser.
"""
from dotenv import load_dotenv
load_dotenv()
import os
import json
from datetime import datetime
from fastapi import FastAPI
from pydantic import BaseModel
from Retrieval import Retriever
import urllib.request

app = FastAPI(title="College Campus Chatbot")
retriever = Retriever("knowledge_base.csv")

UNANSWERED_LOG = "unanswered_questions.log"


class Question(BaseModel):
    query: str


class NewFact(BaseModel):
    topic: str
    info: str

import time

def call_gemini(system_prompt: str, user_message: str, max_retries: int = 3) -> str:
    """Minimal Gemini API call using urllib (no extra dependency needed).
    Free tier via aistudio.google.com — no card required to start.
    Retries on transient errors (503, 429) with exponential backoff."""
    api_key = os.environ.get("GEMINI_API_KEY")
    if not api_key:
        return "[No GEMINI_API_KEY set — set it as an environment variable to enable real answers]"

    model = "gemini-flash-lite-latest"
    url = f"https://generativelanguage.googleapis.com/v1beta/models/{model}:generateContent"

    payload = json.dumps({
        "system_instruction": {"parts": [{"text": system_prompt}]},
        "contents": [{"role": "user", "parts": [{"text": user_message}]}],
        "generationConfig": {"maxOutputTokens": 300},
    }).encode("utf-8")

    req = urllib.request.Request(
        url,
        data=payload,
        headers={"Content-Type": "application/json", "x-goog-api-key": api_key},
    )

    for attempt in range(max_retries):
        try:
            with urllib.request.urlopen(req) as resp:
                data = json.loads(resp.read())
                return data["candidates"][0]["content"]["parts"][0]["text"]
        except urllib.error.HTTPError as e:
            if e.code in (503, 429) and attempt < max_retries - 1:
                wait = 2 ** attempt  # 1s, 2s, 4s
                time.sleep(wait)
                continue
            return f"[Gemini API error {e.code}: {e.reason}. Please try again in a moment.]"
        except Exception as e:
            return f"[Unexpected error calling Gemini: {str(e)}]"

    return "[Gemini API is temporarily unavailable. Please try again shortly.]"

def log_unanswered(query: str):
    with open(UNANSWERED_LOG, "a") as f:
        f.write(f"{datetime.now().isoformat()} | {query}\n")


@app.post("/ask")
def ask(question: Question):
    # STEP 1: Retrieve relevant facts
    results = retriever.retrieve(question.query)

    if not results:
        log_unanswered(question.query)  # so you know what to add later
        return {
            "answer": "I don't have that information yet. Try asking the admin office, or check back soon!",
            "matched_facts": [],
        }

    # STEP 2: Build strict context for the LLM
    context = "\n".join(f"- {fact['topic']}: {fact['info']}" for fact, score in results)

    system_prompt = (
        "You are a helpful college campus assistant. "
        "Answer the student's question using ONLY the information given below. "
        "Do not use any outside knowledge. Keep answers short and clear. "
        "If the answer is not fully covered by the information given, say: "
        "'I don't have complete information on that yet.'\n\n"
        f"CAMPUS INFORMATION:\n{context}"
    )

    # STEP 3: Generate natural-language answer
    answer = call_gemini(system_prompt, question.query)

    return {
        "answer": answer,
        "matched_facts": [fact["topic"] for fact, score in results],
    }


@app.post("/add-fact")
def add_fact(fact: NewFact):
    """This is your answer to 'can I add data later' — no code change needed,
    just a new entry, and it's immediately searchable."""
    retriever.add_fact(fact.topic, fact.info)
    retriever.save("knowledge_base.csv")
    return {"status": "added", "topic": fact.topic}


@app.get("/unanswered")
def get_unanswered():
    """See what students asked that you don't have data for yet."""
    if not os.path.exists(UNANSWERED_LOG):
        return {"unanswered": []}
    with open(UNANSWERED_LOG) as f:
        lines = [line.strip() for line in f.readlines()]
    return {"unanswered": lines}