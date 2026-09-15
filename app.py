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
from fastapi.responses import HTMLResponse
from fastapi import Header, HTTPException

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
def add_fact(fact: NewFact, x_admin_password: str = Header(None)):
    if x_admin_password != ADMIN_PASSWORD:
        raise HTTPException(status_code=401, detail="Invalid password")
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

ADMIN_PASSWORD = os.environ.get("ADMIN_PASSWORD", "changeme")

@app.get("/admin", response_class=HTMLResponse)
def admin_page():
    unanswered_html = ""
    if os.path.exists(UNANSWERED_LOG):
        with open(UNANSWERED_LOG) as f:
            lines = [line.strip() for line in f.readlines()]
        if lines:
            unanswered_html = "".join(
                f'<div class="unanswered-item">{line}</div>' for line in reversed(lines)
            )
        else:
            unanswered_html = '<p class="empty">No unanswered questions yet 🎉</p>'
    else:
        unanswered_html = '<p class="empty">No unanswered questions yet 🎉</p>'

    return f"""
    <html>
    <head>
        <title>Chatbot Admin</title>
        <meta name="viewport" content="width=device-width, initial-scale=1">
        <style>
            * {{ box-sizing: border-box; }}
            body {{
                font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', sans-serif;
                background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
                min-height: 100vh;
                margin: 0;
                padding: 40px 20px;
            }}
            .container {{
                max-width: 640px;
                margin: 0 auto;
            }}
            h1 {{
                color: white;
                text-align: center;
                margin-bottom: 30px;
                font-size: 28px;
            }}
            .card {{
                background: white;
                border-radius: 16px;
                padding: 28px;
                margin-bottom: 24px;
                box-shadow: 0 10px 30px rgba(0,0,0,0.15);
            }}
            .card h2 {{
                margin-top: 0;
                color: #333;
                font-size: 18px;
                display: flex;
                align-items: center;
                gap: 8px;
            }}
            label {{
                display: block;
                font-size: 13px;
                font-weight: 600;
                color: #555;
                margin-bottom: 6px;
                margin-top: 16px;
            }}
            input, textarea {{
                width: 100%;
                padding: 12px;
                border: 1.5px solid #e0e0e0;
                border-radius: 8px;
                font-size: 14px;
                font-family: inherit;
                transition: border-color 0.2s;
            }}
            input:focus, textarea:focus {{
                outline: none;
                border-color: #764ba2;
            }}
            button {{
                width: 100%;
                margin-top: 20px;
                padding: 13px;
                background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
                color: white;
                border: none;
                border-radius: 8px;
                font-size: 15px;
                font-weight: 600;
                cursor: pointer;
                transition: opacity 0.2s;
            }}
            button:hover {{ opacity: 0.9; }}
            #result {{
                margin-top: 14px;
                font-size: 14px;
                font-weight: 500;
            }}
            .unanswered-item {{
                background: #f8f7fc;
                border-left: 3px solid #764ba2;
                padding: 10px 14px;
                margin-bottom: 8px;
                border-radius: 6px;
                font-size: 13px;
                color: #444;
                word-break: break-word;
            }}
            .empty {{
                color: #999;
                text-align: center;
                padding: 20px 0;
            }}
            .scroll-box {{
                max-height: 300px;
                overflow-y: auto;
            }}
        </style>
    </head>
    <body>
        <div class="container">
            <h1>🎓 Campus Chatbot Admin</h1>

            <div class="card">
                <h2>🔐 Admin Access</h2>
                <label>Password</label>
                <input type="password" id="password" placeholder="Enter admin password">
            </div>

            <div class="card">
                <h2>➕ Add New Fact</h2>
                <label>Topic</label>
                <input type="text" id="topic" placeholder="e.g. DBMS Lab">
                <label>Info</label>
                <textarea id="info" rows="3" placeholder="e.g. Located on the third floor of CSIT block"></textarea>
                <button onclick="addFact()">Add Fact</button>
                <p id="result"></p>
            </div>

            <div class="card">
                <h2>❓ Unanswered Questions</h2>
                <div class="scroll-box">
                    {unanswered_html}
                </div>
            </div>
        </div>

        <script>
        async function addFact() {{
            const password = document.getElementById('password').value;
            const topic = document.getElementById('topic').value;
            const info = document.getElementById('info').value;
            const resultEl = document.getElementById('result');

            if (!topic || !info) {{
                resultEl.style.color = '#e74c3c';
                resultEl.innerText = 'Please fill in both fields.';
                return;
            }}

            resultEl.style.color = '#888';
            resultEl.innerText = 'Adding...';

            const res = await fetch('/add-fact', {{
                method: 'POST',
                headers: {{
                    'Content-Type': 'application/json',
                    'x-admin-password': password
                }},
                body: JSON.stringify({{ topic: topic, info: info }})
            }});

            const data = await res.json();
            if (res.ok) {{
                resultEl.style.color = '#27ae60';
                resultEl.innerText = '✅ Added: ' + data.topic;
                document.getElementById('topic').value = '';
                document.getElementById('info').value = '';
            }} else {{
               resultEl.style.color = '#e74c3c';
                resultEl.innerText = 'Error: ' + data.detail;
            }}
        }}
        </script>
    </body>
    </html>
    """