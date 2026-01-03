import json
import numpy as np
from sentence_transformers import SentenceTransformer
import ollama
from utils.transcript_rubric import RUBRIC

client = ollama.Client(host='http://ollama:11434')

# ============================================================
# 1. Embedding 
# ============================================================
embedding_model = SentenceTransformer("sentence-transformers/all-mpnet-base-v2")

def embed_text(text: str) -> np.ndarray:
    """
    Generate embeddings locally
    """
    return embedding_model.encode(text)


# ============================================================
# 2. Local LLM Scoring (FREE)
# ============================================================
def llm_score_answer(question_id: int, question: str, answer: str) -> dict:
    rubric = RUBRIC[question_id]

    prompt = f"""
You are a strict interview evaluator.

Evaluate the answer ONLY based on RELEVANCE using this rubric: 
{json.dumps(rubric, indent=2)}

Question:
{question}

Answer:
{answer}

Choose a score from 0 to 4. Give a short reasoning.

Output JSON ONLY, like:
{{
"score": 3,
"reason": "..."
}}
"""

    response = client.generate(
        model="llama3.2",
        prompt=prompt,
        stream=False
    )

    raw = response["response"].strip()

    # Try strict JSON parse
    try:
        return json.loads(raw)
    except:
        # Attempt to slice JSON only
        start = raw.find("{")
        end = raw.rfind("}") + 1
        if start != -1 and end != -1:
            try:
                return json.loads(raw[start:end])
            except:
                pass
        raise ValueError(f"LLM returned invalid JSON:\n{raw}")


# ============================================================
# 3. Final evaluator
# ============================================================
def evaluate_transcript(question_id: int, question: str, answer: str) -> dict:
    scoring = llm_score_answer(question_id, question, answer)

    return {
        "id": question_id,
        "score": scoring["score"],
        "reason": scoring["reason"]
    }
