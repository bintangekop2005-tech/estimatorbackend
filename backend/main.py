import tiktoken

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel


app = FastAPI(title="OpenAI Token Estimator")


app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


# =========================
# MODEL
# =========================

MODEL = "gpt-4o-mini"

encoding = tiktoken.encoding_for_model(MODEL)


# =========================
# REQUEST
# =========================

class PromptRequest(BaseModel):
    prompt: str


# =========================
# ESTIMATE TOKEN
# =========================

@app.post("/estimate")
def estimate_token(request: PromptRequest):

    prompt = request.prompt

    if not prompt.strip():
        return {
            "success": False,
            "message": "Prompt tidak boleh kosong"
        }

    input_tokens = len(
        encoding.encode(prompt)
    )

    return {
        "success": True,
        "model": MODEL,
        "estimated_input_tokens": input_tokens
    }