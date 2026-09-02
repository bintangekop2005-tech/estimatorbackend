import os
import time

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import google.generativeai as genai
from pinecone import Pinecone

app = FastAPI(
    title="Gemini Token Tracker"
)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"]
)

MODEL = "gemini-3.5-flash"
TOKEN_LIMIT = 1000
PINECONE_INDEX_NAME = "selfai1"
PINECONE_NAMESPACE = "_default_"
PINECONE_TOP_K = 10

# Berapa lama quota per user bertahan sebelum auto-reset.
# Ini yang menggantikan "reset saat refresh" -> reset berkala,
# supaya user tidak bisa bypass limit cuma dengan refresh halaman,
# tapi juga tidak terkunci selamanya.
RESET_INTERVAL_SECONDS = int(
    os.getenv("RESET_INTERVAL_SECONDS", str(24 * 60 * 60))  # default 24 jam
)

# Apakah token "thinking/reasoning" Gemini ikut dihitung ke quota user.
# Default False karena biasanya user hanya ingin menghitung token
# yang benar-benar terlihat (input + output jawaban).
COUNT_THINKING_TOKENS = os.getenv("COUNT_THINKING_TOKENS", "false").lower() == "true"

GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
PINECONE_API_KEY = os.getenv("PINECONE_API_KEY")

if not GEMINI_API_KEY:
    raise RuntimeError("GEMINI_API_KEY belum ditemukan.")

genai.configure(api_key=GEMINI_API_KEY)
model = genai.GenerativeModel(MODEL)

pinecone = None
index = None
if PINECONE_API_KEY:
    try:
        pinecone = Pinecone(api_key=PINECONE_API_KEY)
        index = pinecone.Index(PINECONE_INDEX_NAME)
    except Exception as error:
        print("Pinecone initialization error:", error)

# ============================================================
# IN-MEMORY STORAGE
# ============================================================
# CATATAN PENTING (baca ini!):
# Dict ini disimpan di RAM proses Python. Di Vercel (serverless):
#   - Bisa hilang kapan saja saat instance cold-start / di-scale ulang.
#   - Kalau ada lebih dari 1 instance aktif bersamaan, tiap instance
#     punya hitungan usage SENDIRI-SENDIRI (tidak sinkron).
# Artinya limiter ini TIDAK reliable untuk production yang serius.
# Untuk production, ganti storage ini dengan Redis (misal Upstash,
# yang punya REST API dan cocok untuk serverless) atau database lain
# yang punya TTL bawaan. Untuk sekarang (dev/testing) ini cukup jalan.
user_usage = {}
# Struktur per user: {"used": int, "reset_at": float(epoch)}


def _get_user_record(user_id: str):
    now = time.time()
    record = user_usage.get(user_id)

    if record is None or now >= record["reset_at"]:
        record = {
            "used": 0,
            "reset_at": now + RESET_INTERVAL_SECONDS
        }
        user_usage[user_id] = record

    return record


def get_used_tokens(user_id: str):
    return _get_user_record(user_id)["used"]


def get_reset_at(user_id: str):
    return _get_user_record(user_id)["reset_at"]


def add_used_tokens(user_id: str, tokens: int):
    record = _get_user_record(user_id)
    record["used"] = record["used"] + tokens


def reset_user(user_id: str):
    user_usage.pop(user_id, None)


class ChatRequest(BaseModel):
    user_id: str
    message: str


@app.get("/")
def root():
    return {
        "status": "ok",
        "message": "Gemini Token Tracker API",
        "model": MODEL,
        "token_limit": TOKEN_LIMIT
    }


@app.get("/api/test")
def test():
    return {
        "status": "ok",
        "message": "FastAPI berhasil berjalan."
    }


@app.get("/api/quota/{user_id}")
def quota(user_id: str):
    used = get_used_tokens(user_id)
    remaining = max(TOKEN_LIMIT - used, 0)
    return {
        "user_id": user_id,
        "limit": TOKEN_LIMIT,
        "used": used,
        "remaining": remaining,
        "reset_at": get_reset_at(user_id)
    }


@app.post("/api/reset/{user_id}")
def reset_quota(user_id: str):
    """
    Reset manual untuk testing/debug, supaya tidak perlu redeploy
    hanya untuk mengosongkan quota satu user.
    Kalau ini dipakai di production, sebaiknya dilindungi API key/admin auth.
    """
    reset_user(user_id)
    return {
        "success": True,
        "message": f"Quota untuk user '{user_id}' berhasil di-reset.",
        "user_id": user_id,
        "limit": TOKEN_LIMIT,
        "used": 0,
        "remaining": TOKEN_LIMIT
    }


def search_pinecone(query: str):
    if index is None:
        return ""
    try:
        result = index.search(
            namespace=PINECONE_NAMESPACE,
            query={
                "inputs": {"text": query},
                "top_k": PINECONE_TOP_K
            },
            fields=["chunk_text", "text", "content", "source", "title"]
        )
        contexts = []
        try:
            hits = result["result"]["hits"]
        except Exception:
            return ""

        for hit in hits:
            fields = hit.get("fields", {})
            text = (
                fields.get("chunk_text")
                or fields.get("text")
                or fields.get("content")
            )
            if text:
                contexts.append(str(text))
        return "\n\n".join(contexts)

    except Exception as error:
        print("Pinecone search error:", error)
        return ""


def build_prompt(message: str, context: str):
    if context:
        return f"""
Anda adalah AI assistant.

Gunakan konteks berikut untuk
menjawab pertanyaan user.

KONTEKS:
{context}

PERTANYAAN:
{message}

ATURAN:
- Jawab langsung.
- Jangan mengarang.
- Gunakan informasi paling relevan.
- Hindari pembukaan yang tidak diperlukan.
- Jangan mengulang pertanyaan.
- Jika ruang jawaban terbatas, prioritaskan informasi utama.

JAWABAN:
""".strip()

    return f"""
Anda adalah AI assistant.

PERTANYAAN:
{message}

ATURAN:
- Jawab langsung.
- Jangan mengarang.
- Gunakan jawaban yang ringkas.
- Hindari pembukaan yang tidak diperlukan.
- Jangan mengulang pertanyaan.
- Jika ruang jawaban terbatas, prioritaskan informasi utama.

JAWABAN:
""".strip()


def estimate_input_tokens(prompt: str):
    try:
        result = model.count_tokens(prompt)
        return int(result.total_tokens)
    except Exception as error:
        print("Token estimation error:", error)
        raise RuntimeError("Gagal menghitung estimasi token input.")


@app.post("/api/chat")
def chat(request: ChatRequest):
    user_id = request.user_id.strip()
    message = request.message.strip()

    if not user_id:
        return {"success": False, "blocked": True, "message": "User ID wajib diisi."}

    if not message:
        return {"success": False, "blocked": True, "message": "Pertanyaan wajib diisi."}

    used_before = get_used_tokens(user_id)
    remaining_before = max(TOKEN_LIMIT - used_before, 0)

    # QUOTA HABIS
    if remaining_before <= 0:
        return {
            "success": False,
            "blocked": True,
            "message": "Quota token user sudah habis.",
            "token": {
                "limit": TOKEN_LIMIT,
                "used_before": used_before,
                "remaining_before": 0,
                "estimated_input": 0,
                "max_output_allowed": 0
            }
        }

    context = search_pinecone(message)
    prompt = build_prompt(message, context)
    estimated_input = estimate_input_tokens(prompt)

    remaining_after_input = remaining_before - estimated_input

    if remaining_after_input <= 0:
        return {
            "success": False,
            "blocked": True,
            "message": "Token tersisa tidak cukup untuk memproses pertanyaan.",
            "token": {
                "limit": TOKEN_LIMIT,
                "used_before": used_before,
                "remaining_before": remaining_before,
                "estimated_input": estimated_input,
                "max_output_allowed": 0,
                "remaining_after_input": remaining_after_input
            }
        }

    max_output_tokens = remaining_after_input

    try:
        response = model.generate_content(
            prompt,
            generation_config={
                "max_output_tokens": max_output_tokens,
                "temperature": 0.2
            }
        )
    except Exception as error:
        print("Gemini error:", error)
        return {
            "success": False,
            "blocked": False,
            "message": "Gemini request gagal.",
            "error": str(error)
        }

    try:
        answer = response.text
    except Exception:
        answer = ""

    usage = response.usage_metadata

    actual_input = int(getattr(usage, "prompt_token_count", estimated_input) or 0)
    actual_output = int(getattr(usage, "candidates_token_count", 0) or 0)

    # Beberapa model (yang punya "thinking"/reasoning) mengirim token
    # tambahan lewat thoughts_token_count. Field ini TIDAK termasuk
    # di prompt_token_count maupun candidates_token_count, tapi API
    # sering menjumlahkannya ke total_token_count sehingga total_token_count
    # bisa jauh lebih besar dari (input + output) yang kamu lihat.
    # Ini penyebab bug "used jadi 334 padahal cuma ~100an".
    thinking_tokens = int(getattr(usage, "thoughts_token_count", 0) or 0)

    # Jangan pakai usage.total_token_count mentah-mentah untuk quota.
    # Hitung sendiri supaya konsisten dengan input + output yang ditampilkan.
    if COUNT_THINKING_TOKENS:
        actual_total = actual_input + actual_output + thinking_tokens
    else:
        actual_total = actual_input + actual_output

    add_used_tokens(user_id, actual_total)

    used_after = get_used_tokens(user_id)
    remaining_after = max(TOKEN_LIMIT - used_after, 0)

    return {
        "success": True,
        "blocked": False,
        "answer": answer,
        "token": {
            "limit": TOKEN_LIMIT,
            "used_before": used_before,
            "remaining_before": remaining_before,
            "estimated_input": estimated_input,
            "max_output_allowed": max_output_tokens,
            "actual_input": actual_input,
            "actual_output": actual_output,
            "thinking_tokens": thinking_tokens,
            "actual_total": actual_total,
            "used_after": used_after,
            "remaining_after": remaining_after,
            "reset_at": get_reset_at(user_id)
        }
    }
