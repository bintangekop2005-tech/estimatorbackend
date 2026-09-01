import os

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import google.generativeai as genai
from pinecone import Pinecone


# ============================================================
# FASTAPI
# ============================================================

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


# ============================================================
# CONFIGURATION
# ============================================================

MODEL = "gemini-3.5-flash"

TOKEN_LIMIT = 300

PINECONE_INDEX_NAME = "selfai1"

PINECONE_NAMESPACE = "_default_"

PINECONE_TOP_K = 10


# ============================================================
# API KEYS
# ============================================================

GEMINI_API_KEY = os.getenv(
    "GEMINI_API_KEY"
)

PINECONE_API_KEY = os.getenv(
    "PINECONE_API_KEY"
)


if not GEMINI_API_KEY:

    raise RuntimeError(
        "GEMINI_API_KEY belum ditemukan."
    )


# ============================================================
# GEMINI CONFIGURATION
# ============================================================

genai.configure(
    api_key=GEMINI_API_KEY
)


model = genai.GenerativeModel(
    MODEL
)


# ============================================================
# PINECONE
# ============================================================

pinecone = None
index = None


if PINECONE_API_KEY:

    try:

        pinecone = Pinecone(
            api_key=PINECONE_API_KEY
        )

        index = pinecone.Index(
            PINECONE_INDEX_NAME
        )

    except Exception as error:

        print(
            "Pinecone initialization error:",
            error
        )


# ============================================================
# DEMO TOKEN DATABASE
# ============================================================

# Hanya untuk DEMO.
#
# Vercel Serverless bersifat ephemeral.
# Untuk production gunakan database seperti
# PostgreSQL / Supabase / Redis.

user_usage = {}


def get_used_tokens(
    user_id: str
):

    return user_usage.get(
        user_id,
        0
    )


def add_used_tokens(
    user_id: str,
    tokens: int
):

    current = get_used_tokens(
        user_id
    )

    user_usage[user_id] = (
        current + tokens
    )


# ============================================================
# REQUEST MODEL
# ============================================================

class ChatRequest(BaseModel):

    user_id: str

    message: str


# ============================================================
# ROOT
# ============================================================

@app.get("/")
def root():

    return {

        "status": "ok",

        "message":
            "Gemini Token Tracker API",

        "model":
            MODEL,

        "token_limit":
            TOKEN_LIMIT

    }


# ============================================================
# TEST
# ============================================================

@app.get("/api/test")
def test():

    return {

        "status": "ok",

        "message":
            "FastAPI berhasil berjalan."

    }


# ============================================================
# GET USER QUOTA
# ============================================================

@app.get(
    "/api/quota/{user_id}"
)
def quota(
    user_id: str
):

    used = get_used_tokens(
        user_id
    )

    remaining = max(
        TOKEN_LIMIT - used,
        0
    )

    return {

        "user_id":
            user_id,

        "limit":
            TOKEN_LIMIT,

        "used":
            used,

        "remaining":
            remaining

    }


# ============================================================
# PINECONE SEARCH
# ============================================================

def search_pinecone(
    query: str
):

    if index is None:

        return ""


    try:

        # ----------------------------------------------------
        # Catatan:
        # Format query dapat berbeda tergantung konfigurasi
        # index Pinecone kamu.
        # ----------------------------------------------------

        result = index.search(
            namespace=PINECONE_NAMESPACE,
            query={
                "inputs": {
                    "text": query
                },
                "top_k": PINECONE_TOP_K
            },
            fields=[
                "chunk_text",
                "text",
                "content",
                "source",
                "title"
            ]
        )


        contexts = []


        try:

            hits = (
                result["result"]["hits"]
            )

        except Exception:

            return ""


        for hit in hits:

            fields = hit.get(
                "fields",
                {}
            )


            text = (

                fields.get(
                    "chunk_text"
                )

                or

                fields.get(
                    "text"
                )

                or

                fields.get(
                    "content"
                )

            )


            if text:

                contexts.append(
                    str(text)
                )


        return "\n\n".join(
            contexts
        )


    except Exception as error:

        print(
            "Pinecone search error:",
            error
        )

        return ""


# ============================================================
# BUILD PROMPT
# ============================================================

def build_prompt(
    message: str,
    context: str
):

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


# ============================================================
# ESTIMATE INPUT TOKEN
# ============================================================

def estimate_input_tokens(
    prompt: str
):

    try:

        result = model.count_tokens(
            prompt
        )

        return int(
            result.total_tokens
        )

    except Exception as error:

        print(
            "Token estimation error:",
            error
        )

        raise RuntimeError(
            "Gagal menghitung estimasi token input."
        )


# ============================================================
# CHAT
# ============================================================

@app.post("/api/chat")
def chat(
    request: ChatRequest
):

    user_id = (
        request.user_id.strip()
    )

    message = (
        request.message.strip()
    )


    # ========================================================
    # VALIDATION
    # ========================================================

    if not user_id:

        return {

            "success":
                False,

            "blocked":
                True,

            "message":
                "User ID wajib diisi."

        }


    if not message:

        return {

            "success":
                False,

            "blocked":
                True,

            "message":
                "Pertanyaan wajib diisi."

        }


    # ========================================================
    # USER QUOTA
    # ========================================================

    used_before = get_used_tokens(
        user_id
    )


    remaining_before = max(
        TOKEN_LIMIT - used_before,
        0
    )


    # ========================================================
    # QUOTA HABIS
    # ========================================================

    if remaining_before <= 0:

        return {

            "success":
                False,

            "blocked":
                True,

            "message":
                "Quota token user sudah habis.",

            "token": {

                "limit":
                    TOKEN_LIMIT,

                "used_before":
                    used_before,

                "remaining_before":
                    0,

                "estimated_input":
                    0,

                "max_output_allowed":
                    0

            }

        }


    # ========================================================
    # PINECONE
    # ========================================================

    context = search_pinecone(
        message
    )


    # ========================================================
    # BUILD FINAL PROMPT
    # ========================================================

    prompt = build_prompt(
        message,
        context
    )


    # ========================================================
    # ESTIMATE INPUT
    # ========================================================

    estimated_input = (
        estimate_input_tokens(
            prompt
        )
    )


    # ========================================================
    # CALCULATE OUTPUT BUDGET
    # ========================================================

    remaining_after_input = (

        remaining_before
        - estimated_input

    )


    # ========================================================
    # INPUT SUDAH TERLALU BESAR
    # ========================================================

    if remaining_after_input <= 0:

        return {

            "success":
                False,

            "blocked":
                True,

            "message":
                "Token tersisa tidak cukup untuk memproses pertanyaan.",

            "token": {

                "limit":
                    TOKEN_LIMIT,

                "used_before":
                    used_before,

                "remaining_before":
                    remaining_before,

                "estimated_input":
                    estimated_input,

                "max_output_allowed":
                    0,

                "remaining_after_input":
                    remaining_after_input

            }

        }


    # ========================================================
    # DYNAMIC MAX OUTPUT
    # ========================================================

    max_output_tokens = (
        remaining_after_input
    )


    # ========================================================
    # GEMINI REQUEST
    # ========================================================

    try:

        response = model.generate_content(

            prompt,

            generation_config={

                "max_output_tokens":
                    max_output_tokens,

                "temperature":
                    0.2

            }

        )


    except Exception as error:

        print(
            "Gemini error:",
            error
        )

        return {

            "success":
                False,

            "blocked":
                False,

            "message":
                "Gemini request gagal.",

            "error":
                str(error)

        }


    # ========================================================
    # GET RESPONSE TEXT
    # ========================================================

    try:

        answer = response.text

    except Exception:

        answer = ""


    # ========================================================
    # ACTUAL TOKEN USAGE
    # ========================================================

    usage = (
        response.usage_metadata
    )


    actual_input = int(

        getattr(

            usage,

            "prompt_token_count",

            estimated_input

        )

        or 0

    )


    actual_output = int(

        getattr(

            usage,

            "candidates_token_count",

            0

        )

        or 0

    )


    actual_total = int(

        getattr(

            usage,

            "total_token_count",

            actual_input
            + actual_output

        )

        or 0

    )


    # ========================================================
    # UPDATE TOKEN
    # ========================================================

    add_used_tokens(

        user_id,

        actual_total

    )


    used_after = get_used_tokens(
        user_id
    )


    remaining_after = max(

        TOKEN_LIMIT
        - used_after,

        0

    )


    # ========================================================
    # RESPONSE
    # ========================================================

    return {

        "success":
            True,

        "blocked":
            False,

        "answer":
            answer,

        "token": {

            "limit":
                TOKEN_LIMIT,

            "used_before":
                used_before,

            "remaining_before":
                remaining_before,

            "estimated_input":
                estimated_input,

            "max_output_allowed":
                max_output_tokens,

            "actual_input":
                actual_input,

            "actual_output":
                actual_output,

            "actual_total":
                actual_total,

            "used_after":
                used_after,

            "remaining_after":
                remaining_after

        }

    }
