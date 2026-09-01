import os
from dotenv import load_dotenv
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from pinecone import Pinecone
import google.generativeai as genai


# ============================================================
# LOAD ENVIRONMENT
# ============================================================

load_dotenv()


GEMINI_API_KEY = os.getenv(
    "GEMINI_API_KEY"
)

PINECONE_API_KEY = os.getenv(
    "PINECONE_API_KEY"
)


# ============================================================
# VALIDATE API KEY
# ============================================================

if not GEMINI_API_KEY:

    raise RuntimeError(
        "GEMINI_API_KEY belum ditemukan."
    )


if not PINECONE_API_KEY:

    raise RuntimeError(
        "PINECONE_API_KEY belum ditemukan."
    )


# ============================================================
# CONFIGURATION
# ============================================================

MODEL = "gemini-3.5-flash"

TOKEN_LIMIT = 100

PINECONE_INDEX_NAME = "selfai1"

PINECONE_NAMESPACE = "_default_"

PINECONE_TOP_K = 10


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
# GEMINI
# ============================================================

gemini = genai.Client(
    api_key=GEMINI_API_KEY
)


# ============================================================
# PINECONE
# ============================================================

pinecone = Pinecone(
    api_key=PINECONE_API_KEY
)


index = pinecone.Index(
    PINECONE_INDEX_NAME
)


# ============================================================
# DEMO USER TOKEN STORAGE
# ============================================================

# IMPORTANT:
#
# Ini hanya untuk DEMO.
#
# Data akan hilang ketika serverless
# function di-restart.
#
# Production:
# gunakan Redis / PostgreSQL / Supabase.

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

        "status": "online",

        "model": MODEL,

        "token_limit":
            TOKEN_LIMIT

    }


# ============================================================
# TEST ENDPOINT
# ============================================================

@app.get("/api/test")
def test():

    return {

        "status": "ok",

        "message":
            "FastAPI berjalan di Vercel."

    }


# ============================================================
# QUOTA ENDPOINT
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

    try:

        result = index.search(

            namespace=PINECONE_NAMESPACE,

            query={

                "inputs": {

                    "text":
                        query

                },

                "top_k":
                    PINECONE_TOP_K

            },

            fields=[

                "chunk_text",

                "text",

                "content",

                "source",

                "title"

            ]

        )

        return result

    except Exception as error:

        print(
            "Pinecone error:",
            error
        )

        return None


# ============================================================
# EXTRACT CONTEXT
# ============================================================

def extract_context(
    result
):

    if not result:

        return ""


    contexts = []


    try:

        hits = (
            result[
                "result"
            ][
                "hits"
            ]
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


# ============================================================
# BUILD PROMPT
# ============================================================

def build_prompt(
    message: str,
    context: str
):

    return f"""
Anda adalah AI assistant.

Jawab pertanyaan berdasarkan
konteks yang tersedia.

ATURAN:

- Jawab langsung.
- Jangan mengarang.
- Prioritaskan informasi penting.
- Jangan menggunakan pembukaan yang tidak perlu.
- Gunakan jawaban seefisien mungkin.
- Jika batas output kecil, persingkat jawaban.
- Jangan mengulang pertanyaan user.

KONTEKS:

{context}

PERTANYAAN USER:

{message}

JAWABAN:
""".strip()


# ============================================================
# COUNT TOKENS
# ============================================================

def estimate_input_tokens(
    prompt: str
):

    result = (
        gemini.models.count_tokens(

            model=MODEL,

            contents=prompt

        )
    )

    return int(
        result.total_tokens
    )


# ============================================================
# CHAT
# ============================================================

@app.post("/api/chat")
def chat(
    request: ChatRequest
):

    user_id = request.user_id.strip()

    message = request.message.strip()


    # ========================================================
    # VALIDATION
    # ========================================================

    if not user_id:

        return {

            "success": False,

            "blocked": True,

            "message":
                "User ID wajib diisi."

        }


    if not message:

        return {

            "success": False,

            "blocked": True,

            "message":
                "Pertanyaan wajib diisi."

        }


    # ========================================================
    # CURRENT USER USAGE
    # ========================================================

    used_before = get_used_tokens(
        user_id
    )


    remaining_before = max(

        TOKEN_LIMIT
        - used_before,

        0

    )


    # ========================================================
    # CHECK QUOTA
    # ========================================================

    if remaining_before <= 0:

        return {

            "success": False,

            "blocked": True,

            "message":
                "Quota token user sudah habis.",

            "token": {

                "limit":
                    TOKEN_LIMIT,

                "used_before":
                    used_before,

                "remaining_before":
                    0

            }

        }


    # ========================================================
    # PINECONE
    # ========================================================

    pinecone_result = (
        search_pinecone(
            message
        )
    )


    context = extract_context(
        pinecone_result
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
    # CALCULATE REMAINING
    # ========================================================

    remaining_after_input = (

        remaining_before
        - estimated_input

    )


    # ========================================================
    # INPUT ALREADY EXCEEDS QUOTA
    # ========================================================

    if remaining_after_input <= 0:

        return {

            "success":
                False,

            "blocked":
                True,

            "message":
                "Token tidak cukup untuk memproses input.",

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
                    0

            }

        }


    # ========================================================
    # DYNAMIC OUTPUT TOKEN
    # ========================================================

    max_output_tokens = (
        remaining_after_input
    )


    # ========================================================
    # GEMINI REQUEST
    # ========================================================

    try:

        response = (

            gemini.models.generate_content(

                model=MODEL,

                contents=prompt,

                config=
                    types.GenerateContentConfig(

                        max_output_tokens=
                            max_output_tokens,

                        temperature=0.2

                    )

            )

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
    # TOKEN USAGE
    # ========================================================

    usage = response.usage_metadata


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
    # UPDATE USER TOKEN
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
    # RETURN RESPONSE
    # ========================================================

    return {

        "success":
            True,

        "blocked":
            False,

        "answer":
            response.text,

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
