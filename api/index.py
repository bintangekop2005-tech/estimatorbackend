import os
from dotenv import load_dotenv

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

from google import genai
from google.genai import types
from pinecone import Pinecone


# ============================================================
# LOAD .ENV
# ============================================================

load_dotenv()

GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
PINECONE_API_KEY = os.getenv("PINECONE_API_KEY")


if not GEMINI_API_KEY:
    raise RuntimeError(
        "GEMINI_API_KEY tidak ditemukan."
    )

if not PINECONE_API_KEY:
    raise RuntimeError(
        "PINECONE_API_KEY tidak ditemukan."
    )


# ============================================================
# CONFIGURATION
# ============================================================

# Gemini
MODEL = "gemini-3.5-flash"


# Token quota setiap user
USER_TOKEN_LIMIT = 100


# Pinecone
PINECONE_INDEX_NAME = "selfai1"

PINECONE_TOP_K = 10

PINECONE_NAMESPACE = "__default__"


# ============================================================
# FASTAPI
# ============================================================

app = FastAPI(
    title="Gemini Token Tracker"
)


app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


# ============================================================
# GEMINI CLIENT
# ============================================================

gemini = genai.Client(
    api_key=GEMINI_API_KEY
)


# ============================================================
# PINECONE CLIENT
# ============================================================

pc = Pinecone(
    api_key=PINECONE_API_KEY
)

index = pc.Index(
    PINECONE_INDEX_NAME
)


# ============================================================
# DEMO USER TOKEN DATABASE
# ============================================================

# Untuk DEMO saja.
#
# Production sebaiknya:
# PostgreSQL / Supabase / Redis

user_usage = {}


def get_used_tokens(user_id):

    return user_usage.get(
        user_id,
        0
    )


def add_used_tokens(
    user_id,
    tokens
):

    current = get_used_tokens(
        user_id
    )

    user_usage[user_id] = (
        current + tokens
    )


# ============================================================
# REQUEST
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
            USER_TOKEN_LIMIT

    }


# ============================================================
# GET QUOTA
# ============================================================

@app.get("/api/quota/{user_id}")
def get_quota(
    user_id: str
):

    used = get_used_tokens(
        user_id
    )

    remaining = max(
        USER_TOKEN_LIMIT - used,
        0
    )

    return {

        "user_id":
            user_id,

        "limit":
            USER_TOKEN_LIMIT,

        "used":
            used,

        "remaining":
            remaining

    }


# ============================================================
# PINECONE SEARCH
# ============================================================

def search_pinecone(
    query
):

    try:

        result = index.search(

            namespace=PINECONE_NAMESPACE,

            query={

                "inputs": {

                    "text": query

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

    except Exception as e:

        print(
            "Pinecone error:",
            e
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


# ============================================================
# BUILD RAG PROMPT
# ============================================================

def build_prompt(
    message,
    context
):

    return f"""
Anda adalah AI assistant.

Jawab pertanyaan user berdasarkan
konteks yang diberikan.

ATURAN:

1. Jawab pertanyaan secara langsung.
2. Gunakan konteks sebagai sumber utama.
3. Jangan mengarang informasi.
4. Prioritaskan informasi yang paling penting.
5. Gunakan jawaban seefisien mungkin.
6. Jika output token yang tersedia sedikit,
   persingkat jawaban.
7. Jangan menggunakan pembukaan yang tidak perlu.

KONTEKS:

{context}

PERTANYAAN USER:

{message}

JAWABAN:
""".strip()


# ============================================================
# COUNT INPUT TOKENS
# ============================================================

def count_input_tokens(
    prompt
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


    if not user_id:

        raise HTTPException(
            status_code=400,
            detail="user_id wajib diisi."
        )


    if not message:

        raise HTTPException(
            status_code=400,
            detail="message wajib diisi."
        )


    # ========================================================
    # 1. CHECK USER QUOTA
    # ========================================================

    used_before = get_used_tokens(
        user_id
    )


    remaining_before = max(

        USER_TOKEN_LIMIT
        - used_before,

        0

    )


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
                    USER_TOKEN_LIMIT,

                "used":
                    used_before,

                "remaining":
                    0

            }

        }


    # ========================================================
    # 2. SEARCH PINECONE
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
    # 3. BUILD FINAL PROMPT
    # ========================================================

    final_prompt = build_prompt(

        message,

        context

    )


    # ========================================================
    # 4. COUNT INPUT
    # ========================================================

    estimated_input = (
        count_input_tokens(
            final_prompt
        )
    )


    # ========================================================
    # 5. CALCULATE AVAILABLE OUTPUT
    # ========================================================

    remaining_after_input = (

        remaining_before
        - estimated_input

    )


    # Input saja sudah tidak muat
    if remaining_after_input <= 0:

        return {

            "success":
                False,

            "blocked":
                True,

            "message":
                "Sisa token tidak cukup untuk memproses request.",

            "token": {

                "limit":
                    USER_TOKEN_LIMIT,

                "used_before":
                    used_before,

                "remaining_before":
                    remaining_before,

                "estimated_input":
                    estimated_input,

                "max_output":
                    0

            }

        }


    # ========================================================
    # 6. DYNAMIC OUTPUT LIMIT
    # ========================================================

    max_output_tokens = (
        remaining_after_input
    )


    # ========================================================
    # 7. GENERATE GEMINI
    # ========================================================

    try:

        response = (

            gemini.models.generate_content(

                model=MODEL,

                contents=final_prompt,

                config=types.GenerateContentConfig(

                    max_output_tokens=
                        max_output_tokens,

                    temperature=0.2,

                    thinking_config=
                        types.ThinkingConfig(

                            thinking_budget=0

                        )

                )

            )

        )

    except Exception as e:

        raise HTTPException(

            status_code=500,

            detail=str(e)

        )


    # ========================================================
    # 8. GET ACTUAL TOKEN USAGE
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
    # 9. UPDATE QUOTA
    # ========================================================

    add_used_tokens(

        user_id,

        actual_total

    )


    used_after = get_used_tokens(
        user_id
    )


    remaining_after = max(

        USER_TOKEN_LIMIT
        - used_after,

        0

    )


    # ========================================================
    # 10. RETURN
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
                USER_TOKEN_LIMIT,

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
