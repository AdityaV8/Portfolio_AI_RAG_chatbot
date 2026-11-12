import time
import requests

from config.settings import (
    JINA_API_KEY,
    GROQ_API_KEY,
    EMBEDDING_MODEL,
    LLM_MODEL,
    EMBEDDING_DIMENSIONS
)

# API Endpoints
JINA_API_URL = "https://api.jina.ai/v1/embeddings"
GROQ_API_URL = "https://api.groq.com/openai/v1/chat/completions"

TRANSIENT_STATUS_CODES = {429, 500, 502, 503, 504}


def create_embedding(text: str, is_query: bool = False, max_retries: int = 3) -> list:
    """
    Generate embedding vector using Jina AI Cloud API.

    Args:
        text: The string to embed.
        is_query: If True, uses 'retrieval.query' task; otherwise 'retrieval.passage'.
        max_retries: Number of retries for transient HTTP errors.

    Returns:
        List of floats with length equal to EMBEDDING_DIMENSIONS.
    """
    if not JINA_API_KEY:
        raise ValueError("JINA_API_KEY is missing. Please add it to your .env file.")

    if not text or not text.strip():
        raise ValueError("Text to embed cannot be empty.")

    task = "retrieval.query" if is_query else "retrieval.passage"

    headers = {
        "Authorization": f"Bearer {JINA_API_KEY}",
        "Content-Type": "application/json"
    }

    payload = {
        "model": EMBEDDING_MODEL,
        "task": task,
        "dimensions": EMBEDDING_DIMENSIONS,
        "embedding_type": "float",
        "input": [text]
    }

    attempt = 0
    backoff = 2.0

    while attempt < max_retries:
        try:
            response = requests.post(JINA_API_URL, headers=headers, json=payload, timeout=30)

            if response.status_code == 200:
                data = response.json()
                vector = data["data"][0]["embedding"]

                if len(vector) != EMBEDDING_DIMENSIONS:
                    raise ValueError(
                        f"Expected embedding dimension {EMBEDDING_DIMENSIONS}, got {len(vector)}"
                    )

                return [float(x) for x in vector]

            if response.status_code in TRANSIENT_STATUS_CODES and attempt < max_retries - 1:
                attempt += 1
                time.sleep(backoff)
                backoff *= 2.0
                continue

            # Non-transient or retries exhausted
            raise RuntimeError(
                f"Jina API error (status {response.status_code}): {response.text}"
            )

        except (requests.exceptions.Timeout, requests.exceptions.ConnectionError) as e:
            if attempt < max_retries - 1:
                attempt += 1
                time.sleep(backoff)
                backoff *= 2.0
            else:
                raise RuntimeError(f"Jina API network error: {e}") from e


def generate_answer(prompt: str, max_retries: int = 3) -> str:
    """
    Generate an answer using Groq Cloud API.

    Args:
        prompt: The input prompt for the LLM.
        max_retries: Number of retries for transient errors.

    Returns:
        Generated answer as a string.
    """
    if not GROQ_API_KEY:
        raise ValueError("GROQ_API_KEY is missing. Please add it to your .env file.")

    headers = {
        "Authorization": f"Bearer {GROQ_API_KEY}",
        "Content-Type": "application/json"
    }

    payload = {
        "model": LLM_MODEL,
        "messages": [
            {"role": "user", "content": prompt}
        ],
        "temperature": 0.2,
        "max_tokens": 512
    }

    attempt = 0
    backoff = 2.0

    while attempt < max_retries:
        try:
            response = requests.post(GROQ_API_URL, headers=headers, json=payload, timeout=30)

            if response.status_code == 200:
                data = response.json()
                return data["choices"][0]["message"]["content"].strip()

            if response.status_code in TRANSIENT_STATUS_CODES and attempt < max_retries - 1:
                attempt += 1
                time.sleep(backoff)
                backoff *= 2.0
                continue

            raise RuntimeError(
                f"Groq API error (status {response.status_code}): {response.text}"
            )

        except (requests.exceptions.Timeout, requests.exceptions.ConnectionError) as e:
            if attempt < max_retries - 1:
                attempt += 1
                time.sleep(backoff)
                backoff *= 2.0
            else:
                raise RuntimeError(f"Groq API network error: {e}") from e
