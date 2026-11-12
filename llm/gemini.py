from langchain_google_genai import ChatGoogleGenerativeAI

from config.settings import (
    GEMINI_API_KEY,
    LLM_MODEL
)


def get_llm():
    """
    Create and return the Gemini language model.
    """

    return ChatGoogleGenerativeAI(
        model=LLM_MODEL,
        google_api_key=GEMINI_API_KEY,
        temperature=0.3
    )


def extract_response_text(response):
    """
    Extract plain text from Gemini's response.
    """

    content = response.content

    # Normal string response
    if isinstance(content, str):
        return content

    # Structured response
    if isinstance(content, list):

        text_parts = []

        for part in content:

            if isinstance(part, dict):
                text = part.get("text")

                if text:
                    text_parts.append(text)

            elif isinstance(part, str):
                text_parts.append(part)

        return "".join(text_parts)

    return str(content)