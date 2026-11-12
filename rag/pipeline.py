import sys
import traceback

from retrieval.retriever import retrieve_documents
from llm.providers import generate_answer

from database.session import (
    create_session,
    get_session_messages,
    save_message
)


FALLBACK_MESSAGE = (
    "I don't have this information in my knowledge base yet. "
    "This information will be discovered and updated further."
)


def create_context(documents):
    """
    Combine retrieved documents into one context string.
    """
    context_parts = []

    for document in documents:
        metadata = document.get("metadata", {})
        source = metadata.get("source", "unknown")
        section = metadata.get("section", "unknown")
        text = document.get("text", "")

        context_parts.append(
            f"Source: {source}\n"
            f"Section: {section}\n"
            f"{text}"
        )

    return "\n\n---\n\n".join(context_parts)


def create_history(messages):
    """
    Convert stored session messages into text.
    """
    if not messages:
        return "No previous conversation."

    history_parts = []

    for message in messages:
        role = message.get("role")
        content = message.get("content")

        if role == "user":
            history_parts.append(f"User: {content}")
        elif role == "assistant":
            history_parts.append(f"Assistant: {content}")

    return "\n".join(history_parts)


def ask_rag(question, session_id):
    """
    Answer a question using:
    1. RAG knowledge from MongoDB Vector Search (Jina embeddings)
    2. Previous session conversation
    3. Groq LLM
    """
    # Make sure the session exists
    create_session(session_id)

    # Get previous conversation
    previous_messages = get_session_messages(session_id)

    # Retrieve relevant information from MongoDB
    documents = retrieve_documents(question)

    # If no relevant knowledge was found
    if documents is None:
        answer = FALLBACK_MESSAGE
        save_message(session_id, "user", question)
        save_message(session_id, "assistant", answer)
        return answer

    # Create RAG context
    context = create_context(documents)

    # Create conversation history
    history = create_history(previous_messages)

    # Prompt Groq LLM
    prompt = f"""You are a personal knowledge assistant.

Your job is to answer the user's question using the provided knowledge base and previous conversation.

IMPORTANT RULES:
1. Use the knowledge base as the source of factual information.
2. Use previous conversation to understand references such as "it", "that", "there", "he", etc.
3. Do not invent personal information.
4. Do not use general knowledge to fill missing information.
5. If the answer is not supported by the knowledge base, say that the information is not available.
6. Give a clear and concise answer.

Previous Conversation:
{history}

Knowledge Base Context:
{context}

Current User Question:
{question}
"""

    try:
        answer = generate_answer(prompt)
    except Exception as e:
        print("\n" + "=" * 50, file=sys.stderr)
        print(f"Error generating answer: {e}", file=sys.stderr)
        traceback.print_exc(file=sys.stderr)
        print("=" * 50 + "\n", file=sys.stderr)
        sys.stderr.flush()
        return "Sorry, an error occurred while generating the answer. Please try again."

    # Save both sides of the conversation
    save_message(session_id, "user", question)
    save_message(session_id, "assistant", answer)

    return answer