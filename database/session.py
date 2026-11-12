from datetime import datetime, timezone

from database.connection import db

session_collection = db["sessions"]


def create_session(session_id):
    existing_session = session_collection.find_one({
        "session_id": session_id
    })

    if existing_session is None:
        session_collection.insert_one({
            "session_id": session_id,
            "messages": [],
            "created_at": datetime.now(timezone.utc),
            "updated_at": datetime.now(timezone.utc)
        })


def get_session_messages(session_id):
    session = session_collection.find_one(
        {"session_id": session_id},
        {"_id": 0, "messages": 1}
    )

    if session is None:
        return []

    return session.get("messages", [])


def save_message(session_id, role, content):

    message = {
        "role": role,
        "content": content,
        "timestamp": datetime.now(timezone.utc)
    }

    session_collection.update_one(
        {"session_id": session_id},
        {
            "$push": {
                "messages": message
            },
            "$set": {
                "updated_at": datetime.now(timezone.utc)
            }
        },
        upsert=True
    )


def clear_session(session_id):

    session_collection.update_one(
        {"session_id": session_id},
        {
            "$set": {
                "messages": [],
                "updated_at": datetime.now(timezone.utc)
            }
        }
    )


def get_all_sessions():

    sessions = session_collection.find(
        {},
        {
            "_id": 0,
            "session_id": 1,
            "messages": 1,
            "created_at": 1,
            "updated_at": 1
        }
    ).sort("updated_at", -1)

    return list(sessions)


def get_session_preview(session):

    messages = session.get("messages", [])

    if not messages:
        return "New conversation"

    for message in messages:
        if message["role"] == "user":
            text = message["content"]

            if len(text) > 35:
                return text[:35] + "..."

            return text

    return "New conversation"