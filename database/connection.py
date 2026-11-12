from pymongo import MongoClient

from config.settings import (
    MONGODB_URI,
    DATABASE_NAME,
    COLLECTION_NAME
)


# Create MongoDB client
client = MongoClient(MONGODB_URI)

# Select database
db = client[DATABASE_NAME]

# Select collection
collection = db[COLLECTION_NAME]


def get_collection(collection_name=None):
    """
    Return the MongoDB collection used by the RAG system.
    Defaults to COLLECTION_NAME ('documents_jina').
    """
    if collection_name:
        return db[collection_name]
    return db[COLLECTION_NAME]



def test_connection():
    """
    Test whether MongoDB is reachable.
    """
    try:
        client.admin.command("ping")
        print("MongoDB connection successful.")
        return True

    except Exception as e:
        print("MongoDB connection failed.")
        print("Error:", e)
        return False