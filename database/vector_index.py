
from database.connection import get_collection
from config.settings import (
    VECTOR_INDEX_NAME,
    EMBEDDING_DIMENSIONS
)
from pymongo.operations import SearchIndexModel



def create_vector_index(collection_name=None, index_name=None, dimensions=None):
    from config.settings import COLLECTION_NAME
    col_name = collection_name or COLLECTION_NAME
    idx_name = index_name or VECTOR_INDEX_NAME
    dim = dimensions or EMBEDDING_DIMENSIONS

    from database.connection import db
    if col_name not in db.list_collection_names():
        db.create_collection(col_name)
        print(f"Created collection '{col_name}'.")

    collection = get_collection(col_name)


    # Check existing search indexes
    existing_indexes = list(collection.list_search_indexes())

    for index in existing_indexes:
        if index.get("name") == idx_name:
            print(f"Vector index '{idx_name}' already exists on '{col_name}'.")
            return

    # Define vector search index
    vector_index = SearchIndexModel(
        definition={
            "fields": [
                {
                    "type": "vector",
                    "path": "embedding",
                    "numDimensions": dim,
                    "similarity": "cosine"
                }
            ]
        },
        name=idx_name,
        type="vectorSearch"
    )

    # Create the index
    collection.create_search_index(vector_index)
    print(f"Vector index '{idx_name}' creation requested on collection '{col_name}'.")


if __name__ == "__main__":
    create_vector_index()