import json
from pathlib import Path

from database.connection import get_collection
from llm.providers import create_embedding


# MongoDB collection (defaults to documents_jina)
collection = get_collection()

# Location of data folder
DATA_DIR = Path("data")


def load_json_file(file_path):
    """Load a JSON file and return its contents."""
    with open(file_path, "r", encoding="utf-8") as file:
        return json.load(file)


def create_text_chunks(data):
    """
    Convert top-level JSON sections into separate text chunks.
    Each top-level section becomes one searchable document.
    """
    chunks = []
    for section_name, section_data in data.items():
        text = (
            f"Section: {section_name}\n"
            f"{json.dumps(section_data, indent=2, ensure_ascii=False)}"
        )
        chunks.append({
            "section": section_name,
            "text": text
        })
    return chunks


def get_source_name(file_path):
    """Determine the knowledge source from the filename."""
    filename = file_path.stem.lower()
    if "personal" in filename:
        return "personal"
    if "usbm" in filename:
        return "usbm"
    if "manikeshwari" in filename:
        return "mmu"
    return "unknown"


def ingest_file(file_path):
    """Read one JSON file and store its sections in MongoDB idempotently."""
    print(f"\nProcessing: {file_path.name}")
    data = load_json_file(file_path)
    source = get_source_name(file_path)
    chunks = create_text_chunks(data)
    print(f"Found {len(chunks)} sections.")

    for chunk in chunks:
        print(f"Creating embedding: {chunk['section']}")
        embedding = create_embedding(chunk["text"], is_query=False)

        document = {
            "text": chunk["text"],
            "embedding": embedding,
            "metadata": {
                "source": source,
                "section": chunk["section"],
                "file": file_path.name
            }
        }

        # Idempotent upsert: match on source and section to avoid duplicates
        collection.replace_one(
            {
                "metadata.source": source,
                "metadata.section": chunk["section"]
            },
            document,
            upsert=True
        )
        print(f"Stored: {chunk['section']}")


def main():
    """Process all JSON files in the data folder."""
    json_files = list(DATA_DIR.glob("*.json"))
    if not json_files:
        print("No JSON files found in the data folder.")
        return

    print(f"Found {len(json_files)} JSON files.")
    for file_path in json_files:
        ingest_file(file_path)

    print("\nIngestion completed successfully.")


if __name__ == "__main__":
    main()