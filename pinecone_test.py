import os
from dotenv import load_dotenv
from pinecone import Pinecone

load_dotenv()  # loads .env in repo root

print("ENV VARS (masked):")
print("PINECONE_API_KEY set:", bool(os.environ.get("PINECONE_API_KEY")))
print("PINECONE_ENVIRONMENT:", os.environ.get("PINECONE_ENVIRONMENT"))
print("PINECONE_INDEX_NAME:", os.environ.get("PINECONE_INDEX_NAME"))

try:
    pc = Pinecone(api_key=os.environ.get("PINECONE_API_KEY"))
    print("Pinecone initialized successfully., pc:", pc)

    idxs = Pinecone.list_indexes(pc)
    indexes = [idx["name"] for idx in idxs]
    print("Indexes:", indexes)

    print("pinecone.list_indexes():", idxs)
    if os.environ.get("PINECONE_INDEX_NAME"):
        name = os.environ.get("PINECONE_INDEX_NAME")
        if name in indexes:
            index = pc.Index(name)
            stats = index.describe_index_stats()

            print(f"describe_index_stats({name}):", stats)
        else:
            print(f"Index '{name}' not found in list_indexes()")
except Exception as e:
    print("Error contacting Pinecone:", type(e).__name__, str(e))
