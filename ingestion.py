from time import sleep

from langchain_community.document_loaders import PyPDFDirectoryLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_chroma import Chroma
from langchain_huggingface import HuggingFaceEmbeddings

embeddings_model = HuggingFaceEmbeddings(
    model_name="sentence-transformers/all-MiniLM-L6-v2"
)

loader = PyPDFDirectoryLoader("data")
documents = loader.load()

text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=500,
    chunk_overlap=50
)
docs = text_splitter.split_documents(documents)
print(f" Total chunks (== total vectors): {len(docs)}")

PERSIST_DIR = "chroma_db"
COLLECTION_NAME = "langchain"


def embed_in_batches(docs, batch_size=10, delay=0.2):
    vector_store = Chroma(
        collection_name=COLLECTION_NAME,
        persist_directory=PERSIST_DIR,
        embedding_function=embeddings_model,
        tenant="default_tenant",
        database="default_database"
    )

    total_vectors = 0

    for i in range(0, len(docs), batch_size):
        batch = docs[i:i + batch_size]

        # Tạo ID rõ ràng cho từng vector
        batch_ids = [f"vector_{i + j}" for j in range(len(batch))]

        try:
            vector_store.add_documents(
                documents=batch,
                ids=batch_ids
            )

            print(f"\n Batch {i // batch_size + 1}")
            print(f"  Number of vectors in batch: {len(batch_ids)}")
            print(f"  Vector IDs:")

            for vid, doc in zip(batch_ids, batch):
                print(f"   - {vid} | source: {doc.metadata.get('source', 'unknown')}")

            total_vectors += len(batch_ids)

        except Exception as e:
            print(f" Error embedding batch {i // batch_size + 1}: {e}")

        sleep(delay)

    print("\n Ingestion summary")
    print(f" Total vectors stored: {total_vectors}")


if __name__ == "__main__":
    embed_in_batches(docs)
    print("\n Data ingestion complete!")