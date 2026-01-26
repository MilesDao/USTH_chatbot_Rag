from time import sleep

from langchain_community.document_loaders import TextLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_chroma import Chroma
from langchain_huggingface import HuggingFaceEmbeddings

embeddings_model = HuggingFaceEmbeddings(
    model_name="intfloat/multilingual-e5-base"
)

PERSIST_DIR = "chroma_db"
COLLECTION_NAME = "langchain"

from retriever import normalize_text

def load_and_split_data():
    print("Loading data/finaldata.txt ...")
    loader = TextLoader("data/finaldata.txt", encoding="utf-8")
    documents = loader.load()

    if not documents:
        print("No documents found in data/finaldata.txt")
        return []

    print(f"Loaded {len(documents)} documents.")

    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=300,
        chunk_overlap=50
    )
    docs = text_splitter.split_documents(documents)

    for idx, doc in enumerate(docs):
        doc.metadata["chunk_id"] = idx
        if "source" in doc.metadata:
            doc.metadata["source"] = doc.metadata["source"].split("/")[-1]
        
        # Apply normalization and prefix for E5 model
        doc.page_content = f"passage: {normalize_text(doc.page_content)}"
    
    print(f"Split into {len(docs)} chunks.")
    return docs

import chromadb

def embed_in_batches(docs, batch_size=10, delay=0.2):
    client = chromadb.PersistentClient(path=PERSIST_DIR)
    
    vector_store = Chroma(
        client=client,
        collection_name=COLLECTION_NAME,
        embedding_function=embeddings_model,
    )

    total_vectors = 0

    for i in range(0, len(docs), batch_size):
        batch = docs[i:i + batch_size]
        batch_ids = [f"chunk_{doc.metadata['chunk_id']}" for doc in batch]

        try:
            vector_store.add_documents(
                documents=batch,
                ids=batch_ids
            )

            print(f"\n Batch {i // batch_size + 1}")
            print(f"  Stored {len(batch)} vectors")

            for doc in batch:
                print(
                    f"   - chunk_id={doc.metadata['chunk_id']}, "
                    f"source={doc.metadata['source']} "
                )

            total_vectors += len(batch)

        except Exception as e:
            print(f" Error embedding batch {i // batch_size + 1}: {e}")

        sleep(delay)

    print("\n Ingestion summary")
    print(f" Total vectors stored: {total_vectors}")

def ingest():
    docs = load_and_split_data()
    if docs:
        embed_in_batches(docs)
    else:
        print("Nothing to ingest.")

if __name__ == "__main__":
    ingest()
    print("\n Data ingestion complete!")