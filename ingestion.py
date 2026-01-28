from time import sleep

from langchain_community.document_loaders import DirectoryLoader
from langchain_community.document_loaders import TextLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_experimental.text_splitter import SemanticChunker
from langchain_chroma import Chroma
from langchain_huggingface import HuggingFaceEmbeddings

embeddings_model = HuggingFaceEmbeddings(
    model_name="intfloat/multilingual-e5-base"
)

PERSIST_DIR = "chroma_db"
COLLECTION_NAME = "langchain"

from retriever import normalize_text

def load_and_split_data():
    print("Loading text files from data/valid ...")
    # 1. Load all .txt files from data/valid
    loader = DirectoryLoader("data/valid", glob="**/*.txt", loader_cls=TextLoader, loader_kwargs={"encoding": "utf-8"})
    documents = loader.load()

    if not documents:
        print("No documents found in data/valid")
        return []

    print(f"Loaded {len(documents)} documents.")

    # 2. Pre-split
    print("Pre-splitting documents...")
    pre_splitter = RecursiveCharacterTextSplitter(
        separators=["\n## PART", "\n## PHẦN", "\nArticle", "\nĐiều"],
        chunk_size=3000,
        chunk_overlap=0,
        keep_separator=True,
        strip_whitespace=True
    )
    pre_split_docs = pre_splitter.split_documents(documents)
    print(f"Pre-split into {len(pre_split_docs)} chunks.")

    # 3. Apply Semantic Chunking
    print("Applying Semantic Chunking...")
    text_splitter = SemanticChunker(
        embeddings_model, 
        breakpoint_threshold_type="gradient"
    )
    docs = text_splitter.split_documents(pre_split_docs)

    for idx, doc in enumerate(docs):
        doc.metadata["chunk_id"] = idx
        if "source" in doc.metadata:
            doc.metadata["source"] = doc.metadata["source"].split("\\")[-1]
        
        # Apply normalization and prefix for E5 model
        doc.page_content = f"passage: {normalize_text(doc.page_content)}"
    
    print(f"Final Semantic Split into {len(docs)} chunks.")
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