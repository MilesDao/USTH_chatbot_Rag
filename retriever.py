import re
from typing import List, Tuple, Any
from langchain_chroma import Chroma
from langchain_huggingface import HuggingFaceEmbeddings

PERSIST_DIR = "chroma_db"
COLLECTION_NAME = "langchain"


def normalize_text(text: str) -> str:
    text = text.lower()
    text = re.sub(r"(.)\1{2,}", r"\1", text)
    text = re.sub(r"\s+", " ", text)
    return text.strip()


def e5_query(text: str) -> str:
    return f"query: {normalize_text(text)}"


import chromadb

def get_vectorstore():
    embeddings = HuggingFaceEmbeddings(
        model_name="intfloat/multilingual-e5-base",
        encode_kwargs={"normalize_embeddings": True},
    )

    client = chromadb.PersistentClient(path=PERSIST_DIR)

    vectorstore = Chroma(
        client=client,
        collection_name=COLLECTION_NAME,
        embedding_function=embeddings,
    )
    return vectorstore





def retrieve_with_score(query: str, k: int = 5) -> List[Tuple[Any, float]]:
    """
    Retrieve chunk kèm score
    """
    vectorstore = get_vectorstore()

    docs_and_scores = vectorstore.similarity_search_with_score(
        e5_query(query),
        k=k,
    )

    return docs_and_scores


def get_retriever(k: int = 5):
    """
    Retriever dùng cho chain / agent
    """
    vectorstore = get_vectorstore()

    return vectorstore.as_retriever(
        search_kwargs={
            "k": k,
            "query": e5_query,
        }
    )


if __name__ == "__main__":
    query = "giảng viên Việt Nam và quốc tế"
    results_with_score = retrieve_with_score(query, k=6)

    print(f"\nQuery: {query}")
    print("=" * 60)

    if not results_with_score:
        print("Không retrieve được chunk nào")
    else:
        for i, (doc, score) in enumerate(results_with_score):
            print(f"\nChunk {i+1} (Score: {score:.4f})")
            print(f" Page: {doc.metadata.get('page', 'N/A')}")
            print(f" Chunk ID: {doc.metadata.get('chunk_id', 'N/A')}")
            print(f" Text preview:\n{doc.page_content[:500]}")
