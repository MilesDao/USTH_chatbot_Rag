import re
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


def get_vectorstore():
    embeddings = HuggingFaceEmbeddings(
        model_name="intfloat/multilingual-e5-base",
        encode_kwargs={"normalize_embeddings": True},
    )

    vectorstore = Chroma(
        collection_name=COLLECTION_NAME,
        persist_directory=PERSIST_DIR,
        embedding_function=embeddings,
        tenant="default_tenant",
        database="default_database",
    )
    return vectorstore


def retrieve_with_debug(query: str, k: int = 5):
    """
    Chỉ retrieve chunk – KHÔNG score, KHÔNG lọc
    """
    vectorstore = get_vectorstore()

    docs = vectorstore.similarity_search(
        e5_query(query),
        k=k,
    )

    return docs


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


# -------- Debug nhanh --------
if __name__ == "__main__":
    query = "giảng viên Việt Nam và quốc tế"
    results = retrieve_with_debug(query, k=6)

    print(f"\nQuery: {query}")
    print("=" * 60)

    if not results:
        print("Không retrieve được chunk nào")
    else:
        for i, doc in enumerate(results):
            print(f"\nChunk {i+1}")
            print(f" Page: {doc.metadata.get('page', 'N/A')}")
            print(f" Chunk ID: {doc.metadata.get('chunk_id', 'N/A')}")
            print(f" Text preview:\n{doc.page_content[:500]}")
