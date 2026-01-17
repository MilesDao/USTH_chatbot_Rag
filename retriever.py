import re
from typing import List, Tuple, Any

import chromadb
from langchain_chroma import Chroma
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_core.documents import Document



# config
PERSIST_DIR = "chroma_db"
COLLECTION_NAME = "langchain"
EMBEDDING_MODEL = "intfloat/multilingual-e5-base"



def normalize_text(text: str) -> str:

    text = text.lower()
    text = re.sub(r"\s+", " ", text)
    return text.strip()


def e5_query(text: str) -> str:
    return f"query: {normalize_text(text)}"


def e5_passage(text: str) -> str:
    return f"passage: {normalize_text(text)}"



def get_vectorstore() -> Chroma:
    embeddings = HuggingFaceEmbeddings(
        model_name=EMBEDDING_MODEL,
        encode_kwargs={"normalize_embeddings": True},
    )

    client = chromadb.PersistentClient(path=PERSIST_DIR)

    vectorstore = Chroma(
        client=client,
        collection_name=COLLECTION_NAME,
        embedding_function=embeddings,
    )
    return vectorstore


class E5Retriever:
    def __init__(self, k: int = 5):
        self.vectorstore = get_vectorstore()
        self.k = k

    def get_relevant_documents(self, query: str) -> List[Document]:

        return self.vectorstore.similarity_search(
            e5_query(query),
            k=self.k,
        )

    def retrieve_with_score(self, query: str) -> List[Tuple[Document, float]]:

        return self.vectorstore.similarity_search_with_score(
            e5_query(query),
            k=self.k,
        )



if __name__ == "__main__":
    retriever = E5Retriever(k=8)

    query = "có bao nhiêu loại học bổng USTH"
    results = retriever.retrieve_with_score(query)

    print(f"\nQuery: {query}")
    print("=" * 60)

    if not results:
        print("Không retrieve được chunk nào")
    else:
        for i, (doc, score) in enumerate(results, start=1):
            print(f"\nChunk {i} (Score: {score:.4f})")
            print(f"Chunk ID: {doc.metadata.get('chunk_id', 'N/A')}")
            print("Text preview:")
            print(doc.page_content[:500])
