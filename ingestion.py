from time import sleep

from langchain_community.document_loaders import DirectoryLoader
from langchain_community.document_loaders import TextLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_experimental.text_splitter import SemanticChunker
from langchain_chroma import Chroma
from langchain_huggingface import HuggingFaceEmbeddings
from retriever import normalize_text

import re
from langchain_core.documents import Document


embeddings_model = HuggingFaceEmbeddings(
    model_name="intfloat/multilingual-e5-base"
)

PERSIST_DIR = "chroma_db"
COLLECTION_NAME = "langchain"


def split_text_by_article(text: str) -> list[str]:
    """
    Splits text into articles using regex for 'Article' or 'Điều'.
    Keeps the delimiter at the start of each chunk.
    """
    # Pattern to match "Article <number>" or "Điều <number>" at start of line
    # (?:^|\n) matches start of string or newline
    # ((?:Article|Điều)\s+\d+.*?) matches the header and keeps it in the group if we wanted to split by it
    # But to split and keep delimiter, we can use a lookahead or just standard split and reconstruct.
    
    # A robust way is to find all matches and slice the text.
    pattern = re.compile(r'(?:\n|^)((?:Article|Điều)\s+\d+.*?)(?=(?:\n|^)(?:Article|Điều)\s+\d+|$)', re.DOTALL)
    
    # Alternatively, we can use re.split with capturing group to keep delimiters, 
    # but re.findall or iterating might be clearer if we want to ensure we capture the whole block.
    
    # Let's try splitting by the lookahead positive assert to keep the delimiter 
    # But python re split behavior with groups can be tricky.
    
    # Simpler approach: Iterate over matches
    matches = list(pattern.finditer(text))
    if not matches:
        return [text] # Return whole text if no articles found
        
    chunks = []
    
    # If there is text before the first article, add it as preamble
    if matches[0].start() > 0:
        chunks.append(text[:matches[0].start()].strip())
        
    for i, match in enumerate(matches):
        # Start of this article
        start = match.start()
        # End of this article is start of next match, or end of string
        end = matches[i+1].start() if i + 1 < len(matches) else len(text)
        
        chunk = text[start:end].strip()
        if chunk:
            chunks.append(chunk)
            
    return chunks

def load_and_split_data():
    print("Loading text files from data/valid ...")
    # 1. Load all .txt files from data/valid
    loader = DirectoryLoader("data/valid", glob="**/*.txt", loader_cls=TextLoader, loader_kwargs={"encoding": "utf-8"})
    documents = loader.load()

    if not documents:
        print("No documents found in data/valid")
        return []

    print(f"Loaded {len(documents)} documents.")

    # 2. Split by Article (Rule-based)
    print("Splitting documents by Article/Điều...")
    article_docs = []
    
    for doc in documents:
        articles = split_text_by_article(doc.page_content)
        for article in articles:
            # Create a new Document for each article, preserving metadata
            new_doc = Document(page_content=article, metadata=doc.metadata.copy())
            article_docs.append(new_doc)
            
    print(f"Split into {len(article_docs)} article-level chunks.")

    # 3. Apply Semantic Chunking
    print("Applying Semantic Chunking...")
    text_splitter = SemanticChunker(
        embeddings_model, 
        breakpoint_threshold_type="gradient"
    )
    
    # SemanticChunker expects a list of Documents
    docs = text_splitter.split_documents(article_docs)

    # 4. Post-processing: Merge small chunks (headers) with next chunk
    print(f"Post-processing {len(docs)} chunks to merge small headers...")
    merged_docs = []
    min_chunk_len = 50  # Threshold definition
    
    i = 0
    while i < len(docs):
        current_doc = docs[i]
        current_content = current_doc.page_content.strip()
        
        # Check if current chunk is small
        if len(current_content) < min_chunk_len:
            # If it's not the last chunk, merge with next
            if i + 1 < len(docs):
                next_doc = docs[i+1]
                # Merge content
                combined_content = current_content + "\n" + next_doc.page_content
                # Update next doc content and keep metadata ?? 
                # Better: Modify next_doc and skip current
                next_doc.page_content = combined_content
                # Move to next (which is now merged)
                # But wait, what if the NEXT one is also small? 
                # The loop will handle it in next iteration.
                # Actually, if we merge into [i+1], we should continue loop to process [i+1] (which is now larger)
                # So we just increment i? No, we skip adding current_doc separately.
                # But we updated docs[i+1] inside the list? Yes if mutable.
                docs[i+1].page_content = combined_content
                i += 1
                continue
            else:
                # If it's the very last chunk and small, maybe append to previous?
                if merged_docs:
                    merged_docs[-1].page_content += "\n" + current_content
                else:
                    merged_docs.append(current_doc)
                i += 1
        else:
            merged_docs.append(current_doc)
            i += 1
            
    docs = merged_docs
    print(f"After merging: {len(docs)} chunks.")

    for idx, doc in enumerate(docs):
        doc.metadata["chunk_id"] = idx
        # Keep source logic
        if "source" in doc.metadata:
            # Ensure source is just filename if not already
             doc.metadata["source"] = str(doc.metadata["source"]).split("\\")[-1]

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