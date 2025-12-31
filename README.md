 #📚 **RAG Team Intro AI – Local PDF RAG Chatbot**

This project builds a **Retrieval-Augmented Generation (RAG) chatbot** that runs locally and answers questions based on your **PDF documents**.

The system:
- Loads PDFs from the **`data/`** directory  
- Splits content into text chunks  
- Generates embeddings using **HuggingFace**  
- Stores vectors in **ChromaDB (persistent with tenant + database)**  
- Provides a **Streamlit chatbot UI**

---

## ⚙️ **Technologies**
- **Python**
- **LangChain**
- **ChromaDB (>= 1.3.x)**
- **HuggingFace Embeddings**
- **Streamlit**
- **PDF Loader**
