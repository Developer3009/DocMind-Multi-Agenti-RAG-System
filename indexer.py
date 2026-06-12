"""
Document Indexer — loads PDFs, chunks them, embeds them into ChromaDB.
"""

from __future__ import annotations
import os
import tempfile
import chromadb
from langchain_ollama import OllamaEmbeddings
from langchain_community.document_loaders import PyMuPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_chroma import Chroma


def build_retriever(
    uploaded_files: list,
    ollama_host: str,
    embed_model: str,
    chunk_size: int = 1200,
    chunk_overlap: int = 200,
    k: int = 5,
):
    """
    Index uploaded PDFs and return a LangChain retriever.

    Args:
        uploaded_files: list of Streamlit UploadedFile objects
        ollama_host:    URL of Ollama server (e.g. http://localhost:11434)
        embed_model:    name of Ollama embedding model
        chunk_size:     chars per chunk
        chunk_overlap:  overlap between chunks
        k:              number of chunks to retrieve per query

    Returns:
        (retriever, stats_dict)
    """
    docs = []
    temp_dir = tempfile.TemporaryDirectory()

    file_stats = []
    for file in uploaded_files:
        temp_path = os.path.join(temp_dir.name, file.name)
        with open(temp_path, "wb") as f:
            f.write(file.getvalue())
        loader = PyMuPDFLoader(temp_path)
        file_docs = loader.load()
        docs.extend(file_docs)
        file_stats.append({
            "name": file.name,
            "pages": len(file_docs),
            "size_kb": round(file.size / 1024, 1),
        })

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        separators=["\n\n", "\n", ".", " ", ""],
    )
    chunks = splitter.split_documents(docs)

    embeddings = OllamaEmbeddings(model=embed_model, base_url=ollama_host)

    client = chromadb.PersistentClient(path="./chroma_db")
    try:
        client.delete_collection("rag_collection")
    except Exception:
        pass

    vectordb = Chroma.from_documents(
        documents=chunks,
        embedding=embeddings,
        client=client,
        collection_name="rag_collection",
    )

    retriever = vectordb.as_retriever(
        search_type="mmr",         # Maximal Marginal Relevance — diverse results
        search_kwargs={"k": k, "fetch_k": k * 3},
    )

    stats = {
        "files": file_stats,
        "total_pages": sum(f["pages"] for f in file_stats),
        "total_chunks": len(chunks),
        "embed_model": embed_model,
    }

    return retriever, stats
