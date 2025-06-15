"""
module: vector_db
This module handles the creation and loading of a vector database using Chroma and Ollama embeddings. It provides functionality to ingest a PDF document, split it into chunks, and store the embeddings in a persistent vector database. This vector database can be used for retrieval-augmented generation (RAG) in a document assistant application.
    
"""

import os
import logging
from typing import Final
import ollama
import streamlit as st
from langchain_community.vectorstores import Chroma
from langchain_ollama import OllamaEmbeddings
from utils import ingest_pdf, split_documents

VECTOR_STORE_NAME: Final[str] = "simple-rag"
PERSIST_DIRECTORY: Final[str] = "./chroma_db"
DOC_PATH: Final[str] = "./data/BOI.pdf"
EMBEDDING_MODEL: Final[str] = "nomic-embed-text:latest"

@st.cache_resource
def load_vector_db() -> Chroma | None:
    """Load or create the vector database."""
    # Pull the embedding model if not already available
    ollama.pull(EMBEDDING_MODEL)

    embedding = OllamaEmbeddings(model=EMBEDDING_MODEL)

    if os.path.exists(PERSIST_DIRECTORY):
        vector_db = Chroma(
            embedding_function=embedding,
            collection_name=VECTOR_STORE_NAME,
            persist_directory=PERSIST_DIRECTORY,
        )
        logging.info("Loaded existing vector database.")
    else:
        # Load and process the PDF document
        data = ingest_pdf(DOC_PATH)
        if not data:
            return None

        # Split the documents into chunks
        chunks = split_documents(data)

        vector_db = Chroma.from_documents(
            documents=chunks,
            embedding=embedding,
            collection_name=VECTOR_STORE_NAME,
            persist_directory=PERSIST_DIRECTORY,
        )
        vector_db.persist()
        logging.info("Vector database created and persisted.")
    return vector_db
