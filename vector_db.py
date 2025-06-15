"""
module: vector_db

This module handles the creation and loading of a vector database using Chroma and Ollama embeddings. 
It provides functionality to ingest a PDF document, split it into chunks, and store the embeddings in a 
persistent vector database. This vector database can be used for retrieval-augmented generation (RAG) 
in a document assistant application.

Functions:
    create_or_load_vector_db(file_path: str, vector_store_name: str) -> Chroma | None:
        Loads or creates the vector database for a given PDF file.
    upload_and_process_pdf() -> Chroma | None:
        Handles PDF upload and vector DB creation/loading via Streamlit UI.

Constants:
    DOC_PATH (str): The directory to store uploaded PDF documents.
    EMBEDDING_MODEL (str): The name of the embedding model to use.

Example:
    >>> db = create_or_load_vector_db("./eggs/sample.pdf", "sample.pdf")
    >>> if db:
    >>>     print("Vector DB ready.")
"""

import os
import logging
from typing import Final
import ollama
import streamlit as st
from langchain_community.vectorstores import Chroma
from langchain_ollama import OllamaEmbeddings
from utils import ingest_pdf, split_documents

DOC_PATH: str = "./eggs"
EMBEDDING_MODEL: Final[str] = "nomic-embed-text:latest"

@st.cache_resource
def create_or_load_vector_db(file_path: str = DOC_PATH, vector_store_name: str = "Sample") -> Chroma | None:
    """
    Load or create the vector database for a given PDF document.

    Args:
        file_path (str): The path to the PDF document to ingest.
        vector_store_name (str): The name for the vector store/collection.

    Returns:
        Chroma | None: The loaded or newly created vector database, or None if the process fails.

    Example:
        >>> db = create_or_load_vector_db("./eggs/sample.pdf", "sample.pdf")
        >>> if db:
        >>>     print("Vector DB ready.")

    Raises:
        Exception: If there is an error during vector DB creation/loading.
    """
    st.info("Creating and loading to Knowledge Database...")
    try:
        # Pull the embedding model if not already available
        ollama.pull(EMBEDDING_MODEL)
        embedding = OllamaEmbeddings(model=EMBEDDING_MODEL)

        persist_directory = os.path.join("chroma_db", vector_store_name)

        if os.path.exists(persist_directory):
            vector_db = Chroma(
                embedding_function=embedding,
                collection_name=vector_store_name,
                persist_directory=persist_directory,
            )
            st.info("Loading existing vector database...")
            logging.info("Loaded existing vector database.")
        else:
            # Load and process the PDF document
            data = ingest_pdf(file_path)
            if not data:
                st.error("Failed to ingest the PDF document. Please check the file.")
                logging.error("Failed to process the PDF file.")
                return None

            st.info("PDF document ingested successfully, almost ready..")
            # Split the documents into chunks
            chunks = split_documents(data)
            
            vector_db = Chroma.from_documents(
                documents=chunks,
                embedding=embedding,
                collection_name=vector_store_name,
                persist_directory=persist_directory,
            )
            vector_db.persist()
            st.info("Vector database created and persisted.")
            logging.info("Vector database created and persisted.")

        return vector_db
    except Exception as e:
        logging.error("Error in create_or_load_vector_db: %s", e)
        st.error("An error occurred while creating/loading the vector database. Please try again.")
        return None

def upload_and_process_pdf() -> Chroma | None:
    """
    Upload a PDF file via the Streamlit sidebar, store it in the 'eggs' directory,
    and create or load a vector database for the uploaded file.

    Returns:
        Chroma | None: The loaded or newly created vector database, or None if the upload fails.

    Example:
        >>> vector_db = upload_and_process_pdf()
        >>> if vector_db:
        >>>     print("Vector database loaded successfully.")
    """
    st.sidebar.title("Upload PDF")
    uploaded_file = st.sidebar.file_uploader("Choose a PDF file", type="pdf")

    if uploaded_file is not None:
        try:
            # Save the uploaded file to the 'eggs' directory
            file_path = os.path.join(DOC_PATH, uploaded_file.name)
            os.makedirs(DOC_PATH, exist_ok=True)

            with open(file_path, "wb") as f:
                f.write(uploaded_file.getbuffer())

            st.sidebar.success(f"File uploaded and saved as {uploaded_file.name}")

            # Use the helper function to create or load the vector database
            return create_or_load_vector_db(file_path=file_path, vector_store_name=uploaded_file.name)
        except OSError as e:
            logging.error("Error processing uploaded file: %s", e)
            st.error("An error occurred while processing the uploaded file. Please try again.")

    return None
