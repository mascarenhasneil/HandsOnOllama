"""
module: utils.py
Utility functions for the Document Assistant application.

This module provides functions to ingest PDF documents and split them into smaller chunks for processing.

Example:
    >>> docs = ingest_pdf("./eggs/sample.pdf")
    >>> chunks = split_documents(docs)
"""

import os
import logging
from typing import List
from langchain_community.document_loaders import UnstructuredPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.documents import Document
import streamlit as st

def ingest_pdf(doc_path: str) -> List[Document]:
    """
    Load PDF documents from the specified path.

    Args:
        doc_path (str): The file path to the PDF document.

    Returns:
        List: A list of loaded PDF documents. Returns an empty list if the file is not found or an error occurs.

    Raises:
        FileNotFoundError: If the specified PDF file does not exist.

    Example:
        >>> documents = ingest_pdf("./data/sample.pdf")
        >>> if documents:
        >>>     print("PDF loaded successfully.")
    """
    st.info("Ingesting PDF to Knowledge Database...")
    if os.path.exists(doc_path):
        try:
            loader = UnstructuredPDFLoader(file_path=doc_path)
            data = loader.load()
            logging.info("PDF loaded successfully.")
            return data
        except (IOError, ValueError) as e:
            logging.error("Error loading PDF: %s", e)
            st.error("An error occurred while loading the PDF. Please check the file and try again.")
            return []
    else:
        logging.error("PDF file not found at path: %s", doc_path)
        st.error("PDF file not found at path")
        return []

def split_documents(documents: List) -> List:
    """
    Split documents into smaller chunks for processing.

    Args:
        documents (List): A list of documents to be split into smaller chunks.

    Returns:
        List: A list of document chunks after splitting.

    Example:
        >>> docs = ingest_pdf("./eggs/sample.pdf")
        >>> chunks = split_documents(docs)
    """
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1200, chunk_overlap=300)
    chunks = text_splitter.split_documents(documents)
    logging.info("Documents split into chunks.")
    return chunks

