"""
module: rag_streamlit.py
A simple Streamlit application for a Document Assistant using RAG (Retrieval-Augmented Generation) with LangChain and Ollama.
This module serves as the main entry point for the Streamlit app, integrating components from other modules such as retriever, chain, and vector_db.
This application allows users to ask questions about a PDF document, leveraging a vector database for context retrieval and a language model for response generation.

"""

from typing import Final
import streamlit as st
from langchain_ollama import ChatOllama
from retriever import create_retriever
from chain import create_chain
from vector_db import load_vector_db

MODEL_NAME: Final[str] = "llama3.2:1b"

def main():
    """Main function to run the Streamlit app."""
    pass

if __name__ == "__main__":
    main()
