"""
module: rag_streamlit.py
A simple Streamlit application for a Document Assistant using RAG (Retrieval-Augmented Generation) with LangChain and Ollama.
This module serves as the main entry point for the Streamlit app, integrating components from other modules such as retriever, chain, and vector_db.
This application allows users to ask questions about a PDF document, leveraging a vector database for context retrieval and a language model for response generation.

"""

from typing import Final
import logging
import streamlit as st
from langchain_ollama import ChatOllama
from retriever import create_retriever
from chain import create_chain
from vector_db import load_vector_db

logging.basicConfig(level=logging.INFO)


MODEL_NAME: Final[str] = "llama3.2:1b"
INPUT_PROMPT: Final[str] = "Enter your question:"

def main() -> None:
    """
    Main function to run the Streamlit app.

    This function initializes the language model, loads the vector database,
    creates the retriever and chain, and processes user input to generate a response.
    """
    st.title("Document Assistant")

    user_input : str = st.text_input(INPUT_PROMPT, "")
    if user_input:
        with st.spinner("Generating response..."):
            try:
                response = process_user_input(user_input)
                st.markdown("**Assistant:**")
                logging.info("Response generated successfully.")
                st.write(response)
            except (ValueError, RuntimeError, KeyError, TypeError, ConnectionError) as e:
                logging.error("Error processing user input: %s", e)
                st.error(f"An error occurred: {str(e)}")
    else:
        st.info("Please enter a question to get started.")

def process_user_input(user_input: str) -> str:
    """
    Process user input and return the response.
    This function initializes the language model, loads the vector database,
    creates the retriever and chain, and generates a response based on the user input.
    """
    logging.info("Processing user input.")
    llm = ChatOllama(model=MODEL_NAME)
    vector_db = load_vector_db()
    if vector_db is None:
        raise ValueError("Failed to load or create the vector database.")
    retriever = create_retriever(vector_db, llm)
    chain = create_chain(retriever, llm)
    logging.info("Chain created successfully.")
    return chain.invoke(input=user_input)

if __name__ == "__main__":
    main()
