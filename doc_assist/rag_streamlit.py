"""
module: rag_streamlit.py
A simple Streamlit application for a Document Assistant using RAG (Retrieval-Augmented Generation) with LangChain and Ollama.

This module serves as the main entry point for the Streamlit app, integrating components from other modules such as retriever, chain, and vector_db.
This application allows users to ask questions about a PDF document, leveraging a vector database for context retrieval and a language model for response generation.

Example:
    $ streamlit run rag_streamlit.py
    # Upload a PDF and ask questions in the UI.
"""

import logging
from typing import Final, Union

import streamlit as st
from langchain_ollama import ChatOllama
from langchain_core.runnables import RunnableSerializable
from langchain.retrievers.multi_query import MultiQueryRetriever
from langchain_community.vectorstores import Chroma

from doc_assist.chain import create_chain
from doc_assist.retriever import create_retriever
from doc_assist.vector_db import upload_and_process_pdf

logging.basicConfig(level=logging.INFO)

MODEL_NAME: Final[str] = "llama3.2:1b"
INPUT_PROMPT: Final[str] = "Enter your question:"


def rag_streamlit() -> None:
    """
    Main entry point for the Document Assistant Streamlit application.

    This function initializes the language model and vector database, creates the retriever and chain,
    and processes user input to generate responses. It also provides options via Streamlit buttons to stop or close the application.

    Returns:
        None.

    Example:
        >>> rag_streamlit()
        # Launches the Streamlit app for document Q&A.
    """
    logging.info("Starting Document Assistant Streamlit app.")
    st.title("Document Assistant")

    # Upload and process the PDF file
    vector_db: Chroma | None = upload_and_process_pdf()
    if vector_db is None:
        st.error("Failed to initialize the vector database. Please upload a valid PDF file (PDF format only, not corrupted, and not empty).")
        return

    st.info("Initializing the language model and vector database...")
    chain = initialize_llm_and_vector_db(vector_db=vector_db)

    # Add a stop button to halt the app in the sidebar
    if st.sidebar.button("Stop App"):
        st.warning("The app has been stopped. Please close the terminal to exit.")
        logging.info("Document Assistant Streamlit app stopped.")
        st.stop()

    # Add a close button to halt the app and display a message in the sidebar
    if st.sidebar.button("Close App"):
        st.warning("The app has been closed. Please stop the app manually from the terminal.")
        logging.info("Document Assistant Streamlit app closed.")
        st.stop()

    user_input: str = st.text_input(INPUT_PROMPT, "")

    if user_input:
        logging.info("User input received: %s", user_input)
        # Process the user input
        process_user_input(user_input, chain)
    else:
        st.info("Please enter a question to get started.")


def initialize_llm_and_vector_db(vector_db) -> RunnableSerializable[Union[str, None], str]:
    """
    Initialize the language model, vector database, retriever, and chain.

    Returns:
        RunnablePassthrough: The initialized chain for generating responses.

    Example:
        >>> initialize_llm_and_vector_db(vector_db)
        # Initializes and returns the chain for the document Q&A.
    """
    # Initialize the language model
    llm = ChatOllama(model=MODEL_NAME, seed=42, temperature=1.0)

    # Create the retriever
    retriever: MultiQueryRetriever = create_retriever(vector_db=vector_db, llm=llm)

    # Create the chain
    chain: RunnableSerializable[Union[str, None], str] = create_chain(retriever=retriever, llm=llm)

    return chain


def process_user_input(user_input: str, chain: RunnableSerializable[Union[str, None], str]) -> None:
    """
    Process the user input and generate a response using the provided chain.

    Args:
        user_input (str): The question or input provided by the user.
        chain (RunnableSerializable[Union[str, None], str]): The chain used to generate responses.

    Example:
        >>> process_user_input("What is the capital of France?", chain)
        # Processes the input and displays the response in the Streamlit app.
    """
    logging.info("Processing user input.")
    with st.spinner("Generating response..."):
        try:
            # Get the response
            response = chain.invoke(input=user_input)

            st.markdown("**Assistant:**")
            st.write(response)
            logging.info("Response generated: %s", response)
        except (ValueError, RuntimeError, KeyError, FileNotFoundError) as e:
            logging.error("Error processing user input: %s", str(e))
            st.error(f"An error occurred: {str(e)}")


rag_streamlit()
