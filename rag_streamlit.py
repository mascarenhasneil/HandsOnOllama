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
from langchain_core.runnables import RunnablePassthrough

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

    logging.info("Starting Document Assistant Streamlit app.")

    # Add a stop button to halt the app
    if st.button("Stop App"):
        st.warning("The app has been stopped.")
        logging.info("Document Assistant Streamlit app stopped.")
        st.stop()

    # Add a close button to halt the app and display a message
    if st.button("Close App"):
        st.warning(
            "The app has been closed. Please stop the app manually from the terminal."
        )
        logging.info("Document Assistant Streamlit app closed.")
        st.stop()

    # Initialize the language model and vector database
    st.info("Initializing the language model and vector database...")
    chain = initialize_llm_and_vector_db()

    user_input: str = st.text_input(INPUT_PROMPT, "")

    if user_input:
        logging.info(f"User input received: {user_input}")
        # Process the user input
        process_user_input(user_input, chain)

    else:
        st.info("Please enter a question to get started.")


def initialize_llm_and_vector_db() -> RunnablePassthrough:
    """
    Initialize the language model, vector database, retriever, and chain.

    Returns:
        RunnablePassthrough: The initialized chain for generating responses.
    """
    # Initialize the language model
    llm = ChatOllama(model=MODEL_NAME)

    # Load the vector database
    vector_db = load_vector_db()
    if vector_db is None:
        st.error("Failed to load or create the vector database.")
        st.stop()

    # Create the retriever
    retriever = create_retriever(vector_db, llm)

    # Create the chain
    chain = create_chain(retriever, llm)

    return chain


def process_user_input(user_input: str, chain: RunnablePassthrough) -> None:
    """
    Process the user input, generate a response using the RAG pipeline, and display it in the Streamlit app.

    Args:
        user_input (str): The question or input provided by the user.
        chain (RunnablePassthrough): The initialized chain for generating responses.
    """
    logging.info("Processing user input.")
    with st.spinner("Generating response..."):
        try:
            # Get the response
            response = chain.invoke(input=user_input)

            st.markdown("**Assistant:**")
            st.write(response)
            logging.info(f"Response generated: {response}")
        except (ValueError, RuntimeError, KeyError, FileNotFoundError) as e:
            logging.error(f"Error processing user input: {str(e)}")
            st.error(f"An error occurred: {str(e)}")


if __name__ == "__main__":
    main()
