"""
module: retriever.py

Module for creating a retriever for the Document Assistant application.
This module provides functions to create a multi-query retriever using LangChain.

"""

import logging
from typing import Final
from langchain.prompts import PromptTemplate
from langchain.retrievers.multi_query import MultiQueryRetriever
from langchain_community.vectorstores import Chroma
from langchain_ollama import ChatOllama



def create_retriever(vector_db: Chroma, llm: ChatOllama) -> MultiQueryRetriever:
    """Create a multi-query retriever."""
    
    # Define the prompt template for generating alternative questions
    query_prompt: Final = PromptTemplate(
        input_variables=["question"],
        template="""You are an AI language model assistant. Your task is to generate five
        different versions of the given user question to retrieve relevant documents from
        a vector database. By generating multiple perspectives on the user question, your
        goal is to help the user overcome some of the limitations of the distance-based
        similarity search. Provide these alternative questions separated by newlines.
        Original question: {question}""",
    )

    retriever = MultiQueryRetriever.from_llm(
        vector_db.as_retriever(), llm, prompt=query_prompt
    )
    logging.info("Retriever created.")
    return retriever
