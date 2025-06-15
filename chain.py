"""
This module handles the creation of a processing chain for the RAG (Retrieval-Augmented Generation) pipeline.
# It uses LangChain to create a chain that retrieves context from a vector database and generates responses using a language model.
"""

import logging
from langchain.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser
from langchain.retrievers.multi_query import MultiQueryRetriever
from langchain_ollama import ChatOllama


def create_chain(retriever: MultiQueryRetriever, llm: ChatOllama) -> RunnablePassthrough:
    """
    Create the chain with preserved syntax.
    
    """
    
    
    # Define the prompt template for the chain
    template: str = """Answer the question based ONLY on the following context:
    {context}
    Question: {question}
    """

    prompt = ChatPromptTemplate.from_template(template)

    chain: RunnablePassthrough = (
        {"context": retriever, "question": RunnablePassthrough()}
        | prompt
        | llm
        | StrOutputParser()
    )

    logging.info("Chain created with preserved syntax.")
    return chain
