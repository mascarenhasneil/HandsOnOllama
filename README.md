# HandsOnOllama

HandsOnOllama is a Retrieval-Augmented Generation (RAG) document assistant application built using LangChain and Ollama. It leverages a vector database for efficient document retrieval and a language model for generating context-aware responses.

## Features

- **Document Ingestion**: Ingests PDF documents and splits them into manageable chunks for processing.
- **Vector Database**: Uses Chroma for storing and retrieving document embeddings.
- **Language Model Integration**: Employs Ollama for generating embeddings and responses.
- **Streamlit UI**: Provides an interactive user interface for querying the document assistant.

## Installation

1. Clone the repository:
   ```bash
   git clone https://github.com/mascarenhasneil/HandsOnOllama.git
   ```

2. Navigate to the project directory:
   ```bash
   cd HandsOnOllama
   ```

3. Create and activate a virtual environment:
   ```bash
   python -m venv .venv
   source .venv/bin/activate  # On Windows: .venv\Scripts\activate
   ```

4. Install the required dependencies:
   ```bash
   pip install -r requirements.txt
   ```

## Usage

1. Start the Streamlit application:
   ```bash
   streamlit run rag_streamlit.py
   ```

2. Upload a PDF document and interact with the assistant through the UI.

## Project Structure

- `chain.py`: Handles the creation of the RAG processing chain.
- `retriever.py`: Creates a multi-query retriever for document retrieval.
- `vector_db.py`: Manages the vector database for storing and retrieving embeddings.
- `utils.py`: Utility functions for document ingestion and processing.
- `rag_streamlit.py`: Streamlit application for the document assistant.
- `model_test_1.py`: Demonstrates how to send a JSON POST request to a local API endpoint to generate text using a specified language model.
- `model_test_2.py`: Demonstrates the use of the Ollama API to interact with language models, including listing models, chatting, generating text, and creating custom models.

## Contributing

Contributions are welcome! Please follow the coding guidelines and ensure all changes are well-documented.

## License

This project is licensed under the MIT License. See the `LICENSE` file for details.
