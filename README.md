# HandsOnOllama

HandsOnOllama is a Retrieval-Augmented Generation (RAG) document assistant application built using LangChain and Ollama. It leverages a vector database for efficient document retrieval and a language model for generating context-aware responses.

## Features

- **Document Ingestion**: Ingests PDF documents and splits them into manageable chunks for processing.
- **Vector Database**: Uses Chroma for storing and retrieving document embeddings.
- **Language Model Integration**: Employs Ollama for generating embeddings and responses.
- **Streamlit UI**: Provides an interactive user interface for querying the document assistant.
- **Robust Error Handling**: All modules include improved error handling and input validation.
- **Coding Standards**: The codebase follows PEP 8, Google Python Style Guide, and project-specific guidelines for maintainability and clarity.

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
   - The sidebar allows you to upload a PDF file (stored in the `eggs` directory).
   - The app will create or load a vector database for the uploaded file.
   - Ask questions about the document using the input box.
   - Use the sidebar buttons to stop or close the app with user feedback.

## Project Structure

- `chain.py`: Handles the creation of the RAG processing chain. Includes input validation and Google-style docstrings.
- `retriever.py`: Creates a multi-query retriever for document retrieval. Includes input validation and usage examples.
- `vector_db.py`: Manages the vector database for storing and retrieving embeddings. Features robust error handling and clear docstrings.
- `utils.py`: Utility functions for document ingestion and processing. Includes error handling and usage examples.
- `rag_streamlit.py`: Streamlit application for the document assistant. Provides user feedback and robust error handling in the UI.
- `model_test_1.py`: Demonstrates how to send a JSON POST request to a local API endpoint to generate text using a specified language model.
- `model_test_2.py`: Demonstrates the use of the Ollama API to interact with language models, including listing models, chatting, generating text, and creating custom models.

## Coding Guidelines

- All code must pass Pylance and mypy checks (static type checking and linting).
- Use absolute imports, group imports by standard library, third-party, and local modules.
- Avoid wildcard imports and place all imports at the top of the file after the module docstring.
- Use Google-style docstrings for all public modules, classes, functions, and methods, including usage examples.
- Document all arguments, return values, and exceptions.
- Use type hints for all public APIs.
- Follow PEP 8 and Google Python Style Guide for naming, formatting, and structure.

## Contributing

Contributions are welcome! Please follow the coding guidelines and ensure all changes are well-documented. Run static analysis tools and tests before submitting a pull request.

## License

This project is licensed under the MIT License. See the `LICENSE` file for details.
