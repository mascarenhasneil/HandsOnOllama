# Changelog

## [Unreleased]

### Added
- Google-style docstrings and usage examples for all public functions and classes.
- Input validation for all major APIs (e.g., `create_chain`, `create_retriever`).
- User feedback mechanisms for Streamlit sidebar buttons (Stop/Close App).
- Robust error handling for file operations and vector database interactions.
- Coding guidelines section in README for contributors.
- Migrated all core modules into the `doc_assist` package for improved modularity and maintainability.
- Added a new `main.py` entry point that launches the Streamlit app via `doc_assist.rag_streamlit.rag_streamlit()`.

### Changed
- Refactored `upload_and_process_pdf` to avoid code duplication with `create_or_load_vector_db`.
- Improved error messages and logging throughout the codebase.
- Enhanced docstrings with parameter details and examples.
- Updated README to reflect new coding standards, error handling improvements, and package structure under `doc_assist`.
- Renamed the main Streamlit entry function to `rag_streamlit()` in `doc_assist/rag_streamlit.py`.

### Fixed
- Fixed potential issues with missing or invalid PDF uploads.
- Fixed missing or unclear error messages in the UI and backend.

### Code Quality
- All code now passes Pylance and mypy static checks.
- Imports are grouped and ordered per PEP 8 and Python Style Guide.
- All modules follow project coding standards for maintainability and clarity.
