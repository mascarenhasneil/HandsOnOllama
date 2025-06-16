"""
main.py

Entry point for the Document Assistant Streamlit application package.
This script imports and runs the Streamlit UI from doc_assist.rag_streamlit.

Usage:
    streamlit run main.py
"""

from doc_assist.rag_streamlit import rag_streamlit

if __name__ == "__main__":
    rag_streamlit()
