"""
RAG is a module that provides a high-level interface for performing semantic search queries
against a vector database. It supports multiple vector database backends and
embedding providers for flexible deployment scenarios.
"""

from .rag import RAG
from .vectordb import ChromaVectorDB
from .documents_builder import DocumentsBuilder

__all__ = ['RAG', 'ChromaVectorDB', 'DocumentsBuilder'] 