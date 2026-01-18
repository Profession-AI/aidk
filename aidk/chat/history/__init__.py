"""
Chat history storage module.

This module provides various implementations of chat history storage:
- DictHistory: In-memory storage using Python dictionaries
- JSONHistory: JSON file-based storage
- SQLiteHistory: SQLite database storage
- MongoDBHistory: MongoDB document storage
- FirestoreHistory: Google Cloud Firestore storage

Each implementation inherits from BaseHistory and provides methods for
storing, loading, and managing chat histories.
"""

from .models import Message, BaseHistory
from .dict_history import DictHistory
from .json_history import JSONHistory
from .sqlite_history import SQLiteHistory
from .mongodb_history import MongoDBHistory
from .firestore_history import FirestoreHistory
from .summarizer import HistorySummarizer

__all__ = [
    "Message",
    "BaseHistory",
    "DictHistory",
    "JSONHistory",
    "SQLiteHistory",
    "MongoDBHistory",
    "FirestoreHistory",
    "HistorySummarizer",
]
