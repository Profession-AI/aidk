"""
Chat history management module.

This module provides backward compatibility by re-exporting all history-related
classes from the histories subpackage. It maintains the original API while
allowing for modular organization of code.

For new code, consider importing directly from the histories subpackage:
    from aidk.chat.histories import DictHistory, JSONHistory, SQLiteHistory, etc.
"""

from aidk.chat.histories import (
    Message,
    BaseHistory,
    DictHistory,
    JSONHistory,
    SQLiteHistory,
    MongoDBHistory,
    FirestoreHistory,
    HistorySummarizer,
)

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
