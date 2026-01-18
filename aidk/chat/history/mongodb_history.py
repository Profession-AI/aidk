"""
MongoDB-based history storage implementation.
Stores chat histories in a MongoDB database.
"""

from typing import Optional
from .models import BaseHistory, Message


class MongoDBHistory(BaseHistory):
    """
    MongoDB-based history storage.
    Stores all chat messages in a MongoDB collection.
    Requires pymongo to be installed.
    """
    
    def __init__(self, 
                 db_path: str, 
                 db_name: str = "chat", 
                 collection_name: str = "histories", 
                 last_n: Optional[int] = None,
                 summarizer_provider: Optional[str] = None,
                 summarizer_model: Optional[str] = None,
                 summarizer_max_tokens: Optional[int] = None):
        super().__init__(
            last_n=last_n,
            summarizer_provider=summarizer_provider,
            summarizer_model=summarizer_model,
            summarizer_max_tokens=summarizer_max_tokens
        )

        try:
            from pymongo import MongoClient
        except ImportError:
            raise ImportError("pymongo is not installed. Please install it with 'pip install pymongo'")

        self._uri: str = db_path
        self._db_name: str = db_name
        self._collection_name: str = collection_name
        self._client = MongoClient(self._uri)
        self._db = self._client[self._db_name]
        self._collection = self._db[self._collection_name]

    def get_messages(self, chat_id: str) -> list[Message]:
        doc = self._collection.find_one({"chat_id": chat_id})
        if not doc:
            self.messages = []
            return self.messages
        messages_data = doc.get("messages", [])
        if self._last_n is not None and len(messages_data) > (self._last_n + 1) * 2:
            messages_data = [messages_data[0]] + messages_data[-self._last_n * 2:]
        
        # Convert to Message dataclasses
        self.messages = [Message.from_dict(msg) if isinstance(msg, dict) else msg for msg in messages_data]
        return self.messages

    def save_message(self, chat_id: str, messages: list[Message]) -> None:
        # Get existing messages
        doc = self._collection.find_one({"chat_id": chat_id})
        existing_messages_data = doc.get("messages", []) if doc else []
        
        # Convert existing messages to Message dataclasses
        existing_messages = [Message.from_dict(msg) if isinstance(msg, dict) else msg for msg in existing_messages_data]
        
        # Add the new messages (Message dataclasses)
        new_messages = existing_messages + messages
        
        # Convert to dictionaries for MongoDB storage
        messages_to_store = [msg.to_dict() for msg in new_messages]
        
        self._collection.update_one(
            {"chat_id": chat_id},
            {"$set": {"messages": messages_to_store}},
            upsert=True
        )

    def get_summary(self, chat_id: str) -> Message:
        """Get chat summary for a specific chat ID."""
        doc = self._collection.find_one({"chat_id": chat_id})
        summary_content = doc.get("summary", "") if doc else ""
        return Message(content=summary_content, role="system", provider=None, model=None)

    def save_summary(self, chat_id: str, summary: str) -> None:
        """Save chat summary."""
        self._collection.update_one(
            {"chat_id": chat_id},
            {"$set": {"summary": summary}},
            upsert=True
        )

    def clear(self, chat_id: str) -> None:
        """Clear history for a specific chat."""
        self._collection.delete_one({"chat_id": chat_id})
