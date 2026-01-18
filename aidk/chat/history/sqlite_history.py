"""
SQLite database-based history storage implementation.
Stores chat histories in a SQLite database.
"""

import os
import sqlite3
from typing import Optional
from .models import BaseHistory, Message


class SQLiteHistory(BaseHistory):
    """
    SQLite database-based history storage.
    Stores all chat messages in a single SQLite database file.
    """
    
    _DEFAULT_DB_PATH = "histories/history.db"

    def __init__(self, 
                 path: Optional[str] = None, 
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
        self._db_path: str = path if path is not None else self._DEFAULT_DB_PATH
        self._init_db()
    
    def _init_db(self) -> None:
        """Initialize the SQLite database and create necessary tables."""
        os.makedirs(os.path.dirname(self._db_path), exist_ok=True)
        with sqlite3.connect(self._db_path) as conn:
            conn.execute("""
                CREATE TABLE IF NOT EXISTS messages (
                    chat_id TEXT,
                    order_index INTEGER,
                    role TEXT,
                    content TEXT,
                    PRIMARY KEY (chat_id, order_index)
                )
            """)
            conn.execute("""
                CREATE TABLE IF NOT EXISTS summaries (
                    chat_id TEXT PRIMARY KEY,
                    content TEXT
                )
            """)
    
    def get_messages(self, chat_id: str) -> list[Message]:
        """Load chat history for a specific chat ID."""
        with sqlite3.connect(self._db_path) as conn:
            if self._last_n is not None:
                # Get system message
                cursor = conn.execute(
                    "SELECT role, content FROM messages WHERE chat_id = ? AND order_index = 0",
                    (chat_id,)
                )
                system_message = cursor.fetchone()
                
                # Get last N messages
                cursor = conn.execute(
                    """
                    SELECT role, content 
                    FROM messages 
                    WHERE chat_id = ? 
                    ORDER BY order_index DESC 
                    LIMIT ?
                    """,
                    (chat_id, self._last_n * 2)
                )
                last_messages_data = [{"role": role, "content": content} for role, content in cursor]
                last_messages_data.reverse()  # Reverse to get correct order
                
                # Convert to Message dataclasses
                system_msg = Message(content=system_message[1], role=system_message[0]) if system_message else None
                last_messages = [Message.from_dict(msg) for msg in last_messages_data]
                
                # Combine system message with last N messages
                self.messages = ([system_msg] if system_msg else []) + last_messages
            else:
                cursor = conn.execute(
                    "SELECT role, content FROM messages WHERE chat_id = ? ORDER BY order_index",
                    (chat_id,)
                )
                messages_data = [{"role": role, "content": content} for role, content in cursor]
                self.messages = [Message.from_dict(msg) for msg in messages_data]
        return self.messages

    def save_message(self, chat_id: str, messages: list[Message]) -> None:
        """Store messages in history."""
        with sqlite3.connect(self._db_path) as conn:
            # Get the last order_index
            cursor = conn.execute(
                "SELECT MAX(order_index) FROM messages WHERE chat_id = ?",
                (chat_id,)
            )
            last_index = cursor.fetchone()[0]
            
            # If no messages exist yet, start from -1
            if last_index is None:
                last_index = -1
            
            # Insert the new messages
            for i, message in enumerate(messages, start=last_index + 1):
                conn.execute(
                    "INSERT INTO messages (chat_id, order_index, role, content) VALUES (?, ?, ?, ?)",
                    (chat_id, i, message.role, message.content)
                )
            conn.commit()

    def get_summary(self, chat_id: str) -> Message:
        """Get chat summary for a specific chat ID."""
        with sqlite3.connect(self._db_path) as conn:
            cursor = conn.execute(
                "SELECT content FROM summaries WHERE chat_id = ?",
                (chat_id,)
            )
            result = cursor.fetchone()
            summary_content = result[0] if result else ""
        return Message(content=summary_content, role="system", provider=None, model=None)

    def save_summary(self, chat_id: str, summary: str) -> None:
        """Save chat summary."""
        with sqlite3.connect(self._db_path) as conn:
            conn.execute(
                "INSERT OR REPLACE INTO summaries (chat_id, content) VALUES (?, ?)",
                (chat_id, summary)
            )
            conn.commit()

    def clear(self, chat_id: str) -> None:
        """Clear history for a specific chat."""
        with sqlite3.connect(self._db_path) as conn:
            conn.execute("DELETE FROM messages WHERE chat_id = ?", (chat_id,))
            conn.execute("DELETE FROM summaries WHERE chat_id = ?", (chat_id,))
            conn.commit()
