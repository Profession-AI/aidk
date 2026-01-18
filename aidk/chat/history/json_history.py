"""
JSON file-based history storage implementation.
Stores each chat history as a separate JSON file.
"""

import os
import json
from typing import Optional
from .models import BaseHistory, Message


class JSONHistory(BaseHistory):
    """
    JSON file-based history storage.
    Each chat is stored as a separate JSON file.
    """

    _DEFAULT_HISTORY_PATH = "histories/"
    
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
        self._history_path: str = path if path is not None else self._DEFAULT_HISTORY_PATH

        if not os.path.exists(self._history_path):
            os.makedirs(self._history_path)

    def get_messages(self, chat_id: str) -> list[Message]:
        """Load chat history for a specific chat ID."""
        file_path = os.path.join(self._history_path, chat_id + ".json")
        
        if not os.path.exists(file_path):
            self.messages = []
            return self.messages
        
        with open(file_path, "r") as f:
            messages_data = json.load(f)
        
        # Convert dictionaries to Message dataclasses
        messages = [Message.from_dict(msg) if isinstance(msg, dict) else msg for msg in messages_data]
        
        if self._last_n is not None and len(messages) > (self._last_n + 1) * 2:
            messages = [messages[0]] + messages[-self._last_n * 2:]
        
        self.messages = messages
        return self.messages
    
    def save_message(self, chat_id: str, messages: list[Message]) -> None:
        """Store messages in history."""
        file_path = os.path.join(self._history_path, chat_id + ".json")
        
        # Load existing messages
        if os.path.exists(file_path):
            with open(file_path, "r") as f:
                existing_messages_data = json.load(f)
                # Convert to Message dataclasses
                existing_messages = [Message.from_dict(msg) if isinstance(msg, dict) else msg for msg in existing_messages_data]
        else:
            existing_messages = []
        
        # Add the new messages
        new_messages = existing_messages + messages
        
        # Convert to dictionaries for JSON serialization
        messages_to_save = [msg.to_dict() for msg in new_messages]
        
        with open(file_path, "w") as f:
            json.dump(messages_to_save, f, indent=4)

    def get_summary(self, chat_id: str) -> Message:
        """Get chat summary for a specific chat ID."""
        summary_file = os.path.join(self._history_path, chat_id + ".summary.json")
        
        if not os.path.exists(summary_file):
            return Message(content="", role="system", provider=None, model=None)
        
        with open(summary_file, "r") as f:
            summary_content = json.load(f)
        
        return Message(content=summary_content, role="system", provider=None, model=None)

    def save_summary(self, chat_id: str, summary: str) -> None:
        """Save chat summary."""
        summary_file = os.path.join(self._history_path, chat_id + ".summary.json")
        with open(summary_file, "w") as f:
            json.dump(summary, f, indent=4)

    def clear(self, chat_id: str) -> None:
        """Clear history for a specific chat."""
        file_path = os.path.join(self._history_path, chat_id + ".json")
        summary_file = os.path.join(self._history_path, chat_id + ".summary.json")
        if os.path.exists(file_path):
            os.remove(file_path)
        if os.path.exists(summary_file):
            os.remove(summary_file)
