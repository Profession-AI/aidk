"""
In-memory history storage implementation.
Useful for testing and temporary conversations.
"""

from typing import Optional
from .models import BaseHistory, Message


class DictHistory(BaseHistory):
    """
    In-memory history storage using Python dictionaries.
    Useful for testing and temporary conversations.
    """
    
    def __init__(self, 
                 last_n: Optional[int] = None,
                 summarizer_provider: Optional[str] = None,
                 summarizer_model: Optional[str] = None,
                 summarizer_max_tokens: Optional[int] = None):
        # Call parent __init__ to set up summarizer and last_n
        super().__init__(
            last_n=last_n,
            summarizer_provider=summarizer_provider,
            summarizer_model=summarizer_model,
            summarizer_max_tokens=summarizer_max_tokens
        )
        
        self._histories: dict[str, list[Message]] = {}
        self._summaries: dict[str, str] = {}
    
    def get_messages(self, chat_id: str) -> list[Message]:
        if chat_id not in self._histories:
            self.messages = []
            return self.messages
        
        messages = self._histories[chat_id]
        if self._last_n is not None and len(messages) > (self._last_n + 1) * 2:
            messages = [messages[0]] + messages[-self._last_n * 2:]
        
        self.messages = [Message.from_dict(msg) if isinstance(msg, dict) else msg for msg in messages]
        return self.messages

    def save_message(self, chat_id: str, messages: list[Message]) -> None:
        if chat_id not in self._histories:
            self._histories[chat_id] = []
        self._histories[chat_id].extend(messages)
          
    def get_summary(self, chat_id: str) -> Message:
        """Get chat summary for a specific chat ID."""
        return Message(content=self._summaries.get(chat_id, ""), role="system", provider=None, model=None)
    
    def save_summary(self, chat_id: str, summary: str) -> None:
        self._summaries[chat_id] = summary
    
    def clear(self, chat_id: str) -> None:
        """Clear history for a specific chat."""
        if chat_id in self._histories:
            del self._histories[chat_id]
        if chat_id in self._summaries:
            del self._summaries[chat_id]
