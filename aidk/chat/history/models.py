"""
Data models for chat history management.
Contains base classes and data structures for message handling.
"""

import uuid
import datetime
from dataclasses import dataclass, asdict
from typing import Optional
from aidk.models import Model
from abc import ABC, abstractmethod
from typing import final


@dataclass
class Message:
    """Represents a single message in a chat history."""
    content: str
    role: str
    timestamp: str = None
    provider: str = None
    model: str = None
    
    def __post_init__(self):
        if self.timestamp is None:
            self.timestamp = datetime.datetime.utcnow().isoformat() + 'Z'
    
    def to_dict(self):
        """Convert Message to dictionary for JSON serialization."""
        return asdict(self)
    
    @classmethod
    def from_dict(cls, data: dict):
        """Create Message from dictionary."""
        return cls(**data)


class BaseHistory(ABC):
    """Base class for all history storage implementations."""

    def __init__(self, 
                 last_n: int = None,
                 summarizer_provider: Optional[str] = None, 
                 summarizer_model: Optional[str] = None,
                 summarizer_max_tokens: Optional[int] = None,
        ): 
        self._last_n = last_n

        if summarizer_provider and summarizer_model:
            from .summarizer import HistorySummarizer
            self._summarizer = HistorySummarizer(
                model=Model(
                    provider=summarizer_provider,
                    model=summarizer_model
                ),
                max_tokens=summarizer_max_tokens)
        else:
            self._summarizer = None

    @final
    def generate_chat_id(self):
        """Generate a unique chat ID."""
        return str(uuid.uuid4())

    @final
    def new_chat(self, system_prompt: Message):
        chat_id = self.generate_chat_id()
        self.store_chat(chat_id, [system_prompt])
        return chat_id
    
    @final
    def store_chat(self, chat_id: str, messages: list[Message]):

        self.save_message(chat_id, messages)        

        if self._summarizer is not None:
            # Update summary
            current_summary = self.get_summary(chat_id).content
            new_summary = self._summarizer.summarize(current_summary, messages)
            self.save_summary(chat_id, new_summary)

    def get_system_prompt(self, chat_id: str) -> Optional[Message]:
        message = self.get_messages(chat_id)[0]
        return [Message(content=message.content+"\n\nPer rispondere usa questo riassunto della conversazione fino ad ora:\n"+self.get_summary(chat_id), role="system")]
    
    @abstractmethod
    def get_messages(self, chat_id: str) -> list[Message]:
        """Load chat history."""
        pass

    @abstractmethod
    def save_message(self, chat_id: str, messages: list[Message]):
        """Store messages in history."""
        pass

    @abstractmethod
    def get_summary(self, chat_id: str):
        """Get chat summary."""
        pass

    @abstractmethod
    def save_summary(self, chat_id: str, summary: str):
        """Save chat summary."""
        pass

    @abstractmethod
    def clear(self):
        """Clear chat history."""
        pass
