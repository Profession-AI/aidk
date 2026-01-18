"""Base model abstract class."""

from abc import ABC, abstractmethod
from typing import TYPE_CHECKING, Any, Dict, List, Optional, Union

if TYPE_CHECKING:
    from aidk.rag.rag import RAG  # pragma: no cover
else:
    RAG = Any  # type: ignore


class BaseModel(ABC):
    """Abstract base class for all model implementations."""

    def __init__(
        self,
        max_tokens: int = None,
        rag: Optional['RAG'] = None
    ):
        """
        Initialize base model.

        Args:
            max_tokens: Maximum number of tokens to generate
            rag: RAG object
        """
        self._max_tokens = max_tokens
        self._rag = rag

    @abstractmethod
    def ask(self, prompt: str) -> Union[List[Dict], Dict]:
        """Ask the model synchronously."""

    @abstractmethod
    async def ask_async(self, prompt: str) -> Union[List[Dict], Dict]:
        """Ask the model asynchronously."""

    def _add_rag(self, rag: 'RAG'):
        """Add RAG to the model."""
        self._rag = rag

    def _add_tools(self, tools):
        """Add tools to the model."""
        self._tools = tools
