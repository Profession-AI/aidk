from abc import ABC, abstractmethod
from typing import List, Dict, Union, Optional, TYPE_CHECKING, Any

if TYPE_CHECKING:
    from aidk.rag.rag import RAG  # pragma: no cover
else:
	RAG = Any  # type: ignore

class BaseModel(ABC):

    def __init__(
        self, 
        count_tokens: bool = False, 
        count_cost: bool = False,
        max_tokens: int = None,
        rag: Optional['RAG'] = None
    ):
        """
        Initialize base model with counting preferences.
        
        Args:
            count_tokens: Whether to count tokens for each request
            count_cost: Whether to calculate costs for each request
            max_tokens: Maximum number of tokens to generate
            rag: RAG object
        """
        self._count_tokens = count_tokens
        self._count_cost = count_cost
        self._max_tokens = max_tokens
        self._rag = rag

    @abstractmethod
    def ask(self, prompt: str) -> Union[List[Dict], Dict]:
        """Ask the model synchronously."""
        pass

    @abstractmethod
    async def _ask_async(self, prompt: str) -> Union[List[Dict], Dict]:
        """Ask the model asynchronously."""
        pass

    def _add_rag(self, rag: 'RAG'):
        self._rag=rag

    def _add_tools(self, tools):
        self._tools=tools
