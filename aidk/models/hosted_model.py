from .model import Model
from ._response_processor import ResponseProcessorMixin
from ._prompt_executor import PromptExecutorMixin
from typing import Sequence, Dict, Union
from ..keys.keys_manager import load_key


class HostedModel(Model, ResponseProcessorMixin, PromptExecutorMixin):
    """
    HostedModel is a class for interacting with self-hosted AI language models.
    Currently support models deployed with VLLM.
    
    Examples
    --------
    Basic usage:
    ```
    model = HostedModel(url="http://localhost:8000", version=1, provider="openai", model="gpt-4")
    response = model.ask("What is the capital of France?")
    ```

    """

    def __init__(
        self, 
        url: str,
        version: int = 1,
        provider: str | None = None, 
        model: str | None = None, 
        system_prompt: str | Sequence[str] = (),
        count_tokens: bool = False, 
        count_cost: bool = False,
        max_tokens: int = None
    ):
        """
        Initialize a new Model instance.

        Parameters
        ----------
        provider : str
            Name of the provider (e.g., 'openai', 'anthropic')
        model : str
            Name of the model (e.g., 'gpt-4', 'claude-3')
        system_prompt : str | Sequence[str], optional
            System prompt or sequence of prompts
        count_tokens : bool, optional
            Whether to count tokens for each request
        count_cost : bool, optional
            Whether to calculate costs for each request
        max_tokens : int, optional
            Maximum number of tokens for each request
        """

        super().__init__(
            provider=provider,
            model=model,
            count_tokens=count_tokens, 
            count_cost=count_cost, 
            max_tokens=max_tokens
        )
        
        self.url = url
        self.version = version

