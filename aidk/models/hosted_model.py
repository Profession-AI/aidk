"""Hosted model for self-hosted AI language models."""

from typing import Sequence

from ._prompt_executor import PromptExecutorMixin
from ._response_processor import ResponseProcessorMixin
from .model import Model


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
        max_tokens : int, optional
            Maximum number of tokens for each request
        """

        super().__init__(
            provider=provider,
            model=model,
            max_tokens=max_tokens
        )

        self.url = url
        self.version = version
