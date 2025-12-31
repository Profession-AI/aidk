"""Model class for interacting with AI language models."""

from typing import Dict, Union, AsyncGenerator

from aidk.conf import Conf

from ._base_model import BaseModel
from ..keys.keys_manager import load_key
from ._response_processor import ResponseProcessorMixin
from ._prompt_executor import PromptExecutorMixin
from ..prompts.prompt_chain import PromptChain
from ..prompts.prompt import Prompt

class Model(BaseModel, ResponseProcessorMixin, PromptExecutorMixin):
    """
    Model class for interacting with AI language models.

    This module provides the Model class which serves as the primary interface for interacting
    with various AI language models (like GPT-4, Claude-3, etc.).

    Examples
    --------
    Basic usage:
    ```
    model = Model(provider="openai", model="gpt-4")
    response = model.ask("What is the capital of France?")
    ```

    With prompt:
    ```
    model = Model(
        provider="anthropic",
        model="claude-3",
    )
    prompt = Prompt(
        prompt="What is the capital of {country}?",
        prompt_data={"country": "France"},
        response_type=str
    )
    response = model.ask(prompt)
    ```
    """

    def __init__(
        self,
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
        super().__init__(max_tokens=max_tokens)

        if provider is None:
            provider = Conf()["base_model"]["provider"]
        if model is None:
            model = Conf()["base_model"]["model"]

        load_key(provider)

        self.provider = provider
        self.model = model
        self._web_search = False

    async def _ask_async(
        self, prompt: Union[str, Prompt, PromptChain], metadata: Dict = None
    ) -> Dict:
        """
        Ask the model asynchronously.

        Parameters
        ----------
        prompt : Union[str, Prompt]
            The prompt to process
        metadata : Dict, optional
            Metadata to pass to the completion call

        Returns
        -------
        Dict
            Dictionary containing:
            - response: The model's response
            - prompt: The original prompt
            - model: Dictionary with provider and model name
            - tokens: Token counts (if enabled)
            - cost: Cost calculation (if enabled)

        """
        if metadata is None:
            metadata = {}
        response = await self._execute_async(prompt, metadata)
        return self._process_response(
            prompt,
            response,
        )

    async def ask_stream(
        self, prompt: Union[str, Prompt, PromptChain], metadata: Dict = None
    ) -> AsyncGenerator[Dict, None]:
        """
        Ask the model with streaming response.

        Parameters
        ----------
        prompt : Union[str, Prompt, PromptChain]
            The prompt to process
        metadata : Dict, optional
            Metadata to pass to the completion call

        Yields
        ------
        Dict
            Streaming response chunks
        """
        if metadata is None:
            metadata = {}
        response = ""
        yield self._process_stream_head()
        async for chunk in self._execute_stream(prompt, metadata):
            if "usage" in chunk:
                yield self._process_stream_tail(chunk, prompt, response)
            else:
                processed_chunk = self._process_stream_chunk(chunk)
                if processed_chunk.delta is not None:
                    response += processed_chunk.delta
                    yield processed_chunk

    def ask(self, prompt: Union[str, Prompt, PromptChain], metadata: Dict = None) -> Dict:
        """
        Ask the model.

        Parameters
        ----------
        prompt : Union[str, Prompt]
            The prompt to process
        metadata : Dict, optional
            Metadata to pass to the completion call

        Returns
        -------
        ModelResponse
            - response: The model's response
            - prompt: The original prompt
            - model: Dictionary with provider and model name
            - usage: Token counts and cost
        """
        if metadata is None:
            metadata = {}
        if isinstance(prompt, str):
            prompt = Prompt(prompt=prompt)
        response = self._execute(prompt, metadata)
        return self._process_response(
            prompt,
            response
        )
