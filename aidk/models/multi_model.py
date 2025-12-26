"""Multi model for parallel execution across multiple AI models."""

import asyncio
from typing import Dict, List, Union

from ..prompts.prompt import Prompt
from ..prompts.prompt_chain import PromptChain

from ._base_model import BaseModel
from ._prompt_executor import PromptExecutorMixin
from ._response_processor import ResponseProcessorMixin
from .model import Model


class MultiModel(BaseModel, PromptExecutorMixin, ResponseProcessorMixin):
    """
    A class to execute prompts across multiple AI models in parallel.
    
    MultiModel manages a collection of AI models and enables parallel execution of prompts
    across all models. It's particularly useful for comparing model responses or
    implementing ensemble approaches.

    Examples
    --------
    Basic comparison of models:
    ```
    models = [
        {"provider": "openai", "model": "gpt-4"},
        {"provider": "anthropic", "model": "claude-3"}
    ]
    multi_model = MultiModel(models=models)
    prompt = Prompt(
        prompt="What is 2+2?",
        response_type=int
    )
    responses = multi_model.ask(prompt)
    for resp in responses:
        print(f"{resp['model']['name']}: {resp['response']}")
    ```
    """

    def __init__(
        self,
        models: List[Dict[str, str]]
    ):
        """
        Initialize a new MultiModel instance.

        Parameters
        ----------
        models : List[Dict[str, str]]
            List of dictionaries with provider and model information
        """
        super().__init__()
        self._models = [
            Model(
                provider=model['provider'],
                model=model['model']
            ) for model in models
        ]

    async def _task(self, model: Model, prompt: Union[str, Prompt, PromptChain]) -> Dict:
        """
        Execute a single model task asynchronously.

        Parameters
        ----------
        model : Model
            The model instance to use
        prompt : Union[str, Prompt, PromptChain]
            The prompt to process

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
        response = await self._execute_async(prompt, metadata=None)
        return self._process_response(
            prompt,
            response
        )

    async def _ask_async(self, prompt: Union[str, Prompt, PromptChain]) -> List[Dict]:
        """
        Ask all models asynchronously.

        Parameters
        ----------
        prompt : Union[str, Prompt, PromptChain]
            The prompt to process across all models

        Returns
        -------
        List[Dict]
            List of response dictionaries, one per model, each containing:
            - response: The model's response
            - prompt: The original prompt
            - model: Dictionary with provider and model name
            - tokens: Token counts (if enabled)
            - cost: Cost calculation (if enabled)

        Examples
        --------
        Using async/await:
            >>> responses = await multi_model.ask_async("What is 2+2?")
            >>> for resp in responses:
            ...     print(f"{resp['model']['name']}: {resp['response']}")
        """
        tasks = [self._task(model, prompt) for model in self._models]
        return await asyncio.gather(*tasks)

    def ask(self, prompt: Union[str, Prompt]) -> List[Dict]:
        """
        Ask all models.

        Parameters
        ----------
        prompt : Union[str, Prompt]
            The prompt to process across all models

        Returns
        -------
        List[Dict]
            List of response dictionaries, one per model, each containing:
            - response: The model's response
            - prompt: The original prompt
            - model: Dictionary with provider and model name
            - tokens: Token counts (if enabled)
            - cost: Cost calculation (if enabled)

        """
        return asyncio.run(self._ask_async(prompt))


if __name__ == "__main__":
    # Initialize multi model
    multi_model = MultiModel(
        models=[
            {'provider': 'openai', 'model': 'gpt-4o-mini'},
            {'provider': 'deepseek', 'model': 'deepseek-chat'},
        ]
    )

    # Example with simple prompt
    results = multi_model.ask("What is the capital of France?")
    print("\nSimple prompt results:")
    for i, result in enumerate(results, 1):
        print(f"\nModel {i} Response:")
        print(result["output"])
        if "tokens" in result:
            print("Token Count:")
            print(f"Input tokens: {result['tokens']['input_tokens']:,}")
            print(f"Output tokens: {result['tokens']['output_tokens']:,}")
            print(f"Total tokens: {result['tokens']['total_tokens']:,}")
        if "cost" in result:
            print("Cost:")
            print(f"Input cost: ${result['cost']['input_cost']:.6f}")
            print(f"Output cost: ${result['cost']['output_cost']:.6f}")
            print(f"Total cost: ${result['cost']['total_cost']:.6f}")

    # Example with prompt chain
    chain = PromptChain([
        "What is the capital of France?",
        "Based on the previous answer, what is its population?"
    ])
    results = multi_model.ask(chain)
    print("\nPrompt chain results:")
    for i, result in enumerate(results, 1):
        print(f"\nModel {i} Response:")
        print(result["output"])
        if "tokens" in result:
            print("Token Count:")
            print(f"Input tokens: {result['tokens']['input_tokens']:,}")
            print(f"Output tokens: {result['tokens']['output_tokens']:,}")
            print(f"Total tokens: {result['tokens']['total_tokens']:,}")
        if "cost" in result:
            print("Cost:")
            print(f"Input cost: ${result['cost']['input_cost']:.6f}")
            print(f"Output cost: ${result['cost']['output_cost']:.6f}")
            print(f"Total cost: ${result['cost']['total_cost']:.6f}")
