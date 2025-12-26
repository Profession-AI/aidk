from ._base_model import BaseModel
from .model import Model
from .multi_model import MultiModel
from ._prompt_executor import PromptExecutorMixin
from ._response_processor import ResponseProcessorMixin
from typing import List, Dict, Union
import asyncio
from ..prompts.prompt_chain import PromptChain
from ..prompts.prompt import Prompt

class CollaborativeModel(BaseModel, PromptExecutorMixin, ResponseProcessorMixin):
    """
    A class to implement collaborative decision making across multiple AI models.
    
    CollaborativeModel manages a collection of AI models and an aggregator model.
    It executes prompts across all models in parallel and then uses the aggregator
    to synthesize a final response based on all individual responses.

    Examples
    --------
    Basic collaborative analysis:
    ```
    models = [
        {"provider": "openai", "model": "gpt-4"},
        {"provider": "anthropic", "model": "claude-3"}
    ]
    aggregator = {"provider": "openai", "model": "gpt-4"}
    collab = CollaborativeModel(models=models, aggregator=aggregator)
    response = collab.ask("Explain quantum computing")
    print(response["response"])  # Aggregated response
    for ind_resp in response["individual_responses"]:
        print(f"{ind_resp['model']['name']}: {ind_resp['response']}")
    ```
    """

    def __init__(
        self,
        models: List[Dict[str, str]],
        aggregator: Dict[str, str],
        count_tokens: bool = False,
        count_cost: bool = False
    ):
        """
        Initialize a new CollaborativeModel instance.

        Parameters
        ----------
        models : List[Dict[str, str]]
            List of dictionaries with provider and model information
        aggregator : Dict[str, str]
            Dictionary with provider and model information for the aggregator
        count_tokens : bool, optional
            Whether to count tokens for each request
        count_cost : bool, optional
            Whether to calculate costs for each request
        """
        super().__init__(count_tokens, count_cost)

        self._multi_model = MultiModel(
            models=models,
            count_tokens=count_tokens,
            count_cost=count_cost
        )

        self._aggregator = Model(
            provider=aggregator['provider'],
            model=aggregator['model'],
            count_tokens=count_tokens,
            count_cost=count_cost
        )

    def _format_aggregator_prompt(self, prompt: Union[str, Prompt, PromptChain], responses: List[Dict]) -> str:
        """
        Format the prompt for the aggregator model.

        Parameters
        ----------
        prompt : Union[str, Prompt, PromptChain]
            The original prompt
        responses : List[Dict]
            List of responses from individual models

        Returns
        -------
        str
            Formatted prompt for the aggregator including original question
            and all model responses
        """
        prompt_text = str(prompt)
        model_responses = "\n\n".join([
            f"Model {i+1} ({response['model']['provider']} - {response['model']['name']}):\n{response['response']}"
            for i, response in enumerate(responses)
        ])
        
        return f"""Please analyze the following responses from different models and provide a comprehensive answer:
                    Original Question: {prompt_text}
                    Model Responses:
                    {model_responses}
                    Please provide a well-reasoned response that takes into account all the information above."""

    async def _ask_async(self, prompt: Union[str, Prompt, PromptChain]) -> Dict:
        """
        Ask all models and aggregate their responses asynchronously.

        Parameters
        ----------
        prompt : Union[str, Prompt, PromptChain]
            The prompt to process across all models

        Returns
        -------
        Dict
            Dictionary containing:
            - response: The aggregated response
            - prompt: The original prompt
            - model: Dictionary with aggregator's provider and model name
            - tokens: Token counts (if enabled)
            - cost: Cost calculation (if enabled)
            - individual_responses: List of responses from individual models

        Examples
        --------
        Using async/await:
            >>> response = await collab.ask_async("What is consciousness?")
            >>> print(response["response"])  # Aggregated response
            >>> for resp in response["individual_responses"]:
            ...     print(f"{resp['model']['name']}: {resp['response']}")
        """
        # Get responses from all models
        model_responses = await self._multi_model.ask_async(prompt)
        
        # Get aggregator response
        aggregator_prompt = self._format_aggregator_prompt(prompt, model_responses)
        aggregator_response = await self._execute_async(aggregator_prompt, self._aggregator._agent)
        
        # Process aggregator response
        processed_aggregator = self._process_response(
            aggregator_prompt,
            aggregator_response,
            self._aggregator.provider,
            self._aggregator.model,
            self._count_tokens,
            self._count_cost
        )

        processed_aggregator["individual_responses"] = model_responses
        return processed_aggregator

    def ask(self, prompt: Union[str, Prompt, PromptChain]) -> Dict:
        """
        Ask all models and aggregate their responses synchronously.

        Parameters
        ----------
        prompt : Union[str, Prompt, PromptChain]
            The prompt to process across all models

        Returns
        -------
        Dict
            Dictionary containing:
            - response: The aggregated response
            - prompt: The original prompt
            - model: Dictionary with aggregator's provider and model name
            - tokens: Token counts (if enabled)
            - cost: Cost calculation (if enabled)
            - individual_responses: List of responses from individual models

        """
        return asyncio.run(self.ask_async(prompt))

