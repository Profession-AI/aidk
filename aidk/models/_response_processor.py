"""Response processor mixin for handling model responses."""

from dataclasses import dataclass
from decimal import Decimal
from typing import Dict

from ..prompts.prompt import Prompt

@dataclass
class Model:
    """Model information."""
    provider: str
    name: str

@dataclass
class ModelUsage:
    """Model usage statistics."""
    completion_tokens: int
    prompt_tokens: int
    total_tokens: int
    cost: Decimal

@dataclass
class ModelResponse:
    """Model response."""
    prompt: str
    response: str
    model: Model
    usage:ModelUsage = None

@dataclass
class ModelStreamHead:
    """Model response head for streaming."""
    model: Model

@dataclass
class ModelStreamChunk:
    """Model response chunk for streaming."""
    delta: str

@dataclass
class ModelStreamTail:
    """Model response tail for streaming."""
    prompt: str
    response: str
    model: Model
    usage: ModelUsage

class ResponseProcessorMixin:
    """Mixin class to handle response processing."""

    def _process_response(
        self,
        prompt: Prompt,
        response: Dict
    ) -> ModelResponse:
        """
        Process the response and add token and cost information.

        Args:
            prompt: The input prompt
            response: The model's response

        Returns:
            ModelResponse containing the response and usage stats
        """

        return ModelResponse(
            prompt=str(prompt),
            response=response.choices[0].message.content,
            model=Model(provider=self.provider, name=self.model),
            usage=ModelUsage(
                completion_tokens=response.usage.completion_tokens,
                prompt_tokens=response.usage.prompt_tokens,
                total_tokens=response.usage.total_tokens,
                cost = round(Decimal(response._hidden_params["response_cost"]), 8)
            ),
        )

    def _process_stream_head(self):
        return ModelStreamHead(
            model=Model(provider=self.provider, name=self.model)
        )

    def _process_stream_chunk(self, chunk):
        return ModelStreamChunk(
            delta=chunk.choices[0].delta.content)

    def _process_stream_tail(self, chunk, prompt, response):
        return ModelStreamTail(
            prompt = str(prompt),
            response = response,
            model = Model(provider=self.provider, name=self.model),
            usage=ModelUsage(
                completion_tokens=chunk.usage.completion_tokens,
                prompt_tokens=chunk.usage.prompt_tokens,
                total_tokens=chunk.usage.total_tokens,
                cost=(
                    round(Decimal(chunk._hidden_params["response_cost"]), 8)
                    if chunk._hidden_params["response_cost"] is not None
                    else None
                )
            ),
        )
