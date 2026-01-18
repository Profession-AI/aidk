"""Prompt executor mixin for handling prompt execution."""

import logging
import types
from typing import Dict, Union, AsyncGenerator, List, Optional, Any

from litellm import acompletion, completion
from mcp.types import Tool as MCPTool
from pydantic import BaseModel, create_model

from ..conf import Conf
from ..mcp._mcp_tool_parser import McpToolParser
from ..prompts.prompt import Prompt
from ..tools._tool_parser import ToolParser

# Configura logger globalmente per il modulo se necessario
logger = logging.getLogger(__name__)

class PromptExecutorMixin:
    """Mixin class to handle prompt execution."""

    # Definisco attributi attesi per chiarezza (type hinting)
    _rag: Any
    _tools: List[Any]
    provider: str
    model: str
    url: Optional[str]
    version: Optional[str]
    _max_tokens: Optional[int]

    def _setup_observability(self):
        """Configures observability callbacks based on configuration."""
        observability = Conf()["observability"]
        if observability:
            # Nota: litellm usa variabili globali per i callback
            import litellm
            litellm.success_callback = observability
            litellm.failure_callback = observability

    def _disable_logging(self):
        """Disable logging for specific noisy loggers."""
        loggers = ["LiteLLM Proxy", "LiteLLM Router", "LiteLLM", "httpx"]
        for logger_name in loggers:
            logging.getLogger(logger_name).setLevel(logging.CRITICAL + 1)

    def _get_tools(self) -> Optional[List[Dict]]:
        """Parses and returns tools if available."""
        if not hasattr(self, "_tools") or not self._tools:
            return None

        tools = []
        tp = ToolParser()
        mcp_tp = McpToolParser()
        
        for tool in self._tools:
            if isinstance(tool, types.FunctionType):
                tools.append(tp.parse(tool))
            elif isinstance(tool, MCPTool):
                tools.append(mcp_tp.parse(tool))
        
        return tools if tools else None

    def _resolve_model_params(self) -> dict:
        """Resolves model name and API base URL."""
        model = f"{self.provider}/{self.model}"
        url = None
        
        if getattr(self, "url", None):
            url = f"{self.url}/v{self.version}"
            # Specifico per vLLM hosted o simili proxy
            model = f"hosted_vllm/{model}"
            
        return {"model": model, "base_url": url}

    def _apply_rag(self, prompt: Union[str, Prompt]) -> Union[str, Prompt]:
        """Applies RAG logic to the prompt if enabled."""
        if not hasattr(self, '_rag') or not self._rag:
            return prompt

        # Determina il testo della query
        query_text = str(prompt)
        
        response = self._rag.query(query_text)
        if not response:
            return prompt

        documents = '\n'.join([doc.content for doc in response])
        rag_suffix = Conf()["default_prompt"]["rag"] + documents

        if isinstance(prompt, Prompt):
            # Assumendo che Prompt abbia un attributo modificabile o supporti +=
            # È più sicuro modificare l'attributo interno se esposto, o creare una copia
            if hasattr(prompt, '_prompt'):
                 prompt._prompt += rag_suffix
            else:
                 # Fallback se Prompt si comporta come stringa
                 prompt += rag_suffix 
        else:
            prompt += rag_suffix
            
        return prompt

    def _prepare_messages(self, prompt: Union[str, Prompt, List]) -> List[Dict]:
        """Normalizes the prompt into a list of messages."""
        if isinstance(prompt, Prompt):
            return [prompt.as_dict()]
        elif isinstance(prompt, list):
            return prompt
        else:
            return [{"content": str(prompt), "role": "user"}]

    def _create_response_model(self, response_type: Any) -> Optional[type[BaseModel]]:
        """Creates a dynamic Pydantic model for structured output."""
        if not response_type:
            return None
            
        # Se response_type è già un modello o tipo semplice, litellm spesso lo gestisce,
        # ma qui replico la logica originale di creare un wrapper "Response"
        return create_model(
            'Response',
            response=(response_type, ...),
            __base__=BaseModel
        )

    # --- Execution Methods ---

    async def _execute_stream(self, prompt: Union[str, Prompt], metadata: Dict = None) -> AsyncGenerator[Dict, None]:
        metadata = metadata or {}
        prompt = self._apply_rag(prompt)
        
        # Determina response_type
        response_type = prompt.response_type if isinstance(prompt, Prompt) else None
        
        async for chunk in self._completion_stream(prompt, response_type=response_type, metadata=metadata):
            yield chunk

    def _execute(self, prompt: Union[str, Prompt], metadata: Dict = None) -> Dict:
        metadata = metadata or {}
        prompt = self._apply_rag(prompt)
        return self._completion(prompt, metadata=metadata)

    async def _execute_async(self, prompt: Union[str, Prompt], metadata: Dict = None) -> Dict:
        metadata = metadata or {}
        prompt = self._apply_rag(prompt)
        
        response_type = prompt.response_type if isinstance(prompt, Prompt) else None
        
        return await self._completion_async(prompt, response_type=response_type, metadata=metadata)

    # --- Completion Implementation ---

    def _completion(self, prompt: Union[str, Prompt, List], metadata: Dict = None) -> Dict:
        metadata = metadata or {}
        self._setup_observability()
        self._disable_logging()

        messages = self._prepare_messages(prompt)
        model_params = self._resolve_model_params()
        tools = self._get_tools()

        # Gestione Response Format
        response_type = getattr(prompt, "response_type", None) if isinstance(prompt, Prompt) else None
        # Se non c'è tipo specifico, l'originale creava un modello default Response(response: str).
        # Qui manteniamo la logica se necessario, o passiamo None.
        response_format = self._create_response_model(response_type) if response_type else None
        
        # Se response_format è None e volevamo il default:
        if response_type is None and isinstance(prompt, Prompt):
             # Ripristino logica originale per Prompt senza tipo specifico
             class DefaultResponse(BaseModel):
                 response: str
             response_format = DefaultResponse

        return completion(
            model=model_params["model"],
            base_url=model_params["base_url"],
            messages=messages,
            response_format=response_format,
            tools=tools,
            max_tokens=getattr(self, "_max_tokens", None),
            metadata=metadata,
        )

    async def _completion_stream(
        self, prompt: Union[str, List], response_type: Any = None, metadata: Dict = None
    ) -> AsyncGenerator[Dict, None]:
        metadata = metadata or {}
        self._setup_observability()
        self._disable_logging()

        messages = self._prepare_messages(prompt)
        model_params = self._resolve_model_params()
        tools = self._get_tools()

        response = await acompletion(
            model=model_params["model"],
            base_url=model_params["base_url"],
            messages=messages,
            response_format=response_type, # Stream spesso gestisce i formati diversamente
            stream=True,
            max_tokens=getattr(self, "_max_tokens", None),
            metadata=metadata,
            tools=tools
        )

        async for chunk in response:
            yield chunk

    async def _completion_async(self, prompt: Union[str, List], response_type: Any = None, metadata: Dict = None) -> Dict:
        metadata = metadata or {}
        self._setup_observability()
        self._disable_logging()

        messages = self._prepare_messages(prompt)
        model_params = self._resolve_model_params()
        
        # FIX: Ora usa _get_tools invece di reimplementare la logica (che era buggata su MCP)
        tools = self._get_tools()
        
        # Gestione response format coerente
        response_format = self._create_response_model(response_type) if response_type else None

        return await acompletion(
            model=model_params["model"],
            base_url=model_params["base_url"],
            messages=messages,
            response_format=response_format,
            tools=tools,
            max_tokens=getattr(self, "_max_tokens", None),
            metadata=metadata
        )