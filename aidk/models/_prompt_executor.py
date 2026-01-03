from typing import Dict, Union, AsyncGenerator
import types
from ..prompts.prompt import Prompt
from ..tools._tool_parser import ToolParser
from ..mcp._mcp_tool_parser import McpToolParser
from mcp.types import Tool as MCPTool
from ..conf import Conf
from litellm import completion, acompletion
from litellm import success_callback, failure_callback
import json

class PromptExecutorMixin:
    """Mixin class to handle prompt execution."""

    def _setup_observability(self):
        observability = Conf()["observability"]
        if (len(observability) > 0):
            success_callback = observability 
            failure_callback = observability 


    async def _execute_stream(self, prompt: Union[str, Prompt], metadata: Dict = None) -> AsyncGenerator[Dict, None]:
        """
        Execute a prompt asynchronously with streaming.
        
        Args:
            prompt: The prompt to execute (string, Prompt)
            metadata: Optional metadata to pass to completion calls
            
        Yields:
            Dictionary containing streaming response chunks
        """

        # Handle RAG if available
        if hasattr(self, '_rag') and self._rag:
            response = self._rag.query(prompt)
            if len(response["documents"]) > 0:
                documents = response.get('documents', [])
                documents = [s for doc_list in documents for s in doc_list]
                documents = '\n'.join(documents)
                prompt += Conf()["default_prompt"]["rag"] + documents

        if isinstance(prompt, Prompt):
            async for chunk in self._completion_stream(str(prompt), response_type=prompt.response_type, metadata=metadata):
                yield chunk
        else:
            async for chunk in self._completion_stream(prompt, metadata=metadata):
                yield chunk
    
    def _execute(self, prompt: Union[Prompt], metadata: Dict = {}) -> Dict:
        """
        Execute a prompt synchronously.
        
        Args:
            prompt: The prompt to execute (string, Prompt)
            metadata: Optional metadata to pass to completion calls
            
        Returns:
            Dictionary containing the response
        """

        if self._rag:
            response = self._rag.query(prompt)
            if len(response["documents"])>0:
                documents = response.get('documents', [])
                documents = [s for doc_list in documents for s in doc_list]
                documents = '\n'.join(documents)
                prompt += Conf()["default_prompt"]["rag"] + documents

        return self._completion(prompt, metadata=metadata)

    async def _execute_async(self, prompt: Union[str, Prompt], metadata: Dict = None) -> Dict:
        """
        Execute a prompt asynchronously.
        
        Args:
            prompt: The prompt to execute (string, Prompt)
            metadata: Optional metadata to pass to completion calls
            
        Returns:
            Dictionary containing the response
        """

        if self._rag:
            response = self._rag.query(prompt)
            if len(response["documents"])>0:
                documents = response.get('documents', [])
                documents = [s for doc_list in documents for s in doc_list]
                documents = '\n'.join(documents)
                prompt += Conf()["default_prompt"]["rag"] + documents

        if isinstance(prompt, Prompt):
            return await self._completion_async(str(prompt), response_type=prompt.response_type, metadata=metadata)
        else:
            return await self._completion_async(prompt, metadata=metadata)

    def _get_tools(self):
        tools = None
        if hasattr(self, "_tools") and len(self._tools) > 0:
            tools = []
            tp = ToolParser()
            mcp_tp = McpToolParser()
            for tool in self._tools:
                if isinstance(tool, types.FunctionType):
                    tools.append(tp.parse(tool))      
                elif isinstance(tool, MCPTool):
                    tools.append(mcp_tp.parse(tool))
        return tools

    def _completion(self, prompt: Prompt|list, metadata: Dict | None = None) -> Dict:

        # Avoid mutable defaults and ensure metadata is always a dict
        metadata = metadata or {}

        # Basic observability setup
        self._setup_observability()

        # Determine messages and response type
        from pydantic import BaseModel
        response_type = None
        if isinstance(prompt, Prompt):
            messages = [prompt.as_dict()]
            response_type = getattr(prompt, "response_type", None)
        else:
            messages = prompt

        # Build a Pydantic model for response_format that uses the requested type if provided
        if response_type is None:
            class Response(BaseModel):
                response: str
            response_format = Response
        else:
            # Create a dynamic BaseModel with the desired annotation
            Response = type("Response", (BaseModel,), {"__annotations__": {"response": response_type}})
            response_format = Response

        # Reduce logging noise for upstream libraries
        self._disable_logging()

        # Resolve model and optional base URL
        url = None
        model = f"{self.provider}/{self.model}"
        if getattr(self, "url", None) is not None:
            url = f"{self.url}/v{self.version}"
            model = f"hosted_vllm/{model}"

        tools = self._get_tools()

        # Optional web-search hook (kept for future use)
        """
        if self._web_search:
            web_search_config = {"search_context_size": self._web_search}
        else:
            web_search_config = None
        """

        response = completion(
            model=model,
            messages=messages,
            response_format=response_format,
            base_url=url,
            tools=tools,
            max_tokens=getattr(self, "_max_tokens", None),
            metadata=metadata,
            #web_search_options=web_search_config
        )

        return response

    async def _completion_stream(self, prompt: str|list, response_type: str = None, metadata: Dict = None) -> AsyncGenerator[Dict, None]:
        """
        Execute a streaming completion.
        
        Args:
            prompt: The prompt to execute
            response_type: Optional response type
            metadata: Optional metadata to pass to completion call
            
        Yields:
            Streaming response chunks
        """
        self._setup_observability()
        url = None
        model = self.provider+"/"+self.model
        if hasattr(self, "url") and self.url != None:
            url = self.url+"/v"+str(self.version)
            model = "hosted_vllm/"+model
        
        tools = self._get_tools()

        if isinstance(prompt, str):
            messages = [{ "content": prompt,"role": "user"}]
        else:
            messages = prompt

        self._disable_logging()
        from litellm import acompletion

        response = await acompletion(
            model=model, 
            messages=messages, 
            response_format=response_type,
            base_url=url,
            stream=True,
            max_tokens=self._max_tokens,
            metadata=metadata,
            tools=tools
        )
        
        async for chunk in response:
            yield chunk

    async def _completion_async(self, prompt: str|list, response_type: str = None, metadata: Dict = None) -> Dict:
        """
        Execute a completion asynchronously.
        
        Args:
            prompt: The prompt to execute
            response_type: Optional response type
            metadata: Optional metadata to pass to completion call
            
        Returns:
            Dictionary containing the response
        """
        self._setup_observability()
        from pydantic import BaseModel

        class Response(BaseModel):
            response: response_type

        if response_type!=None:
            response_type = Response

        self._disable_logging()

        url = None
        model = self.provider+"/"+self.model
        
        if hasattr(self, "url") and self.url != None:
            url = self.url+"/v"+str(self.version)
            model = "hosted_vllm/"+model
        
        tools = None

        if hasattr(self, "_tools"):
            tools = []
            tp = ToolParser()
            for tool in self._tools:
                tools.append(tp.parse(tool))      

        if isinstance(prompt, str):
            messages = [{ "content": prompt,"role": "user"}]
        else:
            messages = prompt

        from litellm import acompletion
        return await acompletion(model=model, 
                                messages=messages, 
                                response_format=response_type,
                                base_url = url,
                                tools=tools,
                                max_tokens=self._max_tokens,
                                metadata=metadata)
            

    def _disable_logging(self):
        import logging
        loggers = [
            "LiteLLM Proxy",
            "LiteLLM Router",
            "LiteLLM",
            "httpx"
        ]

        for logger_name in loggers:
            logger = logging.getLogger(logger_name)
            logger.setLevel(logging.CRITICAL + 1) 

