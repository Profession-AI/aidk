from typing import Dict, Union, AsyncGenerator
import types
from ..prompts.prompt import Prompt
from ..prompts.prompt_chain import PromptChain
from ..prompts.iterative_prompt import IterativePrompt
from ..tools._tool_parser import ToolParser
from ..mcp._mcp_tool_parser import McpToolParser
from mcp.types import Tool as MCPTool
from ..conf import Conf
import litellm

class PromptExecutorMixin:
    """Mixin class to handle prompt execution."""

    def _setup_observability(self):
        observability = Conf()["observability"]
        if (len(observability) > 0):
            litellm.success_callback = observability 
            litellm.failure_callback = observability 


    async def _execute_stream(self, prompt: Union[str, Prompt, PromptChain], metadata: Dict = None) -> AsyncGenerator[Dict, None]:
        """
        Execute a prompt asynchronously with streaming.
        
        Args:
            prompt: The prompt to execute (string, Prompt, or PromptChain)
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

        # Handle different prompt types
        if isinstance(prompt, PromptChain):
            async for chunk in self._execute_chain_stream(prompt, metadata):
                yield chunk
        elif isinstance(prompt, IterativePrompt):
            async for chunk in self._execute_iterative_stream(prompt, metadata):
                yield chunk
        elif isinstance(prompt, Prompt):
            async for chunk in self._completion_stream(str(prompt), response_type=prompt.response_type, metadata=metadata):
                yield chunk
        else:
            async for chunk in self._completion_stream(prompt, metadata=metadata):
                yield chunk
    
    def _execute(self, prompt: Union[Prompt, PromptChain], metadata: Dict = {}) -> Dict:
        """
        Execute a prompt synchronously.
        
        Args:
            prompt: The prompt to execute (string, Prompt, or PromptChain)
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

        if isinstance(prompt, PromptChain):
            return self._execute_chain(prompt, metadata)
        elif isinstance(prompt, IterativePrompt):
            return self._execute_iterative(prompt, metadata)
        else:
            return self._completion(prompt, metadata=metadata)

    async def _execute_async(self, prompt: Union[str, Prompt, PromptChain], metadata: Dict = None) -> Dict:
        """
        Execute a prompt asynchronously.
        
        Args:
            prompt: The prompt to execute (string, Prompt, or PromptChain)
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

        if isinstance(prompt, PromptChain):
            return await self._execute_chain_async(prompt, metadata)
        elif isinstance(prompt, IterativePrompt):
            return await self._execute_iterative_async(prompt, metadata)
        elif isinstance(prompt, Prompt):
            return await self._completion_async(str(prompt), response_type=prompt.response_type, metadata=metadata)
        else:
            return await self._completion_async(prompt, metadata=metadata)

    async def _execute_chain_async(self, chain: PromptChain, metadata: Dict = None) -> Dict:
        """
        Execute a prompt chain asynchronously.
        
        Args:
            chain: The prompt chain to execute
            metadata: Optional metadata to pass to completion calls
            
        Returns:
            Dictionary containing the final response
        """
        response = None
        for i in range(chain._size):
            current_prompt = chain._format(i, response.output if response else None)
            response = await self._completion_async(current_prompt, metadata=metadata)
        return response

    async def _execute_chain_stream(self, chain: PromptChain, metadata: Dict = None) -> AsyncGenerator[Dict, None]:
        """
        Execute a prompt chain asynchronously with streaming.
        
        Args:
            chain: The promptChain to execute
            metadata: Optional metadata to pass to completion calls
            
        Yields:
            Streaming response chunks from the final prompt in the chain
        """
        response = None
        for i in range(chain._size):
            current_prompt = chain._format(i, response.output if response else None)
            if i == chain._size - 1:  # Last prompt in chain
                async for chunk in self._completion_stream(current_prompt, metadata=metadata):
                    yield chunk
            else:  # Execute non-streaming for intermediate prompts
                response = self._completion(current_prompt, metadata=metadata)

    def _execute_chain(self, chain: PromptChain, metadata: Dict = None) -> Dict:
        """
        Execute a prompt chain synchronously.
        
        Args:
            chain: The prompt chain to execute
            metadata: Optional metadata to pass to completion calls
            
        Returns:
            Dictionary containing the final response
        """
        response = None
        for i in range(chain._size):
            current_prompt = chain._format(i, response["choices"][0]["message"]["content"] if response else None)
            response = self._completion(current_prompt, metadata=metadata)
        return response
    
    def _execute_iterative(self, prompt: IterativePrompt, metadata: Dict = None) -> Dict:
        """
        Execute an iterative prompt synchronously.
        
        Args:
            prompt: The iterative prompt to execute
            metadata: Optional metadata to pass to completion calls
            
        Returns:
            Dictionary containing the final response
        """
        response = ""
        memory = ""
        for i in range(prompt._size):
            if i > 0 and prompt._has_memory:
                if prompt._retain_all:
                    memory += current_response
                else:
                    memory = current_response   
                current_prompt = prompt._format(i, memory)
            else:
                current_prompt = prompt._format(i)
            current_response = self._completion(current_prompt, metadata=metadata)

            response += current_response
        return response

    async def _execute_iterative_stream(self, prompt: IterativePrompt, metadata: Dict = None) -> AsyncGenerator[Dict, None]:
        """
        Execute an iterative prompt asynchronously with streaming.
        
        Args:
            prompt: The iterative prompt to execute
            metadata: Optional metadata to pass to completion calls
            
        Yields:
            Streaming response chunks from the final iteration
        """
        response = ""
        memory = ""
        for i in range(prompt._size):
            if i > 0 and prompt._has_memory:
                if prompt._retain_all:
                    memory += current_response
                else:
                    memory = current_response   
                current_prompt = prompt._format(i, memory)
            else:
                current_prompt = prompt._format(i)
            
            if i == prompt._size - 1:  # Last iteration
                async for chunk in self._completion_stream(current_prompt, metadata=metadata):
                    yield chunk
            else:  # Execute non-streaming for intermediate iterations
                current_response = self._completion(current_prompt, metadata=metadata)
                response += current_response

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

    def _completion(self, prompt: Prompt|list, response_type: str = None, metadata: Dict = {}) -> Dict:

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
        
        tools = self._get_tools()

        if isinstance(prompt, Prompt):  
            messages = [prompt.as_dict()]
        else:
            messages = prompt
        
        """
        if self._web_search:
            web_search_config = {"search_context_size": self._web_search}
        else:
            web_search_config = None
        """
        
        from litellm import completion
        return completion(model=model, 
                          messages=messages, 
                          response_format=response_type,
                          base_url = url,
                          tools=tools,
                          max_tokens=self._max_tokens,
                          metadata=metadata,
                          #web_search_options=web_search_config
                          )
                          

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

    async def _execute_iterative_async(self, prompt: IterativePrompt, metadata: Dict = None) -> Dict:
        """
        Execute an iterative prompt asynchronously.
        
        Args:
            prompt: The iterative prompt to execute
            metadata: Optional metadata to pass to completion calls
            
        Returns:
            Dictionary containing the final response
        """
        response = ""
        memory = ""
        for i in range(prompt._size):
            if i > 0 and prompt._has_memory:
                if prompt._retain_all:
                    memory += current_response
                else:
                    memory = current_response   
                current_prompt = prompt._format(i, memory)
            else:
                current_prompt = prompt._format(i)
            current_response = await self._completion_async(current_prompt, metadata=metadata)

            response += current_response
        return response

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

