"""
Base classes and mixins for agentic loop implementations.

This module provides the foundational classes and mixins that are used
by all agentic loop implementations. It includes the base agentic loop
class, function calling mixin, and ReAct mixin.
"""

import json
import re
from typing import Any, Dict, List, Optional, Callable

from ...prompts import Prompt
from ...models import Model


class _AgenticLoop:
    """Base class for all agentic loop implementations.
    
    This class provides the core functionality for all AI agents, including
    tool management, message creation, model execution, and streaming support.
    It's designed to be extended by specific classes that implement different
    agentic approaches and reasoning patterns.
    
    The base class handles common operations like:
    - Tool registration and execution
    - Message formatting and conversation management
    - Model interaction (both streaming and non-streaming)
    - Step-based reasoning format parsing
    - Debug output and iteration limits
    
    Attributes
    ----------
    _model : Any
        AI model instance used for execution
    _agentic_prompt : str, Prompt, optional
        Custom prompt for the agent (None for default). Can be a string, 
        a Prompt object, or a path to a .prompt file
    _debug : bool
        Flag to enable debug output
    _max_iter : Optional[int]
        Maximum number of iterations allowed (None for unlimited)
    _stream_callback : Optional[Callable[[str], None]]
        Callback function for handling streaming content
    _tools : Dict[str, Any]
        Dictionary of available tools, mapped by name
    """

    _DEFAULT_PROMPT_PATH = "aidk/agents/prompts/"
    
    def __init__(self, model: Model, agentic_prompt: str=None, debug: bool=False, max_iter: Optional[int]=None, 
                 stream_callback: Optional[Callable[[str], None]]=None, human_feedback: Optional[str]=None) -> None:
        """Initialize the agent with model and configuration.
        
        Parameters
        ----------
        model : Any
            AI model instance to use for execution
        agentic_prompt : str, optional
            Custom prompt for the agent (None to use default)
        debug : bool, default False
            Enable debug output and logging
        max_iter : Optional[int], default None
            Maximum number of iterations allowed (None for unlimited)
        stream_callback : Optional[Callable[[str], None]], default None
            Callback function for handling streaming content chunks
        human_feedback : Optional[str], default None
            Human feedback mode for controlling agent execution. Can be:
            - None: No human feedback required
            - "actions": Pause and request confirmation before executing tool actions
            - "all": Pause after every step for human review
        """
        self._model = model
        self._agentic_prompt = agentic_prompt
        self._debug = debug
        self._max_iter = max_iter
        self._stream_callback = stream_callback
        self._human_feedback = human_feedback
        self._tools = {}
        self._mcp_servers = {}

    def register_tools(self, tools: List[Any]) -> None:
        """Register tools with the agent.
        
        Parameters
        ----------
        tools : List[Any]
            List of tool functions to register. Each tool must have a
            __name__ attribute for identification.
        """
        for tool in tools:
            self._tools[tool.__name__] = tool

    
    def register_mcp_servers(self, mcp_servers: List[Any]) -> None:
        """Register MCP servers with the agent.
        
        Parameters
        ----------
        mcp_servers : List[Any]
            List of MCP server instances to register.
        """
        tools = {}
        self._mcp_servers = {}
        for mcp_server in mcp_servers:
                self._debug_print(f"Connecting to MCP server: {mcp_server.name}")
                # Use context manager for proper connection handling
                self._mcp_servers[mcp_server.name] = mcp_server
                server_tools =  mcp_server.get_tools()
                self._debug_print(f"Retrieved {len(server_tools)} tools from {mcp_server.name}")
                for tool in server_tools:
                    # Create a new tool name with MCP prefix
                    original_name = tool.name
                    tool.name = f"mcp_{mcp_server.name}_{original_name}"
                    tools[tool.name] = tool
                    self._debug_print(f" - {tool.name}")
                
        self._tools.update(tools)

    
    def enable_streaming(self, stream_callback: Optional[Callable[[str], None]] = None) -> None:
        """Enable streaming responses for this agent.
        
        When streaming is enabled, the agent will call the provided callback
        function with each content chunk as it's generated, allowing for
        real-time processing and display of the AI's response.
        
        Parameters
        ----------
        stream_callback : Optional[Callable[[str], None]], default None
            Callback function to handle streaming content chunks.
            If None, uses a default callback that prints content to console.
            The callback receives plain text content strings.
        """
        if stream_callback is None:
            def default_callback(content):
                print(content, end='', flush=True)
            
            self._stream_callback = default_callback
        else:
            self._stream_callback = stream_callback
    
    def disable_streaming(self) -> None:
        """Disable streaming responses for this agent.
        
        After calling this method, the agent will use standard (non-streaming)
        model execution for all subsequent requests.
        """
        self._stream_callback = None
    
    @classmethod
    def create_streaming(cls, model: Any, stream_callback: Optional[Callable[[str], None]] = None, 
                        **kwargs) -> '_AgenticLoop':
        """Create an agent instance with streaming enabled.
        
        This is a convenience class method that creates an agent instance
        with streaming already enabled, avoiding the need to call
        enable_streaming() separately.
        
        Parameters
        ----------
        model : Any
            AI model instance to use
        stream_callback : Optional[Callable[[str], None]], default None
            Callback function for handling streaming content chunks
        **kwargs
            Additional parameters to pass to the constructor
        
        Returns
        -------
        _AgenticLoop
            Agent instance with streaming enabled
        """
        return cls(model, stream_callback=stream_callback, **kwargs)

    def _get_tools(self) -> str:
        """Generate a descriptive string of available tools.
        
        This method creates a formatted string listing all registered tools
        with their signatures and documentation. This string is typically
        included in prompts to help the AI model understand what tools
        are available for use.
        
        Returns
        -------
        str
            Formatted string with descriptions of all available tools,
            one per line with " - " prefix. Returns empty string if no tools.
            
            Example:
             - colab_downloader(colab_url): Downloads a Jupyter notebook from Google Drive and returns its Python code as a string.  
                Args:     colab_url (str): The Google Drive URL of the notebook (should contain 'drive/').
        """
        if not self._tools:
            return ""
        
        tools = []
        for tool_name, tool_func in self._tools.items():
            if tool_name.startswith("mcp_"):
                tools.append(f" - {self._encode_mcp_tool(tool_func)}")
            else:
                tools.append(f" - {self._encode_tool(tool_func)}")
        return "\n".join(tools)

    def _get_base_messages(self, agent_type: str, query: str) -> List[Dict[str, Any]]:
        """Generate base messages for the specific agent type.
        
        This method creates the initial message structure for the agent,
        including the appropriate prompt template and user query. The
        prompt template is selected based on the agent type and includes
        information about available tools.
        
        Parameters
        ----------
        agent_type : str
            Type of agent to determine which prompt template to use
        query : str
            User query to include in the prompt
        
        Returns
        -------
        List[Dict[str, Any]]
            List of base messages for the agent, including the prompt and query
        """
        tools = self._get_tools()
        prompt_data = {"available_tools": tools}
        
        # Handle case where _agentic_prompt is already a Prompt object
        if self._agentic_prompt is not None and hasattr(self._agentic_prompt, 'as_dict'):
            # It's already a Prompt object, just update the data and return
            agentic_prompt = self._agentic_prompt
            # Update prompt data if the prompt supports it
            if hasattr(agentic_prompt, 'prompt_data'):
                agentic_prompt.prompt_data = prompt_data
        else:           
            # Otherwise, determine prompt configuration and create new Prompt
            prompt_config = self._determine_prompt_config(agent_type)
            
            agentic_prompt = Prompt(
                **prompt_config,
                is_system=True,
                prompt_data=prompt_data
            )

        messages = [agentic_prompt.as_dict()]

        if isinstance(query, Prompt):
            messages.append(query.as_dict())
        else:
            messages.append({"role": "user", "content": query})

        return messages
    
    def _determine_prompt_config(self, agent_type: str) -> Dict[str, Any]:
        """Determine the prompt configuration based on agentic_prompt setting.
        
        Parameters
        ----------
        agent_type : str
            Type of agent for fallback prompt selection
            
        Returns
        -------
        Dict[str, Any]
            Configuration dictionary for Prompt initialization
        """
        if self._agentic_prompt is None:
            return {
                "prompt_id": f"{self._DEFAULT_PROMPT_PATH}{agent_type}.prompt"
            }
        
        if isinstance(self._agentic_prompt, str):
            if self._agentic_prompt.endswith(".prompt"):
                return {"prompt_id": self._agentic_prompt}
            else:
                return {"prompt": self._agentic_prompt}
        else:
            # For non-string types (including Prompt objects), treat as prompt content
            return {"prompt": self._agentic_prompt}

    def _debug_print(self, content: str) -> None:
        """Print debug information if debug mode is enabled.
        
        Parameters
        ----------
        content : str
            Content to print in debug mode
        """
        if self._debug:
            print(content)
            print("-------")
    
    def _request_human_feedback(self, step_type: str, content: str, action_data: Optional[Dict] = None) -> bool:
        """Request human feedback for the current step.
        
        This method pauses execution and asks the user for confirmation
        before proceeding with the current step.
        
        Parameters
        ----------
        step_type : str
            Type of step being executed (e.g., "thought", "action", "observation")
        content : str
            Content of the current step
        action_data : Optional[Dict], default None
            Action data if this is an action step
        
        Returns
        -------
        bool
            True if the user approves the step, False if they want to stop
        """
        print(f"\n{'='*60}")
        print(f"HUMAN FEEDBACK REQUIRED - {step_type.upper()} STEP")
        print(f"{'='*60}")
        
        if action_data:
            print(f"Action to execute: {action_data.get('name', 'Unknown')}")
            print(f"Arguments: {action_data.get('arguments', {})}")
        else:
            print(f"Content: {content}")
        
        print(f"\nOptions:")
        print(f"  [y] Yes - Continue with this step")
        print(f"  [n] No - Stop execution")
        print(f"  [s] Skip - Skip this step and continue")
        
        while True:
            try:
                response = input("\nYour choice (y/n/s): ").lower().strip()
                if response in ['y', 'yes']:
                    return True
                elif response in ['n', 'no']:
                    return False
                elif response in ['s', 'skip']:
                    print("Step skipped.")
                    return True  # Continue but skip the current step
                else:
                    print("Please enter 'y' for yes, 'n' for no, or 's' for skip.")
            except KeyboardInterrupt:
                print("\nExecution interrupted by user.")
                return False
    

    def _parse_step_format(self, content: str) -> Optional[Dict[str, Any]]:
        """Parse the step-based format <STEP_TYPE>: <RESULT>.
        
        This method parses agent responses that follow the structured step format
        used by ReAct and similar reasoning patterns. It extracts the step type,
        step number, and content from formatted responses.
        
        Supported step types:
        - Thought: Reasoning and analysis steps
        - Action: Tool calls and actions (must be valid JSON)
        - Observation: Results from tool executions
        - Final answer: Conclusive responses to user queries
        
        Special handling: 
        - If content contains both "Thought" and "Action", the Action will be returned with priority.
        - If content contains both "Thought" and "Final answer", the Final answer will be returned with priority.
        
        Parameters
        ----------
        content : str
            Content to parse in the format <STEP_TYPE>: <RESULT>
        
        Returns
        -------
        Optional[Dict[str, Any]]
            Dictionary containing:
            - step_type: Type of step (thought, action, observation, final answer)
            - step_number: Optional step number if present
            - content: Step content
            - action: Parsed action data (for action steps)
            - final_answer: Final answer content (for final answer steps)
            Returns None if parsing fails
        """
        if not content or not isinstance(content, str):
            return None
        
        content = content.strip()
        
        # Check for multiple step types and prioritize accordingly
        lines = content.split('\n')
        action_start = None
        thought_start = None
        final_answer_start = None
        
        # Find the start lines for Action, Thought, and Final answer
        for i, line in enumerate(lines):
            if re.match(r'^Action(?:\s+\d+)?\s*:', line, re.IGNORECASE):
                action_start = i
            elif re.match(r'^Thought(?:\s+\d+)?\s*:', line, re.IGNORECASE):
                thought_start = i
            elif re.match(r'^Final answer(?:\s+\d+)?\s*:', line, re.IGNORECASE):
                final_answer_start = i
        
        # Priority: Final answer > Action > Thought
        if final_answer_start is not None:
            # Final answer has highest priority
            step_type = "final_answer"
            
            # Extract step number from Final answer line
            final_answer_line = lines[final_answer_start]
            step_number_match = re.search(r'Final answer\s+(\d+)\s*:', final_answer_line, re.IGNORECASE)
            step_number = step_number_match.group(1) if step_number_match else None
            
            # Extract content from Final answer (everything after the colon on the first line)
            final_answer_content = final_answer_line.split(':', 1)[1].strip()
            
            # Add all subsequent lines until we hit another step type or end
            i = final_answer_start + 1
            while i < len(lines):
                line = lines[i].strip()
                if re.match(r'^(Thought|Action|Observation|Final answer)(?:\s+\d+)?\s*:', line, re.IGNORECASE):
                    break
                if line:  # Only add non-empty lines
                    final_answer_content += '\n' + line
                i += 1
            
            step_content = final_answer_content.strip().replace("```json", "").replace("```", "")
            
            result = {
                "step_type": step_type,
                "step_number": step_number,
                "content": step_content,
                "final_answer": step_content
            }
            
            return result
            
        elif action_start is not None and thought_start is not None:
            # Both Thought and Action present, prioritize Action
            step_type = "action"
            
            # Extract step number from Action line
            action_line = lines[action_start]
            step_number_match = re.search(r'Action\s+(\d+)\s*:', action_line, re.IGNORECASE)
            step_number = step_number_match.group(1) if step_number_match else None
            
            # Extract content from Action (everything after the colon on the first line)
            action_content = action_line.split(':', 1)[1].strip()
            
            # Add all subsequent lines until we hit another step type or end
            i = action_start + 1
            while i < len(lines):
                line = lines[i].strip()
                if re.match(r'^(Thought|Action|Observation|Final answer)(?:\s+\d+)?\s*:', line, re.IGNORECASE):
                    break
                if line:  # Only add non-empty lines
                    action_content += '\n' + line
                i += 1
            
            step_content = action_content.strip().replace("```json", "").replace("```", "")
            
            result = {
                "step_type": step_type,
                "step_number": step_number,
                "content": step_content
            }
            
            # Parse action data (must be JSON)
            try:
                action_data = json.loads(step_content)
                result["action"] = action_data
            except json.JSONDecodeError:
                # If not valid JSON, keep content as raw string
                result["action"] = {"raw": step_content}
            
            return result
        
        # Standard single step parsing
        step_pattern = r'^(Thought|Action|Observation|Final answer)(?:\s+(\d+))?\s*:\s*(.*)$'
        match = re.match(step_pattern, content, re.IGNORECASE | re.DOTALL)
        
        if match:
            step_type = match.group(1).lower()
            step_number = match.group(2) if match.group(2) else None
            step_content = match.group(3).strip().replace("```json", "").replace("```", "")
            
            result = {
                "step_type": step_type,
                "step_number": step_number,
                "content": step_content
            }
            
            # Special handling for Action steps (must be JSON)
            if step_type == "action":
                try:
                    action_data = json.loads(step_content)
                    result["action"] = action_data
                except json.JSONDecodeError:
                    # If not valid JSON, keep content as raw string
                    result["action"] = {"raw": step_content}
            
            # Special handling for Final answer steps
            elif step_type == "final_answer":
                result["final_answer"] = step_content
            
            return result
        
        return None

    def _execute_model_step(self, messages: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Execute a model step and return the response.
        
        This method handles both streaming and non-streaming model execution.
        If streaming is enabled and not already in progress, it uses the
        streaming method. Otherwise, it uses standard model execution.
        
        Parameters
        ----------
        messages : List[Dict[str, Any]]
            List of messages to send to the model
        
        Returns
        -------
        Dict[str, Any]
            Model response in standard OpenAI format with usage information
        """
        resp = self._model._execute(messages)
        
        # Extract usage information from the response
        usage_info = None
        if hasattr(resp, 'usage') and resp.usage:
            usage_info = {
                "total_tokens": resp.usage.total_tokens,
                "prompt_tokens": resp.usage.prompt_tokens,
                "completion_tokens": resp.usage.completion_tokens
            }
        
        message = resp["choices"][0]["message"]
        
        # Add usage information to the message
        if usage_info:
            message["usage"] = usage_info
        return message
    
    def _execute_model_step_stream(self, messages: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Execute a model step with streaming and return the complete response.
        
        This method handles asynchronous streaming of model responses, collecting
        all chunks and building the final response. It properly manages async
        resources to avoid memory leaks and task warnings.
        
        Parameters
        ----------
        messages : List[Dict[str, Any]]
            List of messages to send to the model
        
        Returns
        -------
        Dict[str, Any]
            Complete model response in standard OpenAI format
        """

        import asyncio
        from litellm import stream_chunk_builder
        
        # Usa asyncio.run per gestire correttamente il loop
        async def run_streaming():
            chunks = []
            stream = None
            try:
                stream = self._model._execute_stream(messages)
                async for chunk in stream:
                    chunks.append(chunk)
                    content = chunk["choices"][0]["delta"]["content"]

                    if content is not None:
                        self._stream_callback({"type": "text", "delta": content})

            finally:
                if stream is not None:
                    try:
                        await stream.aclose()
                    except Exception:
                        pass
            
            return chunks

        try:
            chunks = asyncio.run(run_streaming())
            resp = stream_chunk_builder(chunks)
            
            # Extract usage information from the response
            usage_info = None
            if hasattr(resp, 'usage') and resp.usage:
                usage_info = {
                    "total_tokens": resp.usage.total_tokens,
                    "prompt_tokens": resp.usage.prompt_tokens,
                    "completion_tokens": resp.usage.completion_tokens
                }
            
            message = resp["choices"][0]["message"]
            
            # Add usage information to the message
            if usage_info:
                message["usage"] = usage_info
                
            return message
        except Exception as e:
            print(f"Streaming error: {e}, falling back to standard execution")
            return self._execute_model_step(messages)
            
    def _create_base_response(self, query: str) -> Dict[str, Any]:
        """Create the base response structure.
        
        Parameters
        ----------
        query : str
            Original user query
        
        Returns
        -------
        Dict[str, Any]
            Dictionary with base response structure:
            - prompt: Original user query
            - iterations: Empty list for iterations
            - usage: Dictionary for tracking token usage
        """
        return {
            "prompt": query, 
            "iterations": [],
            "usage": {
                "total_tokens": 0,
                "prompt_tokens": 0,
                "completion_tokens": 0
            }
        }

    def _update_usage(self, response: Dict[str, Any], iteration_usage: Dict[str, int]) -> None:
        """Update the total usage in the response.
        
        Parameters
        ----------
        response : Dict[str, Any]
            Response dictionary to update
        iteration_usage : Dict[str, int]
            Usage information from current iteration
        """
        if hasattr(iteration_usage, "usage"):
            response["usage"]["total_tokens"] += iteration_usage["usage"]["total_tokens"]
            response["usage"]["prompt_tokens"] += iteration_usage["usage"]["prompt_tokens"]
            response["usage"]["completion_tokens"] += iteration_usage["usage"]["completion_tokens"]

    def _handle_final_answer(self, iteration: Dict[str, Any], response: Dict[str, Any]) -> bool:
        """Handle a final answer, returns True if this is the end.
        
        This method processes iterations that contain final answers and
        updates the response structure accordingly. It supports both
        the old format (final_answer key) and new step format.
        
        Parameters
        ----------
        iteration : Dict[str, Any]
            Current iteration potentially containing a final answer
        response : Dict[str, Any]
            Response dictionary to update
        
        Returns
        -------
        bool
            True if a final answer was found and processed, False otherwise
        
        Notes
        -----
        This method modifies the response object directly.
        """
        if "final_answer" in iteration:
            response["iterations"].append(iteration)
            response["response"] = iteration["final_answer"]
            return True
        elif iteration.get("step_type") == "final answer":
            response["iterations"].append(iteration)
            response["response"] = iteration["content"]
            return True
        return False

    def _pngimagefile_to_base64_data_uri(self, img) -> str:
        """
        Converte un oggetto PIL.PngImageFile in una stringa base64 data URI.
        
        Args:
            img (Image.Image): immagine PIL (PngImageFile o convertita in PNG)
        
        Returns:
            str: stringa "data:image/png;base64,<...>"
        """
        import io
        import base64
        buffer = io.BytesIO()
        img.save(buffer, format="PNG")
        buffer.seek(0)

        # codifica in base64
        encoded = base64.b64encode(buffer.read()).decode("utf-8")

        return f"data:image/png;base64,{encoded}"

    def _handle_tool_action(self, iteration: Dict[str, Any], response: Dict[str, Any], messages: List[Dict[str, Any]]) -> None:
        """Handle a tool action execution.
        
        This method processes iterations that contain tool actions, executes
        the corresponding tools, and updates the conversation with the results.
        It supports both old and new step-based formats.
        
        Parameters
        ----------
        iteration : Dict[str, Any]
            Current iteration containing the tool action
        response : Dict[str, Any]
            Response dictionary to update
        messages : List[Dict[str, Any]]
            Message list to update with observation
        
        Notes
        -----
        This method modifies the response and messages objects directly.
        """
        # Check if human feedback is required for actions
        if self._human_feedback == "actions" or self._human_feedback == "all":
            action_data = iteration.get("action", {})
            if not self._request_human_feedback("action", iteration.get("content", ""), action_data):
                print("Execution stopped by user.")
                return

        if "action" in iteration and iteration["action"].get("name"):
            tool_call = iteration["action"]
            
            if self._stream_callback is not None:
                self._stream_callback({"type": "tool_call", "tool_call": tool_call})

            if tool_call.get("name").startswith("mcp_"):
                tool_result = self._call_mcp_tool(tool_call)
            else:
                tool_result = self._call_tool(tool_call)

            iteration["observation"] = tool_result
            response["iterations"].append(iteration)

            if isinstance(tool_result, dict):
                if "image" in tool_result:
                    img = self._pngimagefile_to_base64_data_uri(tool_result["image"])
                    msg = [{"type": "image_url", "image_url": {"url": img}}, 
                            {"type": "text", "text": "Ecco l'immagine"}]
                    messages.append({"role": "user", "content": msg})
                else:
                    msg = json.dumps({"observation": tool_result})
                    messages.append({"role": "user", "content": msg})
        elif iteration.get("step_type") == "action" and "action" in iteration:
            tool_call = iteration["action"]
            tool_result = self._call_tool(tool_call)
            iteration["observation"] = tool_result
            response["iterations"].append(iteration)
            
            # Add observation in the new system format
            observation_msg = f"Observation {iteration.get('step_number', '')}: {tool_result}".strip()
            messages.append({"role": "user", "content": observation_msg})

    def _handle_default(self, iteration: Dict[str, Any], response: Dict[str, Any], messages: List[Dict[str, Any]]) -> None:
        """Handle default case for unhandled iterations.
        
        This method processes iterations that don't match specific handlers,
        adding them to the response and updating the conversation flow.
        It supports both step-based and JSON formats.
        
        Parameters
        ----------
        iteration : Dict[str, Any]
            Current iteration to handle
        response : Dict[str, Any]
            Response dictionary to update
        messages : List[Dict[str, Any]]
            Message list to update
        
        Notes
        -----
        This method modifies the response and messages objects directly.
        """
        response["iterations"].append(iteration)
        
        # For new format, add content as user message
        if iteration.get("step_type") in ["thought", "observation"]:
            step_type = iteration["step_type"].capitalize()
            step_number = iteration.get("step_number", "")
            content = iteration["content"]
            message_content = f"{step_type} {step_number}: {content}".strip()
            messages.append({"role": "user", "content": message_content})
        else:
            # Fallback to JSON format for compatibility
            messages.append({"role": "user", "content": json.dumps(iteration)})

    def start(self, query: str) -> Dict[str, Any]:
        """Abstract method to start the agentic loop.
        
        This method must be implemented by subclasses to define the specific
        agentic behavior and reasoning pattern.
        
        Parameters
        ----------
        query : str
            User query to process
        
        Returns
        -------
        Dict[str, Any]
            Agent response containing iterations and final result
        
        Raises
        ------
        NotImplementedError
            This method must be implemented by subclasses
        """
        raise NotImplementedError
