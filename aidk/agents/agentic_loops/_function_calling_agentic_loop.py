"""
Function Calling Agent Implementation.

This module provides the FunctionCallingAgenticLoop class that implements
OpenAI's native function calling system for AI agents.
"""

import json
from typing import Any, Dict, List
from ...prompts import Prompt
from ._agentic_loop import _AgenticLoop


class _FunctionCallingMixin:
    """Mixin for handling OpenAI function calling tool execution.
    
    This mixin provides methods for managing tool calls in OpenAI's native
    function calling format, converting tool responses into standardized messages
    that can be used in the conversation flow.
    
    The mixin handles the execution of tools registered with the agent and
    formats their responses according to OpenAI's message format specification.
    """
    
    def _call_tool(self, tool_call: Any) -> Dict[str, str]:
        """Execute a tool call and return the formatted response.
        
        This method takes a tool call object from the AI model's response,
        executes the corresponding tool function, and returns a properly
        formatted message that can be added to the conversation history.
        
        Parameters
        ----------
        tool_call : Any
            Tool call object containing function execution details.
            Must have attributes:
            - function.name: Name of the function to call
            - function.arguments: JSON string of function arguments
            - id: Unique identifier for this tool call
        
        Returns
        -------
        Dict[str, str]
            Formatted tool response message containing:
            - tool_call_id: ID of the original tool call
            - role: Message role (always "tool")
            - name: Name of the executed function
            - content: Tool execution result as string
        
        Raises
        ------
        KeyError
            If the tool function is not registered with the agent
        json.JSONDecodeError
            If the function arguments are not valid JSON
        Exception
            If the tool function execution fails
        """

        function_name = tool_call.function.name
        function_to_call = self._tools[function_name]
        function_args = json.loads(tool_call.function.arguments)                
        function_response = str(function_to_call(**function_args))
        return {
            "tool_call_id": tool_call.id,
            "role": "tool",
            "name": function_name,
            "content": function_response,
        }

    def _call_mcp_tool(self, tool_call: Dict[str, Any]) -> Any:
        """Execute a MCP tool call."""
        tool_name = tool_call.function.name
        tool_arguments = json.loads(tool_call.function.arguments) if tool_call.function.arguments else {}
        tool_call_id = tool_call.id
        
        # Get server name from tool name (format: mcp_servername_toolname)
        server_name = tool_name.split("_", 2)[1]
        if server_name not in self._mcp_servers:
            raise ValueError(f"MCP server '{server_name}' not found")
        # Simple approach: just return an error for now since MCP is not working
        return {
            "tool_call_id": tool_call_id,
            "role": "tool",
            "name": tool_name,
            "content": str(self._mcp_servers[server_name].call_tool(tool_name, tool_arguments))
        }


class FunctionCallingAgenticLoop(_AgenticLoop, _FunctionCallingMixin):
    """Agent that uses OpenAI's native function calling.
    
    This agent implements a loop that leverages OpenAI's native function calling
    system, allowing the model to directly call available functions without
    manual response parsing. This approach is more reliable and efficient
    than text-based tool calling.
    
    The agent automatically handles:
    - Function call detection and execution
    - Tool result integration into conversation
    - Iteration limits to prevent infinite loops
    - Streaming support for real-time responses
    
    Attributes
    ----------
    _model : Any
        OpenAI model with function calling support
    _tools : Dict[str, Any]
        Available tools for the agent
    """
    
    def _get_base_messages(self, query: str) -> List[Dict[str, Any]]:
        """Generate base messages for the specific agent type.
        
        This method creates the initial message structure for the agent,
        including the appropriate prompt template and user query. The
        prompt template is selected based on the agent type and includes
        information about available tools.
        
        Parameters
        ----------
        query : str
            User query to include in the prompt
        
        Returns
        -------
        List[Dict[str, Any]]
            List of base messages for the agent, including the prompt and query
        """
        messages = []
        if self._agentic_prompt is not None:
            if isinstance(self._agentic_prompt, str):
                if self._agentic_prompt.endswith(".prompt"):
                    agentic_prompt = Prompt(prompt_id=self._agentic_prompt, is_system=True)
                else:
                    agentic_prompt = Prompt(prompt=self._agentic_prompt, is_system=True)
            else:
                agentic_prompt = self._agentic_prompt
            messages.append(agentic_prompt.as_dict())
                    
        if isinstance(query, Prompt):
            messages.append(query.as_dict())
        else:
            messages.append({"role": "user", "content": query})

        return messages

    def start(self, query: str) -> Dict[str, Any]:
        """Start the agentic loop using function calling.
        
        This method processes user queries through OpenAI's function calling
        system, automatically executing tools when the model determines
        they are needed.
        
        Parameters
        ----------
        query : str
            User query to process
        
        Returns
        -------
        Dict[str, Any]
            Agent response containing:
            - prompt: Original user query
            - iterations: List of tool calls executed
            - response: Final model response
        """
        # Add all tools to the model
        self._model._add_tools(list(self._tools.values()))
        messages = self._get_base_messages(query)
        response = self._create_base_response(query)
        current_iter = 0
        max_iterations = self._max_iter if self._max_iter is not None else 10  # Limite di sicurezza
        
        while current_iter < max_iterations:
            
            if self._stream_callback is None:
                resp = self._execute_model_step(messages)
            else:
                resp = self._execute_model_step_stream(messages)

            messages.append(resp)
            content = resp["content"]
            self._debug_print(content)
            
            self._update_usage(response, resp)
            
            if resp.get("tool_calls"):
                for tool_call in resp["tool_calls"]:
                    # Check if human feedback is required for actions
                    if self._human_feedback == "actions" or self._human_feedback == "all":
                        action_data = {
                            "name": tool_call.function.name,
                            "arguments": json.loads(tool_call.function.arguments) if tool_call.function.arguments else {}
                        }
                        if not self._request_human_feedback("action", f"Function call: {tool_call.function.name}", action_data):
                            print("Execution stopped by user.")
                            return response
                    
                    if tool_call.function.name.startswith("mcp_"):
                        tool_result = self._call_mcp_tool(tool_call)
                    else:
                        tool_result = self._call_tool(tool_call)
                    iteration_data = {
                        "name": tool_call.function.name,
                        "arguments": tool_call.function.arguments,
                        "result": tool_result["content"]
                    }
                    
                    # Add usage information to the iteration
                    if "usage" in resp:
                        iteration_data["usage"] = resp["usage"]
                    
                    response["iterations"].append(iteration_data)
                    messages.append(tool_result)
            else:
                # Check if human feedback is required for all steps (non-action responses)
                if self._human_feedback == "all":
                    if not self._request_human_feedback("response", content):
                        print("Execution stopped by user.")
                        return response
                
                response["response"] = content
                break
            
            current_iter += 1
        
        # Se arriviamo qui, abbiamo raggiunto il limite di iterazioni
        if self._debug:
            print(f"Raggiunto limite di iterazioni ({max_iterations})")
        
        return response
