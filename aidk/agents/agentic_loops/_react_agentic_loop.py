"""
ReAct-based Agent Implementations.

This module provides ReAct-style agent implementations that use structured
JSON responses for reasoning and tool execution.
"""

from typing import Any, Dict, Optional, Callable
import inspect
from ._agentic_loop import _AgenticLoop
from ._function_calling_agentic_loop import _FunctionCallingMixin


class _ReactMixin:
    """Mixin for handling ReAct-style tool calls with JSON format.
    
    This mixin provides methods for managing tool calls in structured JSON format,
    typical of ReAct-style approaches for AI agents. It handles the encoding
    of tool functions and execution of tool calls based on parsed JSON responses.
    
    The ReAct (Reasoning and Acting) pattern allows agents to reason about
    problems step-by-step and take actions using tools when needed.
    """
    
    def _encode_tool(self, func: Any) -> str:
        """Encode a function into a descriptive string format.
        
        This method creates a human-readable description of a tool function
        that can be included in prompts to help the AI model understand
        what tools are available and how to use them.
        
        Parameters
        ----------
        func : Any
            Function to encode. Must have attributes:
            - __name__: Function name
            - __doc__: Function documentation
        
        Returns
        -------
        str
            Descriptive string in the format:
            "function_name(signature): documentation"
            Newlines are replaced with spaces for single-line format.
        """
        sig = inspect.signature(func)
        doc = inspect.getdoc(func)
        encoded = func.__name__ + str(sig) + ": " + doc
        encoded = encoded.replace("\n", " ")
        return encoded
    
    def _encode_mcp_tool(self, tool: dict) -> str:
        """Converte uno schema tipo JSON Schema in sezione Args Google style."""
        schema = tool.inputSchema
        if schema.get("type") != "object":
            raise ValueError("Lo schema radice deve avere type=object")

        args = []
        args_doc = ["Args:"]
        properties = schema.get("properties", {})
        required = set(schema.get("required", []))

        for name, spec in properties.items():
            args.append(name)
            typ = spec.get("type", "Any")
            desc = spec.get("description", "").strip().replace("\n", " ")
            title = spec.get("title", "")
            default_info = "" if name in required else " Defaults to None."

            arg_line = f"    {name} ({typ}): {desc}{default_info}"
            if title:
                arg_line = f"    {name} ({typ}): {title}. {desc}{default_info}"
            args_doc.append(arg_line)

        encoded = tool.name + "(" + ", ".join(args) + "): "
        encoded += tool.description
        encoded += ". ".join(args_doc)
        return encoded

    
    def _call_tool(self, tool_call: Dict[str, Any]) -> Any:
        """Execute a tool call in ReAct format.
        
        This method executes a tool call based on a structured dictionary
        containing the tool name and arguments. It's designed to work with
        the ReAct pattern where tools are called through JSON-formatted
        action specifications.
        
        Parameters
        ----------
        tool_call : Dict[str, Any]
            Tool call specification containing:
            - name: Name of the tool to call
            - arguments: Dictionary of tool arguments
        
        Returns
        -------
        Any
            Result of the tool execution
        
        Raises
        ------
        KeyError
            If the tool is not registered with the agent
        TypeError
            If the tool arguments don't match the function signature
        Exception
            If the tool execution fails
        """
        tool = self._tools[tool_call["name"]]
        kwargs = list(tool_call["arguments"].values())
        return tool(*kwargs)

    def _call_mcp_tool(self, tool_call: Dict[str, Any]) -> Any:
        """Execute a MCP tool call.
        
        This method executes a MCP tool call based on a structured dictionary
        containing the tool name and arguments.
        """
        # Extract server name from tool name (format: mcp_servername_toolname)
        tool_name_parts = tool_call["name"].split("_", 2)
        if len(tool_name_parts) < 3:
            raise ValueError(f"Invalid MCP tool name format: {tool_call['name']}")
        
        server_name = tool_name_parts[1]
        tool_name = tool_name_parts[2]
        if server_name not in self._mcp_servers:
            raise ValueError(f"MCP server '{server_name}' not found")

        return {
            "tool_call_id": tool_name,
            "role": "tool",
            "name": tool_name,
            "content": str(self._mcp_servers[server_name].call_tool(tool_name, tool_call["arguments"]))
        }


class _BaseReactLoop(_AgenticLoop, _ReactMixin):
    """Base class for all ReAct-style agents.
    
    This class implements the standard loop for agents that use a ReAct-style
    approach, where the model produces structured JSON responses that are
    parsed and handled iteratively. The ReAct pattern combines reasoning
    and acting in a step-by-step manner.
    
    The base loop handles:
    - Step-based reasoning format parsing
    - Tool action execution and observation
    - Final answer detection
    - Custom iteration handlers
    - Error handling and fallbacks
    
    Attributes
    ----------
    _max_iter : Optional[int]
        Maximum number of iterations allowed
    """
    
    def _run_react_loop(self, query: str, agent_type: str, 
                        custom_handlers: Optional[Dict[str, Callable]] = None) -> Dict[str, Any]:
        """Execute the standard ReAct loop.
        
        This method implements the core ReAct reasoning pattern where the
        agent alternates between thinking, acting, and observing until it
        reaches a final answer or hits iteration limits.
        
        Parameters
        ----------
        query : str
            User query to process
        agent_type : str
            Type of agent to determine which prompt template to use
        custom_handlers : Optional[Dict[str, Callable]], optional
            Dictionary of custom handlers for specific iteration types.
            Keys are field names in the iteration, values are functions
            that handle those iterations.
        
        Returns
        -------
        Dict[str, Any]
            Agent response containing:
            - prompt: Original user query
            - iterations: List of processed iterations
            - response: Final response (if present)
        
        Notes
        -----
        This method automatically handles:
        - Final answers (final_answer)
        - Tool actions (action)
        - Custom cases via custom_handlers
        - Default cases for unhandled iterations
        - JSON error handling
        """
        messages = self._get_base_messages(agent_type, query)

        current_iter = 0
        response = self._create_base_response(query)
        
        # Handler personalizzati per casi speciali
        custom_handlers = custom_handlers or {}

        while True:
            if self._max_iter is not None and current_iter >= self._max_iter:
                break
            
            if self._stream_callback is None:
                resp = self._execute_model_step(messages)
            else:
                resp = self._execute_model_step_stream(messages)

            messages.append(resp)
            content = resp["content"]
        
            self._debug_print(content)

            # Update usage for this iteration
            self._update_usage(response, resp)

            if content is not None:
                iteration = self._parse_step_format(content)
                if iteration:
                    # Add usage information to the iteration
                    if "usage" in resp:
                        iteration["usage"] = resp["usage"]
                    
                    # Check if human feedback is required for all steps
                    if self._human_feedback == "all":
                        if not self._request_human_feedback(iteration.get("step_type", "unknown"), 
                                                          iteration.get("content", ""), 
                                                          iteration.get("action")):
                            print("Execution stopped by user.")
                            break
                    
                    # Gestione risposta finale
                    if self._handle_final_answer(iteration, response):
                        break
                    
                    # Gestione azioni di tool
                    if iteration["step_type"]=="action":
                        self._handle_tool_action(iteration, response, messages)
                        continue
                    
                    # Gestione casi personalizzati
                    handled = False
                    for key, handler in custom_handlers.items():
                        if iteration["step_type"] == key:
                            handler(iteration, response, messages)
                            handled = True
                            break
                    
                    if not handled:
                        self._handle_default(iteration, response, messages)
                else:
                    # Se non riesce a parsare, aggiungi come messaggio utente
                    messages.append({"role": "user", "content": content})

            current_iter += 1

        return response


class ReactAgenticLoop(_BaseReactLoop):
    """Standard ReAct agent.
    
    This agent implements the standard ReAct pattern, where the model
    produces JSON responses that are parsed and handled iteratively.
    The ReAct pattern combines reasoning and acting in a structured way.
    """
    
    def start(self, query: str) -> Dict[str, Any]:
        """Start the ReAct agentic loop.
        
        Parameters
        ----------
        query : str
            User query to process
        
        Returns
        -------
        Dict[str, Any]
            Agent response processed through the ReAct loop
        """
        return self._run_react_loop(query, "react")


class ReactWithFCAgenticLoop(_AgenticLoop, _FunctionCallingMixin):
    """Agent that combines ReAct and Function Calling.
    
    This agent combines the ReAct approach with OpenAI's native function
    calling, allowing for hybrid tool call management. This provides
    the flexibility of ReAct reasoning with the reliability of function calling.
    """
    
    def start(self, query: str) -> Dict[str, Any]:
        """Start the hybrid ReAct + Function Calling agentic loop.
        
        Parameters
        ----------
        query : str
            User query to process
        
        Returns
        -------
        Dict[str, Any]
            Agent response (to be implemented)
        
        Notes
        -----
        TODO: Implement combination of ReAct and Function Calling
        """
        # TODO: Implement combination of ReAct and Function Calling
        pass
