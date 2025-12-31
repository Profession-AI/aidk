from ..models import Model
from ..mcp.mcp_server import McpServer
from .agentic_loops._function_calling_agentic_loop import FunctionCallingAgenticLoop
from .agentic_loops._react_agentic_loop import ReactAgenticLoop, ReactWithFCAgenticLoop
from .agentic_loops._other_agentic_loops import (
    PlanAndExecuteAgenticLoop,
    ProgrammaticAgenticLoop,
    ReflexionAgenticLoop,
    SelfAskAgenticLoop,
    SelfAskWithSearchLoop
)
from .agentic_loops._agentic_loop import _AgenticLoop

from ..prompts import Prompt


class Agent:
    """
    AI Agent that implements different reasoning paradigms for autonomous task execution.
    
    The Agent class provides a unified interface for creating and using AI agents
    with various reasoning paradigms. It acts as a high-level wrapper around
    specific agentic loop implementations, offering a consistent API regardless
    of the underlying reasoning pattern.
    
    The agent can be configured with different reasoning approaches, tools,
    and execution parameters to handle complex tasks autonomously or with
    human oversight.
    
    Parameters
    ----------
    model : Model
        AI model instance to use for agent execution
    tools : list, optional
        List of available tools for the agent. Each tool should be a
        callable function. Default is None.
    paradigm : str or _AgenticLoop, optional
        Reasoning paradigm to use. Default is "function_calling".
    agent_prompt : str, optional
        Custom prompt for the agent. If None, uses the default prompt
        for the chosen paradigm. Default is None.
    name : str, optional
        Name identifier for the agent. Default is empty string.
    mcp_servers : list, optional
        List of MCP (Model Context Protocol) servers to register.
        Default is None.
    debug : bool, optional
        Flag to enable debug output during execution. Default is False.
    max_iter : int, optional
        Maximum number of iterations allowed for the agent.
        If None, there are no limits. Default is None.
    native_web_search : str, optional
        Native web search capability level. Must be one of:
        "low", "medium", or "high". Default is None.
    human_feedback : str, optional
        Human feedback mode for controlling agent execution. Can be:
        - None: No human feedback required (default)
        - "actions": Pause and request confirmation before executing tool actions
        - "all": Pause after every step for human review
        Default is None.
    
    Attributes
    ----------
    name : str
        Name identifier for the agent
    _model : Model
        The AI model instance used for execution
    _loop : _AgenticLoop
        The agentic loop implementation for the chosen paradigm
        
    Supported Paradigms
    -------------------
    - **function_calling**: Native OpenAI function calling with tool integration
    - **react**: Reasoning and Acting pattern with iterative problem solving
    - **react_with_function_calling**: Combines ReAct reasoning with function calling
    - **plan-and-execute**: Two-phase approach: planning then execution
    - **programmatic**: Code generation and execution for computational tasks
    - **reflexion**: Self-reflective reasoning with error correction
    - **self_ask**: Self-questioning approach for complex reasoning
    - **self_ask_with_search**: Self-ask with web search capabilities
    
    Examples
    --------
    Basic function calling agent:
    >>> from aidk.models import Model
    >>> from aidk.agents import Agent
    >>> 
    >>> model = Model(provider="openai", model="gpt-4")
    >>> agent = Agent(model=model, paradigm="function_calling")
    >>> response = agent.run("What's the weather like today?")
    
    ReAct agent with custom tools:
    >>> def calculator(expression: str) -> str:
    ...     return str(eval(expression))
    >>> 
    >>> agent = Agent(
    ...     model=model,
    ...     paradigm="react",
    ...     tools=[calculator],
    ...     name="MathAgent"
    ... )
    >>> response = agent.run("Calculate 15 * 23 + 7")
    
    Agent with human feedback:
    >>> agent = Agent(
    ...     model=model,
    ...     paradigm="plan-and-execute",
    ...     human_feedback="actions",
    ...     debug=True
    ... )
    >>> response = agent.run("Plan and execute a data analysis task")
    
    Agent with MCP servers:
    >>> mcp_server = McpServer("coingecko", "http", server_url="https://mcp.api.coingecko.com/sse")
    >>> agent = Agent(
    ...     model=model,
    ...     paradigm="function_calling",
    ...     mcp_servers=[mcp_server]
    ... )
    >>> response = agent.run("What's the price of Bitcoin?")

    Custom paradigm agent:
    >>> class CustomLoop(_AgenticLoop):
    ...     def start(self, prompt):
    ...         # Custom implementation
    ...         return {"response": "Custom result"}
    >>> 
    >>> custom_loop = CustomLoop()
    >>> agent = Agent(model=model, paradigm=custom_loop)
    >>> response = agent.run("Test custom paradigm")
    """
    
    def __init__(self, model: Model, tools=None, paradigm="function_calling", 
                 agent_prompt=None, name="", mcp_servers=None, debug=False, max_iter=None, native_web_search=None, 
                 human_feedback=None):
        """
        Initialize the agent with the specified model and configuration.
        
        Sets up an AI agent with the chosen reasoning paradigm, registers
        any provided tools, and configures execution parameters.
        
        Raises
        ------
        ValueError
            If the specified paradigm is not supported or invalid
        TypeError
            If a custom object is passed that doesn't derive from _AgenticLoop
        """
        self._model = model
        self.name = name

        if native_web_search is not None and native_web_search not in ["low", "medium", "high"]:
            raise ValueError("native_web_search must be 'low', 'medium' or 'high'")
        
        if human_feedback is not None and human_feedback not in ["actions", "all"]:
            raise ValueError("human_feedback must be None, 'actions', or 'all'")
        
        self._model._web_search = native_web_search
        self._human_feedback = human_feedback

        # Handle custom paradigm
        if isinstance(paradigm, _AgenticLoop):
            # Verify that the custom object is valid
            if not hasattr(paradigm, 'start') or not callable(paradigm.start):
                raise TypeError("Custom paradigm must have a callable 'start' method")
            self._loop = paradigm
        else:
            # Predefined paradigms
            loop_kwargs = self._model, agent_prompt, debug, max_iter, None, human_feedback
            
            if paradigm == "function_calling":
                self._loop = FunctionCallingAgenticLoop(*loop_kwargs)
            elif paradigm == "react":
                self._loop = ReactAgenticLoop(*loop_kwargs)
            elif paradigm == "react_with_function_calling":
                self._loop = ReactWithFCAgenticLoop(*loop_kwargs)
            elif paradigm == "plan-and-execute":
                self._loop = PlanAndExecuteAgenticLoop(*loop_kwargs)
            elif paradigm == "programmatic":
                self._loop = ProgrammaticAgenticLoop(*loop_kwargs)
            elif paradigm == "reflexion":
                self._loop = ReflexionAgenticLoop(*loop_kwargs)
            elif paradigm == "self_ask":
                self._loop = SelfAskAgenticLoop(*loop_kwargs)
            elif paradigm == "self_ask_with_search":
                self._loop = SelfAskWithSearchLoop(*loop_kwargs)
            else:
                raise ValueError(f"Paradigm '{paradigm}' not supported. "
                               f"Available paradigms: function_calling, react, "
                               f"react_with_function_calling, plan-and-execute, "
                               f"programmatic, reflexion, self_ask, self_ask_with_search, "
                               f"or a custom object derived from _AgenticLoop")
        
        if tools is not None:
            self._loop.register_tools(tools)
        
        if mcp_servers is not None:
            self._loop.register_mcp_servers(mcp_servers)
        
    def run(self, prompt: str | Prompt):
        """
        Execute the agent with the specified prompt.
        
        Processes a user prompt through the agent's reasoning paradigm,
        returning a structured response that includes the reasoning process
        and final answer.
        
        Parameters
        ----------
        prompt : str or Prompt
            User prompt or query to process. Can be a string or a Prompt object.
        
        Returns
        -------
        Dict[str, Any]
            Agent response containing:
            - prompt: Original user query
            - iterations: List of processed reasoning iterations
            - response: Final response (if available)
            - metadata: Additional execution metadata (paradigm-specific)
        
        Examples
        --------
        >>> agent = Agent(model=model, paradigm="function_calling")
        >>> response = agent.run("What's the weather in New York?")
        >>> print(response['response'])
        
        >>> from aidk.prompts import Prompt
        >>> prompt = Prompt("Analyze this data: [1, 2, 3, 4, 5]")
        >>> response = agent.run(prompt)
        >>> print(response['iterations'])
        
        Notes
        -----
        The response structure varies by paradigm:
        - **Function calling**: Includes tool call details and function results
        - **ReAct**: Includes thought-action-observation cycles
        - **Plan-and-execute**: Includes planning and execution phases
        - **Programmatic**: Includes code generation and execution results
        - **Reflexion**: Includes self-reflection and error correction steps
        - **Self-ask**: Includes question-answer reasoning chains
        - **Self-ask with search**: Includes web search results and reasoning
        - **Custom paradigms**: May include paradigm-specific information
        """
        
        return self._loop.start(prompt)
    
    def enable_streaming(self, stream_callback=None):
        """
        Enable streaming responses for this agent.
        
        When streaming is enabled, the agent will call the provided callback
        function with each content chunk as it's generated, allowing for
        real-time processing and display of the AI's response.
        
        Parameters
        ----------
        stream_callback : callable, optional
            Callback function to handle streaming content chunks.
            If None, uses a default callback that prints content to console.
            The callback receives plain text content strings.
            
            Callback signature: callback(content: str) -> None
            where content is the streaming text chunk.
        
        Raises
        ------
        AttributeError
            If the current paradigm doesn't support streaming
            
        Examples
        --------
        >>> def my_callback(content):
        ...     print(f"Streaming: {content}")
        >>> 
        >>> agent = Agent(model=model, paradigm="function_calling")
        >>> agent.enable_streaming(my_callback)
        >>> response = agent.run("Tell me a story")
        
        Notes
        -----
        Not all paradigms support streaming. Check the specific paradigm
        implementation to ensure streaming is available.
        """
        if hasattr(self._loop, 'enable_streaming'):
            self._loop.enable_streaming(stream_callback)
        else:
            raise AttributeError(f"Paradigm '{self._loop.__class__.__name__}' doesn't support streaming")
    
    def disable_streaming(self):
        """
        Disable streaming responses for this agent.
        
        After calling this method, the agent will use standard (non-streaming)
        model execution for all subsequent requests.
        
        Raises
        ------
        AttributeError
            If the current paradigm doesn't support streaming
            
        Examples
        --------
        >>> agent = Agent(model=model, paradigm="function_calling")
        >>> agent.enable_streaming()
        >>> # ... use streaming ...
        >>> agent.disable_streaming()
        >>> # Now using standard execution
        
        Notes
        -----
        Not all paradigms support streaming. Check the specific paradigm
        implementation to ensure streaming is available.
        """
        if hasattr(self._loop, 'disable_streaming'):
            self._loop.disable_streaming()
        else:
            raise AttributeError(f"Paradigm '{self._loop.__class__.__name__}' doesn't support streaming")