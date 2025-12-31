"""
AIDK Agents Package

This package provides a comprehensive set of agentic loop implementations
that support real-time streaming responses during the entire generation phase.
The agents are designed to work with various AI paradigms and can process
user queries through iterative reasoning, tool usage, and structured output generation.

Key Features:
- Real-time streaming of AI responses
- Multiple agent paradigms (React, Function Calling, Plan-and-Execute, etc.)
- Step-based reasoning format for transparent decision making
- Tool integration and execution
- Configurable iteration limits and debugging
- Asynchronous streaming with proper resource management

Agent Paradigms:
The agents use a structured step-based format for reasoning:
- Thought N: <reasoning process and analysis>
- Action N: {"name": "tool_name", "arguments": {...}}
- Observation N: <tool execution result>
- Final answer: <conclusive response to user query>

Streaming Architecture:
Streaming allows real-time processing of AI responses as they are generated, enabling:
- Immediate feedback to users
- Progress monitoring during long operations
- Better user experience with responsive interfaces
- Debugging and monitoring of agent reasoning

Usage Examples:

1. Basic React Agent with Streaming:
    ```python
    from aidk.agents import Agent
    
    agent = Agent(model, paradigm="react")
    agent.enable_streaming()  # Uses default console output
    result = agent.run("What is the capital of France?")
    ```

2. Custom Streaming Handler:
    ```python
    from aidk.agents import Agent
    import json
    
    def custom_stream_handler(content):
        # Process streaming content in real-time
        print(f"Streaming: {content}", end='', flush=True)
    
    agent = Agent(model, paradigm="react")
    agent.enable_streaming(custom_stream_handler)
    result = agent.run("Your query here")
    ```

3. Function Calling Agent:
    ```python
    from aidk.agents import Agent
    from aidk.tools import search_web
    
    agent = Agent(model, paradigm="function_calling")
    agent.register_tools([search_web])
    agent.enable_streaming()
    result = agent.run("Search for recent AI news")
    ```

4. Plan and Execute Agent:
    ```python
    agent = Agent(model, paradigm="plan-and-execute")
    agent.enable_streaming()
    result = agent.run("Create a detailed project plan")
    ```

Available Agent Types:
- FunctionCallingAgenticLoop: Native OpenAI function calling
- ReactAgenticLoop: ReAct reasoning pattern
- ReactWithFCAgenticLoop: Hybrid ReAct + Function Calling
- ProgrammaticAgenticLoop: Code generation and execution
- PlanAndExecuteAgenticLoop: Planning then execution pattern
- ReflexionAgenticLoop: Self-reflection and improvement
- SelfAskAgenticLoop: Self-questioning approach
- SelfAskWithSearchLoop: Self-ask with web search capabilities

Streaming Callback Format:
The streaming callback receives content as plain text strings. For advanced use cases,
you can access the raw streaming data through the model's streaming methods.

Error Handling:
- Automatic fallback to non-streaming mode on errors
- Proper cleanup of async resources
- Configurable debug output
- Iteration limits to prevent infinite loops

Thread Safety:
This implementation is designed for single-threaded use. For concurrent access,
create separate agent instances for each thread or process.
"""

from .agentic_loops._agentic_loop import _AgenticLoop
from .agentic_loops._function_calling_agentic_loop import FunctionCallingAgenticLoop
from .agentic_loops._react_agentic_loop import ReactAgenticLoop, ReactWithFCAgenticLoop, _BaseReactLoop
from .agentic_loops._other_agentic_loops import (
    ProgrammaticAgenticLoop,
    PlanAndExecuteAgenticLoop,
    ReflexionAgenticLoop,
    SelfAskAgenticLoop,
    SelfAskWithSearchLoop
)

# Main Agent class that provides a unified interface
from .agent import Agent

__all__ = [
    # Base classes
    '_AgenticLoop',
    '_BaseReactLoop',
    
    # Agent implementations
    'FunctionCallingAgenticLoop',
    'ReactAgenticLoop',
    'ReactWithFCAgenticLoop',
    'ProgrammaticAgenticLoop',
    'PlanAndExecuteAgenticLoop',
    'ReflexionAgenticLoop',
    'SelfAskAgenticLoop',
    'SelfAskWithSearchLoop',
    
    # Main interface
    'Agent'
]