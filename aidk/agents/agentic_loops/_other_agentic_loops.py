"""
Specialized Agent Implementations.

This module provides specialized agent implementations for different
reasoning patterns and use cases.
"""

import json
from typing import Any, Dict, List
from ._agentic_loop import _AgenticLoop
from ._react_agentic_loop import _BaseReactLoop


class ProgrammaticAgenticLoop(_BaseReactLoop):
    """Programmatic agent.
    
    This agent implements a programmatic approach where the model
    produces code or structured instructions that are executed.
    It's designed for tasks that require code generation and execution.
    """
    
    def start(self, query: str) -> Dict[str, Any]:
        """Start the programmatic agentic loop.
        
        Parameters
        ----------
        query : str
            User query to process
        
        Returns
        -------
        Dict[str, Any]
            Agent response processed through the programmatic loop
        """
        return self._run_react_loop(query, "programmatic")


class PlanAndExecuteAgenticLoop(_BaseReactLoop):
    """Plan-and-execute agent.
    
    This agent implements the plan-and-execute pattern, where the model
    first plans the actions and then executes them sequentially.
    This approach is useful for complex tasks that require careful planning.
    """
    
    def start(self, query: str) -> Dict[str, Any]:
        """Start the plan-and-execute agentic loop.
        
        Parameters
        ----------
        query : str
            User query to process
        
        Returns
        -------
        Dict[str, Any]
            Agent response processed through the plan-and-execute loop
        """
        return self._run_react_loop(query, "plan_and_execute")


class ReflexionAgenticLoop(_BaseReactLoop):
    """Agent with reflection capabilities.
    
    This agent implements the reflexion pattern, where the model
    reflects on its own actions and decisions to improve performance.
    This self-reflective approach helps the agent learn from mistakes
    and improve its reasoning over time.
    """
    
    def start(self, query: str) -> Dict[str, Any]:
        """Start the reflexion agentic loop.
        
        Parameters
        ----------
        query : str
            User query to process
        
        Returns
        -------
        Dict[str, Any]
            Agent response processed through the reflexion loop
        """
        return self._run_react_loop(query, "reflexion")


class SelfAskAgenticLoop(_BaseReactLoop):
    """Self-ask agent.
    
    This agent implements the self-ask pattern, where the model
    asks itself questions to guide the reasoning process.
    This approach helps break down complex problems into smaller,
    more manageable questions.
    """
    
    def start(self, query: str) -> Dict[str, Any]:
        """Start the self-ask agentic loop.
        
        Parameters
        ----------
        query : str
            User query to process
        
        Returns
        -------
        Dict[str, Any]
            Agent response processed through the self-ask loop
        """
        return self._run_react_loop(query, "self_ask")


class SelfAskWithSearchLoop(_BaseReactLoop):
    """Self-ask agent with web search capabilities.
    
    This agent extends the self-ask pattern with the ability to
    perform web searches to obtain additional information.
    It's particularly useful for questions that require current
    or factual information not available in the model's training data.
    
    Attributes
    ----------
    _handle_search_query : callable
        Method for handling web search queries
    """
    
    def _handle_search_query(self, iteration: Dict[str, Any], response: Dict[str, Any], messages: List[Dict[str, Any]]) -> None:
        """Handle web search queries.
        
        This method processes search queries by executing web searches
        using the Tavily search engine and integrating the results
        into the conversation flow.
        
        Parameters
        ----------
        iteration : Dict[str, Any]
            Current iteration containing the search query
        response : Dict[str, Any]
            Response dictionary to update
        messages : List[Dict[str, Any]]
            Message list to update with search results
        
        Notes
        -----
        This method modifies the response and messages objects directly.
        Uses the Tavily search engine for web searches.
        """
        from ...tools.websearch import search_web
        
        query = iteration["search_query"]
        result = search_web(query, engine="tavily")["text"]
        iteration["search_result"] = result
        
        msg = json.dumps({"query_results": result})
        messages.append({"role": "user", "content": msg})
        response["iterations"].append(iteration)
    
    def start(self, query: str) -> Dict[str, Any]:
        """Start the self-ask with search agentic loop.
        
        Parameters
        ----------
        query : str
            User query to process
        
        Returns
        -------
        Dict[str, Any]
            Agent response processed through the self-ask with search loop
        
        Notes
        -----
        This agent uses a custom handler for web search queries, allowing
        the model to obtain up-to-date information during the process.
        """
        custom_handlers = {"search_query": self._handle_search_query}
        return self._run_react_loop(query, "self_ask_with_search", custom_handlers)
