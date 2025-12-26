from mcp import ClientSession, StdioServerParameters
from mcp.client.stdio import stdio_client
from mcp.client.sse import sse_client
from mcp.client.streamable_http import streamablehttp_client

import asyncio
from contextlib import asynccontextmanager
import json
from anyio import ClosedResourceError

class McpServer:
    """MCP (Model Context Protocol) Server client for connecting to external tools.
    
    This class provides a unified interface for connecting to MCP servers using
    different transport protocols (HTTP/SSE, stdio). It handles connection
    management, tool discovery, and tool execution in both synchronous and
    asynchronous contexts.
    
    The MCP server allows AI agents to access external tools and services
    through a standardized protocol, enabling dynamic tool discovery and
    execution without hardcoded integrations.
        
    Examples
    --------
    >>> # HTTP/SSE server
    >>> server = McpServer(
    ...     name="coingecko",
    ...     server_type="http",
    ...     server_url="https://mcp.api.coingecko.com/sse"
    ... )
    >>> tools = server.get_tools()
    >>> result = server.call_tool("get_price", {"coin": "bitcoin"})
    
    >>> # Stdio server
    >>> server = McpServer(
    ...     name="local_tools",
    ...     server_type="python",
    ...     server_args=["python", "tools_server.py"]
    ... )
    >>> tools = server.get_tools()
    """
    
    def __init__(
        self,
        name: str,
        server_type: str,               # "http" | "node" | "python"
        server_args: list | None = None,
        server_url: str | None = None,
        env: dict | None = None,
        headers: dict | None = None,    # <-- per HTTP/SSE
        connect_timeout: float = 15.0,  # opzionale
    ):
        """Initialize the MCP server client.
        
        Parameters
        ----------
        name : str
            Unique identifier for this MCP server instance
        server_type : str
            Type of server connection ("http", "node", "python")
        server_args : list, optional
            Command line arguments for stdio-based servers
        server_url : str, optional
            URL for HTTP/SSE based servers
        env : dict, optional
            Environment variables for stdio-based servers
        headers : dict, optional
            HTTP headers for HTTP/SSE based servers
        connect_timeout : float, optional
            Connection timeout in seconds (default: 15.0)
        """
        self.name = name
        self._server_type = server_type
        self._server_args = server_args or []
        self._server_url = server_url
        self._env = env or {}
        self._headers = headers or {}

    def _get_client(self):
        """Get the appropriate MCP client for the server type.
        
        Returns
        -------
        async generator
            MCP client context manager for the specified server type
            
        Raises
        ------
        ValueError
            If server_type is "http" but server_url is not provided
        """
        if self._server_type == "http":
            if not self._server_url:
                raise ValueError("URL mancante per server_type='http'")
            
            if self._server_url.endswith("/mcp"):
                return streamablehttp_client(self._server_url, headers=self._headers)
            else:
                return sse_client(self._server_url, headers=self._headers)
        else:
            server_params = StdioServerParameters(
                command="npx" if self._server_type == "node" else "python",
                args=self._server_args,
                env=self._env,
                stderr="pipe",
            )
            
            return stdio_client(server_params)

    async def get_tools_async(self):
        """Asynchronously retrieve available tools from the MCP server.
        
        Returns
        -------
        list
            List of available tools with their schemas and descriptions
            
        Raises
        ------
        RuntimeError
            If unable to connect to the server or retrieve tools
        """
        async with self._get_client() as (read, write):
            async with ClientSession(read, write) as session:
                resp = await session.list_tools()
                return resp.tools

    def get_tools(self):
        """Synchronously retrieve available tools from the MCP server.
        
        This is a synchronous wrapper around the async method that runs
        the operation in a new event loop.
        
        Returns
        -------
        list
            List of available tools with their schemas and descriptions
            
        Raises
        ------
        RuntimeError
            If unable to connect to the server or retrieve tools
        """
        return asyncio.run(self.get_tools_async())

    async def call_tool_async(self, name: str, args: dict):
        """Asynchronously execute a tool on the MCP server.
        
        Parameters
        ----------
        name : str
            Name of the tool to execute. Can include MCP prefix (mcp_servername_toolname)
            or just the tool name
        args : dict
            Arguments to pass to the tool
            
        Returns
        -------
        list
            Tool execution result, formatted as a list of content items
            
        Raises
        ------
        RuntimeError
            If unable to connect to the server or execute the tool
        ValueError
            If the tool name format is invalid
        """
        async with self._get_client() as (read, write):
            async with ClientSession(read, write) as session:
                prefix = f"mcp_{self.name}_"
                tool_name = name[len(prefix):] if name.startswith(prefix) else name
                result = await session.call_tool(tool_name, args)
                                    
                # Handle different response formats
                if hasattr(result, 'content'):
                    if isinstance(result.content, list) and len(result.content) > 0:
                        if hasattr(result.content[0], 'text'):
                            content = json.loads(result.content[0].text)
                        else:
                            content = str(result.content[0])
                    else:
                        content = str(result.content)
                else:
                    content = str(result)

                if isinstance(content, dict):
                    content = [content]
                return content

    def call_tool(self, name: str, args: dict):
        """Synchronously execute a tool on the MCP server.
        
        This is a synchronous wrapper around the async method that runs
        the operation in a new event loop.
        
        Parameters
        ----------
        name : str
            Name of the tool to execute. Can include MCP prefix (mcp_servername_toolname)
            or just the tool name
        args : dict
            Arguments to pass to the tool
            
        Returns
        -------
        list
            Tool execution result, formatted as a list of content items
            
        Raises
        ------
        RuntimeError
            If unable to connect to the server or execute the tool
        ValueError
            If the tool name format is invalid
        """
        return asyncio.run(self.call_tool_async(name, args))