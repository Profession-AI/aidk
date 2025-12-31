from typing import List, Optional, Dict, Any, Callable, Union
from datetime import datetime, timezone
import json
import asyncio

from aidk.models import Model
from aidk.agents import Agent
from .rate_limiter import RateLimiter

class Application:
    """
    FastAPI-based application for serving AI models and agents.
    
    The Application class provides a complete web service wrapper around AI models
    and agents, offering REST API endpoints, WebSocket support, rate limiting,
    and user validation capabilities.
    
    This class automatically creates FastAPI endpoints based on the configured
    models and agents, handling request validation, rate limiting, and response
    formatting.
        
    Examples
    --------
    ```
    Basic usage with a model:
    ```
    from aidk.models import Model
    from aidk.application import Application
    
    model = Model(provider="openai", model="gpt-4o-mini")
    app = Application(name="MyAIApp", model=model)
    app.serve(port=8000)
    ```
    
    ```
    With agents and rate limiting:
    ```
    from aidk.models import Model
    from aidk.agents import Agent
    from aidk.application import Application, RateLimiter
    
    model = Model(provider="openai", model="gpt-4o-mini")
    agent = Agent(model=model, paradigm="react")
    rate_limiter = RateLimiter(requests_per_minute=60)
    
    app = Application(
        name="AgentApp",
        agents=[agent],
        rate_limiter=rate_limiter
    )
    app.serve(port=8000)
    ```
    
    With user validation:
    ```
    def validate_user(user_id: str):
        # Custom validation logic
        if user_id.startswith("user_"):
            return True
        elif user_id.isdigit():
            return f"user_{user_id}"  # Normalize
        return False
    
    app = Application(
        name="SecureApp",
        model=model,
        user_validator=validate_user
    )
    ```
    """

    def __init__(self, name: str, model: Optional[Model] = None, agents: Optional[List[Agent]] = None, 
                 rate_limiter: Optional[RateLimiter] = None, user_validator: Optional[Callable[[str], Union[bool, str]]] = None):
        """
        Initialize the application.
        
        Parameters
        ----------
        name : str
            Application name. Used in API responses and logging.
        model : Optional[Model], default None
            AI model to use. If provided, creates /model endpoints.
            The model will be available at POST /model and POST /model/stream.
        agents : Optional[List[Agent]], default None
            List of available agents. Each agent must have a unique name.
            Creates /agent/{agent_name} endpoints for each agent.
        rate_limiter : Optional[RateLimiter], default None
            Rate limiter to control API usage. Applies to all endpoints.
            If not provided, no rate limiting is enforced.
        user_validator : Optional[Callable[[str], Union[bool, str]]], default None
            Function to validate user_id from requests. Must return:
            - True: user_id is valid and accepted as-is
            - False: user_id is invalid, will fallback to IP-based identification
            - str: user_id is valid but normalized (e.g. "user123" -> "user_123")
            
        Notes
        -----
        At least one of model or agents must be provided to create useful endpoints.
        If neither is provided, only meta endpoints (/, /health) will be available.
        
        The user_validator function is called for every request that includes
        a user_id in the request body. If validation fails, the application
        falls back to using the client IP address for rate limiting.
        """
        self.name = name
        self._model = model
        self._agents: Optional[Dict[str, Agent]] = (
            {a.name: a for a in agents} if agents else None
        )
        self._rate_limiter = rate_limiter
        self._user_validator = user_validator
        self._started_at = datetime.now(timezone.utc)

    @staticmethod
    async def _maybe_await(fn: Callable[..., Any], *args, **kwargs) -> Any:
        """
        Execute a function and await it if it's a coroutine.
        
        This helper method allows the application to work with both
        synchronous and asynchronous model/agent methods.
        
        Parameters
        ----------
        fn : Callable[..., Any]
            Function to execute
        *args
            Positional arguments to pass to the function
        **kwargs
            Keyword arguments to pass to the function
            
        Returns
        -------
        Any
            Result of the function execution
        """
        result = fn(*args, **kwargs)
        if hasattr(result, "__await__"):
            return await result 
        return result

    def _get_user_identifier(self, request, data: Dict[str, Any]) -> str:
        """
        Extract user identifier from the request.
        
        This method implements a multi-step user identification process:
        1. Look for user_id in the request body
        2. Validate user_id using the configured validator (if any)
        3. Fall back to client IP address if user_id is invalid or missing
        
        The method handles various proxy headers (X-Forwarded-For, X-Real-IP)
        to get the real client IP when behind load balancers or proxies.
        
        Parameters
        ----------
        request : Request
            FastAPI request object (can be HTTP Request or WebSocket)
        data : Dict[str, Any]
            Request body data containing potential user_id
            
        Returns
        -------
        str
            User identifier in one of these formats:
            - "user_id" if user_id is provided and valid
            - "ip:192.168.1.1" if using IP-based identification
            - "ip:unknown" if IP cannot be determined
            
        Notes
        -----
        The user identifier is used for rate limiting and request tracking.
        IP-based identifiers are prefixed with "ip:" to distinguish them
        from actual user IDs.
        """
        # Look for user_id in the request body
        user_id = data.get("user_id")
        if user_id:
            user_id_str = str(user_id)
            
            # Validate user_id if validator is configured
            if self._user_validator:
                try:
                    validation_result = self._user_validator(user_id_str)
                    
                    if validation_result is True:
                        # user_id is valid, use as is
                        return user_id_str
                    elif validation_result is False:
                        # user_id is invalid, fallback to IP
                        pass
                    elif isinstance(validation_result, str):
                        # user_id is valid but normalized
                        return validation_result
                    else:
                        # Unknown validation result, fallback to IP
                        pass
                except Exception:
                    # Error during validation, fallback to IP
                    pass
            else:
                # No validator configured, use user_id as is
                return user_id_str
        
        # Fallback to client IP
        # Try different headers to get the real IP
        forwarded_for = request.headers.get("X-Forwarded-For")
        if forwarded_for:
            # X-Forwarded-For can contain multiple IPs, take the first one
            client_ip = forwarded_for.split(",")[0].strip()
        else:
            real_ip = request.headers.get("X-Real-IP")
            if real_ip:
                client_ip = real_ip
            else:
                # Handle both Request and WebSocket
                if hasattr(request, 'client') and request.client:
                    client_ip = request.client.host
                elif hasattr(request, 'url') and hasattr(request.url, 'hostname'):
                    client_ip = request.url.hostname
                else:
                    client_ip = "unknown"
        
        return f"ip:{client_ip}"

    def validate_user_id(self, user_id: str) -> Union[bool, str]:
        """
        Validate a user_id using the configured validator.
        
        This method provides a safe way to validate user IDs, handling
        any exceptions that might occur during validation.
        
        Parameters
        ----------
        user_id : str
            user_id to validate
        
        Returns
        -------
        Union[bool, str]
            - True: user_id is valid and accepted as-is
            - False: user_id is invalid or validation failed
            - str: user_id is valid but normalized (use this value instead)
            
        Notes
        -----
        If no validator is configured, this method always returns True.
        Any exceptions during validation are caught and result in False.
        """
        if not self._user_validator:
            return True  # No validator, always consider valid
        
        try:
            return self._user_validator(user_id)
        except Exception:
            return False  # Error during validation, consider invalid

    def _build_app(self):
        """
        Build and configure the FastAPI application.
        
        This method creates a FastAPI app with all the necessary endpoints,
        middleware, and error handling based on the configured models and agents.
        
        Returns
        -------
        FastAPI
            Configured FastAPI application instance
            
        Raises
        ------
        ImportError
            If FastAPI is not installed
            
        Notes
        -----
        The method dynamically creates endpoints based on what's configured:
        - Model endpoints are created if a model is provided
        - Agent endpoints are created if agents are provided
        - Meta endpoints (/, /health) are always created
        - Rate limiting and user validation are applied to all endpoints
        """

        try:
            from fastapi import FastAPI, Request, HTTPException, status
        except ImportError as e:
            raise ImportError(
                "fastapi is required to build the application. "
                "Install it with: pip install fastapi"
            ) from e

        from fastapi.middleware.cors import CORSMiddleware

        app = FastAPI(title=self.name)

        app.add_middleware(
            CORSMiddleware,
            allow_origins=["*"],
            allow_credentials=True,
            allow_methods=["*"],
            allow_headers=["*"],
        )

        @app.get("/", tags=["meta"], summary="Ping app")
        async def root():
            return {
                "msg": f"App {self.name} successfully started",
                "started_at": self._started_at.isoformat() + "Z",
            }

        @app.get("/health", tags=["meta"], summary="Health check")
        async def health():
            return {"status": "ok", "app": self.name}

        @app.get("/rate-limit/stats", tags=["rate-limit"], summary="Rate limiter statistics")
        async def rate_limit_stats():
            if not self._rate_limiter:
                return {"message": "Rate limiter not configured"}
            
            stats = self._rate_limiter.get_stats()
            return {
                "rate_limiter": str(self._rate_limiter),
                "global_stats": stats
            }

        @app.get("/rate-limit/stats/{user_id}", tags=["rate-limit"], summary="Statistics for a specific user")
        async def rate_limit_user_stats(user_id: str):
            if not self._rate_limiter:
                return {"message": "Rate limiter not configured"}
            
            user_stats = self._rate_limiter.get_stats(user_id)
            usage = self._rate_limiter.get_usage(user_id)
            remaining = self._rate_limiter.get_remaining(user_id)
            
            return {
                "user_id": user_id,
                "usage": usage,
                "remaining": remaining,
                "stats": user_stats
            }

        @app.post("/validate-user", tags=["auth"], summary="Validate a user_id")
        async def validate_user_endpoint(request: Request):
            try:
                data = await request.json()
            except Exception:
                raise HTTPException(status_code=400, detail="Invalid JSON body")
            
            user_id = data.get("user_id")
            if not user_id:
                raise HTTPException(status_code=400, detail="'user_id' is required")
            
            validation_result = self.validate_user_id(str(user_id))
            
            if validation_result is True:
                return {
                    "valid": True,
                    "user_id": user_id,
                    "normalized": None
                }
            elif validation_result is False:
                return {
                    "valid": False,
                    "user_id": user_id,
                    "error": "Invalid user_id"
                }
            elif isinstance(validation_result, str):
                return {
                    "valid": True,
                    "user_id": user_id,
                    "normalized": validation_result
                }
            else:
                return {
                    "valid": False,
                    "user_id": user_id,
                    "error": "Unknown validation result"
                }

        if self._model is not None:
            @app.post(
                "/model",
                tags=["model"],
                summary="Ask the model",
                status_code=status.HTTP_200_OK,
            )
            async def model_route(request: Request):
                try:
                    data = await request.json()
                except Exception:
                    raise HTTPException(status_code=400, detail="Invalid JSON body")

                prompt = (data or {}).get("prompt")
                if not prompt:
                    raise HTTPException(status_code=400, detail="'prompt' is required")

                # Extract user identifier
                user_identifier = self._get_user_identifier(request, data or {})
                
                # Execute the request to the model
                result = await self._maybe_await(self._model.ask, prompt)
                
                # Check and update rate limit if configured
                if self._rate_limiter:
                    # Check rate limit based on the response
                    if not self._rate_limiter.check_with_response(user_identifier, result):
                        raise HTTPException(
                            status_code=429, 
                            detail="Rate limit exceeded. Please try again later."
                        )

                    # Update the rate limiter with the response
                    self._rate_limiter.update_with_response(user_identifier, result)
                
                return result

        if self._agents is not None:
            @app.post(
                "/agent/{agent_name}",
                tags=["agents"],
                summary="Execute an agent",
                status_code=status.HTTP_200_OK,
            )
            async def agent_route(agent_name: str, request: Request):
                if agent_name not in self._agents:
                    raise HTTPException(status_code=404, detail=f"Agent '{agent_name}' not found")

                try:
                    data = await request.json()
                except Exception:
                    raise HTTPException(status_code=400, detail="Invalid JSON body")

                prompt = (data or {}).get("prompt")
                if not prompt:
                    raise HTTPException(status_code=400, detail="'prompt' is required")

                # Extract user identifier
                user_identifier = self._get_user_identifier(request, data or {})
                
                # Execute the agent
                agent = self._agents[agent_name]
                result = await self._maybe_await(agent.run, prompt)
                
                # Check and update rate limit if configured
                if self._rate_limiter:
                    # Check rate limit based on the response
                    if not self._rate_limiter.check_with_response(user_identifier, result):
                        raise HTTPException(
                            status_code=429, 
                            detail="Rate limit exceeded. Please try again later."
                        )

                    # Update the rate limiter with the response
                    self._rate_limiter.update_with_response(user_identifier, result)
                
                return result

        # Model streaming endpoint
        if self._model is not None:
            @app.post(
                "/model/stream",
                tags=["model"],
                summary="Ask the model with streaming",
                status_code=status.HTTP_200_OK,
            )
            async def model_stream_route(request: Request):
                try:
                    data = await request.json()
                except Exception:
                    raise HTTPException(status_code=400, detail="Invalid JSON body")

                prompt = (data or {}).get("prompt")
                if not prompt:
                    raise HTTPException(status_code=400, detail="'prompt' is required")

                # Extract user identifier
                user_identifier = self._get_user_identifier(request, data or {})
                
                # Create a function to handle streaming
                def stream_handler(content: str):
                    return content
                
                # Enable streaming on the model
                if hasattr(self._model, 'enable_streaming'):
                    self._model.enable_streaming(stream_handler)
                
                # Execute the request to the model with streaming
                result = await self._maybe_await(self._model.ask, prompt)
                
                # Check and update rate limit if configured
                if self._rate_limiter:
                    # Check rate limit based on the response
                    if not self._rate_limiter.check_with_response(user_identifier, result):
                        raise HTTPException(
                            status_code=429, 
                            detail="Rate limit exceeded. Please try again later."
                        )

                    # Update the rate limiter with the response
                    self._rate_limiter.update_with_response(user_identifier, result)
                
                return result

        # Agent streaming endpoint
        if self._agents is not None:
            @app.post(
                "/agent/{agent_name}/stream",
                tags=["agents"],
                summary="Execute an agent with streaming",
                status_code=status.HTTP_200_OK,
            )
            async def agent_stream_route(agent_name: str, request: Request):
                if agent_name not in self._agents:
                    raise HTTPException(status_code=404, detail=f"Agent '{agent_name}' not found")

                try:
                    data = await request.json()
                except Exception:
                    raise HTTPException(status_code=400, detail="Invalid JSON body")

                prompt = (data or {}).get("prompt")
                if not prompt:
                    raise HTTPException(status_code=400, detail="'prompt' is required")

                # Extract user identifier
                user_identifier = self._get_user_identifier(request, data or {})
                
                # Create a function to handle streaming
                def stream_handler(content: str):
                    return content
                
                # Enable streaming on the agent
                agent = self._agents[agent_name]
                if hasattr(agent, 'enable_streaming'):
                    agent.enable_streaming(stream_handler)
                
                # Execute the agent with streaming
                result = await self._maybe_await(agent.run, prompt)
                
                # Check and update rate limit if configured
                if self._rate_limiter:
                    # Check rate limit based on the response
                    if not self._rate_limiter.check_with_response(user_identifier, result):
                        raise HTTPException(
                            status_code=429, 
                            detail="Rate limit exceeded. Please try again later."
                        )

                    # Update the rate limiter with the response
                    self._rate_limiter.update_with_response(user_identifier, result)
                
                return result

        # WebSocket endpoint for real-time streaming
        if self._model is not None:
            @app.websocket("/model/ws")
            async def model_websocket(websocket):
                try:
                    from fastapi import WebSocket
                    await websocket.accept()
                    
                    while True:
                        # Receive prompt from client
                        data = await websocket.receive_text()
                        try:
                            request_data = json.loads(data)
                        except json.JSONDecodeError:
                            await websocket.send_text(json.dumps({"error": "Invalid JSON"}))
                            continue
                        
                        prompt = request_data.get("prompt")
                        if not prompt:
                            await websocket.send_text(json.dumps({"error": "Prompt is required"}))
                            continue
                        
                        # Extract user identifier
                        user_identifier = self._get_user_identifier(websocket, request_data)
                        
                        # Create a handler to send chunks via WebSocket
                        def ws_stream_handler(content: str):
                            asyncio.create_task(websocket.send_text(json.dumps({
                                "type": "chunk",
                                "content": content
                            })))
                        
                        # Enable streaming on the model
                        if hasattr(self._model, 'enable_streaming'):
                            self._model.enable_streaming(ws_stream_handler)
                        
                        # Execute the request to the model
                        result = await self._maybe_await(self._model.ask, prompt)
                        
                        # Send the final result
                        await websocket.send_text(json.dumps({
                            "type": "complete",
                            "result": result
                        }))
                        
                except Exception as e:
                    await websocket.send_text(json.dumps({"error": str(e)}))
                finally:
                    await websocket.close()

        if self._agents is not None:
            @app.websocket("/agent/{agent_name}/ws")
            async def agent_websocket(websocket, agent_name: str):
                try:
                    from fastapi import WebSocket
                    await websocket.accept()
                    
                    if agent_name not in self._agents:
                        await websocket.send_text(json.dumps({"error": f"Agent '{agent_name}' not found"}))
                        await websocket.close()
                        return
                    
                    agent = self._agents[agent_name]
                    
                    while True:
                        # Receive prompt from client
                        data = await websocket.receive_text()
                        try:
                            request_data = json.loads(data)
                        except json.JSONDecodeError:
                            await websocket.send_text(json.dumps({"error": "Invalid JSON"}))
                            continue
                        
                        prompt = request_data.get("prompt")
                        if not prompt:
                            await websocket.send_text(json.dumps({"error": "Prompt is required"}))
                            continue
                        
                        # Extract user identifier
                        user_identifier = self._get_user_identifier(websocket, request_data)
                        
                        # Create a handler to send chunks via WebSocket
                        def ws_stream_handler(content: str):
                            asyncio.create_task(websocket.send_text(json.dumps({
                                "type": "chunk",
                                "content": content
                            })))
                        
                        # Enable streaming on the agent
                        if hasattr(agent, 'enable_streaming'):
                            agent.enable_streaming(ws_stream_handler)
                        
                        # Execute the agent
                        result = await self._maybe_await(agent.run, prompt)
                        
                        # Send the final result
                        await websocket.send_text(json.dumps({
                            "type": "complete",
                            "result": result
                        }))
                        
                except Exception as e:
                    await websocket.send_text(json.dumps({"error": str(e)}))
                finally:
                    await websocket.close()

        return app

    def serve(
        self,
        host: str = "0.0.0.0",
        port: int = 8000,
        reload: bool = False,
        workers: Optional[int] = None,
        log_level: str = "info",
    ):

        """
        Serve the application creating endpoints for the model and agents.
        
        This method starts a uvicorn server with the configured FastAPI application.
        The server provides REST API and WebSocket endpoints for interacting with
        AI models and agents.
        
        API Endpoints
        -------------
        When the server is running, the following endpoints are available:
        
        **Model Endpoints** (if model is configured):
            - POST /model                    Ask the model
            - POST /model/stream             Ask the model with streaming
            - WS  /model/ws                  Ask the model with WebSocket streaming
        
        **Agent Endpoints** (if agents are configured):
            - POST /agent/{agent_name}       Execute an agent
            - POST /agent/{agent_name}/stream Execute an agent with streaming
            - WS  /agent/{agent_name}/ws     Execute an agent with WebSocket streaming
        
        **Authentication & Validation:**
            - POST /validate-user            Validate a user_id
        
        **Rate Limiting & Monitoring:**
            - GET  /rate-limit/stats         Rate limiter statistics
            - GET  /rate-limit/stats/{user_id} Statistics for a specific user
        
        **Meta & Health:**
            - GET  /                         Ping the application
            - GET  /health                   Health check
        
        **Request Format:**
        All POST endpoints expect JSON with a "prompt" field:
        ```json
        {
            "prompt": "Your question here",
            "user_id": "optional_user_id"
        }
        ```
        
        **Response Format:**
        - Model responses: Standard model response format
        - Agent responses: Agent execution result with iterations
        - WebSocket: JSON messages with "type" and "content" fields
        
        Parameters
        ----------
        host : str, default "0.0.0.0"
            Host to serve the application. Use "0.0.0.0" for external access.
        port : int, default 8000
            Port to serve the application on.
        reload : bool, default False
            Whether to reload the application when code changes are detected.
            Useful for development, not recommended for production.
        workers : Optional[int], default None
            Number of worker processes to use. If None, uses uvicorn default.
        log_level : str, default "info"
            Log level for uvicorn. Options: "critical", "error", "warning", "info", "debug".
            
        Raises
        ------
        ImportError
            If uvicorn is not installed
            
        Examples
        --------
        Basic serving:
        ```
        app = Application(name="MyApp", model=model)
        app.serve()  # Serves on http://localhost:8000
        ```
        
        Production serving:
        ```
        app.serve(host="0.0.0.0", port=8080, workers=4, log_level="warning")
        ```
        
        Development serving:
        ```
        app.serve(reload=True, log_level="debug")
        ```
        """

        try:
            import uvicorn
        except ImportError as e:
            raise ImportError(
                "uvicorn is required to serve the application. "
                "Install it with: pip install uvicorn"
            ) from e

        uvicorn.run(
            self._build_app(),
            host=host,
            port=port,
            reload=reload,
            workers=workers,
            log_level=log_level,
        )
