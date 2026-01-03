from aidk.chat.history import *
from aidk.models import Model
from aidk.models import ModelResponse, ModelStreamHead, ModelStreamChunk, ModelStreamTail
from aidk.conf.conf import Conf
from aidk.prompts.prompt import Prompt
import os
import base64
import logging
from typing import Union, Optional, AsyncGenerator, List, Dict, Any
from pathlib import Path
from dataclasses import dataclass

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

@dataclass
class ModelInfo:
    """Information about the model used."""
    provider: str
    name: str

@dataclass(kw_only=True)
class ChatResponse(ModelResponse):
    """Response from a chat request."""
    chat_id: str
    history: Optional[List[Message]] = None

@dataclass(kw_only=True)
class ChatStreamHead(ModelStreamHead):
    """Head of a streaming chat response."""
    chat_id: str

@dataclass(kw_only=True)
class ChatStreamChunk(ModelStreamChunk):
    pass

@dataclass(kw_only=True)
class ChatStreamTail(ModelStreamTail):
    pass

@dataclass
class ChatInfo:
    """Information about the current chat."""
    chat_id: str
    provider: str
    model: str
    max_tokens: Optional[int]

class ChatError(Exception):
    """Custom exception for Chat-related errors."""
    pass

class Chat():
    """
    Chat class is responsible for handling the chat interface and messages history.
    
    Examples
    --------
    Basic usage:
    ```
    from aidk.models import Model
    from aidk.chat import Chat
    model = Model(provider="openai", model="gpt-4o-mini")
    chat = Chat(model=model)
    response = chat.send("2+2") # {"chat_id": "...", "response": "4", "model": {...}}
    response = chat.send("+2") # {"chat_id": "...", "response": "6", "model": {...}}
    ```    

    With history:
    ```
    from aidk.models import Model
    from aidk.chat import Chat
    model = Model(provider="openai", model="gpt-4o-mini")
    
    # Create a new chat with JSON history
    chat = Chat(model=model, history="json")
    print(chat.chat_id) # 8cc2bfa3-e9a0-4b82-b46e-3376cd220dd3
    response = chat.send("Hello! I'm Giuseppe") # {"chat_id": "...", "response": "Hello!", "model": {...}}

    # Load a chat with JSON history
    chat = Chat(model=model, history="json", chat_id="8cc2bfa3-e9a0-4b82-b46e-3376cd220dd3")
    response = chat.send("What's my name?") # {"chat_id": "...", "response": "Your name is Giuseppe", "model": {...}}
    # Create a new chat with in-memory dictionary history
    chat = Chat(model=model, history="dict")
    print(chat.chat_id) # 8cc2bfa3-e9a0-4b82-b46e-3376cd220dd3
    response = chat.send("Hello! I'm Giuseppe") # {"chat_id": "...", "response": "Hello!", "model": {...}}
    ```

    With history summarizer:

    ```
    from aidk.models import Model
    
    model = Model(provider="openai", model="gpt-4o-mini")
    chat = Chat(
        model=model,
        history="json", 
        history_summarizer_provider="openai", 
        history_summarizer_model="gpt-4o-mini", 
        history_summarizer_max_tokens=100
    )

    response = chat.send("Hello! I'm Giuseppe") # {"chat_id": "...", "response": "Hello!", "model": {...}}
    response = chat.send("What's my name?") # {"chat_id": "...", "response": "Your name is Giuseppe", "model": {...}}
    ```

    With metadata for observability:

    ```
    from aidk.models import Model
    
    model = Model(provider="openai", model="gpt-4o-mini")
    chat = Chat(model=model)
    
    # Pass metadata for tracking
    response = chat.send(
        "Hello! I'm Giuseppe", 
        metadata={
            "user_id": "12345",
            "session_id": "abc-def-ghi",
            "feature": "chat"
        }
    )
    
    # Streaming with metadata
    async for chunk in chat.send_stream(
        "Tell me a story",
        metadata={"user_id": "12345", "request_type": "story_generation"}
    ):
        print(chunk)
    ```
    """

    _HISTORY_MAP = {
        "json": JSONHistory,
        "sqlite": SQLiteHistory,
        "mongodb": MongoDBHistory,
        "dict": DictHistory
    }

    def __init__(self, 
                 model,
                 system_prompt: Optional[Union[SystemPrompt, str]] = None,
                 max_tokens: Optional[int] = None,
                 history: Union[str, BaseHistory] = "dict", 
                 history_last_n: Optional[int] = None,
                 history_path: Optional[str] = None,
                 chat_id: Optional[str] = None) -> None:

        """
        Initialize a new Chat instance.

        Parameters
        ----------
        model : Model
            Model instance to use for the chat
        system_prompt : str | Prompt, optional
            System prompt or Prompt object. If string ends with '.prompt', 
            it will be treated as a file path to load the prompt from.
        max_tokens : int, optional
            Maximum number of tokens for each request
        history : str | BaseHistory, optional
            The type of history to use for the chat. Options: "json", "sqlite", "mongodb", "dict".
            Default is "dict" for in-memory storage.
        history_last_n : int, optional
            The last n messages to keep in the history. If None, keeps all messages.
        history_path : str, optional
            The path to the history storage (not used for "dict" history type).
            Default is "histories".
        chat_id : str, optional
            The id of the chat to load. If not provided, a new chat will be created.
            If provided but chat doesn't exist, a new chat will be created with this ID.

        Raises
        ------
        ChatError
            If invalid parameters are provided or initialization fails
        """
        try:
            
            self._max_tokens = max_tokens
            self._model = model
            
            # Initialize history
            self._initialize_history(history, history_last_n, history_path)
            
            # Process system prompt
            processed_system_prompt = self._process_system_prompt(system_prompt)
            # Initialize chat
            if chat_id is None:
                self.chat_id = self._history.new(processed_system_prompt)
            else:
                # Check if chat exists, if not create it
                self.chat_id = self._initialize_or_create_chat(chat_id, processed_system_prompt)
            
            self._metadata = {"session_id": self.chat_id}

        except Exception as e:
            logger.error(f"Failed to initialize Chat: {e}")
            raise ChatError(f"Chat initialization failed: {e}")

    def _initialize_or_create_chat(self, chat_id: str, system_prompt: SystemPrompt) -> str:
        """Initialize existing chat or create new one with specified ID.
        
        Parameters
        ----------
        chat_id : str
            The chat ID to check/create
        system_prompt : SystemPrompt
            System prompt to use if creating new chat
            
        Returns
        -------
        str
            The chat ID (either existing or newly created)
        """
        try:
            # Try to load existing chat
            existing_messages = self._history.load(chat_id)
            
            # Check if chat exists and has messages
            if existing_messages and len(existing_messages) > 0:
                logger.info(f"Loaded existing chat: {chat_id}")
                return chat_id
            else:
                # Chat doesn't exist, create it with the specified ID
                logger.info(f"Creating new chat with ID: {chat_id}")
                return self._create_chat_with_id(chat_id, system_prompt)
                
        except Exception as e:
            # If loading fails, assume chat doesn't exist and create it
            logger.info(f"Failed to load chat {chat_id}, creating new one: {e}")
            return self._create_chat_with_id(chat_id, system_prompt)
    
    def _create_chat_with_id(self, chat_id: str, system_prompt: SystemPrompt) -> str:
        """Create a new chat with a specific ID.
        
        Parameters
        ----------
        chat_id : str
            The specific chat ID to use
        system_prompt : SystemPrompt
            System prompt for the new chat
            
        Returns
        -------
        str
            The chat ID
        """
        try:
            # Create Message dataclass for system prompt
            system_message = Message(
                content=system_prompt.content,
                role=system_prompt.role,
                provider=system_prompt.provider,
                model=system_prompt.model
            )
            
            # Store the system message with the specific chat_id
            self._history.store(chat_id, [system_message])
            
            return chat_id
            
        except Exception as e:
            logger.error(f"Failed to create chat with ID {chat_id}: {e}")
            raise ChatError(f"Failed to create chat with ID {chat_id}: {e}")


    def _initialize_history(self, history: Union[str, BaseHistory], history_last_n: Optional[int], history_path: Optional[str]) -> None:
        """Initialize the history system.
        
        Parameters
        ----------
        history : Union[str, BaseHistory]
            History type or instance
        history_last_n : Optional[int]
            Number of last messages to keep
        history_path : Optional[str]
            Path for history storage
            
        Raises
        ------
        ChatError
            If history initialization fails
        """
        try:
            if isinstance(history, str):
                if history == "dict":
                    # DictHistory doesn't need db_path
                    self._history = self._HISTORY_MAP[history](last_n=history_last_n)
                else:
                    self._history = self._HISTORY_MAP[history](last_n=history_last_n, path=history_path)
            else:
                self._history = history
        except Exception as e:
            raise ChatError(f"Failed to initialize history: {e}")


    def _process_system_prompt(self, system_prompt: Optional[Union[Prompt, str]]) -> SystemPrompt:
        """Process and load the system prompt.
        
        Parameters
        ----------
        system_prompt : Optional[Union[Prompt, str]]
            System prompt to process
            
        Returns
        -------
        SystemPrompt
            Processed system prompt object
        """
        try:
            if system_prompt is None:
                system_prompt = self._load_default_system_prompt()
            elif isinstance(system_prompt, str) and system_prompt.endswith(".prompt"):
                system_prompt = self._load_prompt_file(system_prompt)
            elif isinstance(system_prompt, Prompt):
                system_prompt = str(system_prompt)
            else:
                system_prompt = str(system_prompt)
        except Exception as e:
            logger.warning(f"Failed to process system prompt: {e}")
            system_prompt = ""

        return SystemPrompt(content=system_prompt, provider=self._model.provider, model=self._model.model)

    def _load_default_system_prompt(self) -> str:
        """Load the default system prompt from file.
        
        Returns
        -------
        str
            Default system prompt content
        """
        try:
            prompt_path = Conf()["prompts_path"]
            system_prompt_path = Path(prompt_path) / "system.prompt"
            
            if system_prompt_path.exists():
                with open(system_prompt_path, "r", encoding="utf-8") as f:
                    return f.read()
            else:
                logger.info("Default system prompt file not found, using empty prompt")
                return ""
        except Exception as e:
            logger.warning(f"Failed to load default system prompt: {e}")
            return ""

    def _load_prompt_file(self, prompt_file: str) -> str:
        """Load a prompt from a file.
        
        Parameters
        ----------
        prompt_file : str
            Path to the prompt file
            
        Returns
        -------
        str
            Content of the prompt file
            
        Raises
        ------
        ChatError
            If file cannot be loaded
        """
        try:
            prompt_path = Conf()["prompts_path"]
            full_path = Path(prompt_path) / prompt_file
            
            if not full_path.exists():
                raise FileNotFoundError(f"Prompt file not found: {full_path}")
                
            with open(full_path, "r", encoding="utf-8") as f:
                return f.read()
        except Exception as e:
            raise ChatError(f"Failed to load prompt file {prompt_file}: {e}")

    def _process_file_attachment(self, prompt: str, file: Optional[Union[str, bytes]], file_type: Optional[str]) -> Union[str, List[Dict[str, Any]]]:
        """Process file attachment and return modified prompt.
        
        Parameters
        ----------
        prompt : str
            Original prompt text
        file : Optional[Union[str, bytes]]
            File to attach (path, URL, or bytes)
        file_type : Optional[str]
            Type of the file
            
        Returns
        -------
        Union[str, List[Dict[str, Any]]]
            Modified prompt with file content or multimodal content
            
        Raises
        ------
        ChatError
            If file processing fails
        """
        if file is None:
            return prompt
            
        try:
            # Determine file type
            if file_type is None and isinstance(file, str):
                file_type = file.split(".")[-1].lower()
                
            if not file_type:
                raise ChatError("File type could not be determined")
                
            conf = Conf()
            
            # Handle text files
            if file_type in conf["supported_files"]["text"]:
                return self._process_text_file(prompt, file)
            # Handle image files
            elif file_type in conf["supported_files"]["image"]:
                return self._process_image_file(prompt, file)
            else:
                raise ChatError(f"Unsupported file type: {file_type}")
                
        except Exception as e:
            raise ChatError(f"Failed to process file attachment: {e}")

    def _process_text_file(self, prompt: str, file: Union[str, bytes]) -> str:
        """Process text file attachment.
        
        Parameters
        ----------
        prompt : str
            Original prompt text
        file : Union[str, bytes]
            Text file to process
            
        Returns
        -------
        str
            Prompt with file content appended
            
        Raises
        ------
        ChatError
            If text file processing fails
        """
        try:
            if isinstance(file, str):
                # Handle as file path or URL
                if os.path.exists(file):
                    with open(file, "r", encoding="utf-8") as f:
                        file_content = f.read()
                else:
                    # Assume it's a URL or remote file
                    file_content = file
            else:
                # Handle as bytes
                file_content = file.decode("utf-8")
                
            return prompt + Conf()["default_prompt"]["file"] + file_content
            
        except Exception as e:
            raise ChatError(f"Failed to process text file: {e}")

    def _process_image_file(self, prompt: str, file: Union[str, bytes]) -> List[Dict[str, Any]]:
        """Process image file attachment.
        
        Parameters
        ----------
        prompt : str
            Original prompt text
        file : Union[str, bytes]
            Image file to process
            
        Returns
        -------
        List[Dict[str, Any]]
            Multimodal content with text and image
            
        Raises
        ------
        ChatError
            If image file processing fails
        """
        try:
            if isinstance(file, str):
                # Handle as file path or URL
                if os.path.exists(file):
                    # Convert local file to base64
                    with open(file, "rb") as f:
                        file_bytes = f.read()
                    image_url = f"data:image/jpeg;base64,{base64.b64encode(file_bytes).decode('utf-8')}"
                else:
                    # Assume it's already a URL
                    image_url = file
            else:
                # Handle as bytes
                image_url = f"data:image/jpeg;base64,{base64.b64encode(file).decode('utf-8')}"
                
            return [
                {"type": "text", "text": prompt},
                {"type": "image_url", "image_url": image_url}
            ]
            
        except Exception as e:
            raise ChatError(f"Failed to process image file: {e}")

    def _prepare_messages(self, prompt: Union[str, List[Dict[str, Any]]]) -> List[Dict[str, Any]]:
        """Prepare messages for the API call.
        
        Parameters
        ----------
        prompt : Union[str, List[Dict[str, Any]]]
            User prompt or multimodal content
            
        Returns
        -------
        List[Dict[str, Any]]
            Formatted messages for API call
            
        Raises
        ------
        ChatError
            If message preparation fails
        """
        try:
            if self._history._summarizer is None:
                messages = self._history.load(self.chat_id)
            else:
                messages = self._history.get_summary(self.chat_id)
            # Convert Message dataclasses to dictionaries for API call
            messages_dict = [{'role': msg.role, 'content': msg.content} for msg in messages]

            messages_dict.append({"role": "user", "content": prompt})

            return messages_dict
                
        except Exception as e:
            raise ChatError(f"Failed to prepare messages: {e}")

    def _apply_history_summarization(self, messages: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Apply history summarization to messages.
        
        Parameters
        ----------
        messages : List[Dict[str, Any]]
            Full message history
            
        Returns
        -------
        List[Dict[str, Any]]
            Summarized messages for API call
        """
        try:
            summarized = self._history_summarizer.summarize(messages)
            ask_messages = [messages[0], messages[-1]]
            ask_messages[0]["content"] += Conf()["default_prompt"]["summary"] + summarized
            return ask_messages
        except Exception as e:
            logger.warning(f"History summarization failed, using full history: {e}")
            return messages

    def _store_messages(self, messages: List[Dict[str, Any]], response_content: str) -> None:
        """Store messages in history.
        
        Parameters
        ----------
        messages : List[Dict[str, Any]]
            Messages to store
        response_content : str
            Assistant response content
        """
        try:
            # Create Message dataclasses for the last two messages (user + assistant)
            user_message = Message(
                content=messages[-1]["content"],
                role=messages[-1]["role"],
                provider=self._model.provider,
                model=self._model.model
            )
            
            assistant_message = Message(
                content=response_content,
                role="assistant",
                provider=self._model.provider,
                model=self._model.model
            )
            
            self._history.store(self.chat_id, [user_message, assistant_message])
        except Exception as e:
            logger.error(f"Failed to store messages: {e}")


    def send(self, prompt: str, file: Optional[Union[str, bytes]] = None, file_type: Optional[str] = None, return_history: bool = False, metadata: Optional[Dict[str, Any]] = {}) -> ChatResponse:
        """
        Send the model a question.

        Parameters
        ----------
        prompt : str
            The question to send the model
        file : str | bytes, optional
            The file to attach to the message, if it's a string it will be treated as a local path or remote url to a file
        file_type : str, optional
            The type of the file
        return_history : bool, optional
            Whether to return the full history of the chat or only the response
        metadata : Dict[str, Any], optional
            Metadata to pass to the model for observability and tracking

        Returns
        -------
        ChatResponse
            ChatResponse dataclass containing:
            - chat_id: str - The chat identifier
            - response: ModelResponse - The model response
            - history: Optional[List[Message]] - Full chat history if return_history is True

        Raises
        ------
        ChatError
            If the request fails or parameters are invalid
        """

        metadata = self._metadata | metadata

        try:
            # Process file attachment
            processed_prompt = self._process_file_attachment(prompt, file, file_type)
            
            # Prepare messages
            messages = self._prepare_messages(processed_prompt)
            print(messages)
            # Make API call
            model_response = self._model.ask(messages, metadata=metadata)
            
            # Store messages
            self._store_messages(messages, model_response.response)
            
            # Get history if requested
            history = None
            if return_history:
                history = self._history.load(self.chat_id)
            
            return ChatResponse(
                chat_id=self.chat_id,
                history=history,
                **asdict(model_response)
            )
                        
        except Exception as e:
            logger.error(f"Ask request failed: {e}")
            raise ChatError(f"Ask request failed: {e}")

    async def send_stream(self, prompt: str, file: Optional[Union[str, bytes]] = None, file_type: Optional[str] = None, metadata: Optional[Dict[str, Any]] = {}) -> AsyncGenerator[Dict[str, Any], None]:
        """
        Send the model a question and stream the response.

        Parameters
        ----------
        prompt : str
            The question to send the model
        file : str | bytes, optional
            The file to attach to the message
        file_type : str, optional
            The type of the file
        metadata : Dict[str, Any], optional
            Metadata to pass to the model for observability and tracking

        Yields
        ------
        Dict[str, Any]
            Streaming response chunks containing:
            - chat_id: str - The chat identifier (first chunk)
            - delta: str - Response text chunk (subsequent chunks)
            - Other streaming metadata

        Raises
        ------
        ChatError
            If the request fails or parameters are invalid
        """

        metadata = self._metadata | metadata

        try:
            # Process file attachment
            processed_prompt = self._process_file_attachment(prompt, file, file_type)
            
            # Prepare messages
            messages = self._prepare_messages(processed_prompt)
            
            # Make streaming API call
            response = self._model.ask_stream(messages, metadata=metadata)

            async for chunk in response:
                if (isinstance(chunk, ModelStreamHead)):
                    yield ChatStreamHead(chat_id=self.chat_id, **asdict(chunk)) 
                elif (isinstance(chunk, ModelStreamChunk)):
                    yield ChatStreamChunk(**asdict(chunk))
                elif (isinstance(chunk, ModelStreamTail)):
                    yield ChatStreamTail(**asdict(chunk))
                else:
                    yield chunk
            
            # Store messages using Message dataclasses
            #self._store_messages(messages, response_text)
            
        except Exception as e:
            logger.error(f"Stream request failed: {e}")
            raise ChatError(f"Stream request failed: {e}")

    async def send_async(self, prompt: str, file: Optional[Union[str, bytes]] = None, file_type: Optional[str] = None, metadata: Optional[Dict[str, Any]] = None) -> "ChatResponse":
        """
        Send the model a question using the async (non-streaming) model API and return a ChatResponse.

        This method awaits the model's async completion and returns a `ChatResponse` wrapping
        the underlying `ModelResponse`.
        """

        # Merge metadata safely
        metadata = self._metadata | (metadata or {})

        try:
            # Process file attachment
            processed_prompt = self._process_file_attachment(prompt, file, file_type)

            # Prepare messages
            messages = self._prepare_messages(processed_prompt)

            # Await the model's async (non-streaming) API
            model_response = await self._model.ask_async(messages, metadata=metadata)

            # Store messages (use the string payload from ModelResponse)
            self._store_messages(messages, model_response.response)

            # Get history if requested (not applicable here, but kept for parity)
            history = self._history.load(self.chat_id) if hasattr(self._history, 'load') else None

            return ChatResponse(
                chat_id=self.chat_id,
                history=history,
                **asdict(model_response)
            )

        except Exception as e:
            logger.error(f"Async request failed: {e}")
            raise ChatError(f"Async request failed: {e}")

    def get_chat_info(self) -> ChatInfo:
        """
        Get information about the current chat.

        Returns
        -------
        ChatInfo
            ChatInfo dataclass containing chat information
        """
        return ChatInfo(
            chat_id=self.chat_id,
            provider=self._model.provider,
            model=self._model.model,
            max_tokens=self._max_tokens
        )

    def clear_history(self) -> None:
        """
        Clear the chat history.

        Raises
        ------
        ChatError
            If clearing history fails
        """
        try:
            self._history.clear(self.chat_id)
            logger.info(f"Cleared history for chat {self.chat_id}")
        except Exception as e:
            logger.error(f"Failed to clear history: {e}")
            raise ChatError(f"Failed to clear history: {e}")