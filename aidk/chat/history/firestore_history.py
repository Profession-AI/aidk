"""
Firestore-based history storage implementation.
Stores chat histories in Google Cloud Firestore.
"""

import os
import datetime
from typing import Optional
from .models import BaseHistory, Message


class FirestoreHistory(BaseHistory):
    """
    Firestore-based history storage using Google Cloud Firestore.
    
    This class provides persistent storage for chat histories using Google Cloud Firestore,
    a NoSQL document database that scales automatically and provides real-time updates.
    
    Attributes
    ----------
    _collection_name : str
        Name of the Firestore collection to store chat histories
    _last_n : Optional[int]
        Maximum number of recent messages to load
    _client : Any
        Firestore client instance
    _collection : Any
        Firestore collection reference
    """
    
    def __init__(self, collection_name: str = "chat_histories", 
                 credentials_path: Optional[str] = None, 
                 last_n: Optional[int] = None,
                 summarizer_provider: Optional[str] = None,
                 summarizer_model: Optional[str] = None,
                 summarizer_max_tokens: Optional[int] = None):
        """Initialize Firestore history storage.
        
        Parameters
        ----------
        collection_name : str, optional
            Name of the Firestore collection (default: "chat_histories")
        credentials_path : str, optional
            Path to Google Cloud service account credentials JSON file
        last_n : int, optional
            Maximum number of recent messages to load (default: None for all messages)
        summarizer_provider : str, optional
            Provider for message summarization
        summarizer_model : str, optional
            Model for message summarization
        summarizer_max_tokens : int, optional
            Maximum tokens for summarization
        """
        super().__init__(
            last_n=last_n,
            summarizer_provider=summarizer_provider,
            summarizer_model=summarizer_model,
            summarizer_max_tokens=summarizer_max_tokens
        )
        self._collection_name: str = collection_name
        self._credentials_path: Optional[str] = credentials_path
        self._client = None
        self._collection = None
        self._init_firestore()
    
    def _init_firestore(self):
        """Initialize Firestore client and collection reference."""
        try:
            import google.cloud.firestore as firestore
            
            if self._credentials_path and os.path.exists(self._credentials_path):
                # Use service account credentials
                import google.auth
                from google.oauth2 import service_account
                
                credentials = service_account.Credentials.from_service_account_file(
                    self._credentials_path
                )
                # Project ID is extracted from credentials
                self._client = firestore.Client(credentials=credentials)
            else:
                # Use default credentials (from environment or metadata server)
                self._client = firestore.Client()
            
            self._collection = self._client.collection(self._collection_name)
            
        except ImportError:
            raise ImportError(
                "google-cloud-firestore is required for FirestoreHistory. "
                "Install it with: pip install google-cloud-firestore"
            )
        except Exception as e:
            raise RuntimeError(f"Failed to initialize Firestore: {e}")

    def get_messages(self, chat_id: str) -> list[Message]:
        """Load chat history from Firestore.
        
        Parameters
        ----------
        chat_id : str
            Unique identifier for the chat
            
        Returns
        -------
        list
            List of messages in the chat history
        """
        try:
            doc_ref = self._collection.document(chat_id)
            doc = doc_ref.get()
            
            if not doc.exists:
                self.messages = []
                return self.messages
            
            data = doc.to_dict()
            messages_data = data.get("messages", [])
            
            # Apply last_n filter if specified
            if self._last_n is not None and len(messages_data) > (self._last_n + 1) * 2:
                messages_data = [messages_data[0]] + messages_data[-self._last_n * 2:]
            
            # Convert to Message dataclasses
            self.messages = [Message.from_dict(msg) if isinstance(msg, dict) else msg for msg in messages_data]
            return self.messages
            
        except Exception as e:
            # Log error and return empty messages
            print(f"Error loading from Firestore: {e}")
            self.messages = []
            return self.messages
        

    def save_message(self, chat_id: str, messages: list[Message]) -> None:
        """Store messages in Firestore.
        
        Parameters
        ----------
        chat_id : str
            Unique identifier for the chat
        messages : list
            List of messages to store
        """
        
        try:
            doc_ref = self._collection.document(chat_id)
            
            # Get existing messages
            doc = doc_ref.get()
            existing_messages_data = []
            
            if doc.exists:
                data = doc.to_dict()
                existing_messages_data = data.get("messages", [])
            
            # Convert existing messages to Message dataclasses
            existing_messages = [Message.from_dict(msg) if isinstance(msg, dict) else msg for msg in existing_messages_data]
            
            # Add new messages (Message dataclasses)
            new_messages = existing_messages + messages
            
            # Convert to dictionaries for Firestore storage
            messages_to_store = [msg.to_dict() for msg in new_messages]
            
            # Update document with new messages
            doc_ref.set({
                "chat_id": chat_id,
                "messages": messages_to_store,
                "last_updated": datetime.datetime.utcnow().isoformat() + 'Z',
                "message_count": len(messages_to_store)
            }, merge=True)
            
        except Exception as e:
            print(f"Error storing to Firestore: {e}")
            raise RuntimeError(f"Failed to store messages: {e}")
            

    def clear(self, chat_id: str) -> None:
        """Clear chat history.
        
        Parameters
        ----------
        chat_id : str
            Specific chat ID to clear.
        """
        try:
            doc_ref = self._collection.document(chat_id)
            doc_ref.delete()
                    
        except Exception as e:
            print(f"Error clearing from Firestore: {e}")

    def get_summary(self, chat_id: str) -> Message:
        """Get chat summary for a specific chat ID."""
        try:
            doc_ref = self._collection.document(chat_id)
            doc = doc_ref.get()
            
            if not doc.exists:
                return Message(content="", role="system", provider=None, model=None)
            
            data = doc.to_dict()
            summary_content = data.get("summary", "")
            return Message(content=summary_content, role="system", provider=None, model=None)
            
        except Exception as e:
            print(f"Error getting summary from Firestore: {e}")
            return Message(content="", role="system", provider=None, model=None)

    def save_summary(self, chat_id: str, summary: str) -> None:
        """Save chat summary."""
        try:
            doc_ref = self._collection.document(chat_id)
            doc_ref.set({
                "summary": summary,
                "last_updated": datetime.datetime.utcnow().isoformat() + 'Z'
            }, merge=True)
            
        except Exception as e:
            print(f"Error saving summary to Firestore: {e}")
    
    def get_all_chat_ids(self):
        """Get all chat IDs currently stored.
        
        Returns
        -------
        list
            List of all chat IDs
        """
        try:
            docs = self._collection.stream()
            return [doc.id for doc in docs]
        except Exception as e:
            print(f"Error getting chat IDs from Firestore: {e}")
            return []
    
    def get_chat_count(self):
        """Get the total number of chats stored.
        
        Returns
        -------
        int
            Total number of chats
        """
        try:
            docs = self._collection.stream()
            return len(list(docs))
        except Exception as e:
            print(f"Error getting chat count from Firestore: {e}")
            return 0
    
    def search_messages(self, query: str, limit: int = 10):
        """Search for messages containing specific text.
        
        Parameters
        ----------
        query : str
            Text to search for in messages
        limit : int, optional
            Maximum number of results to return (default: 10)
            
        Returns
        -------
        list
            List of matching messages with chat_id and message details
        """
        try:
            # Note: Firestore doesn't support full-text search natively
            # This is a simple substring search implementation
            # For production use, consider using Algolia or similar search service
            
            results = []
            docs = self._collection.stream()
            
            for doc in docs:
                data = doc.to_dict()
                messages = data.get("messages", [])
                
                for msg in messages:
                    if query.lower() in msg.get("content", "").lower():
                        results.append({
                            "chat_id": doc.id,
                            "message": msg,
                            "timestamp": msg.get("timestamp")
                        })
                        
                        if len(results) >= limit:
                            break
                
                if len(results) >= limit:
                    break
            
            return results
            
        except Exception as e:
            print(f"Error searching messages in Firestore: {e}")
            return []
    
    def get_chat_metadata(self, chat_id: str):
        """Get metadata about a specific chat.
        
        Parameters
        ----------
        chat_id : str
            Unique identifier for the chat
            
        Returns
        -------
        dict
            Dictionary containing chat metadata
        """
        try:
            doc_ref = self._collection.document(chat_id)
            doc = doc_ref.get()
            
            if not doc.exists:
                return None
            
            data = doc.to_dict()
            return {
                "chat_id": chat_id,
                "message_count": data.get("message_count", 0),
                "last_updated": data.get("last_updated"),
                "created_at": data.get("created_at")
            }
            
        except Exception as e:
            print(f"Error getting chat metadata from Firestore: {e}")
            return None
    
    def close(self):
        """Close Firestore client connection."""
        if self._client:
            self._client.close()
    
    def __enter__(self):
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        self.close()
