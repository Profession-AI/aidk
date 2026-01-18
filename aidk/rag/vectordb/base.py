from litellm import embedding
from typing import List, Dict, Optional
from dataclasses import dataclass
from aidk.rag.documents_builder.base import Document


@dataclass(kw_only=True)
class DocumentRetrieved(Document):
    """Represents a document retrieved from a vector database query."""
    distance: float

class BaseVectorDB:
    """
    Abstract base class for vector database implementations.
    
    This class defines the interface that all vector database backends must implement.
    It provides common functionality for document embedding and defines the contract
    for database operations like creation, addition, querying, and deletion.
    
    The base class handles:
    - Configuration of embedding models and providers
    - Document vectorization using the specified embedding model
    - Common interface for all vector database operations
    
    Attributes:
        _name (str): Name of the vector database collection
        _vectorizer_provider (str): The embedding provider (e.g., "openai", "anthropic")
        _vectorizer_model (str): The specific embedding model to use
    
    Note:
        This is an abstract base class. Use concrete implementations like
        ChromaVectorDB for actual vector database operations.
    """

    def __init__(self, name: Optional[str] = None, 
                 vectorizer_provider: Optional[str] = None, 
                 vectorizer_model: Optional[str] = None):
        """
        Initialize the base vector database.
        
        Parameters:
        -----------
        name : str, optional
            Name of the vector database collection. If None, no collection
            is created during initialization.
            
        vectorizer_provider : str, optional
            The embedding provider to use for vectorization.
            Examples: "openai", "anthropic", "cohere"
            
        vectorizer_model : str, optional
            The specific embedding model to use.
            Examples: "text-embedding-ada-002", "text-embedding-3-small"
        """
        self._name = name
        self._vectorizer_provider = vectorizer_provider
        self._vectorizer_model = vectorizer_model

    def _embed(self, documents: List[str]) -> List[List[float]]:
        """
        Convert text documents into vector embeddings.
        
        This method uses the configured embedding model to convert text documents
        into numerical vector representations that can be stored and queried
        in the vector database.
        
        Parameters:
        -----------
        documents : List[str]
            List of text documents to convert into embeddings.
            
        Returns:
        --------
        List[List[float]]
            List of vector embeddings, where each embedding is a list of floats.
            
        Examples:
        ---------
        ```python
        # Convert documents to embeddings
        docs = ["Hello world", "Machine learning is fascinating"]
        embeddings = vector_db._embed(docs)
        print(f"Generated {len(embeddings)} embeddings")
        print(f"Each embedding has {len(embeddings[0])} dimensions")
        ```
        """
        result = embedding(
            model=self._vectorizer_model,
            input=documents
        )
        return result

    def new(self, name: str) -> None:
        """
        Create a new vector database collection.
        
        This method should be implemented by concrete subclasses to create
        a new collection with the specified name.
        
        Parameters:
        -----------
        name : str
            Name of the new collection to create.
            
        Note:
        -----
        This is an abstract method that must be implemented by subclasses.
        """
        pass

    def add(self, documents: List[Document]) -> None:
        """
        Add documents to the vector database.
        
        This method should be implemented by concrete subclasses to add
        Document objects to the vector database.
        
        Parameters:
        -----------
        documents : List[Document]
            List of Document objects to add to the database.
            Each Document contains content, metadata, and a unique ID.
            
        Note:
        -----
        This is an abstract method that must be implemented by subclasses.
        """
        pass

    def query(self, query: str, top_k: int = 10) -> List[DocumentRetrieved]:
        """
        Search for similar documents in the vector database.
        
        This method should be implemented by concrete subclasses to perform
        semantic search queries against the stored documents.
        
        Parameters:
        -----------
        query : str
            The text query to search for.
            
        top_k : int, default=10
            Number of most similar documents to return.
            
        Returns:
        --------
        List[DocumentRetrieved]
            List of DocumentRetrieved objects containing the retrieved documents,
            their metadata, IDs, and similarity distances.
            
        Note:
        -----
        This is an abstract method that must be implemented by subclasses.
        """
        pass
    
    def delete(self) -> None:
        """
        Delete the vector database collection.
        
        This method should be implemented by concrete subclasses to remove
        the entire collection and all its data.
        
        Note:
        -----
        This is an abstract method that must be implemented by subclasses.
        """
        pass
