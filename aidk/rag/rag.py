from litellm import embedding
from aidk.keys.keys_manager import load_key
from aidk.rag.vectordb import ChromaVectorDB
from typing import List, Dict, Any, Optional

class RAG:
    """
    Retrieval-Augmented Generation (RAG) system for semantic search and document retrieval.
    
    This class provides a high-level interface for performing semantic search queries
    against a vector database. It supports multiple vector database backends and
    embedding providers for flexible deployment scenarios.
    
    The RAG system works by:
    1. Converting text queries into vector embeddings
    2. Searching the vector database for similar document embeddings
    3. Returning the most relevant documents based on semantic similarity
    
    Attributes:
        _vectorizer (str): The embedding model used for vectorization
        _db (str): Name of the vector database
        _vector_db (ChromaVectorDB): The vector database backend
    
    Examples:
    --------
     Basic usage with default settings:
        
    ```python
    # Initialize RAG with a database name
    rag = RAG(database="my_documents")
        
    # Perform a semantic search
    results = rag.query("What is machine learning?", k=5)
    ```
        
    Using with specific embedding provider:
        
    ```python
    # Initialize with OpenAI embeddings
    rag = RAG(
        database="my_documents",
        provider="openai",
        vectorizer="text-embedding-ada-002"
    )
        
    # Search for relevant documents
    results = rag.query("Explain neural networks", k=10)
    ```
        
    Working with different vector databases:
        
    ```python
    # Currently supports ChromaDB
    rag = RAG(
        database="my_collection",
        vector_db="chroma",
        provider="openai",
        vectorizer="text-embedding-ada-002"
    )
    ```

    Add RAG to a model, so that the model can use the RAG automatically to answer questions:
    ```python
    model = Model(provider="openai", model="gpt-4o-mini")
    model._add_rag(RAG(database="my_documents", vector_db="chroma"))
    ```

    """

    def __init__(self, 
                database: str,
                 provider: Optional[str] = None,
                 vectorizer: Optional[str] = None, 
                 vector_db: str = "chroma"):
        """
        Initialize the RAG system.
        
        Parameters:
        -----------
        database : str
            Name of the vector database/collection to use for storage and retrieval.
            This will be created if it doesn't exist.
            
        provider : str, optional
            The embedding provider to use (e.g., "openai", "anthropic", "cohere").
            If provided, the corresponding API key will be loaded automatically.
            If None, the system will use default embedding settings.
            
        vectorizer : str, optional
            The specific embedding model to use for vectorization.
            Examples: "text-embedding-ada-002", "text-embedding-3-small", "embed-english-v3.0"
            If None, the provider's default model will be used.
            
        vector_db : str, default="chroma"
            The vector database backend to use. Currently supports:
            - "chroma": ChromaDB (default, recommended for most use cases)
            
        Raises:
        -------
        ValueError
            If an unsupported vector database is specified.
            
        Examples:
        ---------
        ```python
        # Minimal initialization
        rag = RAG("my_documents")
        
        # With OpenAI embeddings
        rag = RAG(
            database="research_papers",
            provider="openai",
            vectorizer="text-embedding-ada-002"
        )
        
        # With Anthropic embeddings
        rag = RAG(
            database="articles",
            provider="anthropic",
            vectorizer="text-embedding-3-small"
        )
        ```
        """
        if provider:
            load_key(provider)
            
        self._vectorizer = vectorizer
        self._db = database
        
        if vector_db == "chroma":
            self._vector_db = ChromaVectorDB(
                name=database, 
                vectorizer_provider=provider, 
                vectorizer_model=vectorizer
            )
        else:
            raise ValueError(f"Vector database '{vector_db}' not supported. Currently only 'chroma' is supported.")
        

    def query(self, query: str, k: int = 10) -> Dict[str, Any]:
        """
        Perform a semantic search query against the vector database.
        
        This method converts the input query into a vector embedding and searches
        the database for the most semantically similar documents.
        
        Parameters:
        -----------
        query : str
            The text query to search for. This will be converted to a vector
            embedding and used to find similar documents.
            
        k : int, default=10
            The number of most relevant documents to return. Higher values
            return more results but may include less relevant documents.
            
        Returns:
        --------
        Dict[str, Any]
            A dictionary containing the search results with the following structure:
            {
                'ids': List[List[str]] - Document IDs of the retrieved documents,
                'documents': List[List[str]] - The actual document content,
                'metadatas': List[List[Dict]] - Metadata for each document,
                'distances': List[List[float]] - Similarity scores (lower = more similar)
            }
            
        Examples:
        ---------
        ```python
        # Basic query
        results = rag.query("What is artificial intelligence?")
        
        # Query with more results
        results = rag.query("Machine learning algorithms", k=20)
        
        # Accessing results
        for i, (doc_id, document, metadata, distance) in enumerate(zip(
            results['ids'][0], 
            results['documents'][0], 
            results['metadatas'][0], 
            results['distances'][0]
        )):
            print(f"Result {i+1}:")
            print(f"  ID: {doc_id}")
            print(f"  Content: {document[:100]}...")
            print(f"  Similarity: {1 - distance:.3f}")
            print(f"  Metadata: {metadata}")
            print()
        ```
        
        Notes:
        ------
        - The query is automatically converted to lowercase and processed
        - Results are returned in order of relevance (most similar first)
        - Distance scores are cosine distances (0 = identical, 2 = completely opposite)
        - If fewer than k documents exist in the database, all available documents are returned
        """
        return self._vector_db.query(query, k)





