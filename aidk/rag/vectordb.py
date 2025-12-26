from litellm import embedding
from typing import List, Dict, Any, Optional, Union

class _BaseVectorDB:
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

    def add(self, documents: List[str], metadatas: Optional[List[Dict]] = None, 
            ids: Optional[List[str]] = None) -> None:
        """
        Add documents to the vector database.
        
        This method should be implemented by concrete subclasses to add
        documents along with their metadata and IDs to the vector database.
        
        Parameters:
        -----------
        documents : List[str]
            List of text documents to add to the database.
            
        metadatas : List[Dict], optional
            List of metadata dictionaries for each document.
            If None, no metadata will be stored.
            
        ids : List[str], optional
            List of unique identifiers for each document.
            If None, IDs will be auto-generated.
            
        Note:
        -----
        This is an abstract method that must be implemented by subclasses.
        """
        pass

    def query(self, query: str, k: int = 10) -> Dict[str, Any]:
        """
        Search for similar documents in the vector database.
        
        This method should be implemented by concrete subclasses to perform
        semantic search queries against the stored documents.
        
        Parameters:
        -----------
        query : str
            The text query to search for.
            
        k : int, default=10
            Number of most similar documents to return.
            
        Returns:
        --------
        Dict[str, Any]
            Search results containing documents, metadata, and similarity scores.
            The exact structure depends on the implementation.
            
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
    

class ChromaVectorDB(_BaseVectorDB):
    """
    ChromaDB implementation of the vector database interface.
    
    This class provides a concrete implementation of the vector database
    using ChromaDB as the backend. ChromaDB is an open-source embedding
    database that supports persistent storage and efficient similarity search.
    
    Features:
    - Persistent storage of document embeddings
    - Efficient similarity search with configurable result count
    - Metadata storage for each document
    - Automatic collection creation if it doesn't exist
    - Support for custom embedding models via LiteLLM
    
    Attributes:
        _client (chromadb.PersistentClient): ChromaDB client instance
        _collection (chromadb.Collection): Active collection for operations
    
    Examples:
    --------
    Basic usage:
    
    ```python
    # Initialize with a new collection
    vector_db = ChromaVectorDB(name="my_documents")
    
    # Add documents
    documents = ["Document 1 content", "Document 2 content"]
    metadatas = [{"source": "file1.txt"}, {"source": "file2.txt"}]
    ids = ["doc1", "doc2"]
    
    vector_db.add(documents, metadatas, ids)
    
    # Search for similar documents
    results = vector_db.query("search query", k=5)
    ```
    
    Using with specific embedding model:
    
    ```python
    # Initialize with OpenAI embeddings
    vector_db = ChromaVectorDB(
        name="research_papers",
        vectorizer_provider="openai",
        vectorizer_model="text-embedding-ada-002"
    )
    ```
    """

    def __init__(self, name: Optional[str] = None, 
                 vectorizer_provider: Optional[str] = None, 
                 vectorizer_model: Optional[str] = None):
        """
        Initialize the ChromaDB vector database.
        
        Parameters:
        -----------
        name : str, optional
            Name of the ChromaDB collection. If provided, the collection
            will be created if it doesn't exist, or connected to if it does.
            
        vectorizer_provider : str, optional
            The embedding provider to use for vectorization.
            Examples: "openai", "anthropic", "cohere"
            
        vectorizer_model : str, optional
            The specific embedding model to use.
            Examples: "text-embedding-ada-002", "text-embedding-3-small"
            
        Examples:
        ---------
        ```python
        # Create new collection
        vector_db = ChromaVectorDB("my_documents")
        
        # Connect to existing collection with custom embeddings
        vector_db = ChromaVectorDB(
            name="existing_collection",
            vectorizer_provider="openai",
            vectorizer_model="text-embedding-ada-002"
        )
        ```
        """
        super().__init__(name, vectorizer_provider, vectorizer_model)
        try:
            import chromadb
        except ImportError:
            raise ImportError("chromadb is not installed. Please install it with 'pip install chromadb'")

        self._client = chromadb.PersistentClient()
        if name:
            try:
                self._collection = self._client.get_collection(name)
            except chromadb.errors.NotFoundError:
                self._collection = self._client.create_collection(name)

    def add(self, documents: List[str], metadatas: List[Dict], ids: List[str]) -> None:
        """
        Add documents to the ChromaDB collection.
        
        This method adds documents along with their metadata and IDs to the
        ChromaDB collection. The documents are automatically converted to
        embeddings using the configured embedding model.
        
        Parameters:
        -----------
        documents : List[str]
            List of text documents to add to the database.
            Each document will be converted to a vector embedding.
            
        metadatas : List[Dict]
            List of metadata dictionaries for each document.
            Each metadata dict can contain any key-value pairs for
            document categorization and filtering.
            
        ids : List[str]
            List of unique identifiers for each document.
            IDs must be unique within the collection.
            
        Raises:
        -------
        ValueError
            If the lengths of documents, metadatas, and ids don't match.
            
        Examples:
        ---------
        ```python
        # Add documents with metadata
        documents = [
            "Machine learning is a subset of artificial intelligence.",
            "Deep learning uses neural networks with multiple layers."
        ]
        
        metadatas = [
            {"topic": "machine_learning", "source": "textbook", "year": 2023},
            {"topic": "deep_learning", "source": "research_paper", "year": 2023}
        ]
        
        ids = ["doc_001", "doc_002"]
        
        vector_db.add(documents, metadatas, ids)
        ```
        
        Notes:
        ------
        - All three lists must have the same length
        - IDs must be unique within the collection
        - Documents are automatically embedded using the configured model
        - Metadata can be used for filtering during queries
        """
        if not (len(documents) == len(metadatas) == len(ids)):
            raise ValueError("documents, metadatas, and ids must have the same length")
            
        self._collection.add(
            documents=documents,
            metadatas=metadatas,
            ids=ids
        )

    def query(self, query: str, k: int = 10) -> Dict[str, Any]:
        """
        Search for similar documents in the ChromaDB collection.
        
        This method performs semantic search by converting the query to an
        embedding and finding the most similar document embeddings in the
        collection.
        
        Parameters:
        -----------
        query : str
            The text query to search for. This will be converted to a
            vector embedding and compared against stored documents.
            
        k : int, default=10
            Number of most similar documents to return. Higher values
            return more results but may include less relevant documents.
            
        Returns:
        --------
        Dict[str, Any]
            A dictionary containing search results with the following structure:
            {
                'ids': List[List[str]] - Document IDs of retrieved documents,
                'documents': List[List[str]] - The actual document content,
                'metadatas': List[List[Dict]] - Metadata for each document,
                'distances': List[List[float]] - Similarity scores (lower = more similar)
            }
            
        Examples:
        ---------
        ```python
        # Basic search
        results = vector_db.query("What is machine learning?", k=5)
        
        # Access results
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
        ```
        
        Notes:
        ------
        - Results are returned in order of similarity (most similar first)
        - Distance scores are cosine distances (0 = identical, 2 = opposite)
        - If fewer than k documents exist, all available documents are returned
        - The query is automatically embedded using the same model as stored documents
        """
        results = self._collection.query(
            query_texts=query,
            n_results=k
        )
        return results
    
    
    
