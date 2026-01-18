from aidk.rag.vectordb.base import BaseVectorDB, DocumentRetrieved
from typing import List

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

    def __init__(self, vector_db: BaseVectorDB):
        """
        Initialize the RAG system.
        
        Parameters:
        -----------
        vector_db : BaseVectorDB
            An initialized vector database instance that implements the BaseVectorDB interface.
            This should be pre-configured with the desired collection name, embedding provider,
            and embedding model.
            
        Examples:
        ---------
        ```python
        # Initialize with ChromaDB backend
        from aidk.rag.vectordb import ChromaVectorDB
        
        vector_db = ChromaVectorDB(
            name="my_documents",
            vectorizer_provider="openai",
            vectorizer_model="text-embedding-ada-002"
        )
        rag = RAG(vector_db=vector_db)
        
        # Or using a different backend
        vector_db = SomeOtherVectorDB(...)  # Any BaseVectorDB implementation
        rag = RAG(vector_db=vector_db)
        ```
        """
        self._vector_db = vector_db
        

    def query(self, query: str, top_k: int = 10) -> List[DocumentRetrieved]:
        """
        Perform a semantic search query against the vector database.
        
        This method converts the input query into a vector embedding and searches
        the database for the most semantically similar documents.
        
        Parameters:
        -----------
        query : str
            The text query to search for. This will be converted to a vector
            embedding and used to find similar documents.
            
        top_k : int, default=10
            The number of most relevant documents to return. Higher values
            return more results but may include less relevant documents.
            
        Returns:
        --------
        List[DocumentRetrieved]
            A list of DocumentRetrieved objects, each containing:
            - content: The document text
            - metadata: Dictionary of document metadata
            - doc_id: Unique document identifier
            - distance: Similarity score (lower = more similar, 0 = identical)
            
        Examples:
        ---------
        ```python
        # Basic query
        results = rag.query("What is artificial intelligence?")
        for doc in results:
            print(f"ID: {doc.doc_id}")
            print(f"Content: {doc.content}")
            print(f"Similarity: {doc.distance}")
            print(f"Metadata: {doc.metadata}")
        
        # Query with more results
        results = rag.query("Machine learning algorithms", top_k=20)
        ```
        
        Notes:
        ------
        - Results are returned in order of relevance (most similar first)
        - Distance scores are cosine distances (0 = identical, 2 = completely opposite)
        - If fewer than top_k documents exist in the database, all available documents are returned
        """
        return self._vector_db.query(query, top_k)





