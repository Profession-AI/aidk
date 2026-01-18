from typing import List, Dict, Any, Optional
from .base import BaseVectorDB, DocumentRetrieved
from aidk.rag.documents_builder.base import Document


class ChromaVectorDB(BaseVectorDB):
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

    def add(self, documents: List[Document]) -> None:
        """
        Add documents to the ChromaDB collection.
        
        This method adds Document objects to the ChromaDB collection.
        The documents are automatically converted to embeddings using the
        configured embedding model.
        
        Parameters:
        -----------
        documents : List[Document]
            List of Document objects to add to the database.
            Each Document contains content, metadata, and a unique ID.
            
        Examples:
        ---------
        ```python
        from aidk.rag.documents_builder import DocumentsBuilder, WordChunker, Document
        
        # Using Document objects from DocumentsBuilder
        chunker = WordChunker(chunk_size=1000, chunk_overlap=100)
        builder = DocumentsBuilder(chunker=chunker)
        docs = builder.from_file("document.txt")
        
        # Add to vector database
        vector_db.add(docs)
        
        # Or create Document objects manually
        custom_docs = [
            Document(
                content="Machine learning content",
                metadata={"topic": "ml", "source": "textbook"},
                doc_id="doc_001"
            )
        ]
        vector_db.add(custom_docs)
        ```
        """
        doc_contents = [doc.content for doc in documents]
        doc_metadatas = [doc.metadata for doc in documents]
        doc_ids = [doc.doc_id for doc in documents]
        
        if not (len(doc_contents) == len(doc_metadatas) == len(doc_ids)):
            raise ValueError("All documents must have consistent structure")
            
        self._collection.add(
            documents=doc_contents,
            metadatas=doc_metadatas,
            ids=doc_ids
        )

    def query(self, query: str, top_k: int = 10) -> List[DocumentRetrieved]:
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
            
        top_k : int, default=10
            Number of most similar documents to return. Higher values
            return more results but may include less relevant documents.
            
        Returns:
        --------
        List[DocumentRetrieved]
            List of DocumentRetrieved objects containing the retrieved documents,
            their metadata, IDs, and similarity distances.
            
        Examples:
        ---------
        ```python
        # Basic search
        results = vector_db.query("What is machine learning?", top_k=5)
        
        # Access results
        for doc in results:
            print(f"Content: {doc.content}")
            print(f"Distance: {doc.distance}")
            print(f"ID: {doc.doc_id}")
            print(f"Metadata: {doc.metadata}")
        ```
        
        Notes:
        ------
        - Results are returned in order of similarity (most similar first)
        - Distance scores are cosine distances (0 = identical, 2 = opposite)
        - If fewer than top_k documents exist, all available documents are returned
        - The query is automatically embedded using the same model as stored documents
        """
        results = self._collection.query(
            query_texts=query,
            n_results=top_k
        )
        
        retrieved_docs = []
        if results and results['ids'] and len(results['ids']) > 0:
            for i, doc_id in enumerate(results['ids'][0]):
                retrieved_docs.append(
                    DocumentRetrieved(
                        content=results['documents'][0][i],
                        metadata=results['metadatas'][0][i],
                        doc_id=doc_id,
                        distance=results['distances'][0][i]
                    )
                )
        
        return retrieved_docs
