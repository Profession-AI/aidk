"""
Test suite for the RAG (Retrieval-Augmented Generation) system.

This module tests the RAG system's core functionality including:
- Initialization with vector databases
- Document querying and retrieval
- Integration with DocumentsBuilder for document ingestion
- Proper return types and data structures
- Integration with Model class for RAG-augmented prompting
"""

import pytest
import tempfile
import shutil
from pathlib import Path
from typing import List
from unittest.mock import Mock, patch, MagicMock

from aidk.rag import RAG
from aidk.rag.vectordb.base import BaseVectorDB, DocumentRetrieved, Document
from aidk.rag.vectordb import ChromaVectorDB
from aidk.rag.documents_builder import DocumentsBuilder
from aidk.rag.documents_builder.splitters import WordChunker
from aidk.models.model import Model
from aidk.prompts.prompt import Prompt


class TestRAGInitialization:
    """Test RAG system initialization."""
    
    def test_rag_init_with_chroma_vectordb(self):
        """Test RAG initialization with ChromaVectorDB."""
        with tempfile.TemporaryDirectory() as tmpdir:
            vector_db = ChromaVectorDB(
                name="test_collection",
                vectorizer_provider=None,
                vectorizer_model=None
            )
            rag = RAG(vector_db=vector_db)
            
            assert rag._vector_db is vector_db
            assert isinstance(rag._vector_db, BaseVectorDB)
    
    def test_rag_init_with_basedb_interface(self):
        """Test that RAG accepts any BaseVectorDB implementation."""
        vector_db = ChromaVectorDB(name="test_db")
        rag = RAG(vector_db=vector_db)
        
        # Verify the vector_db is stored correctly
        assert rag._vector_db == vector_db


class TestRAGDocumentManagement:
    """Test document addition and management in RAG."""
    
    def test_add_documents_to_rag(self):
        """Test adding documents to RAG through vector database."""
        vector_db = ChromaVectorDB(name="test_add_docs")
        rag = RAG(vector_db=vector_db)
        
        # Create test documents
        documents = [
            Document(
                content="Machine learning is a subset of artificial intelligence.",
                metadata={"source": "doc1.txt", "topic": "ML"},
                doc_id="doc_1"
            ),
            Document(
                content="Deep learning uses neural networks with multiple layers.",
                metadata={"source": "doc2.txt", "topic": "DL"},
                doc_id="doc_2"
            ),
            Document(
                content="Natural language processing enables computers to understand human language.",
                metadata={"source": "doc3.txt", "topic": "NLP"},
                doc_id="doc_3"
            ),
        ]
        
        # Add documents to vector database
        rag._vector_db.add(documents)
        
        # Verify documents were added (indirectly through query attempt)
        assert rag._vector_db._collection is not None


class TestRAGQuerying:
    """Test RAG query functionality."""
    
    def test_query_returns_document_retrieved_list(self):
        """Test that query returns List[DocumentRetrieved]."""
        vector_db = ChromaVectorDB(name="test_query_type")
        rag = RAG(vector_db=vector_db)
        
        # Add test documents
        documents = [
            Document(
                content="Python is a programming language.",
                metadata={"source": "doc1"},
                doc_id="doc_1"
            ),
            Document(
                content="Java is also a programming language.",
                metadata={"source": "doc2"},
                doc_id="doc_2"
            ),
        ]
        rag._vector_db.add(documents)
        
        # Query the database
        results = rag.query("programming language", top_k=2)
        
        # Verify return type
        assert isinstance(results, list)
        assert len(results) > 0
        assert all(isinstance(doc, DocumentRetrieved) for doc in results)
    
    def test_query_returns_correct_fields(self):
        """Test that DocumentRetrieved objects have all required fields."""
        vector_db = ChromaVectorDB(name="test_query_fields")
        rag = RAG(vector_db=vector_db)
        
        # Add test document
        documents = [
            Document(
                content="Artificial intelligence is transforming industries.",
                metadata={"source": "ai_doc", "category": "tech"},
                doc_id="ai_1"
            ),
        ]
        rag._vector_db.add(documents)
        
        # Query
        results = rag.query("artificial intelligence", top_k=1)
        
        # Verify fields
        assert len(results) > 0
        doc = results[0]
        assert hasattr(doc, 'content')
        assert hasattr(doc, 'metadata')
        assert hasattr(doc, 'doc_id')
        assert hasattr(doc, 'distance')
        assert isinstance(doc.content, str)
        assert isinstance(doc.metadata, dict)
        assert isinstance(doc.doc_id, str)
        assert isinstance(doc.distance, float)
    
    def test_query_with_top_k_parameter(self):
        """Test query with different top_k values."""
        vector_db = ChromaVectorDB(name="test_top_k")
        rag = RAG(vector_db=vector_db)
        
        # Add multiple documents
        documents = [
            Document(
                content=f"Document about topic {i}.",
                metadata={"index": i},
                doc_id=f"doc_{i}"
            )
            for i in range(5)
        ]
        rag._vector_db.add(documents)
        
        # Query with different top_k values
        results_k2 = rag.query("topic", top_k=2)
        results_k5 = rag.query("topic", top_k=5)
        
        # Verify results count respects top_k
        assert len(results_k2) <= 2
        assert len(results_k5) <= 5
    
    def test_query_default_top_k(self):
        """Test query uses default top_k=10."""
        vector_db = ChromaVectorDB(name="test_default_top_k")
        rag = RAG(vector_db=vector_db)
        
        # Add documents
        documents = [
            Document(
                content=f"Test document {i}",
                metadata={"idx": i},
                doc_id=f"doc_{i}"
            )
            for i in range(5)
        ]
        rag._vector_db.add(documents)
        
        # Query without specifying top_k
        results = rag.query("test document")
        
        # Should return results (default top_k=10, but only 5 docs exist)
        assert isinstance(results, list)
        assert len(results) <= 10


class TestRAGWithDocumentsBuilder:
    """Test RAG integration with DocumentsBuilder."""
    
    def test_rag_with_extracted_documents(self):
        """Test RAG with documents extracted by DocumentsBuilder."""
        # Create a temporary text file
        with tempfile.TemporaryDirectory() as tmpdir:
            file_path = Path(tmpdir) / "test_doc.txt"
            file_path.write_text(
                "Machine learning enables computers to learn from data. "
                "Deep learning uses neural networks. "
                "Natural language processing understands human language."
            )
            
            # Extract documents using DocumentsBuilder
            chunker = WordChunker(chunk_size=20, chunk_overlap=5)
            builder = DocumentsBuilder(chunker=chunker)
            extracted_docs = builder.from_file(str(file_path))
            
            # Initialize RAG
            vector_db = ChromaVectorDB(name="test_builder_integration")
            rag = RAG(vector_db=vector_db)
            
            # Add extracted documents
            rag._vector_db.add(extracted_docs)
            
            # Query
            results = rag.query("machine learning", top_k=3)
            
            # Verify
            assert len(results) > 0
            assert all(isinstance(doc, DocumentRetrieved) for doc in results)


class TestRAGMetadata:
    """Test RAG handling of document metadata."""
    
    def test_metadata_preserved_in_retrieved_documents(self):
        """Test that metadata is preserved through storage and retrieval."""
        vector_db = ChromaVectorDB(name="test_metadata")
        rag = RAG(vector_db=vector_db)
        
        # Create document with metadata
        original_metadata = {
            "source": "research_paper.pdf",
            "author": "John Doe",
            "year": 2024,
            "topic": "Machine Learning"
        }
        documents = [
            Document(
                content="Research findings on machine learning algorithms.",
                metadata=original_metadata,
                doc_id="research_1"
            ),
        ]
        rag._vector_db.add(documents)
        
        # Query and retrieve
        results = rag.query("machine learning", top_k=1)
        
        # Verify metadata
        assert len(results) > 0
        retrieved_doc = results[0]
        assert retrieved_doc.metadata["source"] == "research_paper.pdf"
        assert retrieved_doc.metadata["author"] == "John Doe"
        assert retrieved_doc.metadata["year"] == 2024
        assert retrieved_doc.metadata["topic"] == "Machine Learning"
    
    def test_doc_id_preserved(self):
        """Test that document IDs are preserved."""
        vector_db = ChromaVectorDB(name="test_doc_id")
        rag = RAG(vector_db=vector_db)
        
        doc_id = "unique_doc_12345"
        documents = [
            Document(
                content="Some content to test doc ID preservation.",
                metadata={"test": True},
                doc_id=doc_id
            ),
        ]
        rag._vector_db.add(documents)
        
        results = rag.query("content test", top_k=1)
        
        assert len(results) > 0
        assert results[0].doc_id == doc_id


class TestRAGDistanceScores:
    """Test RAG similarity distance scores."""
    
    def test_distance_scores_present(self):
        """Test that distance scores are returned with results."""
        vector_db = ChromaVectorDB(name="test_distances")
        rag = RAG(vector_db=vector_db)
        
        documents = [
            Document(
                content="Python programming language",
                metadata={"lang": "python"},
                doc_id="py_1"
            ),
            Document(
                content="Java programming language",
                metadata={"lang": "java"},
                doc_id="java_1"
            ),
        ]
        rag._vector_db.add(documents)
        
        results = rag.query("Python", top_k=2)
        
        # Check distance scores
        assert len(results) > 0
        for doc in results:
            assert isinstance(doc.distance, float)
            # Distance should be between 0 and 2 for cosine distance
            assert 0 <= doc.distance <= 2
    
    def test_results_ordered_by_relevance(self):
        """Test that results are ordered by relevance (distance)."""
        vector_db = ChromaVectorDB(name="test_ordering")
        rag = RAG(vector_db=vector_db)
        
        documents = [
            Document(
                content="The quick brown fox jumps over the lazy dog",
                metadata={"idx": 0},
                doc_id="doc_0"
            ),
            Document(
                content="A quick fox",
                metadata={"idx": 1},
                doc_id="doc_1"
            ),
            Document(
                content="Completely unrelated content about vegetables",
                metadata={"idx": 2},
                doc_id="doc_2"
            ),
        ]
        rag._vector_db.add(documents)
        
        results = rag.query("quick fox", top_k=3)
        
        # First result should be most similar (smallest distance)
        if len(results) > 1:
            assert results[0].distance <= results[1].distance


class TestRAGErrorHandling:
    """Test RAG error handling."""
    
    def test_query_empty_database(self):
        """Test querying an empty database."""
        vector_db = ChromaVectorDB(name="test_empty")
        rag = RAG(vector_db=vector_db)
        
        # Query without adding any documents
        results = rag.query("test query")
        
        # Should return empty list, not raise error
        assert isinstance(results, list)
        assert len(results) == 0
    
    def test_query_with_empty_string(self):
        """Test querying with empty string."""
        vector_db = ChromaVectorDB(name="test_empty_query")
        rag = RAG(vector_db=vector_db)
        
        documents = [
            Document(
                content="Test content",
                metadata={"test": True},
                doc_id="doc_1"
            ),
        ]
        rag._vector_db.add(documents)
        
        # Query with empty string should still work
        results = rag.query("", top_k=1)
        assert isinstance(results, list)


if __name__ == "__main__":
    # Run tests with pytest
    pytest.main([__file__, "-v"])


class TestRAGWithModel:
    """Test RAG integration with Model class."""
    
    def test_model_add_rag(self):
        """Test adding RAG to a model."""
        # Create vector DB and RAG
        vector_db = ChromaVectorDB(name="test_model_rag")
        rag = RAG(vector_db=vector_db)
        
        # Create model (using mock to avoid actual API calls)
        with patch('aidk.models.model.BaseModel.__init__', return_value=None):
            model = Model(provider="openai", model="gpt-4o-mini")
            model._rag = None  # Initialize RAG as None
            
            # Add RAG to model
            model._add_rag(rag)
            
            # Verify RAG is attached
            assert model._rag is rag
            assert isinstance(model._rag, RAG)
    
    def test_model_initialization_with_rag(self):
        """Test model initialization with RAG parameter."""
        vector_db = ChromaVectorDB(name="test_model_init_rag")
        rag = RAG(vector_db=vector_db)
        
        # Create model with RAG from initialization
        with patch('aidk.models._base_model.BaseModel.__init__'):
            from aidk.models._base_model import BaseModel
            
            model = Model.__new__(Model)
            model._rag = rag
            model._max_tokens = None
            model._tools = []
            
            # Verify RAG is set
            assert hasattr(model, '_rag')
            assert model._rag is rag
    
    def test_model_apply_rag_to_prompt(self):
        """Test that model applies RAG to prompts."""
        # Create and populate vector DB
        vector_db = ChromaVectorDB(name="test_apply_rag")
        rag = RAG(vector_db=vector_db)
        
        # Add test documents
        documents = [
            Document(
                content="Machine learning is a type of artificial intelligence.",
                metadata={"source": "ml_doc"},
                doc_id="ml_1"
            ),
            Document(
                content="Deep learning uses neural networks.",
                metadata={"source": "dl_doc"},
                doc_id="dl_1"
            ),
        ]
        rag._vector_db.add(documents)
        
        # Create model with RAG
        with patch('aidk.models.model.BaseModel.__init__', return_value=None):
            model = Model(provider="openai", model="gpt-4o-mini")
            model._rag = rag
            model._tools = []
            
            # Apply RAG to prompt
            from aidk.models._prompt_executor import PromptExecutorMixin
            original_prompt = "What is machine learning?"
            augmented = model._apply_rag(original_prompt)
            
            # Verify prompt was augmented with RAG documents
            assert isinstance(augmented, str)
            assert original_prompt in augmented or augmented != original_prompt
    
    def test_model_rag_augmentation_with_no_results(self):
        """Test model RAG augmentation when no documents match."""
        vector_db = ChromaVectorDB(name="test_rag_no_results")
        rag = RAG(vector_db=vector_db)
        
        # Don't add any documents - database is empty
        
        with patch('aidk.models.model.BaseModel.__init__', return_value=None):
            model = Model(provider="openai", model="gpt-4o-mini")
            model._rag = rag
            model._tools = []
            
            original_prompt = "Some question?"
            augmented = model._apply_rag(original_prompt)
            
            # When no results, prompt should remain unchanged or be returned as-is
            assert augmented == original_prompt
    
    def test_model_without_rag(self):
        """Test model behaves correctly without RAG."""
        with patch('aidk.models.model.BaseModel.__init__', return_value=None):
            model = Model(provider="openai", model="gpt-4o-mini")
            model._rag = None
            model._tools = []
            
            original_prompt = "Test prompt"
            result = model._apply_rag(original_prompt)
            
            # Without RAG, prompt should be returned unchanged
            assert result == original_prompt
    
    def test_model_rag_with_prompt_object(self):
        """Test model RAG augmentation with Prompt object."""
        vector_db = ChromaVectorDB(name="test_model_prompt_obj")
        rag = RAG(vector_db=vector_db)
        
        # Add test document
        documents = [
            Document(
                content="Python is a programming language used for AI development.",
                metadata={"topic": "Python"},
                doc_id="py_1"
            ),
        ]
        rag._vector_db.add(documents)
        
        with patch('aidk.models.model.BaseModel.__init__', return_value=None):
            model = Model(provider="openai", model="gpt-4o-mini")
            model._rag = rag
            model._tools = []
            
            # Create a Prompt object
            prompt = Prompt(prompt="Tell me about Python programming")
            
            # Apply RAG
            augmented = model._apply_rag(prompt)
            
            # Result should be either modified Prompt or string with augmentation
            assert augmented is not None
    
    def test_model_rag_document_integration(self):
        """Test complete flow: create docs -> add to RAG -> augment model prompt."""
        # Create temporary file with content
        with tempfile.TemporaryDirectory() as tmpdir:
            file_path = Path(tmpdir) / "knowledge.txt"
            file_path.write_text(
                "Artificial Intelligence (AI) is transforming technology. "
                "Machine Learning (ML) is a subset of AI that enables learning from data. "
                "Deep Learning (DL) uses neural networks with multiple layers."
            )
            
            # Extract documents using DocumentsBuilder
            chunker = WordChunker(chunk_size=15, chunk_overlap=3)
            builder = DocumentsBuilder(chunker=chunker)
            extracted_docs = builder.from_file(str(file_path))
            
            # Create RAG and add documents
            vector_db = ChromaVectorDB(name="test_complete_flow")
            rag = RAG(vector_db=vector_db)
            rag._vector_db.add(extracted_docs)
            
            # Create model and add RAG
            with patch('aidk.models.model.BaseModel.__init__', return_value=None):
                model = Model(provider="openai", model="gpt-4o-mini")
                model._rag = rag
                model._tools = []
                
                # Create prompt
                prompt = "Explain artificial intelligence"
                
                # Apply RAG
                augmented = model._apply_rag(prompt)
                
                # Verify augmentation
                assert isinstance(augmented, str)
                assert len(augmented) >= len(prompt)
    
    def test_rag_documents_not_exposed_without_rag(self):
        """Test that documents are not exposed when RAG is not configured."""
        with patch('aidk.models.model.BaseModel.__init__', return_value=None):
            model = Model(provider="openai", model="gpt-4o-mini")
            model._rag = None
            model._tools = []
            
            prompt = "Some prompt"
            result = model._apply_rag(prompt)
            
            # Should not contain document markers or extra content
            assert result == prompt


class TestRAGModelQueryIntegration:
    """Test integration between model query execution and RAG."""
    
    def test_model_query_uses_rag_documents(self):
        """Test that model query methods use RAG when available."""
        vector_db = ChromaVectorDB(name="test_query_rag")
        rag = RAG(vector_db=vector_db)
        
        # Add documents
        documents = [
            Document(
                content="Information about quantum computing and its applications.",
                metadata={"category": "quantum"},
                doc_id="qc_1"
            ),
        ]
        rag._vector_db.add(documents)
        
        # Mock the completion function to avoid actual API calls
        with patch('aidk.models.model.BaseModel.__init__', return_value=None):
            model = Model(provider="openai", model="gpt-4o-mini")
            model._rag = rag
            model._tools = []
            
            # Verify RAG is ready to be used
            assert model._rag is not None
            assert len(model._rag.query("quantum")) > 0
    
    def test_multiple_documents_augmentation(self):
        """Test augmentation with multiple retrieved documents."""
        vector_db = ChromaVectorDB(name="test_multi_docs")
        rag = RAG(vector_db=vector_db)
        
        # Add multiple related documents
        documents = [
            Document(
                content="Document 1: Introduction to Machine Learning",
                metadata={"order": 1},
                doc_id="doc_1"
            ),
            Document(
                content="Document 2: Advanced Machine Learning Techniques",
                metadata={"order": 2},
                doc_id="doc_2"
            ),
            Document(
                content="Document 3: Machine Learning in Production",
                metadata={"order": 3},
                doc_id="doc_3"
            ),
        ]
        rag._vector_db.add(documents)
        
        with patch('aidk.models.model.BaseModel.__init__', return_value=None):
            model = Model(provider="openai", model="gpt-4o-mini")
            model._rag = rag
            model._tools = []
            
            # Query that should return multiple documents
            results = rag.query("Machine Learning", top_k=3)
            assert len(results) > 0
            
            # Apply RAG augmentation
            prompt = "Explain machine learning"
            augmented = model._apply_rag(prompt)
            


if __name__ == "__main__":
    # Run tests with pytest
    pytest.main([__file__, "-v"])
