import pytest
import tempfile
import os
from aidk.rag.documents_builder import DocumentsBuilder


def test_custom_split_function():
    """Test that custom split functions work correctly."""
    
    def custom_split_by_lines(text, chunk_size, chunk_overlap):
        """Custom split function that splits by lines."""
        lines = text.split('\n')
        chunks = []
        for i in range(0, len(lines), chunk_size - chunk_overlap):
            chunk_lines = lines[i:i + chunk_size]
            if chunk_lines:  # Only add non-empty chunks
                chunks.append('\n'.join(chunk_lines))
        return chunks
    
    # Create test text with multiple lines
    test_text = "Line 1\nLine 2\nLine 3\nLine 4\nLine 5\nLine 6\nLine 7\nLine 8"
    
    # Create DocumentsBuilder with custom split function
    builder = DocumentsBuilder(
        chunk_strategy="custom",
        chunk_size=3,  # 3 lines per chunk
        chunk_overlap=1,  # 1 line overlap
        custom_split_func=custom_split_by_lines
    )
    
    # Test the custom split
    chunks = builder._split_text(test_text)
    
    # Expected chunks: ["Line 1\nLine 2\nLine 3", "Line 3\nLine 4\nLine 5", "Line 5\nLine 6\nLine 7", "Line 7\nLine 8"]
    expected_chunks = [
        "Line 1\nLine 2\nLine 3",
        "Line 3\nLine 4\nLine 5", 
        "Line 5\nLine 6\nLine 7",
        "Line 7\nLine 8"
    ]
    
    assert chunks == expected_chunks


def test_custom_split_with_from_str():
    """Test that custom split works with from_str method."""
    
    def custom_split_by_sections(text, chunk_size, chunk_overlap):
        """Custom split function that splits by section markers."""
        sections = text.split('---')
        chunks = []
        for i in range(0, len(sections), chunk_size - chunk_overlap):
            chunk_sections = sections[i:i + chunk_size]
            if chunk_sections:  # Only add non-empty chunks
                chunks.append('---'.join(chunk_sections))
        return chunks
    
    test_text = "Section 1---Section 2---Section 3---Section 4---Section 5"
    
    builder = DocumentsBuilder(
        chunk_strategy="custom",
        chunk_size=2,  # 2 sections per chunk
        chunk_overlap=0,  # No overlap
        custom_split_func=custom_split_by_sections
    )
    
    chunks, metadata, ids = builder.from_str(test_text)
    
    expected_chunks = [
        "Section 1---Section 2",
        "Section 3---Section 4", 
        "Section 5"
    ]
    
    assert chunks == expected_chunks
    assert len(metadata) == len(chunks)
    assert len(ids) == len(chunks)


def test_custom_split_validation():
    """Test validation of custom split function parameters."""
    
    # Test that custom_split_func is required when strategy is "custom"
    with pytest.raises(ValueError, match="custom_split_func must be provided"):
        DocumentsBuilder(chunk_strategy="custom")
    
    # Test that custom_split_func must be callable
    with pytest.raises(ValueError, match="custom_split_func must be callable"):
        DocumentsBuilder(
            chunk_strategy="custom",
            custom_split_func="not a function"
        )
    
    # Test that custom_split_func can be None for other strategies
    builder = DocumentsBuilder(chunk_strategy="word", custom_split_func=None)
    assert builder._custom_split_func is None


def test_custom_split_function_signature():
    """Test that custom split function receives correct parameters."""
    
    def test_split_func(text, chunk_size, chunk_overlap):
        """Test function that records received parameters."""
        test_split_func.called = True
        test_split_func.text = text
        test_split_func.chunk_size = chunk_size
        test_split_func.chunk_overlap = chunk_overlap
        return [text]  # Return single chunk
    
    test_split_func.called = False
    
    builder = DocumentsBuilder(
        chunk_strategy="custom",
        chunk_size=100,
        chunk_overlap=20,
        custom_split_func=test_split_func
    )
    
    test_text = "This is a test text"
    builder._split_text(test_text)
    
    assert test_split_func.called
    assert test_split_func.text == test_text
    assert test_split_func.chunk_size == 100
    assert test_split_func.chunk_overlap == 20


def test_auto_custom_strategy():
    """Test that chunk_strategy is automatically set to 'custom' when custom_split_func is provided."""
    
    def dummy_split_func(text, chunk_size, chunk_overlap):
        return [text]
    
    # Test with explicit "word" strategy but custom function
    builder = DocumentsBuilder(
        chunk_strategy="word",
        chunk_size=100,
        chunk_overlap=10,
        custom_split_func=dummy_split_func
    )
    
    # Strategy should be automatically set to "custom"
    assert builder._chunk_strategy == "custom"
    
    # Test with no strategy specified but custom function
    builder2 = DocumentsBuilder(
        chunk_size=100,
        chunk_overlap=10,
        custom_split_func=dummy_split_func
    )
    
    # Strategy should be automatically set to "custom"
    assert builder2._chunk_strategy == "custom"
    
    # Test that normal strategies work when no custom function is provided
    builder3 = DocumentsBuilder(chunk_strategy="word")
    assert builder3._chunk_strategy == "word"


if __name__ == "__main__":
    pytest.main([__file__]) 