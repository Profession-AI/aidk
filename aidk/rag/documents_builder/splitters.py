from typing import List, Callable, Optional
from abc import ABC, abstractmethod


class Chunker(ABC):
    """
    Abstract base class for text chunking strategies.
    
    Different chunking strategies split text in different ways to optimize
    for various use cases.
    """
    
    def __init__(
        self,
        chunk_size: int = 1000,
        chunk_overlap: int = 0
    ):
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        self._validate()
    
    def _validate(self) -> None:
        """Validate parameters."""
        if self.chunk_overlap >= self.chunk_size:
            raise ValueError(
                f"chunk_overlap ({self.chunk_overlap}) must be less than "
                f"chunk_size ({self.chunk_size})"
            )
        
        if self.chunk_size <= 0:
            raise ValueError(f"chunk_size must be positive, got {self.chunk_size}")
        
        if self.chunk_overlap < 0:
            raise ValueError(f"chunk_overlap must be non-negative, got {self.chunk_overlap}")
    
    @abstractmethod
    def split(self, text: str) -> List[str]:
        """Split text according to the chunking strategy."""
        pass


class WordChunker(Chunker):
    """Split text by word boundaries."""
    
    def split(self, text: str) -> List[str]:
        """Split by word boundaries."""
        words = text.split()
        if len(words) <= self.chunk_size:
            return [text]
        
        chunks = []
        start = 0
        
        while start < len(words):
            end = start + self.chunk_size
            chunk = ' '.join(words[start:end])
            
            if chunk.strip():
                chunks.append(chunk)
            
            new_start = end - self.chunk_overlap
            if new_start <= start:
                new_start = start + 1
            start = new_start
        
        return chunks


class SentenceChunker(Chunker):
    """Split text by sentence boundaries."""
    
    def split(self, text: str) -> List[str]:
        """Split by sentence boundaries."""
        sentence_endings = ['.', '!', '?', '\n\n']
        sentences = []
        last_pos = 0
        
        for i, char in enumerate(text):
            if char in sentence_endings:
                sentence = text[last_pos:i+1].strip()
                if sentence:
                    sentences.append(sentence)
                last_pos = i + 1
        
        if last_pos < len(text):
            last_sentence = text[last_pos:].strip()
            if last_sentence:
                sentences.append(last_sentence)
        
        if len(sentences) <= self.chunk_size:
            return [text]
        
        chunks = []
        start = 0
        
        while start < len(sentences):
            end = start + self.chunk_size
            chunk = ' '.join(sentences[start:end])
            
            if chunk.strip():
                chunks.append(chunk)
            
            new_start = end - self.chunk_overlap
            if new_start <= start:
                new_start = start + 1
            start = new_start
        
        return chunks


class ParagraphChunker(Chunker):
    """Split text by paragraph boundaries."""
    
    def split(self, text: str) -> List[str]:
        """Split by paragraph boundaries."""
        paragraphs = [p.strip() for p in text.split('\n\n') if p.strip()]
        
        if len(paragraphs) <= self.chunk_size:
            return [text]
        
        chunks = []
        start = 0
        
        while start < len(paragraphs):
            end = start + self.chunk_size
            chunk = '\n\n'.join(paragraphs[start:end])
            
            if chunk.strip():
                chunks.append(chunk)
            
            new_start = end - self.chunk_overlap
            if new_start <= start:
                new_start = start + 1
            start = new_start
        
        return chunks


class FixedChunker(Chunker):
    """Split text into fixed-size character chunks."""
    
    def split(self, text: str) -> List[str]:
        """Split into fixed-size chunks."""
        chunks = []
        start = 0
        
        while start < len(text):
            end = start + self.chunk_size
            chunk = text[start:end].strip()
            
            if chunk:
                chunks.append(chunk)
            
            new_start = end - self.chunk_overlap
            if new_start <= start:
                new_start = start + 1
            start = new_start
        
        return chunks


class SemanticChunker(Chunker):
    """Split text by semantic boundaries (headers, lists, etc.)."""
    
    def split(self, text: str) -> List[str]:
        """Split by semantic boundaries (headers, lists, etc.)."""
        semantic_patterns = [
            '\n# ', '\n## ', '\n### ', '\n#### ',
            '\n1. ', '\n2. ', '\n3. ', '\n4. ', '\n5. ',
            '\n• ', '\n- ', '\n* ',
            '\n\n',
            '\n---\n', '\n___\n',
            '\n\nChapter ', '\n\nSection ', '\n\nPart ',
        ]
        
        chunks = []
        current_chunk = ""
        
        parts = [text]
        for pattern in semantic_patterns:
            new_parts = []
            for part in parts:
                if pattern in part:
                    split_parts = part.split(pattern)
                    for i, split_part in enumerate(split_parts):
                        if i > 0:
                            split_part = pattern + split_part
                        if split_part.strip():
                            new_parts.append(split_part)
                else:
                    new_parts.append(part)
            parts = new_parts
        
        for part in parts:
            if len(current_chunk) + len(part) > self.chunk_size and current_chunk:
                chunks.append(current_chunk.strip())
                overlap_start = max(0, len(current_chunk) - self.chunk_overlap)
                current_chunk = current_chunk[overlap_start:] + part
            else:
                current_chunk += part
        
        if current_chunk.strip():
            chunks.append(current_chunk.strip())
        
        return chunks


class CustomChunker(Chunker):
    """Split text using a custom splitting function."""
    
    def __init__(
        self,
        chunk_size: int = 1000,
        chunk_overlap: int = 0,
        custom_func: Callable = None
    ):
        if custom_func is None:
            raise ValueError("custom_func must be provided for CustomChunker")
        if not callable(custom_func):
            raise ValueError("custom_func must be callable")
        
        self.custom_func = custom_func
        super().__init__(chunk_size, chunk_overlap)
    
    def split(self, text: str) -> List[str]:
        """Split using the custom function."""
        return self.custom_func(text, self.chunk_size, self.chunk_overlap)


class TextSplitter:
    """
    Factory class for creating chunkers.
    
    Backward compatible interface that maintains the original API while
    using the new Chunker-based architecture.
    
    Strategies:
    - word: Split by word count
    - sentence: Split by sentence boundaries
    - paragraph: Split by paragraph breaks
    - fixed: Split by fixed character count
    - semantic: Split by semantic markers (headers, lists, etc.)
    - custom: Use custom splitting function
    """
    
    def __init__(
        self,
        strategy: str = "word",
        chunk_size: int = 1000,
        chunk_overlap: int = 0,
        custom_func: Optional[Callable] = None
    ):
        self.strategy = strategy
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        self.custom_func = custom_func
        
        self._chunker = self._create_chunker()
    
    def _create_chunker(self) -> Chunker:
        """Create the appropriate chunker based on strategy."""
        strategy_map = {
            "word": WordChunker,
            "sentence": SentenceChunker,
            "paragraph": ParagraphChunker,
            "fixed": FixedChunker,
            "semantic": SemanticChunker,
            "custom": CustomChunker,
        }
        
        if self.strategy not in strategy_map:
            raise ValueError(f"Unsupported strategy: {self.strategy}")
        
        chunker_class = strategy_map[self.strategy]
        
        if self.strategy == "custom":
            return chunker_class(
                chunk_size=self.chunk_size,
                chunk_overlap=self.chunk_overlap,
                custom_func=self.custom_func
            )
        else:
            return chunker_class(
                chunk_size=self.chunk_size,
                chunk_overlap=self.chunk_overlap
            )
    
    def split(self, text: str) -> List[str]:
        """Split text using the configured strategy."""
        if len(text) <= self.chunk_size:
            return [text]
        
        return self._chunker.split(text)
