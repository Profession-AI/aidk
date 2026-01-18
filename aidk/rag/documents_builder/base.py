import uuid
from typing import List, Dict, Tuple
from dataclasses import dataclass


@dataclass
class Document:
    """Represents a document chunk with metadata and ID."""
    content: str
    metadata: Dict
    doc_id: str


class BaseExtractor:
    """Base class for all extractors."""
    
    def extract(self) -> List[Document]:
        """Extract documents as Document dataclass objects."""
        raise NotImplementedError
    
    @staticmethod
    def _generate_chunks_with_metadata(
        text: str,
        chunks: List[str],
        base_metadata: Dict
    ) -> List[Document]:
        """Generate Document objects from chunks with metadata."""
        documents = []
        
        for i, chunk in enumerate(chunks):
            chunk_id = str(uuid.uuid4())
            metadata = {
                **base_metadata,
                'chunk_index': i,
                'total_chunks': len(chunks),
                'chunk_size': len(chunk)
            }
            doc = Document(
                content=chunk,
                metadata=metadata,
                doc_id=chunk_id
            )
            documents.append(doc)
        
        return documents
