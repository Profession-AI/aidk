from typing import List, Dict, Tuple, Optional

from .splitters import Chunker
from .extractors import FileExtractor, StringExtractor, DocExtractor, PDFExtractor, URLExtractor
from .base import Document


class DocumentsBuilder:
    """
    Main class for building document collections from various sources.
    
    This class provides a high-level API for extracting text from different
    sources (files, URLs, strings) and splitting them into chunks for
    vector database storage using a configured Chunker instance.
    
    Features:
    - File-based document extraction with UTF-8 encoding support
    - Text string processing for in-memory content
    - Web scraping with multiple engine options (requests, tavily, selenium)
    - Word document extraction (.doc and .docx formats)
    - PDF document extraction with metadata
    - Multiple chunking strategies via Chunker instances
    - Rich metadata generation for each document chunk
    - Unique ID generation for database storage
    
    Examples:
    --------
    ```python
    from aidk.rag.documents_builder import DocumentsBuilder, WordChunker
    
    # Initialize chunker with desired configuration
    chunker = WordChunker(chunk_size=1000, chunk_overlap=100)
    
    # Initialize builder with the chunker
    builder = DocumentsBuilder(chunker=chunker)
    
    # Extract from file
    docs = builder.from_file("document.txt")
    
    # Extract from URL
    docs = builder.from_url("https://example.com")
    
    # Extract from PDF with custom page range
    docs = builder.from_pdf("document.pdf", page_range=(1, 10))
    ```
    """
    
    def __init__(self, chunker: Chunker):
        """
        Initialize the DocumentsBuilder with a configured Chunker.
        
        Parameters:
        -----------
        chunker : Chunker
            A Chunker instance (WordChunker, SentenceChunker, etc.) already configured
            with the desired chunk_size and chunk_overlap.
        """
        self.chunker = chunker
    
    def from_file(self, file_path: str) -> List[Document]:
        """
        Read a file and split it into chunks.
        
        Parameters:
        -----------
        file_path : str
            Path to the file to extract
            
        Returns:
        --------
        List[Document]
            List of Document objects
        """
        extractor = FileExtractor(file_path, self.chunker)
        return extractor.extract()
    
    def from_str(self, text: str, source_name: str = "text_string") -> List[Document]:
        """
        Process a text string and split it into chunks.
        
        Parameters:
        -----------
        text : str
            Text to process
        source_name : str
            Name of the text source for metadata
            
        Returns:
        --------
        List[Document]
            List of Document objects
        """
        extractor = StringExtractor(text, source_name, self.chunker)
        return extractor.extract()
    
    def from_doc(self, file_path: str, extraction_method: str = "auto") -> List[Document]:
        """
        Extract text from Word documents (.doc and .docx files).
        
        Parameters:
        -----------
        file_path : str
            Path to the Word document
        extraction_method : str, optional
            Method to use: "auto", "docx", or "docx2txt"
            
        Returns:
        --------
        List[Document]
            List of Document objects
        """
        extractor = DocExtractor(file_path, extraction_method, self.chunker)
        return extractor.extract()
    
    def from_pdf(self, file_path: str, page_range: Optional[Tuple[int, int]] = None) -> List[Document]:
        """
        Extract text from PDF documents.
        
        Parameters:
        -----------
        file_path : str
            Path to the PDF file
        page_range : Optional[Tuple[int, int]]
            Optional (start_page, end_page) range to extract
            
        Returns:
        --------
        List[Document]
            List of Document objects
        """
        extractor = PDFExtractor(file_path, page_range, self.chunker)
        return extractor.extract()
    
    def from_url(self, url: str, engine: str = "requests", deep: bool = False) -> List[Document]:
        """
        Scrape content from a URL and split it into chunks.
        
        Parameters:
        -----------
        url : str
            URL to scrape
        engine : str, optional
            Scraping engine: "requests", "tavily", or "selenium"
        deep : bool, optional
            Whether to perform deep extraction
            
        Returns:
        --------
        List[Document]
            List of Document objects
        """
        extractor = URLExtractor(url, engine, deep, self.chunker)
        return extractor.extract()
