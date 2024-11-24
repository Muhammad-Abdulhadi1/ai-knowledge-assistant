from typing import Dict, List, Optional
import os
from pathlib import Path
from langchain_community.document_loaders import (
    PyPDFLoader,
    TextLoader,
    UnstructuredMarkdownLoader,
    DirectoryLoader
)
from langchain.text_splitter import RecursiveCharacterTextSplitter
from datetime import datetime

from main import vectorstore

class Processor:
    """A processor that ingests various types of documents into a vector store for later retrieval."""
    
    SUPPORTED_EXTENSIONS = {
        '.pdf': PyPDFLoader,
        '.txt': TextLoader,
        '.md': UnstructuredMarkdownLoader
    }
    
    def __init__(self):
        self.text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=1000,
            chunk_overlap=200
        )
        
    def process_document(self, file_path: str, document_type: Optional[str] = None) -> bool:
        """Process a single document file into the vector store."""
        try:
            path = Path(file_path)
            if not path.exists():
                raise FileNotFoundError(f"File not found: {file_path}")
            
            # Get file extension and appropriate loader
            ext = path.suffix.lower()
            if ext not in self.SUPPORTED_EXTENSIONS:
                raise ValueError(f"Unsupported file type: {ext}. Supported types: {list(self.SUPPORTED_EXTENSIONS.keys())}")
            
            print(f"Processing {path.name}...")
            
            loader_class = self.SUPPORTED_EXTENSIONS[ext]
            loader = loader_class(str(path))
            
            # Load and process document
            documents = loader.load()
            
            # Add metadata
            for doc in documents:
                doc.metadata.update({
                    'source': str(path),
                    'filename': path.name,
                    'type': document_type or 'general',
                    'date_processed': datetime.now().isoformat(),
                })
            
            # Split documents into chunks
            splits = self.text_splitter.split_documents(documents)
            
            # Add to vector store
            vectorstore.add_documents(splits)
            print(f"Successfully processed {path.name} ({len(splits)} chunks created)")
            return True
            
        except Exception as e:
            print(f"Error processing document {file_path}: {str(e)}")
            return False

    def process_directory(self, directory_path: str, document_type: Optional[str] = None) -> Dict[str, int]:
        """Process all supported documents in a directory."""
        stats = {
            'total': 0,
            'successful': 0,
            'failed': 0
        }
        
        try:
            path = Path(directory_path)
            if not path.exists() or not path.is_dir():
                raise NotADirectoryError(f"Directory not found: {directory_path}")
            
            # Process each supported file in the directory
            for ext, _ in self.SUPPORTED_EXTENSIONS.items():
                files = list(path.glob(f"**/*{ext}"))
                stats['total'] += len(files)
                
                for file_path in files:
                    if self.process_document(str(file_path), document_type):
                        stats['successful'] += 1
                    else:
                        stats['failed'] += 1
                        
        except Exception as e:
            print(f"Error processing directory {directory_path}: {str(e)}")
            
        return stats

def main():
    processor = Processor()
    
    while True:
        print("\n=== Knowledge Base ===")
        print("1. Process single document")
        print("2. Process directory")
        print("3. Exit")
        
        choice = input("\nEnter your choice (1-3): ")
        
        if choice == "1":
            file_path = input("Enter the path to your document: ")
            doc_type = input("Enter document type (press Enter for 'general'): ")
            if processor.process_document(file_path, doc_type or None):
                print("Document processed successfully!")
            
        elif choice == "2":
            dir_path = input("Enter the directory path: ")
            doc_type = input("Enter document type (press Enter for 'general'): ")
            stats = processor.process_directory(dir_path, doc_type or None)
            print(f"\nProcessing complete:")
            print(f"Total files: {stats['total']}")
            print(f"Successfully processed: {stats['successful']}")
            print(f"Failed: {stats['failed']}")
            
        elif choice == "3":
            break
            
        else:
            print("Invalid choice. Please try again.")

if __name__ == "__main__":
    main()
