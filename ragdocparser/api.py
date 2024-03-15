"""
FastAPI web interface for RAG Document Parser.
"""

from fastapi import FastAPI, File, UploadFile, HTTPException, BackgroundTasks
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from pydantic import BaseModel
from typing import List, Optional, Dict, Any
import tempfile
import os
import logging
from pathlib import Path

from .integration import DocumentProcessor

logger = logging.getLogger(__name__)

# Pydantic models
class SearchRequest(BaseModel):
    query: str
    n_results: int = 5

class SearchResult(BaseModel):
    content: str
    metadata: Dict[str, Any]
    distance: Optional[float] = None
    id: Optional[str] = None

class ProcessingStatus(BaseModel):
    status: str
    message: str
    chunks: int = 0

# Initialize FastAPI app
app = FastAPI(
    title="RAG Document Parser API",
    description="API for processing documents and performing semantic search",
    version="1.0.0"
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Global document processor
processor = None

@app.on_event("startup")
async def startup_event():
    """Initialize document processor on startup."""
    global processor
    try:
        processor = DocumentProcessor()
        logger.info("Document processor initialized successfully")
    except Exception as e:
        logger.error(f"Failed to initialize document processor: {e}")
        processor = None

@app.get("/")
async def root():
    """Root endpoint."""
    return {"message": "RAG Document Parser API", "version": "1.0.0"}

@app.get("/health")
async def health_check():
    """Health check endpoint."""
    if processor is None:
        raise HTTPException(status_code=503, detail="Document processor not initialized")
    
    try:
        stats = processor.get_statistics()
        return {
            "status": "healthy",
            "processor_ready": True,
            "statistics": stats
        }
    except Exception as e:
        logger.error(f"Health check failed: {e}")
        raise HTTPException(status_code=503, detail=f"Health check failed: {str(e)}")

@app.post("/upload", response_model=ProcessingStatus)
async def upload_document(background_tasks: BackgroundTasks, file: UploadFile = File(...)):
    """Upload and process a document."""
    if processor is None:
        raise HTTPException(status_code=503, detail="Document processor not initialized")
    
    # Check file type
    allowed_extensions = {'.pdf', '.txt', '.docx', '.jpg', '.jpeg', '.png', '.tiff'}
    file_extension = Path(file.filename).suffix.lower()
    
    if file_extension not in allowed_extensions:
        raise HTTPException(
            status_code=400, 
            detail=f"Unsupported file type: {file_extension}. Allowed: {allowed_extensions}"
        )
    
    try:
        # Save uploaded file temporarily
        with tempfile.NamedTemporaryFile(delete=False, suffix=file_extension) as tmp_file:
            content = await file.read()
            tmp_file.write(content)
            tmp_file_path = tmp_file.name
        
        # Process document in background
        def process_file():
            try:
                result = processor.process_document(tmp_file_path)
                # Clean up temp file
                os.unlink(tmp_file_path)
                logger.info(f"Processed {file.filename}: {result.get('chunks', 0)} chunks")
            except Exception as e:
                logger.error(f"Background processing failed for {file.filename}: {e}")
                os.unlink(tmp_file_path)
        
        background_tasks.add_task(process_file)
        
        return ProcessingStatus(
            status="accepted",
            message=f"File {file.filename} accepted for processing",
            chunks=0
        )
        
    except Exception as e:
        logger.error(f"Error uploading file {file.filename}: {e}")
        raise HTTPException(status_code=500, detail=f"Error processing file: {str(e)}")

@app.post("/search", response_model=List[SearchResult])
async def search_documents(request: SearchRequest):
    """Search processed documents."""
    if processor is None:
        raise HTTPException(status_code=503, detail="Document processor not initialized")
    
    try:
        results = processor.search_documents(request.query, request.n_results)
        
        search_results = []
        for result in results:
            search_results.append(SearchResult(
                content=result['content'],
                metadata=result['metadata'],
                distance=result.get('distance'),
                id=result.get('id')
            ))
        
        return search_results
        
    except Exception as e:
        logger.error(f"Search error: {e}")
        raise HTTPException(status_code=500, detail=f"Search failed: {str(e)}")

@app.get("/stats")
async def get_statistics():
    """Get processing statistics."""
    if processor is None:
        raise HTTPException(status_code=503, detail="Document processor not initialized")
    
    try:
        return processor.get_statistics()
    except Exception as e:
        logger.error(f"Error getting statistics: {e}")
        raise HTTPException(status_code=500, detail=f"Error getting statistics: {str(e)}")

@app.post("/qa")
async def question_answering(request: SearchRequest):
    """Question answering endpoint using search results."""
    if processor is None:
        raise HTTPException(status_code=503, detail="Document processor not initialized")
    
    try:
        # Search for relevant documents
        search_results = processor.search_documents(request.query, request.n_results)
        
        if not search_results:
            return {
                "answer": "No relevant documents found for your question.",
                "sources": []
            }
        
        # Combine search results for context
        context_parts = []
        sources = []
        
        for result in search_results:
            context_parts.append(result['content'])
            sources.append({
                "file": result['metadata'].get('file_path', 'unknown'),
                "page": result['metadata'].get('page_number', 'unknown'),
                "distance": result.get('distance')
            })
        
        context = "\n\n".join(context_parts)
        
        # For now, return the most relevant chunk as answer
        # In a real implementation, you'd use an LLM to generate an answer
        answer = search_results[0]['content'][:500] + "..." if len(search_results[0]['content']) > 500 else search_results[0]['content']
        
        return {
            "answer": answer,
            "sources": sources,
            "context_used": len(context_parts)
        }
        
    except Exception as e:
        logger.error(f"Q&A error: {e}")
        raise HTTPException(status_code=500, detail=f"Q&A failed: {str(e)}")

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
