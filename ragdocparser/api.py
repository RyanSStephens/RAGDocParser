"""
FastAPI web interface for RAGDocParser.
Provides REST API endpoints for document processing and question answering.
"""

from fastapi import FastAPI, File, UploadFile, HTTPException, BackgroundTasks
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field
from typing import List, Dict, Any, Optional
import asyncio
import tempfile
import os
from pathlib import Path
import logging

from .integration import RAGDocumentProcessor, create_processor
from .config import RAGConfig

logger = logging.getLogger(__name__)

# Pydantic models for API
class QuestionRequest(BaseModel):
    question: str = Field(..., description="Question to ask")
    collection_name: str = Field("documents", description="Collection to search in")
    k: int = Field(5, ge=1, le=20, description="Number of results to retrieve")

class QuestionResponse(BaseModel):
    answer: str
    confidence: float
    sources: List[Dict[str, Any]]
    question: str
    collection: str

class ProcessDocumentsRequest(BaseModel):
    collection_name: str = Field("documents", description="Collection name for storage")
    use_ocr: bool = Field(True, description="Whether to use OCR for image-based documents")

class ProcessingStatus(BaseModel):
    task_id: str
    status: str  # "pending", "processing", "completed", "failed"
    progress: Dict[str, Any]

class CollectionInfo(BaseModel):
    name: str
    document_count: int
    chunk_count: int
    created_at: Optional[str] = None

# Global processor instance
processor: Optional[RAGDocumentProcessor] = None
processing_tasks: Dict[str, Dict] = {}

def create_app(config_path: str = None) -> FastAPI:
    """Create and configure FastAPI application."""
    
    app = FastAPI(
        title="RAGDocParser API",
        description="REST API for document processing and RAG-based question answering",
        version="1.5.0",
        docs_url="/docs",
        redoc_url="/redoc"
    )
    
    # Enable CORS
    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],  # Configure appropriately for production
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )
    
    # Initialize processor
    @app.on_event("startup")
    async def startup_event():
        global processor
        try:
            processor = create_processor(config_path)
            logger.info("RAGDocumentProcessor initialized successfully")
        except Exception as e:
            logger.error(f"Failed to initialize processor: {e}")
            processor = None
    
    @app.on_event("shutdown")
    async def shutdown_event():
        global processor
        if processor:
            processor.cleanup()
            logger.info("RAGDocumentProcessor cleaned up")
    
    return app

app = create_app()

@app.get("/", response_model=Dict[str, str])
async def root():
    """Root endpoint with API information."""
    return {
        "message": "RAGDocParser API",
        "version": "1.0.0",
        "docs": "/docs",
        "status": "online" if processor else "offline"
    }

@app.get("/health", response_model=Dict[str, str])
async def health_check():
    """Health check endpoint."""
    if not processor:
        raise HTTPException(status_code=503, detail="Processor not initialized")
    
    return {
        "status": "healthy",
        "processor": "online",
        "collections": str(len(processor.get_collections_info()))
    }

@app.post("/ask", response_model=QuestionResponse)
async def ask_question(request: QuestionRequest):
    """Ask a question against the document collection."""
    if not processor:
        raise HTTPException(status_code=503, detail="Processor not initialized")
    
    if not processor.rag_manager:
        raise HTTPException(status_code=400, detail="LLM provider not configured")
    
    try:
        result = processor.ask_question(
            request.question,
            request.collection_name,
            request.k
        )
        
        return QuestionResponse(
            answer=result["answer"],
            confidence=result.get("confidence", 0.0),
            sources=result.get("sources", []),
            question=request.question,
            collection=request.collection_name
        )
        
    except Exception as e:
        logger.error(f"Error processing question: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/upload", response_model=Dict[str, str])
async def upload_documents(
    background_tasks: BackgroundTasks,
    files: List[UploadFile] = File(...),
    collection_name: str = "documents",
    use_ocr: bool = True
):
    """Upload and process documents."""
    if not processor:
        raise HTTPException(status_code=503, detail="Processor not initialized")
    
    if not files:
        raise HTTPException(status_code=400, detail="No files provided")
    
    # Generate task ID
    import uuid
    task_id = str(uuid.uuid4())
    
    # Save files to temporary directory
    temp_dir = tempfile.mkdtemp()
    file_paths = []
    
    try:
        for file in files:
            if file.filename:
                file_path = Path(temp_dir) / file.filename
                with open(file_path, "wb") as f:
                    content = await file.read()
                    f.write(content)
                file_paths.append(file_path)
        
        # Start background processing
        background_tasks.add_task(
            process_documents_task,
            task_id,
            file_paths,
            collection_name,
            use_ocr,
            temp_dir
        )
        
        processing_tasks[task_id] = {
            "status": "pending",
            "files": len(file_paths),
            "collection": collection_name
        }
        
        return {"task_id": task_id, "status": "processing_started"}
        
    except Exception as e:
        # Cleanup on error
        import shutil
        shutil.rmtree(temp_dir, ignore_errors=True)
        logger.error(f"Error uploading files: {e}")
        raise HTTPException(status_code=500, detail=str(e))

async def process_documents_task(
    task_id: str,
    file_paths: List[Path],
    collection_name: str,
    use_ocr: bool,
    temp_dir: str
):
    """Background task for processing documents."""
    try:
        processing_tasks[task_id]["status"] = "processing"
        
        # Process documents
        results = processor.process_documents(
            file_paths,
            collection_name,
            use_ocr
        )
        
        processing_tasks[task_id].update({
            "status": "completed",
            "results": results
        })
        
    except Exception as e:
        processing_tasks[task_id].update({
            "status": "failed",
            "error": str(e)
        })
        logger.error(f"Processing task {task_id} failed: {e}")
    
    finally:
        # Cleanup temporary directory
        import shutil
        shutil.rmtree(temp_dir, ignore_errors=True)

@app.get("/tasks/{task_id}", response_model=ProcessingStatus)
async def get_task_status(task_id: str):
    """Get the status of a processing task."""
    if task_id not in processing_tasks:
        raise HTTPException(status_code=404, detail="Task not found")
    
    task_info = processing_tasks[task_id]
    
    return ProcessingStatus(
        task_id=task_id,
        status=task_info["status"],
        progress=task_info
    )

@app.get("/collections", response_model=List[CollectionInfo])
async def list_collections():
    """List all available collections."""
    if not processor:
        raise HTTPException(status_code=503, detail="Processor not initialized")
    
    try:
        collections = processor.get_collections_info()
        return [
            CollectionInfo(
                name=col.get("name", "unknown"),
                document_count=col.get("document_count", 0),
                chunk_count=col.get("count", 0),
                created_at=col.get("created_at")
            )
            for col in collections
        ]
    except Exception as e:
        logger.error(f"Error listing collections: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.delete("/collections/{collection_name}")
async def delete_collection(collection_name: str):
    """Delete a collection."""
    if not processor:
        raise HTTPException(status_code=503, detail="Processor not initialized")
    
    try:
        # This would need to be implemented in the vector database manager
        # processor.vectordb.delete_collection(collection_name)
        return {"message": f"Collection {collection_name} deletion requested"}
    except Exception as e:
        logger.error(f"Error deleting collection: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/search/{collection_name}")
async def search_collection(
    collection_name: str,
    query: str,
    k: int = 5
):
    """Search within a specific collection."""
    if not processor:
        raise HTTPException(status_code=503, detail="Processor not initialized")
    
    try:
        results = processor.vectordb.search_similar(
            query,
            k=k,
            collection_name=collection_name
        )
        
        return {
            "query": query,
            "collection": collection_name,
            "results": results
        }
        
    except Exception as e:
        logger.error(f"Error searching collection: {e}")
        raise HTTPException(status_code=500, detail=str(e))

if __name__ == "__main__":
    import uvicorn
    
    # Configure logging
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    
    uvicorn.run(
        "ragdocparser.api:app",
        host="0.0.0.0",
        port=8000,
        reload=True,
        log_level="info"
    ) 