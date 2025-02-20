# RAG Document Parser

A comprehensive document processing system for Retrieval-Augmented Generation (RAG) applications.

## Features

### Document Processing
- **Multi-format support**: PDF, DOCX, TXT, images (JPG, PNG, TIFF)
- **OCR capabilities**: Extract text from image-based documents using Tesseract
- **Web scraping**: Process content from URLs with BeautifulSoup and Selenium
- **Intelligent parsing**: Automatic format detection and optimized extraction

### Text Chunking
- **Multiple strategies**: Fixed-size, sentence-based, paragraph-based chunking
- **Semantic chunking**: AI-powered chunking based on semantic similarity
- **Configurable overlap**: Customizable chunk size and overlap settings
- **Metadata preservation**: Rich metadata attached to each chunk

### Vector Database Integration
- **ChromaDB support**: Efficient vector storage and similarity search
- **Multiple embeddings**: OpenAI, Sentence Transformers, or custom providers
- **Scalable indexing**: Handle large document collections efficiently
- **Advanced search**: Semantic search with filtering and ranking

### AI Integration
- **LLM providers**: OpenAI GPT and Anthropic Claude integration
- **Semantic analysis**: Automated topic extraction, sentiment analysis
- **Document summarization**: AI-powered extractive and abstractive summaries
- **Entity extraction**: Named entity recognition and classification

### Web API
- **FastAPI interface**: RESTful API for document processing and search
- **Async processing**: Background document processing with status tracking
- **CORS support**: Cross-origin requests for web applications
- **Health monitoring**: Built-in health checks and performance metrics

### Monitoring & Analytics
- **Performance tracking**: Real-time performance metrics and statistics
- **Error monitoring**: Comprehensive error tracking and reporting
- **Health checks**: System health monitoring with alerting
- **Export capabilities**: Metrics export for external monitoring systems

## Quick Start

### Installation

```bash
pip install ragdocparser
```

### Basic Usage

```python
from ragdocparser import DocumentParser, TextChunker, DocumentVectorStore

# Parse a document
parser = DocumentParser()
document = parser.parse("path/to/document.pdf")

# Chunk the text
chunker = TextChunker(strategy="sentence", chunk_size=1000)
chunks = chunker.chunk_document(document)

# Store in vector database
vector_store = DocumentVectorStore()
vector_store.add_document(document, chunker)

# Search for similar content
results = vector_store.search_documents("your search query")
```

### Command Line Interface

```bash
# Parse a document
ragdocparser parse document.pdf --output parsed.json

# Index documents
ragdocparser index ./documents --collection my_docs

# Search indexed documents
ragdocparser search "machine learning" --collection my_docs
```

### Web API

```bash
# Start the API server
uvicorn ragdocparser.api:app --host 0.0.0.0 --port 8000

# Or use Docker
docker-compose up
```

## Configuration

Create a `.env` file with your API keys and configuration:

```env
OPENAI_API_KEY=your_openai_api_key
ANTHROPIC_API_KEY=your_anthropic_api_key
CHROMA_PERSIST_DIRECTORY=./chroma_db
TESSERACT_PATH=/usr/bin/tesseract
LOG_LEVEL=INFO
```

## Docker Deployment

```bash
# Build and run with Docker Compose
docker-compose up --build

# Access the API at http://localhost:8000
# Access the web interface at http://localhost:3000
```

## Advanced Features

### Semantic Analysis

```python
from ragdocparser.ai_integration import SemanticAnalyzer

analyzer = SemanticAnalyzer()
analysis = analyzer.analyze_document(document)

print(f"Summary: {analysis.summary}")
print(f"Key topics: {analysis.key_topics}")
print(f"Sentiment: {analysis.sentiment_score}")
```

### Performance Monitoring

```python
from ragdocparser.monitoring import performance_monitor

# Get performance statistics
stats = performance_monitor.get_statistics(hours=24)
print(f"Documents processed: {stats['counters']['documents_processed']}")

# Get health status
health = performance_monitor.get_health_status()
print(f"System status: {health['status']}")
```

### Web Scraping

```python
from ragdocparser.scraper import URLDocumentParser

url_parser = URLDocumentParser()
web_document = url_parser.parse("https://example.com/article")
```

## API Documentation

When running the web API, visit `http://localhost:8000/docs` for interactive API documentation.

## Requirements

- Python 3.8+
- See `requirements.txt` for detailed dependencies

### Optional Dependencies

- **OCR**: `tesseract` system package
- **Web scraping**: Chrome/Chromium for Selenium
- **AI features**: OpenAI or Anthropic API keys

## License

MIT License - see LICENSE file for details.

## Contributing

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Add tests
5. Submit a pull request

## Support

For issues and questions:
- GitHub Issues: [Report bugs or request features](https://github.com/RyanSStephens/RAGDocParser/issues)
- Documentation: See `/docs` directory for detailed documentation

## Changelog

### v1.5.0 (2025-02-20)
- Added comprehensive AI integration and semantic analysis
- Implemented performance monitoring and analytics
- Enhanced Docker support with health checks
- Added comprehensive test suite
- Improved error handling and logging
- Production-ready deployment configuration

### v1.0.0 (2024-03-15)
- Initial stable release
- FastAPI web interface
- Complete document processing pipeline
- Vector database integration
- Multi-format document support

### v0.3.0 (2024-02-10)
- Added LLM provider integrations
- Implemented document integration service
- Enhanced chunking strategies

### v0.2.0 (2023-12-19)
- Added CLI interface
- Vector database support with ChromaDB
- Text chunking functionality

### v0.1.0 (2023-12-15)
- Initial release
- Basic document parsing
- Configuration management
