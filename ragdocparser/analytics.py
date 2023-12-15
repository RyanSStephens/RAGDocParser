"""
Advanced analytics module for RAGDocParser.
Provides comprehensive analytics and insights for RAG system performance.
"""

import logging
from typing import Dict, List, Any, Optional, Tuple
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from collections import defaultdict, Counter
import json
import statistics
from pathlib import Path
import sqlite3
import threading
import uuid

logger = logging.getLogger(__name__)

@dataclass
class QueryAnalytics:
    """Analytics data for a single query."""
    query_id: str
    query_text: str
    timestamp: datetime
    response_time: float
    retrieved_docs: int
    relevance_scores: List[float]
    user_feedback: Optional[str] = None
    collection_name: str = "default"
    embedding_method: str = "default"
    retrieval_strategy: str = "semantic"
    
    @property
    def avg_relevance(self) -> float:
        """Average relevance score."""
        return statistics.mean(self.relevance_scores) if self.relevance_scores else 0.0
    
    @property
    def max_relevance(self) -> float:
        """Maximum relevance score."""
        return max(self.relevance_scores) if self.relevance_scores else 0.0

@dataclass
class DocumentAnalytics:
    """Analytics data for document processing."""
    doc_id: str
    filename: str
    file_size_bytes: int
    processing_time: float
    chunk_count: int
    word_count: int
    language: str
    timestamp: datetime
    collection_name: str = "default"
    
    @property
    def processing_speed_wps(self) -> float:
        """Processing speed in words per second."""
        return self.word_count / self.processing_time if self.processing_time > 0 else 0.0

@dataclass
class UsageStats:
    """Overall system usage statistics."""
    total_queries: int = 0
    total_documents: int = 0
    total_chunks: int = 0
    avg_response_time: float = 0.0
    avg_relevance_score: float = 0.0
    most_common_queries: List[Tuple[str, int]] = field(default_factory=list)
    peak_usage_hours: List[int] = field(default_factory=list)
    collection_stats: Dict[str, Dict[str, Any]] = field(default_factory=dict)
    language_distribution: Dict[str, int] = field(default_factory=dict)

class AnalyticsDatabase:
    """SQLite database for storing analytics data."""
    
    def __init__(self, db_path: Path):
        """Initialize analytics database."""
        self.db_path = db_path
        self.lock = threading.Lock()
        self._init_database()
    
    def _init_database(self):
        """Initialize database tables."""
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()
            
            # Query analytics table
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS query_analytics (
                    query_id TEXT PRIMARY KEY,
                    query_text TEXT,
                    timestamp TEXT,
                    response_time REAL,
                    retrieved_docs INTEGER,
                    relevance_scores TEXT,
                    user_feedback TEXT,
                    collection_name TEXT,
                    embedding_method TEXT,
                    retrieval_strategy TEXT
                )
            """)
            
            # Document analytics table
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS document_analytics (
                    doc_id TEXT PRIMARY KEY,
                    filename TEXT,
                    file_size_bytes INTEGER,
                    processing_time REAL,
                    chunk_count INTEGER,
                    word_count INTEGER,
                    language TEXT,
                    timestamp TEXT,
                    collection_name TEXT
                )
            """)
            
            # Create indexes
            cursor.execute("CREATE INDEX IF NOT EXISTS idx_query_timestamp ON query_analytics(timestamp)")
            cursor.execute("CREATE INDEX IF NOT EXISTS idx_doc_timestamp ON document_analytics(timestamp)")
            cursor.execute("CREATE INDEX IF NOT EXISTS idx_collection ON query_analytics(collection_name)")
            
            conn.commit()
    
    def store_query_analytics(self, analytics: QueryAnalytics):
        """Store query analytics data."""
        with self.lock:
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.cursor()
                cursor.execute("""
                    INSERT OR REPLACE INTO query_analytics 
                    (query_id, query_text, timestamp, response_time, retrieved_docs, 
                     relevance_scores, user_feedback, collection_name, embedding_method, retrieval_strategy)
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """, (
                    analytics.query_id,
                    analytics.query_text,
                    analytics.timestamp.isoformat(),
                    analytics.response_time,
                    analytics.retrieved_docs,
                    json.dumps(analytics.relevance_scores),
                    analytics.user_feedback,
                    analytics.collection_name,
                    analytics.embedding_method,
                    analytics.retrieval_strategy
                ))
                conn.commit()
    
    def store_document_analytics(self, analytics: DocumentAnalytics):
        """Store document analytics data."""
        with self.lock:
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.cursor()
                cursor.execute("""
                    INSERT OR REPLACE INTO document_analytics 
                    (doc_id, filename, file_size_bytes, processing_time, chunk_count, 
                     word_count, language, timestamp, collection_name)
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
                """, (
                    analytics.doc_id,
                    analytics.filename,
                    analytics.file_size_bytes,
                    analytics.processing_time,
                    analytics.chunk_count,
                    analytics.word_count,
                    analytics.language,
                    analytics.timestamp.isoformat(),
                    analytics.collection_name
                ))
                conn.commit()
    
    def get_query_analytics(
        self, 
        start_date: Optional[datetime] = None,
        end_date: Optional[datetime] = None,
        collection_name: Optional[str] = None,
        limit: int = 1000
    ) -> List[QueryAnalytics]:
        """Retrieve query analytics data."""
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()
            
            query = "SELECT * FROM query_analytics WHERE 1=1"
            params = []
            
            if start_date:
                query += " AND timestamp >= ?"
                params.append(start_date.isoformat())
            
            if end_date:
                query += " AND timestamp <= ?"
                params.append(end_date.isoformat())
            
            if collection_name:
                query += " AND collection_name = ?"
                params.append(collection_name)
            
            query += " ORDER BY timestamp DESC LIMIT ?"
            params.append(limit)
            
            cursor.execute(query, params)
            rows = cursor.fetchall()
            
            analytics = []
            for row in rows:
                analytics.append(QueryAnalytics(
                    query_id=row[0],
                    query_text=row[1],
                    timestamp=datetime.fromisoformat(row[2]),
                    response_time=row[3],
                    retrieved_docs=row[4],
                    relevance_scores=json.loads(row[5]) if row[5] else [],
                    user_feedback=row[6],
                    collection_name=row[7],
                    embedding_method=row[8],
                    retrieval_strategy=row[9]
                ))
            
            return analytics

    def get_document_analytics(
        self,
        start_date: Optional[datetime] = None,
        end_date: Optional[datetime] = None,
        collection_name: Optional[str] = None,
        limit: int = 1000
    ) -> List[DocumentAnalytics]:
        """Retrieve document analytics data."""
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()
            
            query = "SELECT * FROM document_analytics WHERE 1=1"
            params = []
            
            if start_date:
                query += " AND timestamp >= ?"
                params.append(start_date.isoformat())
            
            if end_date:
                query += " AND timestamp <= ?"
                params.append(end_date.isoformat())
            
            if collection_name:
                query += " AND collection_name = ?"
                params.append(collection_name)
            
            query += " ORDER BY timestamp DESC LIMIT ?"
            params.append(limit)
            
            cursor.execute(query, params)
            rows = cursor.fetchall()
            
            analytics = []
            for row in rows:
                analytics.append(DocumentAnalytics(
                    doc_id=row[0],
                    filename=row[1],
                    file_size_bytes=row[2],
                    processing_time=row[3],
                    chunk_count=row[4],
                    word_count=row[5],
                    language=row[6],
                    timestamp=datetime.fromisoformat(row[7]),
                    collection_name=row[8]
                ))
            
            return analytics

class AnalyticsEngine:
    """Main analytics engine for RAGDocParser."""
    
    def __init__(self, db_path: Optional[Path] = None):
        """Initialize analytics engine."""
        if db_path is None:
            db_path = Path("analytics.db")
        
        self.database = AnalyticsDatabase(db_path)
        self.session_queries: List[QueryAnalytics] = []
        self.session_documents: List[DocumentAnalytics] = []
    
    def track_query(
        self,
        query_text: str,
        response_time: float,
        retrieved_docs: int,
        relevance_scores: List[float],
        collection_name: str = "default",
        embedding_method: str = "default",
        retrieval_strategy: str = "semantic",
        user_feedback: Optional[str] = None
    ) -> str:
        """Track a query and its performance metrics."""
        query_id = str(uuid.uuid4())
        
        analytics = QueryAnalytics(
            query_id=query_id,
            query_text=query_text,
            timestamp=datetime.now(),
            response_time=response_time,
            retrieved_docs=retrieved_docs,
            relevance_scores=relevance_scores,
            user_feedback=user_feedback,
            collection_name=collection_name,
            embedding_method=embedding_method,
            retrieval_strategy=retrieval_strategy
        )
        
        self.database.store_query_analytics(analytics)
        self.session_queries.append(analytics)
        
        logger.debug(f"Tracked query: {query_id}")
        return query_id
    
    def track_document_processing(
        self,
        filename: str,
        file_size_bytes: int,
        processing_time: float,
        chunk_count: int,
        word_count: int,
        language: str,
        collection_name: str = "default"
    ) -> str:
        """Track document processing metrics."""
        doc_id = str(uuid.uuid4())
        
        analytics = DocumentAnalytics(
            doc_id=doc_id,
            filename=filename,
            file_size_bytes=file_size_bytes,
            processing_time=processing_time,
            chunk_count=chunk_count,
            word_count=word_count,
            language=language,
            timestamp=datetime.now(),
            collection_name=collection_name
        )
        
        self.database.store_document_analytics(analytics)
        self.session_documents.append(analytics)
        
        logger.debug(f"Tracked document processing: {doc_id}")
        return doc_id
    
    def get_usage_stats(
        self,
        start_date: Optional[datetime] = None,
        end_date: Optional[datetime] = None
    ) -> UsageStats:
        """Get comprehensive usage statistics."""
        
        # Get analytics data
        query_analytics = self.database.get_query_analytics(start_date, end_date)
        doc_analytics = self.database.get_document_analytics(start_date, end_date)
        
        if not query_analytics and not doc_analytics:
            return UsageStats()
        
        # Calculate query statistics
        response_times = [q.response_time for q in query_analytics]
        relevance_scores = [score for q in query_analytics for score in q.relevance_scores]
        query_texts = [q.query_text.lower() for q in query_analytics]
        
        # Calculate peak usage hours
        query_hours = [q.timestamp.hour for q in query_analytics]
        hour_counts = Counter(query_hours)
        peak_hours = [hour for hour, count in hour_counts.most_common(3)]
        
        # Collection statistics
        collection_stats = defaultdict(lambda: {"queries": 0, "documents": 0, "avg_response_time": 0.0})
        
        for q in query_analytics:
            collection_stats[q.collection_name]["queries"] += 1
        
        for d in doc_analytics:
            collection_stats[d.collection_name]["documents"] += 1
        
        # Calculate average response times per collection
        collection_response_times = defaultdict(list)
        for q in query_analytics:
            collection_response_times[q.collection_name].append(q.response_time)
        
        for collection, times in collection_response_times.items():
            collection_stats[collection]["avg_response_time"] = statistics.mean(times)
        
        # Language distribution
        language_dist = Counter(d.language for d in doc_analytics)
        
        return UsageStats(
            total_queries=len(query_analytics),
            total_documents=len(doc_analytics),
            total_chunks=sum(d.chunk_count for d in doc_analytics),
            avg_response_time=statistics.mean(response_times) if response_times else 0.0,
            avg_relevance_score=statistics.mean(relevance_scores) if relevance_scores else 0.0,
            most_common_queries=Counter(query_texts).most_common(10),
            peak_usage_hours=peak_hours,
            collection_stats=dict(collection_stats),
            language_distribution=dict(language_dist)
        )
    
    def get_performance_trends(
        self,
        days: int = 30,
        collection_name: Optional[str] = None
    ) -> Dict[str, List[Dict[str, Any]]]:
        """Get performance trends over time."""
        end_date = datetime.now()
        start_date = end_date - timedelta(days=days)
        
        query_analytics = self.database.get_query_analytics(
            start_date, end_date, collection_name
        )
        
        # Group by day
        daily_stats = defaultdict(lambda: {
            "queries": 0,
            "total_response_time": 0.0,
            "relevance_scores": []
        })
        
        for q in query_analytics:
            day_key = q.timestamp.date().isoformat()
            daily_stats[day_key]["queries"] += 1
            daily_stats[day_key]["total_response_time"] += q.response_time
            daily_stats[day_key]["relevance_scores"].extend(q.relevance_scores)
        
        # Calculate trends
        trends = []
        for day_key in sorted(daily_stats.keys()):
            stats = daily_stats[day_key]
            trends.append({
                "date": day_key,
                "query_count": stats["queries"],
                "avg_response_time": stats["total_response_time"] / stats["queries"] if stats["queries"] > 0 else 0,
                "avg_relevance": statistics.mean(stats["relevance_scores"]) if stats["relevance_scores"] else 0
            })
        
        return {"daily_trends": trends}
    
    def get_collection_insights(self, collection_name: str) -> Dict[str, Any]:
        """Get detailed insights for a specific collection."""
        query_analytics = self.database.get_query_analytics(collection_name=collection_name)
        doc_analytics = self.database.get_document_analytics(collection_name=collection_name)
        
        if not query_analytics and not doc_analytics:
            return {"error": f"No data found for collection: {collection_name}"}
        
        # Query patterns
        query_lengths = [len(q.query_text.split()) for q in query_analytics]
        common_words = Counter()
        for q in query_analytics:
            common_words.update(q.query_text.lower().split())
        
        # Document insights
        file_types = Counter(Path(d.filename).suffix.lower() for d in doc_analytics)
        processing_speeds = [d.processing_speed_wps for d in doc_analytics]
        
        return {
            "collection_name": collection_name,
            "query_insights": {
                "total_queries": len(query_analytics),
                "avg_query_length": statistics.mean(query_lengths) if query_lengths else 0,
                "common_query_words": common_words.most_common(20),
                "avg_response_time": statistics.mean([q.response_time for q in query_analytics]) if query_analytics else 0
            },
            "document_insights": {
                "total_documents": len(doc_analytics),
                "file_type_distribution": dict(file_types),
                "avg_processing_speed_wps": statistics.mean(processing_speeds) if processing_speeds else 0,
                "total_chunks": sum(d.chunk_count for d in doc_analytics),
                "language_distribution": dict(Counter(d.language for d in doc_analytics))
            }
        }
    
    def export_analytics_report(self, output_path: Path, format: str = "json") -> None:
        """Export comprehensive analytics report."""
        stats = self.get_usage_stats()
        trends = self.get_performance_trends()
        
        report = {
            "report_generated": datetime.now().isoformat(),
            "usage_statistics": {
                "total_queries": stats.total_queries,
                "total_documents": stats.total_documents,
                "total_chunks": stats.total_chunks,
                "avg_response_time": stats.avg_response_time,
                "avg_relevance_score": stats.avg_relevance_score,
                "collection_stats": stats.collection_stats,
                "language_distribution": stats.language_distribution
            },
            "performance_trends": trends,
            "insights": {
                "most_common_queries": stats.most_common_queries,
                "peak_usage_hours": stats.peak_usage_hours
            }
        }
        
        if format.lower() == "json":
            with open(output_path, 'w') as f:
                json.dump(report, f, indent=2)
        else:
            raise ValueError(f"Unsupported format: {format}")
        
        logger.info(f"Analytics report exported to {output_path}")

# Global analytics engine
_global_analytics: Optional[AnalyticsEngine] = None

def get_analytics_engine() -> AnalyticsEngine:
    """Get or create the global analytics engine."""
    global _global_analytics
    if _global_analytics is None:
        _global_analytics = AnalyticsEngine()
    return _global_analytics

def track_query_performance(
    query_text: str,
    response_time: float,
    retrieved_docs: int,
    relevance_scores: List[float],
    **kwargs
) -> str:
    """Convenience function to track query performance."""
    engine = get_analytics_engine()
    return engine.track_query(
        query_text, response_time, retrieved_docs, relevance_scores, **kwargs
    ) 