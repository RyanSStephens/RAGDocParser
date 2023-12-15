"""
Monitoring and metrics module for RAGDocParser.
Tracks performance, usage statistics, and system health.
"""

import time
import threading
from typing import Dict, List, Any, Optional
from dataclasses import dataclass, field
from datetime import datetime, timedelta
import logging
import json
from pathlib import Path
from collections import defaultdict, deque

logger = logging.getLogger(__name__)

@dataclass
class ProcessingMetric:
    """Single processing operation metric."""
    operation: str
    start_time: float
    end_time: float
    success: bool
    error_message: Optional[str] = None
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    @property
    def duration(self) -> float:
        """Operation duration in seconds."""
        return self.end_time - self.start_time

@dataclass
class SystemStats:
    """System-wide statistics."""
    total_documents_processed: int = 0
    total_chunks_created: int = 0
    total_questions_answered: int = 0
    total_api_requests: int = 0
    average_processing_time: float = 0.0
    average_query_time: float = 0.0
    error_count: int = 0
    uptime_seconds: float = 0.0
    collections: Dict[str, int] = field(default_factory=dict)

class PerformanceMonitor:
    """Monitor system performance and collect metrics."""
    
    def __init__(self, max_metrics: int = 10000):
        """Initialize performance monitor.
        
        Args:
            max_metrics: Maximum number of metrics to keep in memory
        """
        self.max_metrics = max_metrics
        self.metrics: deque = deque(maxlen=max_metrics)
        self.start_time = time.time()
        self.stats = SystemStats()
        self.lock = threading.Lock()
        
        # Performance tracking
        self.operation_counts = defaultdict(int)
        self.operation_durations = defaultdict(list)
        self.error_counts = defaultdict(int)
        
        logger.info("Performance monitor initialized")
    
    def record_metric(self, metric: ProcessingMetric):
        """Record a processing metric."""
        with self.lock:
            self.metrics.append(metric)
            self.operation_counts[metric.operation] += 1
            
            if metric.success:
                self.operation_durations[metric.operation].append(metric.duration)
                
                # Update stats based on operation type
                if metric.operation == "document_processing":
                    self.stats.total_documents_processed += 1
                    self.stats.total_chunks_created += metric.metadata.get("chunks_created", 0)
                elif metric.operation == "question_answering":
                    self.stats.total_questions_answered += 1
                elif metric.operation == "api_request":
                    self.stats.total_api_requests += 1
            else:
                self.error_counts[metric.operation] += 1
                self.stats.error_count += 1
                
            # Update averages
            self._update_averages()
    
    def _update_averages(self):
        """Update average processing times."""
        if self.operation_durations["document_processing"]:
            self.stats.average_processing_time = sum(
                self.operation_durations["document_processing"]
            ) / len(self.operation_durations["document_processing"])
        
        if self.operation_durations["question_answering"]:
            self.stats.average_query_time = sum(
                self.operation_durations["question_answering"]
            ) / len(self.operation_durations["question_answering"])
        
        self.stats.uptime_seconds = time.time() - self.start_time
    
    def get_stats(self) -> SystemStats:
        """Get current system statistics."""
        with self.lock:
            self._update_averages()
            return self.stats
    
    def get_operation_stats(self, operation: str) -> Dict[str, Any]:
        """Get statistics for a specific operation."""
        with self.lock:
            durations = self.operation_durations.get(operation, [])
            
            if not durations:
                return {
                    "count": self.operation_counts.get(operation, 0),
                    "errors": self.error_counts.get(operation, 0),
                    "average_duration": 0.0,
                    "min_duration": 0.0,
                    "max_duration": 0.0
                }
            
            return {
                "count": self.operation_counts.get(operation, 0),
                "errors": self.error_counts.get(operation, 0),
                "average_duration": sum(durations) / len(durations),
                "min_duration": min(durations),
                "max_duration": max(durations),
                "total_duration": sum(durations)
            }
    
    def get_recent_metrics(self, minutes: int = 60) -> List[ProcessingMetric]:
        """Get metrics from the last N minutes."""
        cutoff_time = time.time() - (minutes * 60)
        
        with self.lock:
            return [
                metric for metric in self.metrics 
                if metric.start_time >= cutoff_time
            ]
    
    def get_error_summary(self) -> Dict[str, Any]:
        """Get summary of recent errors."""
        recent_errors = []
        
        with self.lock:
            for metric in self.metrics:
                if not metric.success and metric.error_message:
                    recent_errors.append({
                        "operation": metric.operation,
                        "timestamp": datetime.fromtimestamp(metric.start_time).isoformat(),
                        "error": metric.error_message,
                        "metadata": metric.metadata
                    })
        
        # Get last 50 errors
        recent_errors = recent_errors[-50:]
        
        # Count errors by type
        error_types = defaultdict(int)
        for error in recent_errors:
            error_types[error["operation"]] += 1
        
        return {
            "total_errors": len(recent_errors),
            "error_types": dict(error_types),
            "recent_errors": recent_errors[-10:]  # Last 10 errors
        }
    
    def export_metrics(self, filepath: Path, format: str = "json"):
        """Export metrics to file."""
        data = {
            "stats": {
                "total_documents_processed": self.stats.total_documents_processed,
                "total_chunks_created": self.stats.total_chunks_created,
                "total_questions_answered": self.stats.total_questions_answered,
                "total_api_requests": self.stats.total_api_requests,
                "average_processing_time": self.stats.average_processing_time,
                "average_query_time": self.stats.average_query_time,
                "error_count": self.stats.error_count,
                "uptime_seconds": self.stats.uptime_seconds,
                "export_timestamp": datetime.now().isoformat()
            },
            "operation_stats": {
                op: self.get_operation_stats(op) 
                for op in self.operation_counts.keys()
            },
            "error_summary": self.get_error_summary()
        }
        
        if format.lower() == "json":
            with open(filepath, 'w') as f:
                json.dump(data, f, indent=2)
        else:
            raise ValueError(f"Unsupported format: {format}")
        
        logger.info(f"Metrics exported to {filepath}")


class MetricsCollector:
    """Context manager for collecting operation metrics."""
    
    def __init__(self, monitor: PerformanceMonitor, operation: str, **metadata):
        """Initialize metrics collector.
        
        Args:
            monitor: Performance monitor instance
            operation: Name of the operation being monitored
            **metadata: Additional metadata to record
        """
        self.monitor = monitor
        self.operation = operation
        self.metadata = metadata
        self.start_time = None
        self.error_message = None
    
    def __enter__(self):
        """Start timing the operation."""
        self.start_time = time.time()
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        """Finish timing and record the metric."""
        end_time = time.time()
        success = exc_type is None
        
        if not success:
            self.error_message = str(exc_val) if exc_val else "Unknown error"
        
        metric = ProcessingMetric(
            operation=self.operation,
            start_time=self.start_time,
            end_time=end_time,
            success=success,
            error_message=self.error_message,
            metadata=self.metadata
        )
        
        self.monitor.record_metric(metric)
        
        # Don't suppress exceptions
        return False
    
    def add_metadata(self, **metadata):
        """Add additional metadata to the metric."""
        self.metadata.update(metadata)


# Global monitor instance
_global_monitor: Optional[PerformanceMonitor] = None

def get_monitor() -> PerformanceMonitor:
    """Get or create the global performance monitor."""
    global _global_monitor
    if _global_monitor is None:
        _global_monitor = PerformanceMonitor()
    return _global_monitor

def monitor_operation(operation: str, **metadata):
    """Decorator or context manager for monitoring operations."""
    monitor = get_monitor()
    return MetricsCollector(monitor, operation, **metadata)

def track_api_request(endpoint: str, method: str = "GET"):
    """Track an API request."""
    return monitor_operation("api_request", endpoint=endpoint, method=method)

def track_document_processing(file_count: int = 1):
    """Track document processing operation."""
    return monitor_operation("document_processing", file_count=file_count)

def track_question_answering(collection: str = "default"):
    """Track question answering operation."""
    return monitor_operation("question_answering", collection=collection)

def get_health_status() -> Dict[str, Any]:
    """Get system health status for monitoring endpoints."""
    monitor = get_monitor()
    stats = monitor.get_stats()
    recent_errors = monitor.get_error_summary()
    
    # Determine health status
    error_rate = recent_errors["total_errors"] / max(stats.total_api_requests, 1)
    health_status = "healthy"
    
    if error_rate > 0.1:  # More than 10% error rate
        health_status = "degraded"
    elif error_rate > 0.2:  # More than 20% error rate
        health_status = "unhealthy"
    
    return {
        "status": health_status,
        "uptime_seconds": stats.uptime_seconds,
        "total_requests": stats.total_api_requests,
        "error_rate": error_rate,
        "average_response_time": stats.average_query_time,
        "last_updated": datetime.now().isoformat()
    } 