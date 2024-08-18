"""
Performance monitoring and analytics for RAG Document Parser.
"""

import time
import logging
import threading
from typing import Dict, Any, List, Optional
from dataclasses import dataclass, field
from datetime import datetime, timedelta
import json

logger = logging.getLogger(__name__)


@dataclass
class PerformanceMetric:
    """Performance metric data class."""
    name: str
    value: float
    timestamp: datetime
    metadata: Dict[str, Any] = field(default_factory=dict)


class PerformanceMonitor:
    """Monitor performance metrics for document processing."""
    
    def __init__(self, max_metrics: int = 10000):
        self.max_metrics = max_metrics
        self.metrics: List[PerformanceMetric] = []
        self.lock = threading.Lock()
        
        # Performance counters
        self.counters = {
            'documents_processed': 0,
            'chunks_created': 0,
            'searches_performed': 0,
            'errors_encountered': 0,
        }
        
        # Timing statistics
        self.timing_stats = {
            'document_parsing': [],
            'text_chunking': [],
            'vector_embedding': [],
            'search_queries': [],
        }
    
    def record_metric(self, name: str, value: float, metadata: Dict[str, Any] = None):
        """Record a performance metric."""
        with self.lock:
            metric = PerformanceMetric(
                name=name,
                value=value,
                timestamp=datetime.now(),
                metadata=metadata or {}
            )
            
            self.metrics.append(metric)
            
            # Trim metrics if we exceed max size
            if len(self.metrics) > self.max_metrics:
                self.metrics = self.metrics[-self.max_metrics:]
    
    def increment_counter(self, counter_name: str, amount: int = 1):
        """Increment a counter."""
        with self.lock:
            if counter_name in self.counters:
                self.counters[counter_name] += amount
            else:
                self.counters[counter_name] = amount
    
    def record_timing(self, operation: str, duration: float, metadata: Dict[str, Any] = None):
        """Record timing for an operation."""
        with self.lock:
            if operation in self.timing_stats:
                self.timing_stats[operation].append({
                    'duration': duration,
                    'timestamp': datetime.now(),
                    'metadata': metadata or {}
                })
                
                # Keep only recent timings (last 1000)
                if len(self.timing_stats[operation]) > 1000:
                    self.timing_stats[operation] = self.timing_stats[operation][-1000:]
        
        # Also record as a general metric
        self.record_metric(f"{operation}_duration", duration, metadata)
    
    def get_statistics(self, hours: int = 24) -> Dict[str, Any]:
        """Get performance statistics for the last N hours."""
        cutoff_time = datetime.now() - timedelta(hours=hours)
        
        with self.lock:
            # Filter recent metrics
            recent_metrics = [m for m in self.metrics if m.timestamp >= cutoff_time]
            
            # Calculate statistics
            stats = {
                'period_hours': hours,
                'total_metrics': len(recent_metrics),
                'counters': self.counters.copy(),
                'timing_statistics': {}
            }
            
            # Calculate timing statistics
            for operation, timings in self.timing_stats.items():
                recent_timings = [t for t in timings if t['timestamp'] >= cutoff_time]
                
                if recent_timings:
                    durations = [t['duration'] for t in recent_timings]
                    stats['timing_statistics'][operation] = {
                        'count': len(durations),
                        'avg_duration': sum(durations) / len(durations),
                        'min_duration': min(durations),
                        'max_duration': max(durations),
                        'total_duration': sum(durations)
                    }
                else:
                    stats['timing_statistics'][operation] = {
                        'count': 0,
                        'avg_duration': 0,
                        'min_duration': 0,
                        'max_duration': 0,
                        'total_duration': 0
                    }
            
            return stats
    
    def get_health_status(self) -> Dict[str, Any]:
        """Get system health status."""
        recent_stats = self.get_statistics(hours=1)
        
        # Calculate health indicators
        error_rate = 0
        if self.counters['documents_processed'] > 0:
            error_rate = self.counters['errors_encountered'] / self.counters['documents_processed']
        
        # Check for performance issues
        avg_parse_time = recent_stats['timing_statistics'].get('document_parsing', {}).get('avg_duration', 0)
        avg_search_time = recent_stats['timing_statistics'].get('search_queries', {}).get('avg_duration', 0)
        
        health_status = "healthy"
        issues = []
        
        if error_rate > 0.1:  # More than 10% error rate
            health_status = "degraded"
            issues.append(f"High error rate: {error_rate:.2%}")
        
        if avg_parse_time > 30:  # More than 30 seconds average parse time
            health_status = "degraded"
            issues.append(f"Slow document parsing: {avg_parse_time:.2f}s average")
        
        if avg_search_time > 5:  # More than 5 seconds average search time
            health_status = "degraded"
            issues.append(f"Slow search queries: {avg_search_time:.2f}s average")
        
        return {
            'status': health_status,
            'error_rate': error_rate,
            'issues': issues,
            'uptime_metrics': recent_stats,
            'last_updated': datetime.now().isoformat()
        }
    
    def export_metrics(self, file_path: str, hours: int = 24):
        """Export metrics to a JSON file."""
        stats = self.get_statistics(hours)
        
        with open(file_path, 'w') as f:
            json.dump(stats, f, indent=2, default=str)
        
        logger.info(f"Metrics exported to {file_path}")


# Global performance monitor instance
performance_monitor = PerformanceMonitor()


def timed_operation(operation_name: str):
    """Decorator to time operations."""
    def decorator(func):
        def wrapper(*args, **kwargs):
            start_time = time.time()
            try:
                result = func(*args, **kwargs)
                duration = time.time() - start_time
                performance_monitor.record_timing(operation_name, duration)
                return result
            except Exception as e:
                duration = time.time() - start_time
                performance_monitor.record_timing(operation_name, duration, {'error': str(e)})
                performance_monitor.increment_counter('errors_encountered')
                raise
        return wrapper
    return decorator
