"""
Performance optimizations for RAGDocParser.
Year-end improvements for better efficiency and resource utilization.
"""

import logging
import gc
import threading
from typing import Dict, List, Any, Optional, Callable
from functools import wraps, lru_cache
import time
import weakref
from concurrent.futures import ThreadPoolExecutor
import asyncio

logger = logging.getLogger(__name__)

class MemoryOptimizer:
    """Memory optimization utilities."""
    
    @staticmethod
    def force_garbage_collection():
        """Force garbage collection to free memory."""
        collected = gc.collect()
        logger.debug(f"Garbage collection freed {collected} objects")
        return collected
    
    @staticmethod
    def get_memory_usage():
        """Get current memory usage statistics."""
        import psutil
        process = psutil.Process()
        memory_info = process.memory_info()
        return {
            "rss_mb": memory_info.rss / 1024 / 1024,
            "vms_mb": memory_info.vms / 1024 / 1024,
            "percent": process.memory_percent()
        }
    
    @staticmethod
    def optimize_memory_settings():
        """Optimize Python memory settings."""
        # Adjust garbage collection thresholds for better performance
        gc.set_threshold(700, 10, 10)
        logger.info("Memory optimization settings applied")

class CacheManager:
    """Advanced caching system for frequently accessed data."""
    
    def __init__(self, max_size: int = 1000):
        """Initialize cache manager."""
        self.max_size = max_size
        self._cache = {}
        self._access_times = {}
        self._lock = threading.RLock()
    
    def get(self, key: str) -> Any:
        """Get item from cache."""
        with self._lock:
            if key in self._cache:
                self._access_times[key] = time.time()
                return self._cache[key]
            return None
    
    def set(self, key: str, value: Any) -> None:
        """Set item in cache."""
        with self._lock:
            if len(self._cache) >= self.max_size:
                self._evict_oldest()
            
            self._cache[key] = value
            self._access_times[key] = time.time()
    
    def _evict_oldest(self) -> None:
        """Evict least recently used item."""
        if not self._access_times:
            return
        
        oldest_key = min(self._access_times.keys(), key=lambda k: self._access_times[k])
        del self._cache[oldest_key]
        del self._access_times[oldest_key]
    
    def clear(self) -> None:
        """Clear all cached items."""
        with self._lock:
            self._cache.clear()
            self._access_times.clear()
    
    def size(self) -> int:
        """Get current cache size."""
        return len(self._cache)

# Global cache instance
_global_cache = CacheManager()

def cached_result(ttl_seconds: int = 300):
    """Decorator for caching function results."""
    def decorator(func: Callable) -> Callable:
        cache_key_prefix = f"{func.__module__}.{func.__qualname__}"
        
        @wraps(func)
        def wrapper(*args, **kwargs):
            # Create cache key from function arguments
            cache_key = f"{cache_key_prefix}:{hash((args, tuple(sorted(kwargs.items()))))}"
            
            # Check cache first
            cached_value = _global_cache.get(cache_key)
            if cached_value is not None:
                result, timestamp = cached_value
                if time.time() - timestamp < ttl_seconds:
                    logger.debug(f"Cache hit for {cache_key}")
                    return result
            
            # Compute result and cache it
            result = func(*args, **kwargs)
            _global_cache.set(cache_key, (result, time.time()))
            logger.debug(f"Cached result for {cache_key}")
            
            return result
        
        return wrapper
    return decorator

class BatchProcessor:
    """Batch processing for improved efficiency."""
    
    def __init__(self, batch_size: int = 100, max_workers: int = 4):
        """Initialize batch processor."""
        self.batch_size = batch_size
        self.max_workers = max_workers
        self.executor = ThreadPoolExecutor(max_workers=max_workers)
    
    def process_batch(self, items: List[Any], process_func: Callable) -> List[Any]:
        """Process items in batches."""
        if not items:
            return []
        
        results = []
        for i in range(0, len(items), self.batch_size):
            batch = items[i:i + self.batch_size]
            
            # Process batch in parallel
            futures = [
                self.executor.submit(process_func, item) 
                for item in batch
            ]
            
            # Collect results
            batch_results = []
            for future in futures:
                try:
                    result = future.result(timeout=30)  # 30 second timeout
                    batch_results.append(result)
                except Exception as e:
                    logger.error(f"Batch processing error: {e}")
                    batch_results.append(None)
            
            results.extend(batch_results)
            
            # Optional memory cleanup between batches
            if i % (self.batch_size * 5) == 0:
                MemoryOptimizer.force_garbage_collection()
        
        return results
    
    def cleanup(self):
        """Cleanup batch processor resources."""
        self.executor.shutdown(wait=True)

class ConnectionPool:
    """Connection pooling for database and API connections."""
    
    def __init__(self, create_connection: Callable, max_connections: int = 10):
        """Initialize connection pool."""
        self.create_connection = create_connection
        self.max_connections = max_connections
        self._pool = []
        self._lock = threading.Lock()
        self._in_use = set()
    
    def get_connection(self):
        """Get connection from pool."""
        with self._lock:
            # Reuse existing connection if available
            if self._pool:
                conn = self._pool.pop()
                self._in_use.add(id(conn))
                return conn
            
            # Create new connection if under limit
            if len(self._in_use) < self.max_connections:
                conn = self.create_connection()
                self._in_use.add(id(conn))
                return conn
            
            # Pool is full, wait and retry
            logger.warning("Connection pool exhausted, waiting...")
            return None
    
    def return_connection(self, conn):
        """Return connection to pool."""
        with self._lock:
            conn_id = id(conn)
            if conn_id in self._in_use:
                self._in_use.remove(conn_id)
                self._pool.append(conn)
    
    def close_all(self):
        """Close all connections in pool."""
        with self._lock:
            for conn in self._pool:
                try:
                    if hasattr(conn, 'close'):
                        conn.close()
                except Exception as e:
                    logger.error(f"Error closing connection: {e}")
            
            self._pool.clear()
            self._in_use.clear()

class PerformanceProfiler:
    """Simple performance profiler for optimization insights."""
    
    def __init__(self):
        """Initialize profiler."""
        self.timings = {}
        self.call_counts = {}
        self._lock = threading.Lock()
    
    def profile_function(self, func: Callable) -> Callable:
        """Decorator to profile function performance."""
        func_name = f"{func.__module__}.{func.__qualname__}"
        
        @wraps(func)
        def wrapper(*args, **kwargs):
            start_time = time.perf_counter()
            
            try:
                result = func(*args, **kwargs)
                return result
            finally:
                end_time = time.perf_counter()
                duration = end_time - start_time
                
                with self._lock:
                    if func_name not in self.timings:
                        self.timings[func_name] = []
                        self.call_counts[func_name] = 0
                    
                    self.timings[func_name].append(duration)
                    self.call_counts[func_name] += 1
                    
                    # Keep only last 1000 timings to prevent memory bloat
                    if len(self.timings[func_name]) > 1000:
                        self.timings[func_name] = self.timings[func_name][-1000:]
        
        return wrapper
    
    def get_performance_report(self) -> Dict[str, Any]:
        """Get performance report for all profiled functions."""
        report = {}
        
        with self._lock:
            for func_name, timings in self.timings.items():
                if timings:
                    report[func_name] = {
                        "call_count": self.call_counts[func_name],
                        "avg_time": sum(timings) / len(timings),
                        "min_time": min(timings),
                        "max_time": max(timings),
                        "total_time": sum(timings)
                    }
        
        return report
    
    def reset_stats(self):
        """Reset all profiling statistics."""
        with self._lock:
            self.timings.clear()
            self.call_counts.clear()

class OptimizationManager:
    """Main manager for all performance optimizations."""
    
    def __init__(self):
        """Initialize optimization manager."""
        self.memory_optimizer = MemoryOptimizer()
        self.cache_manager = CacheManager()
        self.profiler = PerformanceProfiler()
        self.batch_processors = {}
        self.connection_pools = {}
        
        # Apply initial optimizations
        self.apply_startup_optimizations()
    
    def apply_startup_optimizations(self):
        """Apply optimizations at startup."""
        logger.info("Applying startup performance optimizations...")
        
        # Memory optimizations
        self.memory_optimizer.optimize_memory_settings()
        
        # Force initial garbage collection
        self.memory_optimizer.force_garbage_collection()
        
        logger.info("Startup optimizations completed")
    
    def create_batch_processor(self, name: str, batch_size: int = 100, max_workers: int = 4) -> BatchProcessor:
        """Create a named batch processor."""
        processor = BatchProcessor(batch_size, max_workers)
        self.batch_processors[name] = processor
        return processor
    
    def create_connection_pool(self, name: str, create_connection: Callable, max_connections: int = 10) -> ConnectionPool:
        """Create a named connection pool."""
        pool = ConnectionPool(create_connection, max_connections)
        self.connection_pools[name] = pool
        return pool
    
    def periodic_cleanup(self):
        """Perform periodic cleanup operations."""
        logger.debug("Running periodic cleanup...")
        
        # Force garbage collection
        collected = self.memory_optimizer.force_garbage_collection()
        
        # Clear old cache entries
        cache_size_before = self.cache_manager.size()
        # Cache cleanup happens automatically through LRU eviction
        
        logger.debug(f"Cleanup completed: {collected} objects collected, cache size: {cache_size_before}")
    
    def get_optimization_report(self) -> Dict[str, Any]:
        """Get comprehensive optimization report."""
        memory_usage = self.memory_optimizer.get_memory_usage()
        performance_report = self.profiler.get_performance_report()
        
        return {
            "memory_usage": memory_usage,
            "cache_size": self.cache_manager.size(),
            "performance_profile": performance_report,
            "active_batch_processors": len(self.batch_processors),
            "active_connection_pools": len(self.connection_pools)
        }
    
    def cleanup_all(self):
        """Cleanup all optimization resources."""
        logger.info("Cleaning up optimization resources...")
        
        # Cleanup batch processors
        for processor in self.batch_processors.values():
            processor.cleanup()
        self.batch_processors.clear()
        
        # Cleanup connection pools
        for pool in self.connection_pools.values():
            pool.close_all()
        self.connection_pools.clear()
        
        # Clear caches
        self.cache_manager.clear()
        
        # Final garbage collection
        self.memory_optimizer.force_garbage_collection()
        
        logger.info("Optimization cleanup completed")

# Global optimization manager
_global_optimizer: Optional[OptimizationManager] = None

def get_optimization_manager() -> OptimizationManager:
    """Get or create the global optimization manager."""
    global _global_optimizer
    if _global_optimizer is None:
        _global_optimizer = OptimizationManager()
    return _global_optimizer

def profile_performance(func: Callable) -> Callable:
    """Convenience decorator for performance profiling."""
    manager = get_optimization_manager()
    return manager.profiler.profile_function(func)

def optimize_memory():
    """Convenience function for memory optimization."""
    manager = get_optimization_manager()
    manager.periodic_cleanup()

def get_performance_report() -> Dict[str, Any]:
    """Convenience function to get performance report."""
    manager = get_optimization_manager()
    return manager.get_optimization_report() 