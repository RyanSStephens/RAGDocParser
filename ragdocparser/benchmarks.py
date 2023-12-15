"""
Performance benchmarking module for RAGDocParser.
Provides comprehensive benchmarking tools for RAG system evaluation.
"""

import time
import logging
from typing import Dict, List, Any, Optional, Callable, Tuple
from dataclasses import dataclass, field
from datetime import datetime, timedelta
import json
import statistics
from pathlib import Path
import asyncio
from concurrent.futures import ThreadPoolExecutor, as_completed
import psutil
import gc

logger = logging.getLogger(__name__)

@dataclass
class BenchmarkResult:
    """Result of a single benchmark run."""
    name: str
    duration: float
    memory_usage_mb: float
    cpu_percent: float
    success: bool
    error_message: Optional[str] = None
    metadata: Dict[str, Any] = field(default_factory=dict)
    timestamp: datetime = field(default_factory=datetime.now)

@dataclass
class BenchmarkSuite:
    """Collection of benchmark results."""
    name: str
    results: List[BenchmarkResult] = field(default_factory=list)
    start_time: datetime = field(default_factory=datetime.now)
    end_time: Optional[datetime] = None
    total_duration: float = 0.0
    
    def add_result(self, result: BenchmarkResult):
        """Add a benchmark result to the suite."""
        self.results.append(result)
    
    def finalize(self):
        """Finalize the benchmark suite."""
        self.end_time = datetime.now()
        self.total_duration = (self.end_time - self.start_time).total_seconds()
    
    def get_summary(self) -> Dict[str, Any]:
        """Get benchmark suite summary."""
        if not self.results:
            return {"error": "No results available"}
        
        successful_results = [r for r in self.results if r.success]
        failed_results = [r for r in self.results if not r.success]
        
        durations = [r.duration for r in successful_results]
        memory_usages = [r.memory_usage_mb for r in successful_results]
        cpu_usages = [r.cpu_percent for r in successful_results]
        
        return {
            "suite_name": self.name,
            "total_tests": len(self.results),
            "successful_tests": len(successful_results),
            "failed_tests": len(failed_results),
            "success_rate": len(successful_results) / len(self.results),
            "total_duration": self.total_duration,
            "performance_stats": {
                "duration": {
                    "mean": statistics.mean(durations) if durations else 0,
                    "median": statistics.median(durations) if durations else 0,
                    "std_dev": statistics.stdev(durations) if len(durations) > 1 else 0,
                    "min": min(durations) if durations else 0,
                    "max": max(durations) if durations else 0
                },
                "memory": {
                    "mean_mb": statistics.mean(memory_usages) if memory_usages else 0,
                    "max_mb": max(memory_usages) if memory_usages else 0,
                    "min_mb": min(memory_usages) if memory_usages else 0
                },
                "cpu": {
                    "mean_percent": statistics.mean(cpu_usages) if cpu_usages else 0,
                    "max_percent": max(cpu_usages) if cpu_usages else 0
                }
            },
            "errors": [{"test": r.name, "error": r.error_message} for r in failed_results]
        }

class PerformanceBenchmarker:
    """Main benchmarking class for RAG system performance."""
    
    def __init__(self):
        """Initialize the benchmarker."""
        self.current_suite: Optional[BenchmarkSuite] = None
        self.suites: List[BenchmarkSuite] = []
        
    def start_suite(self, name: str) -> BenchmarkSuite:
        """Start a new benchmark suite."""
        self.current_suite = BenchmarkSuite(name=name)
        logger.info(f"Started benchmark suite: {name}")
        return self.current_suite
    
    def end_suite(self) -> Optional[BenchmarkSuite]:
        """End the current benchmark suite."""
        if self.current_suite:
            self.current_suite.finalize()
            self.suites.append(self.current_suite)
            logger.info(f"Completed benchmark suite: {self.current_suite.name}")
            suite = self.current_suite
            self.current_suite = None
            return suite
        return None
    
    def benchmark_function(
        self,
        name: str,
        func: Callable,
        *args,
        iterations: int = 1,
        warmup_iterations: int = 0,
        **kwargs
    ) -> List[BenchmarkResult]:
        """Benchmark a single function."""
        results = []
        
        # Warmup runs
        for i in range(warmup_iterations):
            try:
                func(*args, **kwargs)
            except Exception as e:
                logger.warning(f"Warmup iteration {i} failed: {e}")
        
        # Actual benchmark runs
        for i in range(iterations):
            gc.collect()  # Clean up memory before each run
            
            # Measure memory and CPU before
            process = psutil.Process()
            memory_before = process.memory_info().rss / 1024 / 1024  # MB
            cpu_before = process.cpu_percent()
            
            start_time = time.perf_counter()
            success = True
            error_message = None
            
            try:
                result = func(*args, **kwargs)
                metadata = {"result_type": type(result).__name__ if result else "None"}
                if hasattr(result, '__len__'):
                    metadata["result_length"] = len(result)
            except Exception as e:
                success = False
                error_message = str(e)
                metadata = {}
                logger.error(f"Benchmark iteration {i} failed: {e}")
            
            end_time = time.perf_counter()
            duration = end_time - start_time
            
            # Measure memory and CPU after
            memory_after = process.memory_info().rss / 1024 / 1024  # MB
            cpu_after = process.cpu_percent()
            
            benchmark_result = BenchmarkResult(
                name=f"{name}_iter_{i}",
                duration=duration,
                memory_usage_mb=memory_after - memory_before,
                cpu_percent=max(cpu_after - cpu_before, 0),  # Ensure non-negative
                success=success,
                error_message=error_message,
                metadata={
                    **metadata,
                    "iteration": i,
                    "memory_before_mb": memory_before,
                    "memory_after_mb": memory_after
                }
            )
            
            results.append(benchmark_result)
            
            if self.current_suite:
                self.current_suite.add_result(benchmark_result)
        
        return results
    
    def benchmark_rag_pipeline(
        self,
        processor,
        test_documents: List[str],
        test_queries: List[str],
        collection_name: str = "benchmark_collection"
    ) -> Dict[str, Any]:
        """Benchmark the complete RAG pipeline."""
        pipeline_suite = self.start_suite("RAG_Pipeline_Benchmark")
        
        try:
            # 1. Document Processing Benchmark
            logger.info("Benchmarking document processing...")
            doc_results = self.benchmark_function(
                "document_processing",
                processor.process_documents,
                test_documents,
                collection_name,
                iterations=3
            )
            
            # 2. Vector Search Benchmark
            logger.info("Benchmarking vector search...")
            search_results = []
            for i, query in enumerate(test_queries[:5]):  # Limit to 5 queries
                query_results = self.benchmark_function(
                    f"vector_search_query_{i}",
                    processor.vectordb.search_similar,
                    query,
                    5,  # k=5
                    collection_name,
                    iterations=5
                )
                search_results.extend(query_results)
            
            # 3. Question Answering Benchmark (if available)
            if processor.rag_manager:
                logger.info("Benchmarking question answering...")
                qa_results = []
                for i, query in enumerate(test_queries[:3]):  # Limit to 3 queries
                    qa_query_results = self.benchmark_function(
                        f"question_answering_query_{i}",
                        processor.ask_question,
                        query,
                        collection_name,
                        5,  # k=5
                        iterations=2
                    )
                    qa_results.extend(qa_query_results)
            
            # 4. Collection Info Benchmark
            logger.info("Benchmarking collection info retrieval...")
            info_results = self.benchmark_function(
                "collection_info",
                processor.get_collections_info,
                iterations=10
            )
            
        except Exception as e:
            logger.error(f"Pipeline benchmark failed: {e}")
        
        suite = self.end_suite()
        return suite.get_summary() if suite else {}
    
    def benchmark_scalability(
        self,
        processor,
        base_documents: List[str],
        scale_factors: List[int] = [1, 2, 5, 10],
        test_query: str = "What is the main topic?"
    ) -> Dict[str, Any]:
        """Benchmark system scalability with different data sizes."""
        scalability_suite = self.start_suite("Scalability_Benchmark")
        
        scalability_results = {}
        
        for scale_factor in scale_factors:
            logger.info(f"Testing scalability with scale factor: {scale_factor}")
            
            # Scale up documents
            scaled_docs = base_documents * scale_factor
            collection_name = f"scale_test_{scale_factor}"
            
            try:
                # Process documents
                doc_results = self.benchmark_function(
                    f"process_docs_scale_{scale_factor}",
                    processor.process_documents,
                    scaled_docs,
                    collection_name,
                    iterations=1
                )
                
                # Test search performance
                search_results = self.benchmark_function(
                    f"search_scale_{scale_factor}",
                    processor.vectordb.search_similar,
                    test_query,
                    5,
                    collection_name,
                    iterations=5
                )
                
                scalability_results[scale_factor] = {
                    "document_count": len(scaled_docs),
                    "processing_time": doc_results[0].duration if doc_results else 0,
                    "avg_search_time": statistics.mean([r.duration for r in search_results if r.success]),
                    "memory_usage": max([r.memory_usage_mb for r in doc_results + search_results])
                }
                
            except Exception as e:
                logger.error(f"Scalability test failed for scale {scale_factor}: {e}")
                scalability_results[scale_factor] = {"error": str(e)}
        
        suite = self.end_suite()
        return {
            "scalability_results": scalability_results,
            "suite_summary": suite.get_summary() if suite else {}
        }
    
    def benchmark_concurrent_access(
        self,
        processor,
        queries: List[str],
        concurrent_users: List[int] = [1, 2, 5, 10],
        collection_name: str = "concurrent_test"
    ) -> Dict[str, Any]:
        """Benchmark concurrent access performance."""
        concurrent_suite = self.start_suite("Concurrent_Access_Benchmark")
        concurrent_results = {}
        
        def execute_query(query: str, user_id: int) -> Tuple[float, bool, str]:
            """Execute a single query and measure performance."""
            start_time = time.perf_counter()
            try:
                processor.vectordb.search_similar(query, 5, collection_name)
                end_time = time.perf_counter()
                return end_time - start_time, True, ""
            except Exception as e:
                end_time = time.perf_counter()
                return end_time - start_time, False, str(e)
        
        for num_users in concurrent_users:
            logger.info(f"Testing concurrent access with {num_users} users")
            
            # Create tasks for concurrent execution
            tasks = []
            with ThreadPoolExecutor(max_workers=num_users) as executor:
                futures = []
                start_time = time.perf_counter()
                
                for user_id in range(num_users):
                    query = queries[user_id % len(queries)]
                    future = executor.submit(execute_query, query, user_id)
                    futures.append(future)
                
                # Collect results
                durations = []
                successes = 0
                errors = []
                
                for future in as_completed(futures):
                    duration, success, error = future.result()
                    durations.append(duration)
                    if success:
                        successes += 1
                    else:
                        errors.append(error)
                
                end_time = time.perf_counter()
                total_time = end_time - start_time
            
            concurrent_results[num_users] = {
                "total_time": total_time,
                "successful_queries": successes,
                "failed_queries": len(errors),
                "success_rate": successes / num_users,
                "avg_query_time": statistics.mean(durations),
                "max_query_time": max(durations),
                "min_query_time": min(durations),
                "queries_per_second": num_users / total_time,
                "errors": errors[:5]  # First 5 errors
            }
            
            # Add to suite
            self.current_suite.add_result(BenchmarkResult(
                name=f"concurrent_access_{num_users}_users",
                duration=total_time,
                memory_usage_mb=0,  # Would need more detailed measurement
                cpu_percent=0,
                success=successes > 0,
                metadata=concurrent_results[num_users]
            ))
        
        suite = self.end_suite()
        return {
            "concurrent_results": concurrent_results,
            "suite_summary": suite.get_summary() if suite else {}
        }
    
    def export_results(self, filepath: Path, format: str = "json") -> None:
        """Export all benchmark results to file."""
        data = {
            "benchmark_export": {
                "timestamp": datetime.now().isoformat(),
                "total_suites": len(self.suites),
                "suites": [
                    {
                        "name": suite.name,
                        "start_time": suite.start_time.isoformat(),
                        "end_time": suite.end_time.isoformat() if suite.end_time else None,
                        "duration": suite.total_duration,
                        "summary": suite.get_summary(),
                        "detailed_results": [
                            {
                                "name": r.name,
                                "duration": r.duration,
                                "memory_usage_mb": r.memory_usage_mb,
                                "cpu_percent": r.cpu_percent,
                                "success": r.success,
                                "error_message": r.error_message,
                                "metadata": r.metadata,
                                "timestamp": r.timestamp.isoformat()
                            }
                            for r in suite.results
                        ]
                    }
                    for suite in self.suites
                ]
            }
        }
        
        if format.lower() == "json":
            with open(filepath, 'w') as f:
                json.dump(data, f, indent=2)
        else:
            raise ValueError(f"Unsupported format: {format}")
        
        logger.info(f"Benchmark results exported to {filepath}")

# Global benchmarker instance
_global_benchmarker: Optional[PerformanceBenchmarker] = None

def get_benchmarker() -> PerformanceBenchmarker:
    """Get or create the global benchmarker."""
    global _global_benchmarker
    if _global_benchmarker is None:
        _global_benchmarker = PerformanceBenchmarker()
    return _global_benchmarker

def quick_benchmark(func: Callable, *args, iterations: int = 5, **kwargs) -> Dict[str, Any]:
    """Quick benchmark function for ad-hoc testing."""
    benchmarker = get_benchmarker()
    results = benchmarker.benchmark_function(
        func.__name__,
        func,
        *args,
        iterations=iterations,
        **kwargs
    )
    
    successful_results = [r for r in results if r.success]
    if not successful_results:
        return {"error": "All benchmark iterations failed"}
    
    durations = [r.duration for r in successful_results]
    
    return {
        "function": func.__name__,
        "iterations": len(successful_results),
        "avg_duration": statistics.mean(durations),
        "min_duration": min(durations),
        "max_duration": max(durations),
        "std_dev": statistics.stdev(durations) if len(durations) > 1 else 0,
        "total_duration": sum(durations)
    } 