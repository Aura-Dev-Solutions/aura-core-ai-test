"""
Advanced performance optimization techniques for the document analyzer.
"""

import time
import asyncio
import threading
from typing import Dict, List, Any, Optional, Callable, Tuple
from datetime import datetime, timedelta
from dataclasses import dataclass, field
from collections import defaultdict, deque
from concurrent.futures import ThreadPoolExecutor, ProcessPoolExecutor
import multiprocessing as mp
from functools import wraps, lru_cache
import weakref

from src.core.logging import LoggerMixin
from src.core.config import settings


@dataclass
class PerformanceMetrics:
    operation_name: str
    execution_time: float
    memory_usage: int
    cpu_usage: float
    throughput: float
    error_count: int = 0
    success_count: int = 0


@dataclass
class OptimizationConfig:
    enable_caching: bool = True
    enable_batching: bool = True
    enable_parallel_processing: bool = True
    enable_memory_optimization: bool = True
    batch_size: int = 10
    max_workers: int = 4
    cache_size: int = 1000


class BatchProcessor(LoggerMixin):
    """
    Batch processor for optimizing throughput by grouping operations.
    """
    
    def __init__(self, batch_size: int = 10, max_wait_time: float = 1.0):
        self.batch_size = batch_size
        self.max_wait_time = max_wait_time
        self.pending_items: Dict[str, List[Any]] = defaultdict(list)
        self.batch_processors: Dict[str, Callable] = {}
        self.last_batch_time: Dict[str, float] = {}
        self.lock = threading.Lock()
    
    def register_batch_processor(self, operation_name: str, processor_func: Callable):
        """Register a batch processing function for an operation."""
        self.batch_processors[operation_name] = processor_func
        self.last_batch_time[operation_name] = time.time()
    
    async def add_to_batch(self, operation_name: str, item: Any) -> Any:
        """
        Add item to batch and process when batch is full or timeout reached.
        
        Args:
            operation_name: Name of the operation
            item: Item to process
            
        Returns:
            Processing result
        """
        with self.lock:
            self.pending_items[operation_name].append(item)
            
            # Check if batch is ready for processing
            should_process = (
                len(self.pending_items[operation_name]) >= self.batch_size or
                time.time() - self.last_batch_time[operation_name] > self.max_wait_time
            )
            
            if should_process and operation_name in self.batch_processors:
                # Process current batch
                batch = self.pending_items[operation_name].copy()
                self.pending_items[operation_name].clear()
                self.last_batch_time[operation_name] = time.time()
                
                # Process batch asynchronously
                processor = self.batch_processors[operation_name]
                results = await self._process_batch(processor, batch)
                
                # Return result for current item (last in batch)
                return results[-1] if results else None
            
            # If batch not ready, wait for next opportunity
            return await self._wait_for_batch_result(operation_name, item)
    
    async def _process_batch(self, processor: Callable, batch: List[Any]) -> List[Any]:
        """Process a batch of items."""
        try:
            if asyncio.iscoroutinefunction(processor):
                return await processor(batch)
            else:
                # Run in thread pool for CPU-bound operations
                loop = asyncio.get_event_loop()
                return await loop.run_in_executor(None, processor, batch)
        except Exception as e:
            self.logger.error(f"Batch processing failed: {str(e)}")
            return [None] * len(batch)
    
    async def _wait_for_batch_result(self, operation_name: str, item: Any) -> Any:
        """Wait for batch processing result."""
        # Simplified implementation - in production would use proper synchronization
        await asyncio.sleep(0.1)
        return None


class MemoryOptimizer(LoggerMixin):
    """
    Memory optimization techniques including object pooling and weak references.
    """
    
    def __init__(self):
        self.object_pools: Dict[str, List[Any]] = defaultdict(list)
        self.weak_references: Dict[str, weakref.WeakSet] = defaultdict(weakref.WeakSet)
        self.memory_usage_history = deque(maxlen=100)
        self.gc_threshold = 0.8  # Trigger GC when memory usage > 80%
    
    def get_pooled_object(self, object_type: str, factory_func: Callable) -> Any:
        """
        Get object from pool or create new one.
        
        Args:
            object_type: Type identifier for the object
            factory_func: Function to create new object if pool is empty
            
        Returns:
            Pooled or new object
        """
        pool = self.object_pools[object_type]
        
        if pool:
            obj = pool.pop()
            self.logger.debug(f"Retrieved {object_type} from pool")
            return obj
        else:
            obj = factory_func()
            self.logger.debug(f"Created new {object_type}")
            return obj
    
    def return_to_pool(self, object_type: str, obj: Any, reset_func: Optional[Callable] = None):
        """
        Return object to pool for reuse.
        
        Args:
            object_type: Type identifier for the object
            obj: Object to return to pool
            reset_func: Optional function to reset object state
        """
        if reset_func:
            reset_func(obj)
        
        pool = self.object_pools[object_type]
        if len(pool) < 100:  # Limit pool size
            pool.append(obj)
            self.logger.debug(f"Returned {object_type} to pool")
    
    def register_weak_reference(self, ref_type: str, obj: Any):
        """Register object with weak reference for memory tracking."""
        self.weak_references[ref_type].add(obj)
    
    def get_memory_usage_stats(self) -> Dict[str, Any]:
        """Get memory usage statistics."""
        import psutil
        import os
        
        process = psutil.Process(os.getpid())
        memory_info = process.memory_info()
        
        stats = {
            "rss_mb": memory_info.rss / 1024 / 1024,
            "vms_mb": memory_info.vms / 1024 / 1024,
            "percent": process.memory_percent(),
            "pool_sizes": {k: len(v) for k, v in self.object_pools.items()},
            "weak_ref_counts": {k: len(v) for k, v in self.weak_references.items()}
        }
        
        self.memory_usage_history.append(stats["percent"])
        
        return stats
    
    def should_trigger_gc(self) -> bool:
        """Check if garbage collection should be triggered."""
        if not self.memory_usage_history:
            return False
        
        current_usage = self.memory_usage_history[-1]
        return current_usage > self.gc_threshold * 100
    
    def optimize_memory(self):
        """Perform memory optimization."""
        import gc
        
        if self.should_trigger_gc():
            # Force garbage collection
            collected = gc.collect()
            self.logger.info(f"Garbage collection freed {collected} objects")
            
            # Clear oversized pools
            for pool_type, pool in self.object_pools.items():
                if len(pool) > 50:
                    removed = len(pool) - 25
                    self.object_pools[pool_type] = pool[:25]
                    self.logger.info(f"Trimmed {pool_type} pool by {removed} objects")


class ParallelExecutor(LoggerMixin):
    """
    Advanced parallel execution with dynamic worker scaling.
    """
    
    def __init__(self, max_workers: int = None):
        self.max_workers = max_workers or mp.cpu_count()
        self.thread_pool = ThreadPoolExecutor(max_workers=self.max_workers)
        self.process_pool = ProcessPoolExecutor(max_workers=self.max_workers)
        self.active_tasks: Dict[str, int] = defaultdict(int)
        self.performance_history = deque(maxlen=100)
    
    async def execute_parallel(
        self,
        tasks: List[Tuple[Callable, Tuple, Dict]],
        execution_mode: str = "thread"
    ) -> List[Any]:
        """
        Execute tasks in parallel with optimal resource allocation.
        
        Args:
            tasks: List of (function, args, kwargs) tuples
            execution_mode: "thread" for I/O bound, "process" for CPU bound
            
        Returns:
            List of results
        """
        start_time = time.time()
        
        try:
            if execution_mode == "thread":
                results = await self._execute_in_threads(tasks)
            elif execution_mode == "process":
                results = await self._execute_in_processes(tasks)
            else:
                # Adaptive mode - choose based on task characteristics
                results = await self._execute_adaptive(tasks)
            
            execution_time = time.time() - start_time
            self.performance_history.append({
                "task_count": len(tasks),
                "execution_time": execution_time,
                "throughput": len(tasks) / execution_time,
                "mode": execution_mode
            })
            
            return results
            
        except Exception as e:
            self.logger.error(f"Parallel execution failed: {str(e)}")
            raise
    
    async def _execute_in_threads(self, tasks: List[Tuple[Callable, Tuple, Dict]]) -> List[Any]:
        """Execute tasks using thread pool."""
        loop = asyncio.get_event_loop()
        futures = []
        
        for func, args, kwargs in tasks:
            future = loop.run_in_executor(self.thread_pool, lambda: func(*args, **kwargs))
            futures.append(future)
        
        return await asyncio.gather(*futures, return_exceptions=True)
    
    async def _execute_in_processes(self, tasks: List[Tuple[Callable, Tuple, Dict]]) -> List[Any]:
        """Execute tasks using process pool."""
        loop = asyncio.get_event_loop()
        futures = []
        
        for func, args, kwargs in tasks:
            future = loop.run_in_executor(self.process_pool, lambda: func(*args, **kwargs))
            futures.append(future)
        
        return await asyncio.gather(*futures, return_exceptions=True)
    
    async def _execute_adaptive(self, tasks: List[Tuple[Callable, Tuple, Dict]]) -> List[Any]:
        """Adaptively choose execution mode based on task characteristics."""
        # Simple heuristic: use processes for CPU-intensive tasks
        cpu_intensive_tasks = []
        io_intensive_tasks = []
        
        for task in tasks:
            func, args, kwargs = task
            # Heuristic: functions with "compute", "calculate", "process" are CPU-intensive
            func_name = getattr(func, '__name__', str(func))
            if any(keyword in func_name.lower() for keyword in ['compute', 'calculate', 'process', 'analyze']):
                cpu_intensive_tasks.append(task)
            else:
                io_intensive_tasks.append(task)
        
        results = []
        
        if cpu_intensive_tasks:
            cpu_results = await self._execute_in_processes(cpu_intensive_tasks)
            results.extend(cpu_results)
        
        if io_intensive_tasks:
            io_results = await self._execute_in_threads(io_intensive_tasks)
            results.extend(io_results)
        
        return results
    
    def get_performance_stats(self) -> Dict[str, Any]:
        """Get performance statistics for parallel execution."""
        if not self.performance_history:
            return {}
        
        recent_stats = list(self.performance_history)
        
        return {
            "average_throughput": sum(s["throughput"] for s in recent_stats) / len(recent_stats),
            "total_tasks_processed": sum(s["task_count"] for s in recent_stats),
            "average_execution_time": sum(s["execution_time"] for s in recent_stats) / len(recent_stats),
            "active_threads": self.thread_pool._threads,
            "active_processes": len(self.process_pool._processes) if hasattr(self.process_pool, '_processes') else 0
        }


class PerformanceProfiler(LoggerMixin):
    """
    Performance profiler for identifying bottlenecks and optimization opportunities.
    """
    
    def __init__(self):
        self.operation_metrics: Dict[str, List[PerformanceMetrics]] = defaultdict(list)
        self.bottlenecks: List[Dict[str, Any]] = []
        self.optimization_suggestions: List[str] = []
    
    def profile_operation(self, operation_name: str):
        """Decorator to profile operation performance."""
        def decorator(func):
            @wraps(func)
            async def async_wrapper(*args, **kwargs):
                return await self._profile_async_operation(operation_name, func, args, kwargs)
            
            @wraps(func)
            def sync_wrapper(*args, **kwargs):
                return self._profile_sync_operation(operation_name, func, args, kwargs)
            
            if asyncio.iscoroutinefunction(func):
                return async_wrapper
            else:
                return sync_wrapper
        
        return decorator
    
    async def _profile_async_operation(self, operation_name: str, func: Callable, args: Tuple, kwargs: Dict) -> Any:
        """Profile async operation."""
        import psutil
        import os
        
        process = psutil.Process(os.getpid())
        start_time = time.time()
        start_memory = process.memory_info().rss
        start_cpu = process.cpu_percent()
        
        try:
            result = await func(*args, **kwargs)
            success = True
            error_count = 0
        except Exception as e:
            self.logger.error(f"Operation {operation_name} failed: {str(e)}")
            result = None
            success = False
            error_count = 1
        
        end_time = time.time()
        end_memory = process.memory_info().rss
        end_cpu = process.cpu_percent()
        
        execution_time = end_time - start_time
        memory_usage = end_memory - start_memory
        cpu_usage = end_cpu - start_cpu
        
        metrics = PerformanceMetrics(
            operation_name=operation_name,
            execution_time=execution_time,
            memory_usage=memory_usage,
            cpu_usage=cpu_usage,
            throughput=1.0 / execution_time if execution_time > 0 else 0.0,
            error_count=error_count,
            success_count=1 if success else 0
        )
        
        self.operation_metrics[operation_name].append(metrics)
        self._analyze_performance(operation_name, metrics)
        
        return result
    
    def _profile_sync_operation(self, operation_name: str, func: Callable, args: Tuple, kwargs: Dict) -> Any:
        """Profile synchronous operation."""
        import psutil
        import os
        
        process = psutil.Process(os.getpid())
        start_time = time.time()
        start_memory = process.memory_info().rss
        
        try:
            result = func(*args, **kwargs)
            success = True
            error_count = 0
        except Exception as e:
            self.logger.error(f"Operation {operation_name} failed: {str(e)}")
            result = None
            success = False
            error_count = 1
        
        end_time = time.time()
        end_memory = process.memory_info().rss
        
        execution_time = end_time - start_time
        memory_usage = end_memory - start_memory
        
        metrics = PerformanceMetrics(
            operation_name=operation_name,
            execution_time=execution_time,
            memory_usage=memory_usage,
            cpu_usage=0.0,  # Simplified for sync operations
            throughput=1.0 / execution_time if execution_time > 0 else 0.0,
            error_count=error_count,
            success_count=1 if success else 0
        )
        
        self.operation_metrics[operation_name].append(metrics)
        self._analyze_performance(operation_name, metrics)
        
        return result
    
    def _analyze_performance(self, operation_name: str, metrics: PerformanceMetrics):
        """Analyze performance metrics and identify bottlenecks."""
        operation_history = self.operation_metrics[operation_name]
        
        if len(operation_history) < 5:
            return  # Need more data points
        
        # Calculate averages
        avg_execution_time = sum(m.execution_time for m in operation_history) / len(operation_history)
        avg_memory_usage = sum(m.memory_usage for m in operation_history) / len(operation_history)
        
        # Identify bottlenecks
        if metrics.execution_time > avg_execution_time * 2:
            bottleneck = {
                "operation": operation_name,
                "type": "execution_time",
                "current_value": metrics.execution_time,
                "average_value": avg_execution_time,
                "severity": "high" if metrics.execution_time > avg_execution_time * 3 else "medium"
            }
            self.bottlenecks.append(bottleneck)
        
        if metrics.memory_usage > avg_memory_usage * 2:
            bottleneck = {
                "operation": operation_name,
                "type": "memory_usage",
                "current_value": metrics.memory_usage,
                "average_value": avg_memory_usage,
                "severity": "high" if metrics.memory_usage > avg_memory_usage * 3 else "medium"
            }
            self.bottlenecks.append(bottleneck)
        
        # Generate optimization suggestions
        self._generate_optimization_suggestions(operation_name, operation_history)
    
    def _generate_optimization_suggestions(self, operation_name: str, history: List[PerformanceMetrics]):
        """Generate optimization suggestions based on performance history."""
        if len(history) < 10:
            return
        
        avg_execution_time = sum(m.execution_time for m in history) / len(history)
        avg_memory_usage = sum(m.memory_usage for m in history) / len(history)
        error_rate = sum(m.error_count for m in history) / len(history)
        
        suggestions = []
        
        if avg_execution_time > 1.0:  # Slow operations
            suggestions.append(f"Consider caching results for {operation_name}")
            suggestions.append(f"Implement batch processing for {operation_name}")
        
        if avg_memory_usage > 100 * 1024 * 1024:  # High memory usage (100MB)
            suggestions.append(f"Implement memory pooling for {operation_name}")
            suggestions.append(f"Consider streaming processing for {operation_name}")
        
        if error_rate > 0.1:  # High error rate
            suggestions.append(f"Add retry logic for {operation_name}")
            suggestions.append(f"Improve error handling for {operation_name}")
        
        self.optimization_suggestions.extend(suggestions)
    
    def get_performance_report(self) -> Dict[str, Any]:
        """Generate comprehensive performance report."""
        report = {
            "operations": {},
            "bottlenecks": self.bottlenecks[-10:],  # Last 10 bottlenecks
            "optimization_suggestions": list(set(self.optimization_suggestions[-20:])),  # Unique suggestions
            "summary": {}
        }
        
        total_operations = 0
        total_execution_time = 0.0
        
        for operation_name, metrics_list in self.operation_metrics.items():
            if not metrics_list:
                continue
            
            operation_stats = {
                "total_calls": len(metrics_list),
                "average_execution_time": sum(m.execution_time for m in metrics_list) / len(metrics_list),
                "average_memory_usage": sum(m.memory_usage for m in metrics_list) / len(metrics_list),
                "average_throughput": sum(m.throughput for m in metrics_list) / len(metrics_list),
                "error_rate": sum(m.error_count for m in metrics_list) / len(metrics_list),
                "success_rate": sum(m.success_count for m in metrics_list) / len(metrics_list)
            }
            
            report["operations"][operation_name] = operation_stats
            total_operations += len(metrics_list)
            total_execution_time += sum(m.execution_time for m in metrics_list)
        
        report["summary"] = {
            "total_operations": total_operations,
            "total_execution_time": total_execution_time,
            "average_operation_time": total_execution_time / total_operations if total_operations > 0 else 0.0,
            "bottleneck_count": len(self.bottlenecks),
            "optimization_suggestions_count": len(set(self.optimization_suggestions))
        }
        
        return report


# Global instances
batch_processor = BatchProcessor()
memory_optimizer = MemoryOptimizer()
parallel_executor = ParallelExecutor()
performance_profiler = PerformanceProfiler()
