"""
Performance utilities for caching and optimization.

This module provides utilities for improving app performance through
intelligent caching, memory management, and data optimization.
"""

import streamlit as st
import pandas as pd
import time
from functools import wraps
from typing import Any, Callable, Dict, Optional
import hashlib
import json


class PerformanceMonitor:
    """Monitor and track application performance metrics."""

    def __init__(self):
        """Initialize the performance monitor."""
        self.timings = {}
        self.cache_hits = {}
        self.cache_misses = {}

    def time_operation(self, operation_name: str):
        """Decorator to time operations."""
        def decorator(func: Callable) -> Callable:
            @wraps(func)
            def wrapper(*args, **kwargs) -> Any:
                start_time = time.time()
                try:
                    result = func(*args, **kwargs)
                    success = True
                except Exception as e:
                    result = e
                    success = False
                finally:
                    end_time = time.time()
                    duration = end_time - start_time

                    # Store timing
                    if operation_name not in self.timings:
                        self.timings[operation_name] = []
                    self.timings[operation_name].append({
                        'duration': duration,
                        'success': success,
                        'timestamp': start_time
                    })

                    # Show timing in sidebar if in debug mode
                    if st.session_state.get('debug_mode', False):
                        st.sidebar.info(f"{operation_name}: {duration:.2f}s")

                if not success:
                    raise result

                return result
            return wrapper
        return decorator

    def get_performance_stats(self) -> Dict:
        """Get performance statistics."""
        stats = {}
        for operation, timings in self.timings.items():
            durations = [t['duration'] for t in timings]
            stats[operation] = {
                'count': len(durations),
                'avg_time': sum(durations) / len(durations),
                'min_time': min(durations),
                'max_time': max(durations),
                'total_time': sum(durations),
                'success_rate': sum(1 for t in timings if t['success']) / len(timings)
            }
        return stats


class DataCache:
    """Enhanced caching system for large datasets."""

    def __init__(self, max_size_mb: int = 100):
        """
        Initialize the data cache.

        Args:
            max_size_mb: Maximum cache size in megabytes
        """
        self.max_size_mb = max_size_mb
        self.cache = {}
        self.access_times = {}
        self.sizes = {}

    def _calculate_size_mb(self, data: Any) -> float:
        """Estimate size of data in MB."""
        if isinstance(data, pd.DataFrame):
            return data.memory_usage(deep=True).sum() / (1024 * 1024)
        elif isinstance(data, (dict, list)):
            return len(str(data)) / (1024 * 1024)
        else:
            return 0.1  # Default small size

    def _generate_key(self, *args, **kwargs) -> str:
        """Generate cache key from arguments."""
        key_data = {'args': args, 'kwargs': kwargs}
        key_str = json.dumps(key_data, sort_keys=True, default=str)
        return hashlib.md5(key_str.encode()).hexdigest()

    def _evict_lru(self):
        """Evict least recently used items until under size limit."""
        while self._get_total_size() > self.max_size_mb and self.cache:
            # Find least recently used key
            lru_key = min(self.access_times.keys(), key=lambda k: self.access_times[k])
            self._remove_item(lru_key)

    def _remove_item(self, key: str):
        """Remove item from cache."""
        if key in self.cache:
            del self.cache[key]
        if key in self.access_times:
            del self.access_times[key]
        if key in self.sizes:
            del self.sizes[key]

    def _get_total_size(self) -> float:
        """Get total cache size in MB."""
        return sum(self.sizes.values())

    def get(self, key: str) -> Optional[Any]:
        """Get item from cache."""
        if key in self.cache:
            self.access_times[key] = time.time()
            return self.cache[key]
        return None

    def put(self, key: str, value: Any):
        """Put item in cache."""
        size = self._calculate_size_mb(value)

        # Don't cache items that are too large
        if size > self.max_size_mb * 0.5:
            return

        # Remove existing item if present
        if key in self.cache:
            self._remove_item(key)

        # Add new item
        self.cache[key] = value
        self.access_times[key] = time.time()
        self.sizes[key] = size

        # Evict if necessary
        self._evict_lru()

    def cached_function(self, func: Callable) -> Callable:
        """Decorator for caching function results."""
        @wraps(func)
        def wrapper(*args, **kwargs) -> Any:
            cache_key = self._generate_key(func.__name__, *args, **kwargs)

            # Try to get from cache
            result = self.get(cache_key)
            if result is not None:
                return result

            # Compute and cache
            result = func(*args, **kwargs)
            self.put(cache_key, result)
            return result

        return wrapper

    def clear(self):
        """Clear all cache."""
        self.cache.clear()
        self.access_times.clear()
        self.sizes.clear()

    def get_stats(self) -> Dict:
        """Get cache statistics."""
        return {
            'total_items': len(self.cache),
            'total_size_mb': self._get_total_size(),
            'max_size_mb': self.max_size_mb,
            'utilization': self._get_total_size() / self.max_size_mb,
            'items': list(self.cache.keys())
        }


class DataOptimizer:
    """Optimize DataFrames for better performance."""

    @staticmethod
    def optimize_dataframe(df: pd.DataFrame) -> pd.DataFrame:
        """
        Optimize DataFrame memory usage.

        Args:
            df: DataFrame to optimize

        Returns:
            Optimized DataFrame
        """
        if df.empty:
            return df

        optimized = df.copy()

        # Convert object columns to categories where appropriate
        for col in optimized.select_dtypes(include=['object']).columns:
            if optimized[col].nunique() / len(optimized) < 0.5:  # Less than 50% unique values
                optimized[col] = optimized[col].astype('category')

        # Optimize numeric columns
        for col in optimized.select_dtypes(include=['int64', 'float64']).columns:
            # Check if we can use smaller int types
            if optimized[col].dtype == 'int64':
                if optimized[col].min() >= -128 and optimized[col].max() <= 127:
                    optimized[col] = optimized[col].astype('int8')
                elif optimized[col].min() >= -32768 and optimized[col].max() <= 32767:
                    optimized[col] = optimized[col].astype('int16')
                elif optimized[col].min() >= -2147483648 and optimized[col].max() <= 2147483647:
                    optimized[col] = optimized[col].astype('int32')

            # Check if we can use float32 instead of float64
            elif optimized[col].dtype == 'float64':
                if optimized[col].min() >= -3.4e38 and optimized[col].max() <= 3.4e38:
                    optimized[col] = optimized[col].astype('float32')

        return optimized

    @staticmethod
    def sample_large_dataset(df: pd.DataFrame, max_rows: int = 10000) -> pd.DataFrame:
        """
        Sample large datasets for visualization.

        Args:
            df: DataFrame to sample
            max_rows: Maximum number of rows to keep

        Returns:
            Sampled DataFrame
        """
        if len(df) <= max_rows:
            return df

        # Stratified sampling to preserve distribution
        if 'events' in df.columns:
            # Sample proportionally by event type
            sampled_dfs = []
            for event in df['events'].unique():
                event_df = df[df['events'] == event]
                event_sample_size = min(len(event_df), max(1, int(max_rows * len(event_df) / len(df))))
                sampled_dfs.append(event_df.sample(n=event_sample_size, random_state=42))
            return pd.concat(sampled_dfs, ignore_index=True)
        else:
            # Simple random sampling
            return df.sample(n=max_rows, random_state=42)


class ProgressTracker:
    """Track progress of long-running operations."""

    def __init__(self):
        """Initialize the progress tracker."""
        self.operations = {}

    def start_operation(self, operation_id: str, description: str, total_steps: int = 100):
        """Start tracking an operation."""
        self.operations[operation_id] = {
            'description': description,
            'total_steps': total_steps,
            'current_step': 0,
            'start_time': time.time(),
            'status': 'running'
        }

    def update_progress(self, operation_id: str, current_step: int, status_text: str = ""):
        """Update operation progress."""
        if operation_id in self.operations:
            self.operations[operation_id]['current_step'] = current_step
            if status_text:
                self.operations[operation_id]['status_text'] = status_text

    def complete_operation(self, operation_id: str):
        """Mark operation as complete."""
        if operation_id in self.operations:
            self.operations[operation_id]['status'] = 'complete'
            self.operations[operation_id]['end_time'] = time.time()

    def get_progress(self, operation_id: str) -> Optional[Dict]:
        """Get progress for an operation."""
        return self.operations.get(operation_id)


# Global instances
_performance_monitor = None
_data_cache = None
_progress_tracker = None


def get_performance_monitor() -> PerformanceMonitor:
    """Get global performance monitor instance."""
    global _performance_monitor
    if _performance_monitor is None:
        _performance_monitor = PerformanceMonitor()
    return _performance_monitor


def get_data_cache() -> DataCache:
    """Get global data cache instance."""
    global _data_cache
    if _data_cache is None:
        _data_cache = DataCache()
    return _data_cache


def get_progress_tracker() -> ProgressTracker:
    """Get global progress tracker instance."""
    global _progress_tracker
    if _progress_tracker is None:
        _progress_tracker = ProgressTracker()
    return _progress_tracker


# Decorators for easy use
def monitor_performance(operation_name: str):
    """Decorator to monitor performance of functions."""
    return get_performance_monitor().time_operation(operation_name)


def cache_result(func: Callable) -> Callable:
    """Decorator to cache function results."""
    return get_data_cache().cached_function(func)


def optimize_for_display(df: pd.DataFrame, max_rows: int = 5000) -> pd.DataFrame:
    """
    Optimize DataFrame for display in Streamlit.

    Args:
        df: DataFrame to optimize
        max_rows: Maximum rows for display

    Returns:
        Optimized DataFrame
    """
    # Sample if too large
    if len(df) > max_rows:
        df = DataOptimizer.sample_large_dataset(df, max_rows)

    # Optimize memory usage
    df = DataOptimizer.optimize_dataframe(df)

    return df


def show_performance_metrics():
    """Show performance metrics in sidebar (for debugging)."""
    if st.session_state.get('debug_mode', False):
        with st.sidebar.expander("⚡ Performance Metrics"):
            monitor = get_performance_monitor()
            stats = monitor.get_performance_stats()

            for operation, metrics in stats.items():
                st.write(f"**{operation}**")
                st.write(f"Avg: {metrics['avg_time']:.3f}s")
                st.write(f"Count: {metrics['count']}")

            cache = get_data_cache()
            cache_stats = cache.get_stats()
            st.write(f"**Cache Usage**")
            st.write(f"Items: {cache_stats['total_items']}")
            st.write(f"Size: {cache_stats['total_size_mb']:.2f} MB")


def enable_debug_mode():
    """Enable debug mode for performance monitoring."""
    st.session_state['debug_mode'] = True