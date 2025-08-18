"""
Advanced caching strategies for the document analyzer system.
"""

import time
import hashlib
import pickle
from typing import Any, Dict, Optional, List, Callable, Tuple
from datetime import datetime, timedelta
from enum import Enum
from dataclasses import dataclass
from collections import OrderedDict
import threading
import asyncio

from src.core.logging import LoggerMixin
from src.core.config import settings


class CacheStrategy(Enum):
    LRU = "lru"  # Least Recently Used
    LFU = "lfu"  # Least Frequently Used
    TTL = "ttl"  # Time To Live
    ADAPTIVE = "adaptive"  # Adaptive replacement cache


@dataclass
class CacheEntry:
    key: str
    value: Any
    created_at: float
    last_accessed: float
    access_count: int
    ttl: Optional[float] = None
    size: int = 0


class LRUCache:
    """Least Recently Used cache implementation."""
    
    def __init__(self, max_size: int = 1000):
        self.max_size = max_size
        self.cache: OrderedDict = OrderedDict()
        self.lock = threading.RLock()
    
    def get(self, key: str) -> Optional[Any]:
        """Get value from cache and mark as recently used."""
        with self.lock:
            if key in self.cache:
                # Move to end (most recently used)
                value = self.cache.pop(key)
                self.cache[key] = value
                return value.value
            return None
    
    def put(self, key: str, value: Any, ttl: Optional[float] = None) -> None:
        """Put value in cache."""
        with self.lock:
            current_time = time.time()
            
            if key in self.cache:
                # Update existing entry
                self.cache.pop(key)
            elif len(self.cache) >= self.max_size:
                # Remove least recently used
                self.cache.popitem(last=False)
            
            entry = CacheEntry(
                key=key,
                value=value,
                created_at=current_time,
                last_accessed=current_time,
                access_count=1,
                ttl=ttl,
                size=self._calculate_size(value)
            )
            
            self.cache[key] = entry
    
    def delete(self, key: str) -> bool:
        """Delete entry from cache."""
        with self.lock:
            if key in self.cache:
                del self.cache[key]
                return True
            return False
    
    def clear(self) -> None:
        """Clear all entries from cache."""
        with self.lock:
            self.cache.clear()
    
    def _calculate_size(self, value: Any) -> int:
        """Calculate approximate size of value in bytes."""
        try:
            return len(pickle.dumps(value))
        except:
            return 1024  # Default size estimate


class LFUCache:
    """Least Frequently Used cache implementation."""
    
    def __init__(self, max_size: int = 1000):
        self.max_size = max_size
        self.cache: Dict[str, CacheEntry] = {}
        self.frequencies: Dict[int, List[str]] = {}
        self.min_frequency = 0
        self.lock = threading.RLock()
    
    def get(self, key: str) -> Optional[Any]:
        """Get value from cache and increment frequency."""
        with self.lock:
            if key not in self.cache:
                return None
            
            entry = self.cache[key]
            self._increment_frequency(key, entry)
            entry.last_accessed = time.time()
            
            return entry.value
    
    def put(self, key: str, value: Any, ttl: Optional[float] = None) -> None:
        """Put value in cache."""
        with self.lock:
            current_time = time.time()
            
            if key in self.cache:
                # Update existing entry
                entry = self.cache[key]
                entry.value = value
                entry.created_at = current_time
                entry.ttl = ttl
                self._increment_frequency(key, entry)
                return
            
            if len(self.cache) >= self.max_size:
                self._evict_lfu()
            
            entry = CacheEntry(
                key=key,
                value=value,
                created_at=current_time,
                last_accessed=current_time,
                access_count=1,
                ttl=ttl,
                size=self._calculate_size(value)
            )
            
            self.cache[key] = entry
            
            # Add to frequency 1
            if 1 not in self.frequencies:
                self.frequencies[1] = []
            self.frequencies[1].append(key)
            self.min_frequency = 1
    
    def _increment_frequency(self, key: str, entry: CacheEntry) -> None:
        """Increment frequency of a cache entry."""
        old_freq = entry.access_count
        new_freq = old_freq + 1
        
        # Remove from old frequency list
        self.frequencies[old_freq].remove(key)
        if not self.frequencies[old_freq] and old_freq == self.min_frequency:
            self.min_frequency += 1
        
        # Add to new frequency list
        if new_freq not in self.frequencies:
            self.frequencies[new_freq] = []
        self.frequencies[new_freq].append(key)
        
        entry.access_count = new_freq
    
    def _evict_lfu(self) -> None:
        """Evict least frequently used entry."""
        if self.min_frequency in self.frequencies and self.frequencies[self.min_frequency]:
            key_to_remove = self.frequencies[self.min_frequency].pop(0)
            del self.cache[key_to_remove]
    
    def _calculate_size(self, value: Any) -> int:
        """Calculate approximate size of value in bytes."""
        try:
            return len(pickle.dumps(value))
        except:
            return 1024


class TTLCache:
    """Time To Live cache implementation."""
    
    def __init__(self, max_size: int = 1000, default_ttl: float = 3600):
        self.max_size = max_size
        self.default_ttl = default_ttl
        self.cache: Dict[str, CacheEntry] = {}
        self.lock = threading.RLock()
    
    def get(self, key: str) -> Optional[Any]:
        """Get value from cache if not expired."""
        with self.lock:
            if key not in self.cache:
                return None
            
            entry = self.cache[key]
            current_time = time.time()
            
            # Check if expired
            if entry.ttl and (current_time - entry.created_at) > entry.ttl:
                del self.cache[key]
                return None
            
            entry.last_accessed = current_time
            entry.access_count += 1
            
            return entry.value
    
    def put(self, key: str, value: Any, ttl: Optional[float] = None) -> None:
        """Put value in cache with TTL."""
        with self.lock:
            current_time = time.time()
            
            # Clean expired entries if cache is full
            if len(self.cache) >= self.max_size:
                self._clean_expired()
                
                # If still full, remove oldest entry
                if len(self.cache) >= self.max_size:
                    oldest_key = min(self.cache.keys(), key=lambda k: self.cache[k].created_at)
                    del self.cache[oldest_key]
            
            entry = CacheEntry(
                key=key,
                value=value,
                created_at=current_time,
                last_accessed=current_time,
                access_count=1,
                ttl=ttl or self.default_ttl,
                size=self._calculate_size(value)
            )
            
            self.cache[key] = entry
    
    def _clean_expired(self) -> None:
        """Remove expired entries from cache."""
        current_time = time.time()
        expired_keys = []
        
        for key, entry in self.cache.items():
            if entry.ttl and (current_time - entry.created_at) > entry.ttl:
                expired_keys.append(key)
        
        for key in expired_keys:
            del self.cache[key]
    
    def _calculate_size(self, value: Any) -> int:
        """Calculate approximate size of value in bytes."""
        try:
            return len(pickle.dumps(value))
        except:
            return 1024


class AdaptiveCache:
    """Adaptive Replacement Cache (ARC) implementation."""
    
    def __init__(self, max_size: int = 1000):
        self.max_size = max_size
        self.p = 0  # Target size for T1
        
        # T1: Recent cache entries
        self.t1: OrderedDict = OrderedDict()
        # T2: Frequent cache entries  
        self.t2: OrderedDict = OrderedDict()
        # B1: Ghost entries for T1
        self.b1: OrderedDict = OrderedDict()
        # B2: Ghost entries for T2
        self.b2: OrderedDict = OrderedDict()
        
        self.lock = threading.RLock()
    
    def get(self, key: str) -> Optional[Any]:
        """Get value from adaptive cache."""
        with self.lock:
            current_time = time.time()
            
            if key in self.t1:
                entry = self.t1.pop(key)
                entry.last_accessed = current_time
                entry.access_count += 1
                self.t2[key] = entry
                return entry.value
            
            if key in self.t2:
                entry = self.t2.pop(key)
                entry.last_accessed = current_time
                entry.access_count += 1
                self.t2[key] = entry  # Move to end
                return entry.value
            
            return None
    
    def put(self, key: str, value: Any, ttl: Optional[float] = None) -> None:
        """Put value in adaptive cache."""
        with self.lock:
            current_time = time.time()
            
            entry = CacheEntry(
                key=key,
                value=value,
                created_at=current_time,
                last_accessed=current_time,
                access_count=1,
                ttl=ttl,
                size=self._calculate_size(value)
            )
            
            # Case 1: Key in B1 (was recently evicted from T1)
            if key in self.b1:
                self.p = min(self.max_size, self.p + max(1, len(self.b2) // len(self.b1)))
                self._replace(key)
                self.b1.pop(key)
                self.t2[key] = entry
                return
            
            # Case 2: Key in B2 (was recently evicted from T2)
            if key in self.b2:
                self.p = max(0, self.p - max(1, len(self.b1) // len(self.b2)))
                self._replace(key)
                self.b2.pop(key)
                self.t2[key] = entry
                return
            
            # Case 3: Key not in cache or ghost lists
            if len(self.t1) + len(self.t2) < self.max_size:
                # Cache not full
                if len(self.t1) + len(self.b1) >= self.max_size:
                    if len(self.t1) < self.max_size:
                        self.b1.popitem(last=False)
                        self._replace(key)
                    else:
                        self.t1.popitem(last=False)
                elif len(self.t1) + len(self.t2) + len(self.b1) + len(self.b2) >= self.max_size:
                    if len(self.t1) + len(self.t2) + len(self.b1) + len(self.b2) >= 2 * self.max_size:
                        self.b2.popitem(last=False)
                    self._replace(key)
            else:
                # Cache is full
                self._replace(key)
            
            self.t1[key] = entry
    
    def _replace(self, key: str) -> None:
        """Replace cache entry according to ARC algorithm."""
        if len(self.t1) >= 1 and ((key in self.b2 and len(self.t1) == self.p) or len(self.t1) > self.p):
            # Move from T1 to B1
            old_key, old_entry = self.t1.popitem(last=False)
            self.b1[old_key] = old_entry
        else:
            # Move from T2 to B2
            if self.t2:
                old_key, old_entry = self.t2.popitem(last=False)
                self.b2[old_key] = old_entry
    
    def _calculate_size(self, value: Any) -> int:
        """Calculate approximate size of value in bytes."""
        try:
            return len(pickle.dumps(value))
        except:
            return 1024


class CacheManager(LoggerMixin):
    """Manager for different caching strategies."""
    
    def __init__(self):
        self.caches: Dict[str, Any] = {}
        self.default_strategy = CacheStrategy.LRU
        self.hit_counts: Dict[str, int] = {}
        self.miss_counts: Dict[str, int] = {}
    
    def create_cache(self, name: str, strategy: CacheStrategy, max_size: int = 1000, **kwargs) -> None:
        """Create a cache with specified strategy."""
        if strategy == CacheStrategy.LRU:
            cache = LRUCache(max_size)
        elif strategy == CacheStrategy.LFU:
            cache = LFUCache(max_size)
        elif strategy == CacheStrategy.TTL:
            cache = TTLCache(max_size, kwargs.get('default_ttl', 3600))
        elif strategy == CacheStrategy.ADAPTIVE:
            cache = AdaptiveCache(max_size)
        else:
            raise ValueError(f"Unknown cache strategy: {strategy}")
        
        self.caches[name] = cache
        self.hit_counts[name] = 0
        self.miss_counts[name] = 0
        
        self.logger.info(f"Created {strategy.value} cache: {name}")
    
    def get(self, cache_name: str, key: str) -> Optional[Any]:
        """Get value from specified cache."""
        if cache_name not in self.caches:
            return None
        
        value = self.caches[cache_name].get(key)
        
        if value is not None:
            self.hit_counts[cache_name] += 1
        else:
            self.miss_counts[cache_name] += 1
        
        return value
    
    def put(self, cache_name: str, key: str, value: Any, **kwargs) -> None:
        """Put value in specified cache."""
        if cache_name not in self.caches:
            return
        
        self.caches[cache_name].put(key, value, **kwargs)
    
    def get_cache_stats(self, cache_name: str) -> Dict[str, Any]:
        """Get statistics for a cache."""
        if cache_name not in self.caches:
            return {}
        
        hits = self.hit_counts[cache_name]
        misses = self.miss_counts[cache_name]
        total = hits + misses
        
        return {
            "hits": hits,
            "misses": misses,
            "hit_rate": hits / total if total > 0 else 0.0,
            "total_requests": total
        }
    
    def clear_cache(self, cache_name: str) -> None:
        """Clear specified cache."""
        if cache_name in self.caches:
            self.caches[cache_name].clear()
            self.hit_counts[cache_name] = 0
            self.miss_counts[cache_name] = 0


# Global cache manager instance
cache_manager = CacheManager()

# Initialize default caches
cache_manager.create_cache("embeddings", CacheStrategy.LRU, max_size=500)
cache_manager.create_cache("classifications", CacheStrategy.TTL, max_size=1000, default_ttl=1800)
cache_manager.create_cache("documents", CacheStrategy.ADAPTIVE, max_size=200)
cache_manager.create_cache("search_results", CacheStrategy.LFU, max_size=300)
