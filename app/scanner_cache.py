import hashlib
import json
from collections import OrderedDict
from typing import Dict, List, Optional, Any
import structlog

from tueri.input_scanners.base import Scanner as InputScanner
from tueri.output_scanners.base import Scanner as OutputScanner
from tueri.vault import Vault
from .config import ScannerConfig

LOGGER = structlog.getLogger(__name__)

class ScannerCacheManager:
    """
    Manages caching of scanner instances using LRU (Least Recently Used) eviction.
    """

    def __init__(self, max_cache_size: int = 64):
        """Initialize the scanner cache manager with LRU eviction."""
        self._input_cache: OrderedDict[str, InputScanner] = OrderedDict()
        self._output_cache: OrderedDict[str, OutputScanner] = OrderedDict()
        self._max_cache_size = max_cache_size

    def _create_scanner_cache_key(self, scanner_type: str, params: Dict[str, Any]) -> str:
        """Create a cache key for a scanner based on its type and parameters."""
        cache_data = {
            "scanner_type": scanner_type,
            "params": params
        }
        config_str = json.dumps(cache_data, sort_keys=True)
        return hashlib.md5(config_str.encode()).hexdigest()

    def _evict_lru_if_needed(self, cache: OrderedDict) -> None:
        """Remove least recently used entry if cache is at capacity."""
        if len(cache) >= self._max_cache_size:
            # Remove least recently used (first item in OrderedDict)
            evicted_key, _ = cache.popitem(last=False)
            LOGGER.debug("Evicted LRU scanner from cache", cache_key=evicted_key)

    def get_input_scanner(self, scanner_type: str, params: Dict[str, Any], scanner_factory_func: callable) -> InputScanner:
        """Get an input scanner from cache or create a new one using LRU eviction."""
        cache_key = self._create_scanner_cache_key(scanner_type, params)

        # Check if we have a cached scanner
        if cache_key in self._input_cache:
            # Move to end (mark as most recently used)
            self._input_cache.move_to_end(cache_key)
            LOGGER.debug("Using cached input scanner", scanner_type=scanner_type, cache_key=cache_key)
            return self._input_cache[cache_key]

        # Create new scanner
        scanner = scanner_factory_func(scanner_type, params)

        # Evict LRU if cache is full
        self._evict_lru_if_needed(self._input_cache)

        # Cache the new scanner (automatically becomes most recently used)
        self._input_cache[cache_key] = scanner
        LOGGER.debug("Cached new input scanner", scanner_type=scanner_type, cache_key=cache_key)

        return scanner

    def get_output_scanner(self, scanner_type: str, params: Dict[str, Any], scanner_factory_func: callable) -> OutputScanner:
        """Get an output scanner from cache or create a new one using LRU eviction."""
        cache_key = self._create_scanner_cache_key(scanner_type, params)

        # Check if we have a cached scanner
        if cache_key in self._output_cache:
            # Move to end (mark as most recently used)
            self._output_cache.move_to_end(cache_key)
            LOGGER.debug("Using cached output scanner", scanner_type=scanner_type, cache_key=cache_key)
            return self._output_cache[cache_key]

        # Create new scanner
        scanner = scanner_factory_func(scanner_type, params)

        # Evict LRU if cache is full
        self._evict_lru_if_needed(self._output_cache)

        # Cache the new scanner (automatically becomes most recently used)
        self._output_cache[cache_key] = scanner
        LOGGER.debug("Cached new output scanner", scanner_type=scanner_type, cache_key=cache_key)

        return scanner

    def get_cache_stats(self) -> Dict[str, Any]:
        """Get statistics about the cache usage."""
        return {
            "input_cache_size": len(self._input_cache),
            "output_cache_size": len(self._output_cache),
            "max_cache_size": self._max_cache_size,
            "eviction_policy": "LRU"
        }

    def clear_cache(self) -> None:
        """Clear all cached scanners from both input and output caches."""
        self._input_cache.clear()
        self._output_cache.clear()
        LOGGER.info("Cleared all scanner caches")


# Global scanner cache manager instance with LRU eviction
_scanner_cache_manager = ScannerCacheManager()


def get_scanner_cache_manager() -> ScannerCacheManager:
    """Get the global scanner cache manager instance."""
    return _scanner_cache_manager
