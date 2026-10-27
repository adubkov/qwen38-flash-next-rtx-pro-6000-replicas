"""shared_ram PLE backend plugin (SGLang entry-point group sglang.srt.plugins)."""
from .plugin import register  # noqa: F401  (entry point target)

__all__ = ["register"]
