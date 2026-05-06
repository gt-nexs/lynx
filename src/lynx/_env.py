"""Cached env-var lookups for the Lynx plugin gate. We read these once
at plugin install time so the off-by-default path costs nothing on the
hot path.
"""

import os
from functools import lru_cache


@lru_cache(maxsize=1)
def is_enabled() -> bool:
    """``True`` iff ``VLLM_LYNX_ENABLED`` is set to a truthy value
    (``1``, ``true``, ``yes``, case-insensitive). Cached: first call
    fixes the answer for the lifetime of the process."""
    return os.environ.get("VLLM_LYNX_ENABLED", "").lower() in ("1", "true", "yes")


@lru_cache(maxsize=1)
def metrics_enabled() -> bool:
    return os.environ.get("VLLM_LYNX_METRICS", "").lower() in ("1", "true", "yes")


def config_file_override() -> str | None:
    """Optional explicit policy-file path. Overrides the registry lookup.
    Not cached (env may be set late by ops tooling)."""
    val = os.environ.get("VLLM_LYNX_CONFIG_FILE", "").strip()
    return val or None


def profile_dir() -> str:
    """Output dir for ablation telemetry CSVs (only consulted when
    metrics_enabled())."""
    return os.environ.get("VLLM_LYNX_PROFILE_DIR", "./profiling_output")
