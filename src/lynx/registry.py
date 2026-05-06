"""Lookup table mapping HuggingFace model identifiers to bundled Lynx
routing-policy configs. Used when ``--lynx`` is set without an explicit
``--lynx-config-file``.

Keys are case-insensitive substrings of the ``--model`` argument; the
first match wins. Values are paths relative to the ``configs`` directory
within this package.
"""

from importlib import resources

# Substring → bundled config path. Order matters when multiple substrings
# could match the same model name (the first match in dict insertion order
# is used). Keep the more specific keys first.
LYNX_MODEL_REGISTRY: dict[str, str] = {
    # Qwen3 dense MoE families (128 experts × top-8). The
    # ``quant_alpha3_beta2_optimized`` policy is the more aggressive of
    # the two validated configs and is what produces the headline TPOT
    # speedup; switch to ``quant_alpha1_beta1_optimized`` (β/α=1.0) via
    # ``--lynx-config-file`` if you want the conservative variant.
    "qwen3-235b-a22b": "qwen3_235b/quant_alpha1_beta2_optimized.json",
    "qwen3-30b-a3b": "qwen3_30b/quant_alpha3_beta2_optimized.json",
    # Qwen2 MoE family (64 experts × top-8)
    "qwen2-57b-a14b": "qwen2/quant_alpha3_beta4_optimized.json",
    # Mixtral families (8 experts × top-2)
    "mixtral-8x7b": "mixtral/quant_alpha0.7_beta1_optimized.json",
    "mixtral-8x22b": "mixtral/quant_alpha0.7_beta1_optimized.json",
}


def register_model(model_substring: str, config_path: str) -> None:
    """Add or override a model→config mapping at runtime. ``config_path``
    must be either an absolute filesystem path OR a path relative to the
    bundled ``configs/`` directory.

    Lookup is case-insensitive substring match against the user-supplied
    ``--model`` argument; first matching key in dict insertion order
    wins. Use this to teach Lynx about a new model family without
    forking the package.

    Example:
        >>> import lynx
        >>> lynx.register_model("my-org/my-7b-moe", "/etc/lynx/policy.json")
    """
    if not model_substring:
        raise ValueError("model_substring must be non-empty")
    if not config_path:
        raise ValueError("config_path must be non-empty")
    LYNX_MODEL_REGISTRY[model_substring.lower()] = config_path


def lookup(model_name: str) -> str | None:
    """Return the absolute path to the bundled Lynx config for the given
    model, or None if no entry matches.

    Matching is case-insensitive substring. The first matching key in
    ``LYNX_MODEL_REGISTRY`` insertion order wins.
    """
    if not model_name:
        return None
    needle = model_name.lower()
    configs_root = resources.files(__package__).joinpath("configs")
    for key, rel_path in LYNX_MODEL_REGISTRY.items():
        if key in needle:
            # Absolute path: pass through. Relative path: resolve against
            # bundled configs/ directory.
            import os
            if os.path.isabs(rel_path):
                return rel_path
            return str(configs_root.joinpath(rel_path))
    return None
