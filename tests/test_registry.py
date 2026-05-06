"""Registry lookup + register_model behaviour."""

from pathlib import Path

import lynx


def test_lookup_known_model():
    """Bundled model substring resolves to a real config file."""
    path = lynx.lookup("Qwen/Qwen2-57B-A14B-Instruct")
    assert path is not None
    assert Path(path).is_file()
    assert path.endswith("quant_alpha3_beta4_optimized.json")


def test_lookup_qwen3_30b_uses_aggressive_default():
    """Default for Qwen3-30B-A3B is the aggressive alpha3_beta2 config,
    matching the headline perf numbers in the README."""
    path = lynx.lookup("Qwen/Qwen3-30B-A3B-Instruct-2507")
    assert path is not None
    assert "alpha3_beta2_optimized" in path


def test_lookup_qwen3_235b_uses_alpha3_beta2():
    path = lynx.lookup("Qwen/Qwen3-235B-A22B-Thinking-2507")
    assert path is not None
    assert "qwen3_235b/quant_alpha3_beta2_optimized" in path


def test_lookup_mixtral_variants_pick_correct_config():
    """Mixtral 8x7B uses alpha0.7_beta1; 8x22B uses alpha1_beta1.
    These are validated separately in latency-scripts/run_mixtral_4bmk.sh
    and run_mixtral_8x22b_full_benchmarks.sh."""
    p7b = lynx.lookup("mistralai/Mixtral-8x7B-Instruct-v0.1")
    p22b = lynx.lookup("mistralai/Mixtral-8x22B-Instruct-v0.1")
    assert p7b is not None and "quant_alpha0.7_beta1_optimized" in p7b
    assert p22b is not None and "quant_alpha1_beta1_optimized" in p22b


def test_lookup_deepseek_coder_v2():
    path = lynx.lookup("deepseek-ai/DeepSeek-Coder-V2-Instruct")
    assert path is not None
    assert "deepseek_v2_coder/quant_alpha1_beta1_optimized" in path


def test_lookup_gpt_oss_120b():
    path = lynx.lookup("openai/gpt-oss-120b")
    assert path is not None
    assert "gpt_oss_120b/quant_alpha3_beta2_optimized" in path


def test_all_registry_paths_resolve_to_real_files():
    """Every entry in LYNX_MODEL_REGISTRY must point at a config file
    that's actually shipped in the package."""
    from lynx.registry import LYNX_MODEL_REGISTRY
    for substring in LYNX_MODEL_REGISTRY:
        path = lynx.lookup(f"any/{substring}-instruct")
        assert path is not None and Path(path).is_file(), (
            f"registry entry {substring!r} resolves to {path!r} which doesn't exist"
        )


def test_lookup_unknown_model_returns_none():
    assert lynx.lookup("some/totally-unknown-model") is None
    assert lynx.lookup("") is None


def test_register_model_runtime_override(tmp_path):
    policy_file = tmp_path / "custom_policy.json"
    policy_file.write_text('{"policy": "do-nothing"}')
    lynx.register_model("acme/super-7b-moe", str(policy_file))
    resolved = lynx.lookup("acme/super-7b-moe-Instruct")
    assert resolved == str(policy_file)


def test_register_model_rejects_empty():
    import pytest
    with pytest.raises(ValueError):
        lynx.register_model("", "/path/to/policy.json")
    with pytest.raises(ValueError):
        lynx.register_model("model-name", "")


def test_lookup_is_case_insensitive():
    # Qwen2 substring "qwen2-57b-a14b" should match different casings.
    assert lynx.lookup("QWEN/QWEN2-57B-A14B-INSTRUCT") is not None
    assert lynx.lookup("qwen/Qwen2-57B-a14b-Chat") is not None
