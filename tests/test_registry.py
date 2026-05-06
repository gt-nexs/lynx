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
