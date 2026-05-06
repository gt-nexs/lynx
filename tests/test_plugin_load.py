"""Plugin entry-point + install() behaviour tests.

These tests do not require a GPU — they only validate that the plugin
discovers correctly through Python's entry-point mechanism and that
install() is a true no-op when the env-var gate is unset.
"""

import os
import subprocess
import sys
from importlib.metadata import entry_points


def test_entry_point_registered():
    """`vllm.general_plugins['lynx']` must resolve to lynx._plugin:install."""
    eps = entry_points(group="vllm.general_plugins")
    matches = [ep for ep in eps if ep.name == "lynx"]
    assert len(matches) == 1, f"expected exactly one 'lynx' plugin, found: {matches}"
    assert matches[0].value == "lynx._plugin:install"


def test_install_is_noop_when_disabled(monkeypatch):
    """Calling install() with VLLM_LYNX_ENABLED unset must NOT mutate
    any vllm symbol."""
    monkeypatch.delenv("VLLM_LYNX_ENABLED", raising=False)

    # Reset the env-var cache (lru_cache holds the previous answer).
    from lynx import _env
    _env.is_enabled.cache_clear()

    from lynx import _plugin
    _plugin.install()

    # Make sure the FusedMoE patch is NOT applied.
    from vllm.model_executor.layers.fused_moe.layer import FusedMoE
    assert not getattr(FusedMoE.__init__, "_lynx_patched", False), (
        "lynx plugin should be a no-op when VLLM_LYNX_ENABLED is unset, "
        "but FusedMoE.__init__ was patched anyway"
    )


def test_install_patches_when_enabled():
    """Run install() in a subprocess with VLLM_LYNX_ENABLED=1 and confirm
    FusedMoE.__init__ ends up patched.

    A subprocess is used because the no-op-when-off test in this same
    process already imported vllm without lynx, and once vllm's
    FusedMoE has been imported, applying the patch in the SAME process
    would carry into other tests. Subprocess gives a clean env."""
    script = (
        "import os; "
        "from lynx import _plugin; "
        "_plugin.install(); "
        "from vllm.model_executor.layers.fused_moe.layer import FusedMoE; "
        "import sys; "
        "sys.exit(0 if getattr(FusedMoE.__init__, '_lynx_patched', False) else 1)"
    )
    env = {**os.environ, "VLLM_LYNX_ENABLED": "1"}
    result = subprocess.run(
        [sys.executable, "-c", script],
        env=env,
        capture_output=True,
        text=True,
        timeout=120,
    )
    assert result.returncode == 0, (
        f"plugin failed to patch FusedMoE under VLLM_LYNX_ENABLED=1.\n"
        f"stdout: {result.stdout!r}\nstderr: {result.stderr!r}"
    )
