"""Confirm that installing lynx-vllm without setting VLLM_LYNX_ENABLED
leaves vllm completely untouched. This is the "do no harm" promise.
"""

import os
import subprocess
import sys


def test_vllm_unchanged_when_lynx_unset():
    """Spawn a fresh subprocess (no env var), import vllm, verify that
    none of the lynx-targeted symbols are patched."""
    script = (
        "import sys; "
        "import vllm; "
        "from vllm.model_executor.layers.fused_moe.layer import FusedMoE; "
        "from vllm.config.model import ModelConfig; "
        "from vllm.v1.worker.gpu_worker import Worker; "
        "from vllm.v1.worker.gpu_model_runner import GPUModelRunner; "
        "patched = ["
        "    getattr(FusedMoE.__init__, '_lynx_patched', False),"
        "    getattr(ModelConfig.__post_init__, '_lynx_patched', False),"
        "    getattr(Worker.__init__, '_lynx_patched', False),"
        "    getattr(Worker.initialize_from_config, '_lynx_patched', False),"
        "    getattr(GPUModelRunner.execute_model, '_lynx_patched', False),"
        "    getattr(GPUModelRunner._dummy_run, '_lynx_patched', False),"
        "]; "
        "sys.exit(0 if not any(patched) else 1)"
    )
    env = {k: v for k, v in os.environ.items() if k != "VLLM_LYNX_ENABLED"}
    result = subprocess.run(
        [sys.executable, "-c", script],
        env=env,
        capture_output=True,
        text=True,
        timeout=120,
    )
    assert result.returncode == 0, (
        "vllm symbols were modified despite VLLM_LYNX_ENABLED being unset.\n"
        f"stdout: {result.stdout!r}\nstderr: {result.stderr!r}"
    )
