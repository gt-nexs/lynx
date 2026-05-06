"""End-to-end integration: spin up `vllm serve` for Qwen2-57B-A14B-Instruct
with VLLM_LYNX_ENABLED=1, hit it with a single completion, kill it,
and verify the server log contains the Lynx state-init markers.

Requires GPUs 0,1 free and ~60GB GPU memory each. Skipped unless
LYNX_RUN_INTEGRATION=1 is set.
"""

import os
import signal
import subprocess
import sys
import time
from pathlib import Path

import pytest


pytestmark = pytest.mark.skipif(
    os.environ.get("LYNX_RUN_INTEGRATION", "").lower() not in ("1", "true", "yes"),
    reason="set LYNX_RUN_INTEGRATION=1 to run (requires 2x H100, ~3 min)",
)


MODEL = "Qwen/Qwen2-57B-A14B-Instruct"
PORT = 8765
LOG_PATH = Path("/data/vgupta345/prowl_related_data/prowl-open-source/lynx/.integration-test-server.log")


def _wait_for_text(path: Path, needle: str, timeout_s: float = 600) -> bool:
    deadline = time.time() + timeout_s
    while time.time() < deadline:
        if path.exists():
            content = path.read_text(errors="ignore")
            if needle in content:
                return True
            if "Engine core init" in content and "failed" in content:
                return False
            if "RuntimeError" in content or "Traceback" in content.split("\n", 1)[0]:
                # don't fail too eagerly — vllm prints stack traces during normal startup
                # only treat fatal lines as failure
                pass
        time.sleep(2)
    return False


def test_lynx_kicks_in_via_env_var():
    """Boot vllm with VLLM_LYNX_ENABLED=1, hit /v1/completions, confirm
    the server log shows Lynx state-init markers (proving the patches
    fired in worker processes)."""
    import requests

    env = {
        **os.environ,
        "VLLM_LYNX_ENABLED": "1",
        "CUDA_VISIBLE_DEVICES": "0,1",
        "PATH": (
            f"{os.path.dirname(sys.executable)}:/usr/bin:/bin:"
            f"/nethome/vgupta345/.local/bin"
        ),
        "CC": "/usr/bin/gcc",
        "CXX": "/usr/bin/g++",
    }
    for var in ("CPATH", "C_INCLUDE_PATH", "CPLUS_INCLUDE_PATH",
                "LIBRARY_PATH", "LD_LIBRARY_PATH", "CONDA_BUILD_SYSROOT",
                "SDKROOT"):
        env.pop(var, None)

    LOG_PATH.write_text("")
    server = subprocess.Popen(
        [
            "vllm", "serve", MODEL,
            "--tensor-parallel-size", "2",
            "--max-model-len", "4096",
            "--gpu-memory-utilization", "0.9",
            "--port", str(PORT),
        ],
        env=env,
        stdout=open(LOG_PATH, "w"),
        stderr=subprocess.STDOUT,
        start_new_session=True,
    )

    try:
        ready = _wait_for_text(LOG_PATH, "Application startup complete", timeout_s=600)
        assert ready, (
            "server never became ready in 10 min. tail of log:\n"
            + LOG_PATH.read_text()[-3000:]
        )

        # Plugin install markers must be present.
        log = LOG_PATH.read_text()
        assert "lynx: state initialized on worker" in log, (
            "lynx plugin did not install LynxState in the worker — patches "
            "may not have fired. Log tail:\n" + log[-2000:]
        )
        assert "lynx: profiling complete on worker" in log, (
            "post-warmup profile-complete flip did not run. Log tail:\n"
            + log[-2000:]
        )

        # Sanity completion.
        resp = requests.post(
            f"http://127.0.0.1:{PORT}/v1/completions",
            json={
                "model": MODEL,
                "prompt": "What is 2+2?",
                "max_tokens": 16,
                "temperature": 0,
            },
            timeout=60,
        )
        resp.raise_for_status()
        body = resp.json()
        assert "choices" in body, body
        text = body["choices"][0]["text"]
        assert text.strip(), f"empty completion: {body!r}"

    finally:
        try:
            os.killpg(os.getpgid(server.pid), signal.SIGKILL)
        except Exception:
            pass
        server.wait(timeout=10)
        # Give workers a moment to actually free GPU memory.
        subprocess.run(
            ["pkill", "-9", "-u", os.environ.get("USER", "vgupta345"),
             "-f", "vllm|VLLM|EngineCore|Worker_TP"],
            check=False,
        )
        time.sleep(2)
