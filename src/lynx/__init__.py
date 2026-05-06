"""Lynx — opt-in MoE expert-skipping for vLLM.

Public surface (this package's stable API):

* ``lynx.register_model(model_substring, config_path)`` — register a new
  model→policy mapping at runtime.
* ``lynx.lookup(model_name)`` — resolve the policy file for a model name.
* ``lynx.LynxState`` — the per-worker routing state singleton.

Activation contract:

The package installs as a vLLM general plugin via the
``vllm.general_plugins`` entry point declared in ``pyproject.toml``. The
plugin runs in every vLLM process (api server, engine core, every TP
worker) and gates on the ``VLLM_LYNX_ENABLED`` environment variable. When
that variable is unset, ``install()`` returns immediately and vLLM
behaves exactly as it would without lynx installed.

Usage:

.. code-block:: bash

    pip install vllm==0.20.1 lynx-vllm
    VLLM_LYNX_ENABLED=1 vllm serve Qwen/Qwen2-57B-A14B-Instruct \\
        --tensor-parallel-size 2

Adding a new model:

.. code-block:: python

    import lynx
    lynx.register_model("my-org/my-7b-moe", "/path/to/policy.json")
"""

from lynx.registry import LYNX_MODEL_REGISTRY, lookup, register_model
from lynx.state import LynxState

__all__ = ["LYNX_MODEL_REGISTRY", "lookup", "register_model", "LynxState"]
__version__ = "0.1.0"
