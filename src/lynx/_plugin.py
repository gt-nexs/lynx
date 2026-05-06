"""Plugin entry point — registered in ``pyproject.toml`` under the
``vllm.general_plugins`` group. vLLM's plugin loader calls
:func:`install` once per process (api server, engine core, every TP
worker).
"""

import logging

from lynx._env import is_enabled

logger = logging.getLogger(__name__)


def install() -> None:
    """Apply all Lynx monkey-patches.

    No-op when ``VLLM_LYNX_ENABLED`` is unset, so users can safely
    install ``lynx-vllm`` without affecting their default vLLM behaviour.
    """
    if not is_enabled():
        logger.debug("lynx: VLLM_LYNX_ENABLED unset; plugin is a no-op")
        return

    from lynx import _patches

    _patches.install_all()
    logger.info("lynx: plugin installed (env-gate ON)")
