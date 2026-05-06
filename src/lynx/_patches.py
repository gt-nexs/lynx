"""Runtime monkey-patches that splice Lynx into vLLM without modifying
its source. Each patch is idempotent (re-running ``install_all()`` is a
no-op) and gated on ``VLLM_LYNX_ENABLED`` via :mod:`lynx._env`.

Patch summary:

1. ``FusedMoE.__init__`` — inject the right Lynx routing variant.
2. ``ModelConfig.__post_init__`` — load the policy JSON onto ``hf_config``.
3. ``GPUModelRunner.execute_model`` — per-batch prefill detection +
   ``LynxState.on_batch_start_worker`` + force eager on prefill.
4. ``GPUModelRunner._dummy_run`` — set ``is_prefill=False`` so cudagraph
   capture exercises the Lynx kernel (else captured graphs bypass it).
5. ``Worker.__init__`` — create the per-worker ``LynxState`` singleton.
6. ``Worker.initialize_from_config`` — flip ``profile_complete=True``
   after kernel warmup.

The patches are designed to fail loudly (with a logged warning) rather
than silently when the underlying vLLM API moves. We pin ``vllm`` to
``>=0.20.1,<0.21`` in ``pyproject.toml`` to keep this surface stable.
"""

from __future__ import annotations

import json
import logging
import os
from typing import Any

from lynx._env import (
    config_file_override,
    is_enabled,
    metrics_enabled,
    profile_dir,
)

logger = logging.getLogger(__name__)


_SENTINEL = "_lynx_patched"


def _already_patched(target: Any) -> bool:
    return getattr(target, _SENTINEL, False)


def install_all() -> None:
    """Apply every Lynx patch, in dependency order. Idempotent."""
    if not is_enabled():
        return
    _patch_model_config()
    _patch_fused_moe()
    _patch_gpu_worker()
    _patch_gpu_model_runner()
    logger.info("lynx: all patches installed")


# ---------------------------------------------------------------------------
# 1. FusedMoE.__init__ — inject the right Lynx routing variant.
# ---------------------------------------------------------------------------


def _patch_fused_moe() -> None:
    from vllm.model_executor.layers.fused_moe.layer import FusedMoE

    if _already_patched(FusedMoE.__init__):
        return

    _orig_init = FusedMoE.__init__

    def _patched_init(self: Any, *args: Any, **kwargs: Any) -> None:
        # Pick the right Lynx variant based on the routing-style kwargs.
        # We import lazily so plugin load doesn't pay the triton-jit cost.
        if kwargs.get("use_grouped_topk", False):
            from lynx.routing import (
                custom_routing_function_grouped_topk as _crf,
            )
            kwargs["enable_eplb"] = False  # untested combo
        elif kwargs.get("scoring_func", "softmax") == "sigmoid":
            from lynx.routing import (
                custom_routing_function_sigmoid as _crf,
            )
            kwargs["enable_eplb"] = False
        else:
            from lynx.routing import custom_routing_function as _crf

        # Override any caller-supplied routing function (e.g. Llama4MoE).
        kwargs["custom_routing_function"] = _crf
        return _orig_init(self, *args, **kwargs)

    _patched_init._lynx_patched = True  # type: ignore[attr-defined]
    FusedMoE.__init__ = _patched_init  # type: ignore[method-assign]
    logger.debug("lynx: patched FusedMoE.__init__")


# ---------------------------------------------------------------------------
# 2. ModelConfig.__post_init__ — inject the policy JSON onto hf_config.
# ---------------------------------------------------------------------------


def _patch_model_config() -> None:
    from vllm.config.model import ModelConfig

    if _already_patched(ModelConfig.__post_init__):
        return

    _orig_post_init = ModelConfig.__post_init__

    def _patched_post_init(self: Any, *args: Any, **kwargs: Any) -> None:
        result = _orig_post_init(self, *args, **kwargs)
        try:
            _resolve_and_apply_lynx_config(self)
        except Exception as e:
            # Don't kill the engine on a Lynx config mistake — log and
            # continue. LynxState.create_instance will fail loudly later
            # if hf_config didn't get the required keys.
            logger.warning("lynx: config injection failed: %s", e)
        return result

    _patched_post_init._lynx_patched = True  # type: ignore[attr-defined]
    ModelConfig.__post_init__ = _patched_post_init  # type: ignore[method-assign]
    logger.debug("lynx: patched ModelConfig.__post_init__")


def _resolve_and_apply_lynx_config(model_config: Any) -> None:
    """Resolve the policy file (explicit override, then registry lookup),
    load it, and ``setattr`` its keys onto ``model_config.hf_config``."""
    from lynx.registry import lookup as registry_lookup

    path = config_file_override() or registry_lookup(model_config.model)
    if path is None:
        raise ValueError(
            f"VLLM_LYNX_ENABLED=1 but no Lynx config registered for "
            f"model {model_config.model!r}. Either register one with "
            f"lynx.register_model(...) or set VLLM_LYNX_CONFIG_FILE."
        )
    logger.info("lynx: using policy %s for %s", path, model_config.model)
    with open(path) as f:
        policy_cfg = json.load(f)
    for key, value in policy_cfg.items():
        setattr(model_config.hf_config, key, value)


# ---------------------------------------------------------------------------
# 3 + 4. GPUModelRunner: per-batch hook + dummy_run is_prefill flip.
# ---------------------------------------------------------------------------


def _patch_gpu_model_runner() -> None:
    from vllm.v1.worker.gpu_model_runner import GPUModelRunner

    if _already_patched(GPUModelRunner.execute_model):
        return

    _orig_execute = GPUModelRunner.execute_model
    _orig_dummy = GPUModelRunner._dummy_run

    def _patched_execute(self: Any, scheduler_output: Any, *args: Any, **kwargs: Any) -> Any:
        # Per-batch state update for Lynx routing decisions.
        try:
            from lynx.state import LynxState

            num_new = len(getattr(scheduler_output, "scheduled_new_reqs", []) or [])
            num_chunked_prefill = 0
            num_pure_decode = 0
            is_prefill_batch = num_new > 0

            kvt = getattr(self.vllm_config, "kv_transfer_config", None)
            if is_prefill_batch and kvt is not None and getattr(kvt, "is_kv_consumer", False):
                # Disagg KV consumer: "new" requests are decode, KV cached upstream.
                is_prefill_batch = False
                num_pure_decode += num_new

            cached = getattr(scheduler_output, "scheduled_cached_reqs", None)
            if cached is not None:
                for i, req_id in enumerate(getattr(cached, "req_ids", []) or []):
                    req_state = self.requests.get(req_id)
                    if req_state is None:
                        continue
                    num_computed = cached.num_computed_tokens[i]
                    prompt_len = len(req_state.prompt_token_ids)
                    if num_computed < prompt_len:
                        num_chunked_prefill += 1
                        is_prefill_batch = True
                    else:
                        num_pure_decode += 1

            state = LynxState.get_instance()
            if state is not None:
                state.on_batch_start_worker(
                    num_new, num_pure_decode, num_chunked_prefill, is_prefill_batch
                )
        except Exception as e:  # never break the engine on telemetry hiccups
            logger.debug("lynx: per-batch hook skipped: %s", e)

        return _orig_execute(self, scheduler_output, *args, **kwargs)

    def _patched_dummy(self: Any, *args: Any, **kwargs: Any) -> Any:
        # Force is_prefill=False, profile_complete=True so cudagraph
        # capture exercises the Lynx kernel. Without this, captured
        # graphs route through vanilla fused_topk and runtime replays
        # silently bypass Lynx — exactly the bug we hit in the in-tree
        # port.
        try:
            from lynx.state import LynxState

            state = LynxState.get_instance()
            if state is not None:
                state.is_prefill = False
                state.profile_complete = True
        except Exception:
            pass
        return _orig_dummy(self, *args, **kwargs)

    _patched_execute._lynx_patched = True  # type: ignore[attr-defined]
    _patched_dummy._lynx_patched = True  # type: ignore[attr-defined]
    GPUModelRunner.execute_model = _patched_execute  # type: ignore[method-assign]
    GPUModelRunner._dummy_run = _patched_dummy  # type: ignore[method-assign]
    logger.debug("lynx: patched GPUModelRunner.execute_model + _dummy_run")


# ---------------------------------------------------------------------------
# 5 + 6. Worker.__init__ + Worker.initialize_from_config.
# ---------------------------------------------------------------------------


def _patch_gpu_worker() -> None:
    from vllm.v1.worker.gpu_worker import Worker

    if _already_patched(Worker.__init__):
        return

    _orig_init = Worker.__init__
    _orig_initialize = Worker.initialize_from_config

    def _patched_init(self: Any, *args: Any, **kwargs: Any) -> None:
        _orig_init(self, *args, **kwargs)
        # Best-effort lazy LynxState creation. The hf_config has already
        # been mutated by _patch_model_config in this process.
        try:
            from lynx.state import LynxState

            LynxState.create_instance(
                self.vllm_config.model_config.hf_config,
                metrics_enabled=metrics_enabled(),
            )
            if metrics_enabled():
                _create_metric_store(self.vllm_config.model_config.hf_config)
        except Exception as e:
            logger.warning("lynx: LynxState creation failed: %s", e)

    def _patched_initialize(self: Any, *args: Any, **kwargs: Any) -> Any:
        result = _orig_initialize(self, *args, **kwargs)
        # By this point kernel_warmup has run inside the original
        # initialize_from_config. Flip profile_complete so subsequent
        # forwards exercise the Lynx kernel.
        try:
            from lynx.state import LynxState

            state = LynxState.get_instance()
            if state is not None:
                state.mark_profiling_done()
            if metrics_enabled():
                _mark_metric_store_done()
        except Exception as e:
            logger.warning("lynx: profile-complete flip failed: %s", e)
        return result

    _patched_init._lynx_patched = True  # type: ignore[attr-defined]
    _patched_initialize._lynx_patched = True  # type: ignore[attr-defined]
    Worker.__init__ = _patched_init  # type: ignore[method-assign]
    Worker.initialize_from_config = _patched_initialize  # type: ignore[method-assign]
    logger.debug("lynx: patched Worker.__init__ + initialize_from_config")


# ---------------------------------------------------------------------------
# Optional metric setup (no-op when VLLM_LYNX_METRICS unset).
# ---------------------------------------------------------------------------


def _create_metric_store(hf_config: Any) -> None:
    """Lazy import — pulls in ddsketch only when metrics are enabled."""
    import atexit

    os.environ.setdefault("VLLM_LYNX_PROFILE_DIR", profile_dir())
    from lynx.metrics.metric_logging import MetricStore

    store = MetricStore.get_or_create_instance(hf_config)

    def _flush() -> None:
        try:
            store.plot_metrics()
        except Exception as e:  # pragma: no cover
            logger.warning("lynx metrics flush failed: %s", e)

    atexit.register(_flush)


def _mark_metric_store_done() -> None:
    from lynx.metrics.metric_logging import MetricStore

    store = MetricStore.get_instance()
    if store is not None:
        store.mark_profiling_done()
