"""Per-engine state for Lynx routing.

A single ``LynxState`` instance is created in the V1 engine when ``--lynx``
is passed. The routing kernels in ``routing.py`` consult it on the hot
path to decide whether to apply expert pruning.

Key invariants:

* ``profile_complete`` starts ``False`` and is flipped to ``True`` after
  worker kernel warmup. Pruning only runs once it is ``True`` so warmup
  shapes don't get distorted.
* ``is_prefill`` is updated per batch by ``on_batch_start_worker``. We
  bypass pruning during prefill (full top-k is needed for the first
  token's quality and for KV-cache initialization).
* ``metrics_enabled`` is ``True`` only when ``--lynx-metrics`` was passed;
  hot-path metrics calls in ``routing.py`` are gated on it.
"""

from __future__ import annotations

from typing import Any

try:
    # Prefer vllm's logger init so INFO lines appear alongside vllm's
    # own logs (vllm silences third-party loggers by default).
    from vllm.logger import init_logger as _init_logger
    logger = _init_logger("lynx.state")
except Exception:
    import logging
    logger = logging.getLogger(__name__)


class LynxState:
    """Singleton holder for Lynx routing state."""

    _instance: "LynxState | None" = None

    def __init__(
        self,
        hf_config: Any,
        *,
        metrics_enabled: bool = False,
    ) -> None:
        # Routing-policy fields, read off the (already JSON-injected)
        # hf_config. Required keys; raise loudly if missing so config
        # mistakes don't fall through to bad routing.
        try:
            self.policy: str = hf_config.policy
            self.alpha: float = float(hf_config.alpha)
            self.beta: float = float(hf_config.beta)
            # NOTE: field names follow the routing-kernel module
            # (`num_total_experts`, `num_experts_per_token`) rather than
            # the HF naming (`num_local_experts`, `num_experts_per_tok`).
            self.num_total_experts: int = int(hf_config.num_local_experts)
            self.num_experts_per_token: int = int(hf_config.num_experts_per_tok)
        except AttributeError as e:
            raise ValueError(
                "Lynx config is missing a required field. The JSON loaded "
                "via --lynx-config-file (or the bundled default) must set "
                "policy, alpha, beta, num_local_experts, and "
                "num_experts_per_tok."
            ) from e

        # Optional knobs (older policies use them; newer ones may not).
        self.min_experts: int = int(getattr(hf_config, "min_experts", 0))
        self.threshold_percentile: float = float(
            getattr(hf_config, "threshold_percentile", 0.0)
        )
        self.count_of_topk: int = int(getattr(hf_config, "count_of_topk", 0))
        self.num_experts_to_keep: int = int(
            getattr(hf_config, "num_experts_to_keep", 0)
        )
        self.num_experts_to_drop: int = max(
            0, self.num_total_experts - self.num_experts_to_keep
        )

        # Lifecycle flags.
        self.profile_complete: bool = False
        self.is_prefill: bool = True
        self.batch_idx: int = 0
        self.layer_idx: int | None = None

        # Metrics gate. The metrics module (Phase E) installs hooks that
        # only fire when this is True.
        self.metrics_enabled: bool = metrics_enabled

        logger.info(
            "Lynx: state initialized (policy=%s, alpha=%s, beta=%s, "
            "num_total_experts=%s, top_k=%s, metrics=%s)",
            self.policy,
            self.alpha,
            self.beta,
            self.num_total_experts,
            self.num_experts_per_token,
            self.metrics_enabled,
        )

    # ---- singleton plumbing ------------------------------------------------

    @classmethod
    def get_instance(cls) -> "LynxState | None":
        return cls._instance

    @classmethod
    def create_instance(
        cls, hf_config: Any, *, metrics_enabled: bool = False
    ) -> "LynxState | None":
        if cls._instance is not None:
            return cls._instance
        # Skip silently for dense models (no MoE experts). This lets a
        # user pass --lynx for a dense model without the engine crashing —
        # routing simply never kicks in. The tradeoff is a confusing
        # no-op; we log it.
        if not hasattr(hf_config, "num_local_experts"):
            logger.warning(
                "Lynx: model has no `num_local_experts` on hf_config; "
                "Lynx will be a no-op for this model."
            )
            return None
        cls._instance = cls(hf_config, metrics_enabled=metrics_enabled)
        return cls._instance

    @classmethod
    def reset(cls) -> None:
        """Drop the singleton. Useful for tests that spin up multiple
        engines in one process."""
        cls._instance = None

    # ---- lifecycle hooks ---------------------------------------------------

    def mark_profiling_done(self) -> None:
        self.profile_complete = True
        logger.info("Lynx: profiling complete; expert pruning is now active")

    def on_batch_start_worker(
        self,
        num_new_reqs: int,
        num_pure_decode_reqs: int,
        num_chunked_prefill_reqs: int,
        is_prefill_batch: bool,
    ) -> None:
        """Called by the GPU model runner before each forward pass to
        update the prefill/decode flag. Routing kernels read
        ``self.is_prefill`` to skip pruning on prefill batches."""
        if self.profile_complete:
            self.is_prefill = is_prefill_batch

    def on_batch_end_worker(self) -> None:
        self.batch_idx += 1

    def set_layer_idx(self, layer_idx: int) -> None:
        self.layer_idx = layer_idx

    def get_layer_idx(self) -> int | None:
        return self.layer_idx


# Backwards-compatible alias for any code paths that still refer to the
# old class name from the v0.10.1-era fork. Drop in a future release.
MixtralLogitStore = LynxState
