"""Lynx routing kernels.

Custom routing functions plugged into vLLM's :class:`FusedMoE` via the
``custom_routing_function=`` argument. These select a sparse subset of
experts per decode step using a quantization-confidence policy. They are
no-ops during prefill and during worker warmup; the gating is on
``LynxState.profile_complete and not LynxState.is_prefill``.

Three entry points are exposed and registered by per-model glue in the
``vllm.model_executor.models`` tree:

* ``custom_routing_function`` — softmax-style gate (Mixtral/Qwen MoE).
* ``custom_routing_function_grouped_topk`` — DeepSeekV2 grouped routing.
* ``custom_routing_function_sigmoid`` — Llama4 sigmoid gating.

All three accept ``(hidden_states, gating_output, topk, renormalize)``
kwargs (matching :class:`vllm.model_executor.layers.fused_moe.router.
custom_routing_router.CustomRoutingRouter`) and return a 2-tuple
``(topk_weights, topk_ids)``.
"""

from __future__ import annotations

import math
from typing import Optional, Tuple

import torch
import triton
import triton.language as tl

from vllm.model_executor.layers.fused_moe.router.fused_topk_router import (
    fused_topk,
)
from lynx.state import LynxState

# Backwards-compatible alias used by the original kernel source.
MixtralLogitStore = LynxState

# ---------------------------------------------------------------------------
# Metrics shim. The full metric path is opt-in via --lynx-metrics; the hot
# path here only consults LynxState.metrics_enabled, so make these no-ops
# unless metrics are enabled and the optional module is importable.
# ---------------------------------------------------------------------------


class _NullMetricStore:
    """Stand-in used when ``--lynx-metrics`` is off."""

    @classmethod
    def get_instance(cls) -> "_NullMetricStore":
        return cls()

    def put_metric(self, *_args, **_kwargs) -> None:  # pragma: no cover
        return None


class _NullTaskCM:
    """Stand-in context manager used when ``--lynx-metrics`` is off."""

    def __init__(self, *_args, **_kwargs) -> None:
        pass

    def __enter__(self) -> "_NullTaskCM":
        return self

    def __exit__(self, *_args) -> None:
        return None


class _NullTaskType:  # pragma: no cover
    def __getattr__(self, _name: str) -> str:
        return _name


def _resolve_metrics():
    """Return ``(MetricStore, TaskType, TaskLoggingContextManagerGPU)`` —
    real ones if ``--lynx-metrics`` is on and the optional module is
    importable, no-op stand-ins otherwise."""
    state = LynxState.get_instance()
    if state is None or not state.metrics_enabled:
        return _NullMetricStore, _NullTaskType(), _NullTaskCM
    try:
        from lynx.metrics.metric_logging import (
            MetricStore as _MS,
            TaskLoggingContextManagerGPU as _CM,
            TaskType as _TT,
        )

        return _MS, _TT, _CM
    except ImportError:
        return _NullMetricStore, _NullTaskType(), _NullTaskCM


MetricStore, TaskType, TaskLoggingContextManagerGPU = _resolve_metrics()


def add_num_selected_experts_policy(_value: int) -> None:  # pragma: no cover
    return None


def add_num_selected_experts_base(_value: int) -> None:  # pragma: no cover
    return None


def get_num_selected_experts_policy() -> int:  # pragma: no cover
    return 0


def get_num_selected_experts_base() -> int:  # pragma: no cover
    return 0


# ---------------------------------------------------------------------------
# optimize_expert_selection_parameterized + helpers (vendored from the
# v0.10.1-era fork of fused_moe.py; only the ``advanced_parametrized``
# policy uses these).
# ---------------------------------------------------------------------------


def extract_important_graded(
    topk_weights: torch.Tensor,
    epsilon: float = 1e-8,
) -> torch.Tensor:
    """Continuous confidence score in [-1, 1]: top half of the routing
    distribution minus bottom half, normalised by the total."""
    mid = topk_weights.size(1) // 2
    top_half_sum = topk_weights[:, :mid].sum(dim=1)
    bottom_half_sum = topk_weights[:, mid:].sum(dim=1)
    return (top_half_sum - bottom_half_sum) / (top_half_sum + epsilon)


def apply_expert_mask(
    gating_output: torch.Tensor,
    experts_to_drop_mask: torch.Tensor,
) -> torch.Tensor:
    large_negative_value = torch.finfo(gating_output.dtype).min
    gating_output.masked_fill_(experts_to_drop_mask.unsqueeze(0), large_negative_value)
    return gating_output


def optimize_expert_selection_parameterized(
    topk_ids: torch.Tensor,
    topk_weights: torch.Tensor,
    gating_output: torch.Tensor,
    num_experts: int,
    min_experts: int = 2,
    topk: int = 6,
    beta: float = 0.5,
    alpha: float = 0.5,
) -> torch.Tensor:
    """CUDA-graph-friendly per-token expert reduction. Returns the count
    of experts kept across the batch; mutates ``gating_output`` in place
    to mask dropped experts to ``-inf``."""
    device = topk_ids.device
    B, K = topk_ids.shape

    confidence = extract_important_graded(topk_weights)
    keep_counts = (confidence * K * beta).floor().to(torch.int64) + alpha

    pos = torch.arange(K, device=device).expand(B, K)
    keep_mask = pos < keep_counts.unsqueeze(-1)

    kept_ids = torch.where(keep_mask, topk_ids, -1)
    flat_ids = kept_ids.reshape(-1)
    shifted = flat_ids + 1

    counts_plus1 = torch.zeros(num_experts + 1, dtype=torch.int32, device=device)
    ones = torch.ones_like(shifted, dtype=counts_plus1.dtype)
    counts_plus1.scatter_add_(0, shifted.to(torch.int64), ones)

    experts_to_keep_mask = counts_plus1[1:] > 0
    num_kept_total = experts_to_keep_mask.sum()

    apply_expert_mask(gating_output, experts_to_drop_mask=~experts_to_keep_mask)
    return num_kept_total

def next_pow2(x: int) -> int:
    return 1 << (x - 1).bit_length()

@triton.jit
def fused_topk_sum_qh_kernel_beta(
    router_logits_ptr,   # *any  [M, E]  (bf16/f16/f32)
    topk_sum_ptr,        # *f32  [M]
    q_out_ptr,           # *f32  [M, E]
    h_out_ptr,           # *f32  [M, E]
    M, E,
    alpha,               # f32 scalar
    beta,                # f32 scalar
    logB,                # f32 scalar
    topk: tl.constexpr,
    BLOCK_E: tl.constexpr,
    BLOCK_K: tl.constexpr,
):
    pid = tl.program_id(0)
    if pid >= M:
        return

    row_start = pid * E
    offs_e = tl.arange(0, BLOCK_E)
    mask_e = offs_e < E

    # load as input dtype, immediately upcast to f32
    logits = tl.load(router_logits_ptr + row_start + offs_e,
                     mask=mask_e, other=-float('inf')).to(tl.float32)

    # per-row max
    row_max = tl.max(logits, axis=0)

    # ----- q & h in f32 -----
    q = alpha * (logits - row_max)
    q = tl.where(q <= -beta, -beta, -tl.floor(-q))
    h = tl.exp(q * logB)

    tl.store(q_out_ptr + row_start + offs_e, q, mask=mask_e)
    tl.store(h_out_ptr + row_start + offs_e, h, mask=mask_e)

    # ----- pre-renorm top-k sum over softmax probs -----
    exps  = tl.exp(logits - row_max)
    denom = tl.sum(exps, axis=0)
    probs = exps / denom

    scratch = probs
    offs_k = tl.arange(0, BLOCK_K)
    top_vals = tl.zeros((BLOCK_K,), dtype=tl.float32)
    for i in range(topk):
        mx  = tl.max(scratch, axis=0)
        idx = tl.argmax(scratch, axis=0).to(tl.int32)
        top_vals = tl.where(offs_k == i, mx, top_vals)
        scratch  = tl.where(offs_e == idx, 0.0, scratch)

    pre_sum = tl.sum(tl.where(offs_k < topk, top_vals, 0.0), axis=0)
    tl.store(topk_sum_ptr + pid, pre_sum)


def fused_topk_sum_qh_beta(router_logits, topk, alpha, beta, B):
    M, E = router_logits.shape
    device = router_logits.device

    topk_sum = torch.empty((M,), dtype=torch.float32, device=device)
    q_out    = torch.empty((M, E), dtype=torch.float32, device=device)
    h_out    = torch.empty((M, E), dtype=torch.float32, device=device)

    BLOCK_E = 1 << (E - 1).bit_length()
    BLOCK_K = 1 << (topk - 1).bit_length()

    fused_topk_sum_qh_kernel_beta[(M,)](
        router_logits.contiguous(),              # no host cast
        topk_sum, q_out, h_out,
        M, E,
        float(alpha), float(beta), float(math.log(B)),
        topk=topk,
        BLOCK_E=BLOCK_E, BLOCK_K=BLOCK_K,
        num_warps=4, num_stages=2,
    )
    return topk_sum.view(M, 1), q_out, h_out

@triton.jit
def fused_topk_sum_qh_kernel(
    router_logits_ptr,   # *any  [M, E]  (bf16/f16/f32)
    topk_sum_ptr,        # *f32  [M]
    q_out_ptr,           # *f32  [M, E]
    h_out_ptr,           # *f32  [M, E]
    M, E,
    alpha,               # f32 scalar
    logB,                # f32 scalar
    topk: tl.constexpr,
    BLOCK_E: tl.constexpr,
    BLOCK_K: tl.constexpr,
):
    pid = tl.program_id(0)
    if pid >= M:
        return

    row_start = pid * E
    offs_e = tl.arange(0, BLOCK_E)
    mask_e = offs_e < E

    # load as input dtype, immediately upcast to f32
    logits = tl.load(router_logits_ptr + row_start + offs_e,
                     mask=mask_e, other=-float('inf')).to(tl.float32)

    # per-row max
    row_max = tl.max(logits, axis=0)

    # ----- q & h in f32 -----
    q = alpha * (logits - row_max)
    q = -tl.floor(-q)                     # ceil
    h = tl.exp(q * logB)

    tl.store(q_out_ptr + row_start + offs_e, q, mask=mask_e)
    tl.store(h_out_ptr + row_start + offs_e, h, mask=mask_e)

    # ----- pre-renorm top-k sum over softmax probs -----
    exps  = tl.exp(logits - row_max)
    denom = tl.sum(exps, axis=0)
    probs = exps / denom

    scratch = probs
    offs_k = tl.arange(0, BLOCK_K)
    top_vals = tl.zeros((BLOCK_K,), dtype=tl.float32)
    for i in range(topk):
        mx  = tl.max(scratch, axis=0)
        idx = tl.argmax(scratch, axis=0).to(tl.int32)
        top_vals = tl.where(offs_k == i, mx, top_vals)
        scratch  = tl.where(offs_e == idx, 0.0, scratch)

    pre_sum = tl.sum(tl.where(offs_k < topk, top_vals, 0.0), axis=0)
    tl.store(topk_sum_ptr + pid, pre_sum)


def fused_topk_sum_qh(router_logits, topk, alpha, B):
    M, E = router_logits.shape
    device = router_logits.device

    topk_sum = torch.empty((M,), dtype=torch.float32, device=device)
    q_out    = torch.empty((M, E), dtype=torch.float32, device=device)
    h_out    = torch.empty((M, E), dtype=torch.float32, device=device)

    BLOCK_E = 1 << (E - 1).bit_length()
    BLOCK_K = 1 << (topk - 1).bit_length()

    fused_topk_sum_qh_kernel[(M,)](
        router_logits.contiguous(),              # no host cast
        topk_sum, q_out, h_out,
        M, E,
        float(alpha),
        topk=topk,
        BLOCK_E=BLOCK_E, BLOCK_K=BLOCK_K,
        num_warps=4, num_stages=2,
    )
    return topk_sum.view(M, 1), q_out, h_out

def fused_topk_sum_qh_alpha3_beta4(router_logits, topk, B):
    M, E = router_logits.shape
    device = router_logits.device

    topk_sum = torch.empty((M,), dtype=torch.float32, device=device)
    q_out    = torch.empty((M, E), dtype=torch.float32, device=device)
    h_out    = torch.empty((M, E), dtype=torch.float32, device=device)

    BLOCK_E = 1 << (E - 1).bit_length()
    BLOCK_K = 1 << (topk - 1).bit_length()

    fused_topk_sum_qh_kernel_alpha3_beta4[(M,)](
        router_logits.contiguous(),              # no host cast
        topk_sum, q_out, h_out,
        M, E,
        float(math.log2(B)),
        topk=topk,
        BLOCK_E=BLOCK_E, BLOCK_K=BLOCK_K,
        num_warps=4, num_stages=2,
    )
    return topk_sum.view(M, 1), q_out, h_out

@triton.jit
def fused_topk_sum_qh_kernel_alpha3_beta4(
    router_logits_ptr,   # *any  [M, E]  (bf16/f16/f32)
    topk_sum_ptr,        # *f32  [M]
    q_out_ptr,           # *f32  [M, E]
    h_out_ptr,           # *f32  [M, E]
    M, E,
    log2B,                # f32 scalar
    topk: tl.constexpr,
    BLOCK_E: tl.constexpr,
    BLOCK_K: tl.constexpr,
):
    pid = tl.program_id(0)
    if pid >= M:
        return

    row_start = pid * E
    offs_e = tl.arange(0, BLOCK_E)
    mask_e = offs_e < E

    # load as input dtype, immediately upcast to f32
    logits = tl.load(router_logits_ptr + row_start + offs_e,
                     mask=mask_e, other=-float('inf')).to(tl.float32)

    # per-row max
    row_max = tl.max(logits, axis=0)

    # ----- q & h in f32 -----
    q = 3 * (logits - row_max)
    q = tl.where(q <= -4.0, -4.0, -tl.floor(-q))
    h = tl.exp2(q * log2B)

    tl.store(q_out_ptr + row_start + offs_e, q, mask=mask_e)
    tl.store(h_out_ptr + row_start + offs_e, h, mask=mask_e)

    # ----- pre-renorm top-k sum over softmax probs -----
    exps  = tl.exp(logits - row_max)
    denom = tl.sum(exps, axis=0)
    probs = exps / denom

    scratch = probs
    offs_k = tl.arange(0, BLOCK_K)
    top_vals = tl.zeros((BLOCK_K,), dtype=tl.float32)
    for i in range(topk):
        mx  = tl.max(scratch, axis=0)
        idx = tl.argmax(scratch, axis=0).to(tl.int32)
        top_vals = tl.where(offs_k == i, mx, top_vals)
        scratch  = tl.where(offs_e == idx, 0.0, scratch)

    pre_sum = tl.sum(tl.where(offs_k < topk, top_vals, 0.0), axis=0)
    tl.store(topk_sum_ptr + pid, pre_sum)

@triton.jit
def fused_topk_sum_qh_alpha3_kernel(
    router_logits_ptr, topk_sum_ptr, q_out_ptr, h_out_ptr,
    M, E, log2B,
    topk: tl.constexpr, BLOCK_E: tl.constexpr, BLOCK_K: tl.constexpr,
):
    pid = tl.program_id(0)
    if pid >= M:
        return
    row_start = pid * E
    offs_e = tl.arange(0, BLOCK_E)
    mask_e = offs_e < E
    logits = tl.load(router_logits_ptr + row_start + offs_e, mask=mask_e, other=-float('inf')).to(tl.float32)
    row_max = tl.max(logits, axis=0)
    # alpha=3, baked in
    q = 3 * (logits - row_max)
    h = tl.exp2(q * log2B)
    tl.store(q_out_ptr + row_start + offs_e, q, mask=mask_e)
    tl.store(h_out_ptr + row_start + offs_e, h, mask=mask_e)
    exps = tl.exp(logits - row_max)
    denom = tl.sum(exps, axis=0)
    probs = exps / denom
    scratch = probs
    offs_k = tl.arange(0, BLOCK_K)
    top_vals = tl.zeros((BLOCK_K,), dtype=tl.float32)
    for i in range(topk):
        mx = tl.max(scratch, axis=0)
        idx = tl.argmax(scratch, axis=0).to(tl.int32)
        top_vals = tl.where(offs_k == i, mx, top_vals)
        scratch = tl.where(offs_e == idx, 0.0, scratch)
    pre_sum = tl.sum(tl.where(offs_k < topk, top_vals, 0.0), axis=0)
    tl.store(topk_sum_ptr + pid, pre_sum)

@triton.jit
def fused_topk_sum_qh_alpha3_beta2_kernel(
    router_logits_ptr, topk_sum_ptr, q_out_ptr, h_out_ptr,
    M, E, log2B,
    topk: tl.constexpr, BLOCK_E: tl.constexpr, BLOCK_K: tl.constexpr,
):
    pid = tl.program_id(0)
    if pid >= M:
        return
    row_start = pid * E
    offs_e = tl.arange(0, BLOCK_E)
    mask_e = offs_e < E
    logits = tl.load(router_logits_ptr + row_start + offs_e, mask=mask_e, other=-float('inf')).to(tl.float32)
    row_max = tl.max(logits, axis=0)
    # alpha=3, beta=2 baked in
    q = 3 * (logits - row_max)
    q = tl.where(q <= -2.0, -2.0, -tl.floor(-q))
    h = tl.exp2(q * log2B)
    tl.store(q_out_ptr + row_start + offs_e, q, mask=mask_e)
    tl.store(h_out_ptr + row_start + offs_e, h, mask=mask_e)
    exps = tl.exp(logits - row_max)
    denom = tl.sum(exps, axis=0)
    probs = exps / denom
    scratch = probs
    offs_k = tl.arange(0, BLOCK_K)
    top_vals = tl.zeros((BLOCK_K,), dtype=tl.float32)
    for i in range(topk):
        mx = tl.max(scratch, axis=0)
        idx = tl.argmax(scratch, axis=0).to(tl.int32)
        top_vals = tl.where(offs_k == i, mx, top_vals)
        scratch = tl.where(offs_e == idx, 0.0, scratch)
    pre_sum = tl.sum(tl.where(offs_k < topk, top_vals, 0.0), axis=0)
    tl.store(topk_sum_ptr + pid, pre_sum)

def fused_topk_sum_qh_alpha3_beta2(router_logits, topk, B):
    M, E = router_logits.shape
    device = router_logits.device
    topk_sum = torch.empty((M,), dtype=torch.float32, device=device)
    q_out    = torch.empty((M, E), dtype=torch.float32, device=device)
    h_out    = torch.empty((M, E), dtype=torch.float32, device=device)
    BLOCK_E = 1 << (E - 1).bit_length()
    BLOCK_K = 1 << (topk - 1).bit_length()
    fused_topk_sum_qh_alpha3_beta2_kernel[(M,)](
        router_logits.contiguous(), topk_sum, q_out, h_out,
        M, E, float(math.log2(B)),
        topk=topk, BLOCK_E=BLOCK_E, BLOCK_K=BLOCK_K,
        num_warps=4, num_stages=2,
    )
    return topk_sum.view(M, 1), q_out, h_out

@triton.jit
def fused_topk_sum_qh_beta1_kernel(
    router_logits_ptr,   # *any  [M, E]  (bf16/f16/f32)
    topk_sum_ptr,        # *f32  [M]
    q_out_ptr,           # *f32  [M, E]
    h_out_ptr,           # *f32  [M, E]
    M, E,
    alpha,               # f32 scalar
    log2B,                # f32 scalar
    topk: tl.constexpr,
    BLOCK_E: tl.constexpr,
    BLOCK_K: tl.constexpr,
):
    pid = tl.program_id(0)
    if pid >= M:
        return

    row_start = pid * E
    offs_e = tl.arange(0, BLOCK_E)
    mask_e = offs_e < E

    # load as input dtype, immediately upcast to f32
    logits = tl.load(router_logits_ptr + row_start + offs_e,
                     mask=mask_e, other=-float('inf')).to(tl.float32)

    # per-row max
    row_max = tl.max(logits, axis=0)

    # ----- q & h in f32 -----
    q = alpha * (logits - row_max)
    q = tl.where(q <= -1.0, -1.0, -tl.floor(-q))
    h = tl.exp2(q * log2B)

    tl.store(q_out_ptr + row_start + offs_e, q, mask=mask_e)
    tl.store(h_out_ptr + row_start + offs_e, h, mask=mask_e)

    # ----- pre-renorm top-k sum over softmax probs -----
    exps  = tl.exp(logits - row_max)
    denom = tl.sum(exps, axis=0)
    probs = exps / denom

    scratch = probs
    offs_k = tl.arange(0, BLOCK_K)
    top_vals = tl.zeros((BLOCK_K,), dtype=tl.float32)
    for i in range(topk):
        mx  = tl.max(scratch, axis=0)
        idx = tl.argmax(scratch, axis=0).to(tl.int32)
        top_vals = tl.where(offs_k == i, mx, top_vals)
        scratch  = tl.where(offs_e == idx, 0.0, scratch)

    pre_sum = tl.sum(tl.where(offs_k < topk, top_vals, 0.0), axis=0)
    tl.store(topk_sum_ptr + pid, pre_sum)


@triton.jit
def fused_topk_sum_qh_alpha70_beta1_kernel(
    router_logits_ptr,   # *any  [M, E]  (bf16/f16/f32)
    topk_sum_ptr,        # *f32  [M]
    q_out_ptr,           # *f32  [M, E]
    h_out_ptr,           # *f32  [M, E]
    M, E,
    log2B,                # f32 scalar
    topk: tl.constexpr,
    BLOCK_E: tl.constexpr,
    BLOCK_K: tl.constexpr,
):
    pid = tl.program_id(0)
    if pid >= M:
        return

    row_start = pid * E
    offs_e = tl.arange(0, BLOCK_E)
    mask_e = offs_e < E

    # load as input dtype, immediately upcast to f32
    logits = tl.load(router_logits_ptr + row_start + offs_e,
                     mask=mask_e, other=-float('inf')).to(tl.float32)

    # per-row max
    row_max = tl.max(logits, axis=0)

    # ----- q & h in f32 -----
    q = 0.7 * (logits - row_max)
    q = tl.where(q <= -1.0, -1.0, -tl.floor(-q))
    h = tl.exp2(q * log2B)

    tl.store(q_out_ptr + row_start + offs_e, q, mask=mask_e)
    tl.store(h_out_ptr + row_start + offs_e, h, mask=mask_e)

    # ----- pre-renorm top-k sum over softmax probs -----
    exps  = tl.exp(logits - row_max)
    denom = tl.sum(exps, axis=0)
    probs = exps / denom

    scratch = probs
    offs_k = tl.arange(0, BLOCK_K)
    top_vals = tl.zeros((BLOCK_K,), dtype=tl.float32)
    for i in range(topk):
        mx  = tl.max(scratch, axis=0)
        idx = tl.argmax(scratch, axis=0).to(tl.int32)
        top_vals = tl.where(offs_k == i, mx, top_vals)
        scratch  = tl.where(offs_e == idx, 0.0, scratch)

    pre_sum = tl.sum(tl.where(offs_k < topk, top_vals, 0.0), axis=0)
    tl.store(topk_sum_ptr + pid, pre_sum)


def fused_topk_sum_qh_beta1(router_logits, topk, alpha, B):
    M, E = router_logits.shape
    device = router_logits.device

    topk_sum = torch.empty((M,), dtype=torch.float32, device=device)
    q_out    = torch.empty((M, E), dtype=torch.float32, device=device)
    h_out    = torch.empty((M, E), dtype=torch.float32, device=device)

    BLOCK_E = 1 << (E - 1).bit_length()
    BLOCK_K = 1 << (topk - 1).bit_length()

    fused_topk_sum_qh_beta1_kernel[(M,)](
        router_logits.contiguous(),              # no host cast
        topk_sum, q_out, h_out,
        M, E,
        float(alpha), float(math.log2(B)),
        topk=topk,
        BLOCK_E=BLOCK_E, BLOCK_K=BLOCK_K,
        num_warps=4, num_stages=2,
    )
    return topk_sum.view(M, 1), q_out, h_out

def fused_topk_sum_qh_alpha70_beta1(router_logits, topk, B):
    M, E = router_logits.shape
    device = router_logits.device

    topk_sum = torch.empty((M,), dtype=torch.float32, device=device)
    q_out    = torch.empty((M, E), dtype=torch.float32, device=device)
    h_out    = torch.empty((M, E), dtype=torch.float32, device=device)

    BLOCK_E = 1 << (E - 1).bit_length()
    BLOCK_K = 1 << (topk - 1).bit_length()

    fused_topk_sum_qh_alpha70_beta1_kernel[(M,)](
        router_logits.contiguous(),              # no host cast
        topk_sum, q_out, h_out,
        M, E,
        float(math.log2(B)),
        topk=topk,
        BLOCK_E=BLOCK_E, BLOCK_K=BLOCK_K,
        num_warps=4, num_stages=2,
    )
    return topk_sum.view(M, 1), q_out, h_out

@triton.jit
def fused_topk_sum_qh_alpha3_beta6_kernel(
    router_logits_ptr, topk_sum_ptr, q_out_ptr, h_out_ptr,
    M, E, log2B,
    topk: tl.constexpr, BLOCK_E: tl.constexpr, BLOCK_K: tl.constexpr,
):
    pid = tl.program_id(0)
    if pid >= M:
        return
    row_start = pid * E
    offs_e = tl.arange(0, BLOCK_E)
    mask_e = offs_e < E
    logits = tl.load(router_logits_ptr + row_start + offs_e, mask=mask_e, other=-float('inf')).to(tl.float32)
    row_max = tl.max(logits, axis=0)
    q = 3 * (logits - row_max)
    q = tl.where(q <= -6.0, -6.0, -tl.floor(-q))
    h = tl.exp2(q * log2B)
    tl.store(q_out_ptr + row_start + offs_e, q, mask=mask_e)
    tl.store(h_out_ptr + row_start + offs_e, h, mask=mask_e)
    exps = tl.exp(logits - row_max)
    denom = tl.sum(exps, axis=0)
    probs = exps / denom
    scratch = probs
    offs_k = tl.arange(0, BLOCK_K)
    top_vals = tl.zeros((BLOCK_K,), dtype=tl.float32)
    for i in range(topk):
        mx = tl.max(scratch, axis=0)
        idx = tl.argmax(scratch, axis=0).to(tl.int32)
        top_vals = tl.where(offs_k == i, mx, top_vals)
        scratch = tl.where(offs_e == idx, 0.0, scratch)
    pre_sum = tl.sum(tl.where(offs_k < topk, top_vals, 0.0), axis=0)
    tl.store(topk_sum_ptr + pid, pre_sum)

def fused_topk_sum_qh_alpha3_beta6(router_logits, topk, B):
    M, E = router_logits.shape
    device = router_logits.device
    topk_sum = torch.empty((M,), dtype=torch.float32, device=device)
    q_out    = torch.empty((M, E), dtype=torch.float32, device=device)
    h_out    = torch.empty((M, E), dtype=torch.float32, device=device)
    BLOCK_E = 1 << (E - 1).bit_length()
    BLOCK_K = 1 << (topk - 1).bit_length()
    fused_topk_sum_qh_alpha3_beta6_kernel[(M,)](
        router_logits.contiguous(), topk_sum, q_out, h_out,
        M, E, float(math.log2(B)),
        topk=topk, BLOCK_E=BLOCK_E, BLOCK_K=BLOCK_K,
        num_warps=4, num_stages=2,
    )
    return topk_sum.view(M, 1), q_out, h_out

@triton.jit
def fused_topk_sum_qh_alpha4_beta5_kernel(
    router_logits_ptr, topk_sum_ptr, q_out_ptr, h_out_ptr,
    M, E, log2B,
    topk: tl.constexpr, BLOCK_E: tl.constexpr, BLOCK_K: tl.constexpr,
):
    pid = tl.program_id(0)
    if pid >= M:
        return
    row_start = pid * E
    offs_e = tl.arange(0, BLOCK_E)
    mask_e = offs_e < E
    logits = tl.load(router_logits_ptr + row_start + offs_e, mask=mask_e, other=-float('inf')).to(tl.float32)
    row_max = tl.max(logits, axis=0)
    q = 4 * (logits - row_max)
    q = tl.where(q <= -5.0, -5.0, -tl.floor(-q))
    h = tl.exp2(q * log2B)
    tl.store(q_out_ptr + row_start + offs_e, q, mask=mask_e)
    tl.store(h_out_ptr + row_start + offs_e, h, mask=mask_e)
    exps = tl.exp(logits - row_max)
    denom = tl.sum(exps, axis=0)
    probs = exps / denom
    scratch = probs
    offs_k = tl.arange(0, BLOCK_K)
    top_vals = tl.zeros((BLOCK_K,), dtype=tl.float32)
    for i in range(topk):
        mx = tl.max(scratch, axis=0)
        idx = tl.argmax(scratch, axis=0).to(tl.int32)
        top_vals = tl.where(offs_k == i, mx, top_vals)
        scratch = tl.where(offs_e == idx, 0.0, scratch)
    pre_sum = tl.sum(tl.where(offs_k < topk, top_vals, 0.0), axis=0)
    tl.store(topk_sum_ptr + pid, pre_sum)

def fused_topk_sum_qh_alpha4_beta5(router_logits, topk, B):
    M, E = router_logits.shape
    device = router_logits.device
    topk_sum = torch.empty((M,), dtype=torch.float32, device=device)
    q_out    = torch.empty((M, E), dtype=torch.float32, device=device)
    h_out    = torch.empty((M, E), dtype=torch.float32, device=device)
    BLOCK_E = 1 << (E - 1).bit_length()
    BLOCK_K = 1 << (topk - 1).bit_length()
    fused_topk_sum_qh_alpha4_beta5_kernel[(M,)](
        router_logits.contiguous(), topk_sum, q_out, h_out,
        M, E, float(math.log2(B)),
        topk=topk, BLOCK_E=BLOCK_E, BLOCK_K=BLOCK_K,
        num_warps=4, num_stages=2,
    )
    return topk_sum.view(M, 1), q_out, h_out

@triton.jit
def fused_topk_sum_qh_alpha4_beta7_kernel(
    router_logits_ptr, topk_sum_ptr, q_out_ptr, h_out_ptr,
    M, E, log2B,
    topk: tl.constexpr, BLOCK_E: tl.constexpr, BLOCK_K: tl.constexpr,
):
    pid = tl.program_id(0)
    if pid >= M:
        return
    row_start = pid * E
    offs_e = tl.arange(0, BLOCK_E)
    mask_e = offs_e < E
    logits = tl.load(router_logits_ptr + row_start + offs_e, mask=mask_e, other=-float('inf')).to(tl.float32)
    row_max = tl.max(logits, axis=0)
    q = 4 * (logits - row_max)
    q = tl.where(q <= -7.0, -7.0, -tl.floor(-q))
    h = tl.exp2(q * log2B)
    tl.store(q_out_ptr + row_start + offs_e, q, mask=mask_e)
    tl.store(h_out_ptr + row_start + offs_e, h, mask=mask_e)
    exps = tl.exp(logits - row_max)
    denom = tl.sum(exps, axis=0)
    probs = exps / denom
    scratch = probs
    offs_k = tl.arange(0, BLOCK_K)
    top_vals = tl.zeros((BLOCK_K,), dtype=tl.float32)
    for i in range(topk):
        mx = tl.max(scratch, axis=0)
        idx = tl.argmax(scratch, axis=0).to(tl.int32)
        top_vals = tl.where(offs_k == i, mx, top_vals)
        scratch = tl.where(offs_e == idx, 0.0, scratch)
    pre_sum = tl.sum(tl.where(offs_k < topk, top_vals, 0.0), axis=0)
    tl.store(topk_sum_ptr + pid, pre_sum)

def fused_topk_sum_qh_alpha4_beta7(router_logits, topk, B):
    M, E = router_logits.shape
    device = router_logits.device
    topk_sum = torch.empty((M,), dtype=torch.float32, device=device)
    q_out    = torch.empty((M, E), dtype=torch.float32, device=device)
    h_out    = torch.empty((M, E), dtype=torch.float32, device=device)
    BLOCK_E = 1 << (E - 1).bit_length()
    BLOCK_K = 1 << (topk - 1).bit_length()
    fused_topk_sum_qh_alpha4_beta7_kernel[(M,)](
        router_logits.contiguous(), topk_sum, q_out, h_out,
        M, E, float(math.log2(B)),
        topk=topk, BLOCK_E=BLOCK_E, BLOCK_K=BLOCK_K,
        num_warps=4, num_stages=2,
    )
    return topk_sum.view(M, 1), q_out, h_out

@triton.jit
def fused_topk_sum_qh_alpha1_beta2_kernel(
    router_logits_ptr, topk_sum_ptr, q_out_ptr, h_out_ptr,
    M, E, log2B,
    topk: tl.constexpr, BLOCK_E: tl.constexpr, BLOCK_K: tl.constexpr,
):
    pid = tl.program_id(0)
    if pid >= M:
        return
    row_start = pid * E
    offs_e = tl.arange(0, BLOCK_E)
    mask_e = offs_e < E
    logits = tl.load(router_logits_ptr + row_start + offs_e, mask=mask_e, other=-float('inf')).to(tl.float32)
    row_max = tl.max(logits, axis=0)
    q = 1 * (logits - row_max)
    q = tl.where(q <= -2.0, -2.0, -tl.floor(-q))
    h = tl.exp2(q * log2B)
    tl.store(q_out_ptr + row_start + offs_e, q, mask=mask_e)
    tl.store(h_out_ptr + row_start + offs_e, h, mask=mask_e)
    exps = tl.exp(logits - row_max)
    denom = tl.sum(exps, axis=0)
    probs = exps / denom
    scratch = probs
    offs_k = tl.arange(0, BLOCK_K)
    top_vals = tl.zeros((BLOCK_K,), dtype=tl.float32)
    for i in range(topk):
        mx = tl.max(scratch, axis=0)
        idx = tl.argmax(scratch, axis=0).to(tl.int32)
        top_vals = tl.where(offs_k == i, mx, top_vals)
        scratch = tl.where(offs_e == idx, 0.0, scratch)
    pre_sum = tl.sum(tl.where(offs_k < topk, top_vals, 0.0), axis=0)
    tl.store(topk_sum_ptr + pid, pre_sum)

def fused_topk_sum_qh_alpha1_beta2(router_logits, topk, B):
    M, E = router_logits.shape
    device = router_logits.device
    topk_sum = torch.empty((M,), dtype=torch.float32, device=device)
    q_out    = torch.empty((M, E), dtype=torch.float32, device=device)
    h_out    = torch.empty((M, E), dtype=torch.float32, device=device)
    BLOCK_E = 1 << (E - 1).bit_length()
    BLOCK_K = 1 << (topk - 1).bit_length()
    fused_topk_sum_qh_alpha1_beta2_kernel[(M,)](
        router_logits.contiguous(), topk_sum, q_out, h_out,
        M, E, float(math.log2(B)),
        topk=topk, BLOCK_E=BLOCK_E, BLOCK_K=BLOCK_K,
        num_warps=4, num_stages=2,
    )
    return topk_sum.view(M, 1), q_out, h_out

@triton.jit
def fused_topk_sum_qh_alpha2_beta2_kernel(
    router_logits_ptr, topk_sum_ptr, q_out_ptr, h_out_ptr,
    M, E, log2B,
    topk: tl.constexpr, BLOCK_E: tl.constexpr, BLOCK_K: tl.constexpr,
):
    pid = tl.program_id(0)
    if pid >= M:
        return
    row_start = pid * E
    offs_e = tl.arange(0, BLOCK_E)
    mask_e = offs_e < E
    logits = tl.load(router_logits_ptr + row_start + offs_e, mask=mask_e, other=-float('inf')).to(tl.float32)
    row_max = tl.max(logits, axis=0)
    q = 2 * (logits - row_max)
    q = tl.where(q <= -2.0, -2.0, -tl.floor(-q))
    h = tl.exp2(q * log2B)
    tl.store(q_out_ptr + row_start + offs_e, q, mask=mask_e)
    tl.store(h_out_ptr + row_start + offs_e, h, mask=mask_e)
    exps = tl.exp(logits - row_max)
    denom = tl.sum(exps, axis=0)
    probs = exps / denom
    scratch = probs
    offs_k = tl.arange(0, BLOCK_K)
    top_vals = tl.zeros((BLOCK_K,), dtype=tl.float32)
    for i in range(topk):
        mx = tl.max(scratch, axis=0)
        idx = tl.argmax(scratch, axis=0).to(tl.int32)
        top_vals = tl.where(offs_k == i, mx, top_vals)
        scratch = tl.where(offs_e == idx, 0.0, scratch)
    pre_sum = tl.sum(tl.where(offs_k < topk, top_vals, 0.0), axis=0)
    tl.store(topk_sum_ptr + pid, pre_sum)

def fused_topk_sum_qh_alpha2_beta2(router_logits, topk, B):
    M, E = router_logits.shape
    device = router_logits.device
    topk_sum = torch.empty((M,), dtype=torch.float32, device=device)
    q_out    = torch.empty((M, E), dtype=torch.float32, device=device)
    h_out    = torch.empty((M, E), dtype=torch.float32, device=device)
    BLOCK_E = 1 << (E - 1).bit_length()
    BLOCK_K = 1 << (topk - 1).bit_length()
    fused_topk_sum_qh_alpha2_beta2_kernel[(M,)](
        router_logits.contiguous(), topk_sum, q_out, h_out,
        M, E, float(math.log2(B)),
        topk=topk, BLOCK_E=BLOCK_E, BLOCK_K=BLOCK_K,
        num_warps=4, num_stages=2,
    )
    return topk_sum.view(M, 1), q_out, h_out

@triton.jit
def fused_topk_sum_qh_alpha2_beta3_kernel(
    router_logits_ptr, topk_sum_ptr, q_out_ptr, h_out_ptr,
    M, E, log2B,
    topk: tl.constexpr, BLOCK_E: tl.constexpr, BLOCK_K: tl.constexpr,
):
    pid = tl.program_id(0)
    if pid >= M:
        return
    row_start = pid * E
    offs_e = tl.arange(0, BLOCK_E)
    mask_e = offs_e < E
    logits = tl.load(router_logits_ptr + row_start + offs_e, mask=mask_e, other=-float('inf')).to(tl.float32)
    row_max = tl.max(logits, axis=0)
    q = 2 * (logits - row_max)
    q = tl.where(q <= -3.0, -3.0, -tl.floor(-q))
    h = tl.exp2(q * log2B)
    tl.store(q_out_ptr + row_start + offs_e, q, mask=mask_e)
    tl.store(h_out_ptr + row_start + offs_e, h, mask=mask_e)
    exps = tl.exp(logits - row_max)
    denom = tl.sum(exps, axis=0)
    probs = exps / denom
    scratch = probs
    offs_k = tl.arange(0, BLOCK_K)
    top_vals = tl.zeros((BLOCK_K,), dtype=tl.float32)
    for i in range(topk):
        mx = tl.max(scratch, axis=0)
        idx = tl.argmax(scratch, axis=0).to(tl.int32)
        top_vals = tl.where(offs_k == i, mx, top_vals)
        scratch = tl.where(offs_e == idx, 0.0, scratch)
    pre_sum = tl.sum(tl.where(offs_k < topk, top_vals, 0.0), axis=0)
    tl.store(topk_sum_ptr + pid, pre_sum)

@triton.jit
def fused_topk_sum_qh_alpha2_beta4_kernel(
    router_logits_ptr, topk_sum_ptr, q_out_ptr, h_out_ptr,
    M, E, log2B,
    topk: tl.constexpr, BLOCK_E: tl.constexpr, BLOCK_K: tl.constexpr,
):
    pid = tl.program_id(0)
    if pid >= M:
        return
    row_start = pid * E
    offs_e = tl.arange(0, BLOCK_E)
    mask_e = offs_e < E
    logits = tl.load(router_logits_ptr + row_start + offs_e, mask=mask_e, other=-float('inf')).to(tl.float32)
    row_max = tl.max(logits, axis=0)
    q = 2 * (logits - row_max)
    q = tl.where(q <= -4.0, -4.0, -tl.floor(-q))
    h = tl.exp2(q * log2B)
    tl.store(q_out_ptr + row_start + offs_e, q, mask=mask_e)
    tl.store(h_out_ptr + row_start + offs_e, h, mask=mask_e)
    exps = tl.exp(logits - row_max)
    denom = tl.sum(exps, axis=0)
    probs = exps / denom
    scratch = probs
    offs_k = tl.arange(0, BLOCK_K)
    top_vals = tl.zeros((BLOCK_K,), dtype=tl.float32)
    for i in range(topk):
        mx = tl.max(scratch, axis=0)
        idx = tl.argmax(scratch, axis=0).to(tl.int32)
        top_vals = tl.where(offs_k == i, mx, top_vals)
        scratch = tl.where(offs_e == idx, 0.0, scratch)
    pre_sum = tl.sum(tl.where(offs_k < topk, top_vals, 0.0), axis=0)
    tl.store(topk_sum_ptr + pid, pre_sum)

def fused_topk_sum_qh_alpha2_beta3(router_logits, topk, B):
    M, E = router_logits.shape
    device = router_logits.device
    topk_sum = torch.empty((M,), dtype=torch.float32, device=device)
    q_out    = torch.empty((M, E), dtype=torch.float32, device=device)
    h_out    = torch.empty((M, E), dtype=torch.float32, device=device)
    BLOCK_E = 1 << (E - 1).bit_length()
    BLOCK_K = 1 << (topk - 1).bit_length()
    fused_topk_sum_qh_alpha2_beta3_kernel[(M,)](
        router_logits.contiguous(), topk_sum, q_out, h_out,
        M, E, float(math.log2(B)),
        topk=topk, BLOCK_E=BLOCK_E, BLOCK_K=BLOCK_K,
        num_warps=4, num_stages=2,
    )
    return topk_sum.view(M, 1), q_out, h_out

def fused_topk_sum_qh_alpha2_beta4(router_logits, topk, B):
    M, E = router_logits.shape
    device = router_logits.device
    topk_sum = torch.empty((M,), dtype=torch.float32, device=device)
    q_out    = torch.empty((M, E), dtype=torch.float32, device=device)
    h_out    = torch.empty((M, E), dtype=torch.float32, device=device)
    BLOCK_E = 1 << (E - 1).bit_length()
    BLOCK_K = 1 << (topk - 1).bit_length()
    fused_topk_sum_qh_alpha2_beta4_kernel[(M,)](
        router_logits.contiguous(), topk_sum, q_out, h_out,
        M, E, float(math.log2(B)),
        topk=topk, BLOCK_E=BLOCK_E, BLOCK_K=BLOCK_K,
        num_warps=4, num_stages=2,
    )
    return topk_sum.view(M, 1), q_out, h_out

@triton.jit
def fused_topk_sum_qh_alpha15_beta2_kernel(
    router_logits_ptr, topk_sum_ptr, q_out_ptr, h_out_ptr,
    M, E, log2B,
    topk: tl.constexpr, BLOCK_E: tl.constexpr, BLOCK_K: tl.constexpr,
):
    pid = tl.program_id(0)
    if pid >= M:
        return
    row_start = pid * E
    offs_e = tl.arange(0, BLOCK_E)
    mask_e = offs_e < E
    logits = tl.load(router_logits_ptr + row_start + offs_e, mask=mask_e, other=-float('inf')).to(tl.float32)
    row_max = tl.max(logits, axis=0)
    q = 1.5 * (logits - row_max)
    q = tl.where(q <= -2.0, -2.0, -tl.floor(-q))
    h = tl.exp2(q * log2B)
    tl.store(q_out_ptr + row_start + offs_e, q, mask=mask_e)
    tl.store(h_out_ptr + row_start + offs_e, h, mask=mask_e)
    exps = tl.exp(logits - row_max)
    denom = tl.sum(exps, axis=0)
    probs = exps / denom
    scratch = probs
    offs_k = tl.arange(0, BLOCK_K)
    top_vals = tl.zeros((BLOCK_K,), dtype=tl.float32)
    for i in range(topk):
        mx = tl.max(scratch, axis=0)
        idx = tl.argmax(scratch, axis=0).to(tl.int32)
        top_vals = tl.where(offs_k == i, mx, top_vals)
        scratch = tl.where(offs_e == idx, 0.0, scratch)
    pre_sum = tl.sum(tl.where(offs_k < topk, top_vals, 0.0), axis=0)
    tl.store(topk_sum_ptr + pid, pre_sum)

def fused_topk_sum_qh_alpha15_beta2(router_logits, topk, B):
    M, E = router_logits.shape
    device = router_logits.device
    topk_sum = torch.empty((M,), dtype=torch.float32, device=device)
    q_out    = torch.empty((M, E), dtype=torch.float32, device=device)
    h_out    = torch.empty((M, E), dtype=torch.float32, device=device)
    BLOCK_E = 1 << (E - 1).bit_length()
    BLOCK_K = 1 << (topk - 1).bit_length()
    fused_topk_sum_qh_alpha15_beta2_kernel[(M,)](
        router_logits.contiguous(), topk_sum, q_out, h_out,
        M, E, float(math.log2(B)),
        topk=topk, BLOCK_E=BLOCK_E, BLOCK_K=BLOCK_K,
        num_warps=4, num_stages=2,
    )
    return topk_sum.view(M, 1), q_out, h_out

@triton.jit
def fused_topk_sum_qh_alpha15_beta3_kernel(
    router_logits_ptr, topk_sum_ptr, q_out_ptr, h_out_ptr,
    M, E, log2B,
    topk: tl.constexpr, BLOCK_E: tl.constexpr, BLOCK_K: tl.constexpr,
):
    pid = tl.program_id(0)
    if pid >= M:
        return
    row_start = pid * E
    offs_e = tl.arange(0, BLOCK_E)
    mask_e = offs_e < E
    logits = tl.load(router_logits_ptr + row_start + offs_e, mask=mask_e, other=-float('inf')).to(tl.float32)
    row_max = tl.max(logits, axis=0)
    q = 1.5 * (logits - row_max)
    q = tl.where(q <= -3.0, -3.0, -tl.floor(-q))
    h = tl.exp2(q * log2B)
    tl.store(q_out_ptr + row_start + offs_e, q, mask=mask_e)
    tl.store(h_out_ptr + row_start + offs_e, h, mask=mask_e)
    exps = tl.exp(logits - row_max)
    denom = tl.sum(exps, axis=0)
    probs = exps / denom
    scratch = probs
    offs_k = tl.arange(0, BLOCK_K)
    top_vals = tl.zeros((BLOCK_K,), dtype=tl.float32)
    for i in range(topk):
        mx = tl.max(scratch, axis=0)
        idx = tl.argmax(scratch, axis=0).to(tl.int32)
        top_vals = tl.where(offs_k == i, mx, top_vals)
        scratch = tl.where(offs_e == idx, 0.0, scratch)
    pre_sum = tl.sum(tl.where(offs_k < topk, top_vals, 0.0), axis=0)
    tl.store(topk_sum_ptr + pid, pre_sum)

def fused_topk_sum_qh_alpha15_beta3(router_logits, topk, B):
    M, E = router_logits.shape
    device = router_logits.device
    topk_sum = torch.empty((M,), dtype=torch.float32, device=device)
    q_out    = torch.empty((M, E), dtype=torch.float32, device=device)
    h_out    = torch.empty((M, E), dtype=torch.float32, device=device)
    BLOCK_E = 1 << (E - 1).bit_length()
    BLOCK_K = 1 << (topk - 1).bit_length()
    fused_topk_sum_qh_alpha15_beta3_kernel[(M,)](
        router_logits.contiguous(), topk_sum, q_out, h_out,
        M, E, float(math.log2(B)),
        topk=topk, BLOCK_E=BLOCK_E, BLOCK_K=BLOCK_K,
        num_warps=4, num_stages=2,
    )
    return topk_sum.view(M, 1), q_out, h_out

@triton.jit
def fused_topk_sum_qh_alpha1125_beta2_kernel(
    router_logits_ptr, topk_sum_ptr, q_out_ptr, h_out_ptr,
    M, E, log2B,
    topk: tl.constexpr, BLOCK_E: tl.constexpr, BLOCK_K: tl.constexpr,
):
    pid = tl.program_id(0)
    if pid >= M:
        return
    row_start = pid * E
    offs_e = tl.arange(0, BLOCK_E)
    mask_e = offs_e < E
    logits = tl.load(router_logits_ptr + row_start + offs_e, mask=mask_e, other=-float('inf')).to(tl.float32)
    row_max = tl.max(logits, axis=0)
    q = 1.125 * (logits - row_max)
    q = tl.where(q <= -2.0, -2.0, -tl.floor(-q))
    h = tl.exp2(q * log2B)
    tl.store(q_out_ptr + row_start + offs_e, q, mask=mask_e)
    tl.store(h_out_ptr + row_start + offs_e, h, mask=mask_e)
    exps = tl.exp(logits - row_max)
    denom = tl.sum(exps, axis=0)
    probs = exps / denom
    scratch = probs
    offs_k = tl.arange(0, BLOCK_K)
    top_vals = tl.zeros((BLOCK_K,), dtype=tl.float32)
    for i in range(topk):
        mx = tl.max(scratch, axis=0)
        idx = tl.argmax(scratch, axis=0).to(tl.int32)
        top_vals = tl.where(offs_k == i, mx, top_vals)
        scratch = tl.where(offs_e == idx, 0.0, scratch)
    pre_sum = tl.sum(tl.where(offs_k < topk, top_vals, 0.0), axis=0)
    tl.store(topk_sum_ptr + pid, pre_sum)

def fused_topk_sum_qh_alpha1125_beta2(router_logits, topk, B):
    M, E = router_logits.shape
    device = router_logits.device
    topk_sum = torch.empty((M,), dtype=torch.float32, device=device)
    q_out    = torch.empty((M, E), dtype=torch.float32, device=device)
    h_out    = torch.empty((M, E), dtype=torch.float32, device=device)
    BLOCK_E = 1 << (E - 1).bit_length()
    BLOCK_K = 1 << (topk - 1).bit_length()
    fused_topk_sum_qh_alpha1125_beta2_kernel[(M,)](
        router_logits.contiguous(), topk_sum, q_out, h_out,
        M, E, float(math.log2(B)),
        topk=topk, BLOCK_E=BLOCK_E, BLOCK_K=BLOCK_K,
        num_warps=4, num_stages=2,
    )
    return topk_sum.view(M, 1), q_out, h_out

@triton.jit
def fused_topk_sum_qh_alpha6_beta8_kernel(
    router_logits_ptr, topk_sum_ptr, q_out_ptr, h_out_ptr,
    M, E, log2B,
    topk: tl.constexpr, BLOCK_E: tl.constexpr, BLOCK_K: tl.constexpr,
):
    pid = tl.program_id(0)
    if pid >= M:
        return
    row_start = pid * E
    offs_e = tl.arange(0, BLOCK_E)
    mask_e = offs_e < E
    logits = tl.load(router_logits_ptr + row_start + offs_e, mask=mask_e, other=-float('inf')).to(tl.float32)
    row_max = tl.max(logits, axis=0)
    q = 6 * (logits - row_max)
    q = tl.where(q <= -8.0, -8.0, -tl.floor(-q))
    h = tl.exp2(q * log2B)
    tl.store(q_out_ptr + row_start + offs_e, q, mask=mask_e)
    tl.store(h_out_ptr + row_start + offs_e, h, mask=mask_e)
    exps = tl.exp(logits - row_max)
    denom = tl.sum(exps, axis=0)
    probs = exps / denom
    scratch = probs
    offs_k = tl.arange(0, BLOCK_K)
    top_vals = tl.zeros((BLOCK_K,), dtype=tl.float32)
    for i in range(topk):
        mx = tl.max(scratch, axis=0)
        idx = tl.argmax(scratch, axis=0).to(tl.int32)
        top_vals = tl.where(offs_k == i, mx, top_vals)
        scratch = tl.where(offs_e == idx, 0.0, scratch)
    pre_sum = tl.sum(tl.where(offs_k < topk, top_vals, 0.0), axis=0)
    tl.store(topk_sum_ptr + pid, pre_sum)

def fused_topk_sum_qh_alpha6_beta8(router_logits, topk, B):
    M, E = router_logits.shape
    device = router_logits.device
    topk_sum = torch.empty((M,), dtype=torch.float32, device=device)
    q_out    = torch.empty((M, E), dtype=torch.float32, device=device)
    h_out    = torch.empty((M, E), dtype=torch.float32, device=device)
    BLOCK_E = 1 << (E - 1).bit_length()
    BLOCK_K = 1 << (topk - 1).bit_length()
    fused_topk_sum_qh_alpha6_beta8_kernel[(M,)](
        router_logits.contiguous(), topk_sum, q_out, h_out,
        M, E, float(math.log2(B)),
        topk=topk, BLOCK_E=BLOCK_E, BLOCK_K=BLOCK_K,
        num_warps=4, num_stages=2,
    )
    return topk_sum.view(M, 1), q_out, h_out

@triton.jit
def fused_topk_sum_qh_alpha3_beta3_kernel(
    router_logits_ptr, topk_sum_ptr, q_out_ptr, h_out_ptr,
    M, E, log2B,
    topk: tl.constexpr, BLOCK_E: tl.constexpr, BLOCK_K: tl.constexpr,
):
    pid = tl.program_id(0)
    if pid >= M:
        return
    row_start = pid * E
    offs_e = tl.arange(0, BLOCK_E)
    mask_e = offs_e < E
    logits = tl.load(router_logits_ptr + row_start + offs_e, mask=mask_e, other=-float('inf')).to(tl.float32)
    row_max = tl.max(logits, axis=0)
    q = 3 * (logits - row_max)
    q = tl.where(q <= -3.0, -3.0, -tl.floor(-q))
    h = tl.exp2(q * log2B)
    tl.store(q_out_ptr + row_start + offs_e, q, mask=mask_e)
    tl.store(h_out_ptr + row_start + offs_e, h, mask=mask_e)
    exps = tl.exp(logits - row_max)
    denom = tl.sum(exps, axis=0)
    probs = exps / denom
    scratch = probs
    offs_k = tl.arange(0, BLOCK_K)
    top_vals = tl.zeros((BLOCK_K,), dtype=tl.float32)
    for i in range(topk):
        mx = tl.max(scratch, axis=0)
        idx = tl.argmax(scratch, axis=0).to(tl.int32)
        top_vals = tl.where(offs_k == i, mx, top_vals)
        scratch = tl.where(offs_e == idx, 0.0, scratch)
    pre_sum = tl.sum(tl.where(offs_k < topk, top_vals, 0.0), axis=0)
    tl.store(topk_sum_ptr + pid, pre_sum)

def fused_topk_sum_qh_alpha3_beta3(router_logits, topk, B):
    M, E = router_logits.shape
    device = router_logits.device
    topk_sum = torch.empty((M,), dtype=torch.float32, device=device)
    q_out    = torch.empty((M, E), dtype=torch.float32, device=device)
    h_out    = torch.empty((M, E), dtype=torch.float32, device=device)
    BLOCK_E = 1 << (E - 1).bit_length()
    BLOCK_K = 1 << (topk - 1).bit_length()
    fused_topk_sum_qh_alpha3_beta3_kernel[(M,)](
        router_logits.contiguous(), topk_sum, q_out, h_out,
        M, E, float(math.log2(B)),
        topk=topk, BLOCK_E=BLOCK_E, BLOCK_K=BLOCK_K,
        num_warps=4, num_stages=2,
    )
    return topk_sum.view(M, 1), q_out, h_out

@triton.jit
def fused_topk_sum_qh_alpha14_beta2_kernel(
    router_logits_ptr, topk_sum_ptr, q_out_ptr, h_out_ptr,
    M, E, log2B,
    topk: tl.constexpr, BLOCK_E: tl.constexpr, BLOCK_K: tl.constexpr,
):
    pid = tl.program_id(0)
    if pid >= M:
        return
    row_start = pid * E
    offs_e = tl.arange(0, BLOCK_E)
    mask_e = offs_e < E
    logits = tl.load(router_logits_ptr + row_start + offs_e, mask=mask_e, other=-float('inf')).to(tl.float32)
    row_max = tl.max(logits, axis=0)
    q = 1.4 * (logits - row_max)
    q = tl.where(q <= -2.0, -2.0, -tl.floor(-q))
    h = tl.exp2(q * log2B)
    tl.store(q_out_ptr + row_start + offs_e, q, mask=mask_e)
    tl.store(h_out_ptr + row_start + offs_e, h, mask=mask_e)
    exps = tl.exp(logits - row_max)
    denom = tl.sum(exps, axis=0)
    probs = exps / denom
    scratch = probs
    offs_k = tl.arange(0, BLOCK_K)
    top_vals = tl.zeros((BLOCK_K,), dtype=tl.float32)
    for i in range(topk):
        mx = tl.max(scratch, axis=0)
        idx = tl.argmax(scratch, axis=0).to(tl.int32)
        top_vals = tl.where(offs_k == i, mx, top_vals)
        scratch = tl.where(offs_e == idx, 0.0, scratch)
    pre_sum = tl.sum(tl.where(offs_k < topk, top_vals, 0.0), axis=0)
    tl.store(topk_sum_ptr + pid, pre_sum)

def fused_topk_sum_qh_alpha14_beta2(router_logits, topk, B):
    M, E = router_logits.shape
    device = router_logits.device
    topk_sum = torch.empty((M,), dtype=torch.float32, device=device)
    q_out    = torch.empty((M, E), dtype=torch.float32, device=device)
    h_out    = torch.empty((M, E), dtype=torch.float32, device=device)
    BLOCK_E = 1 << (E - 1).bit_length()
    BLOCK_K = 1 << (topk - 1).bit_length()
    fused_topk_sum_qh_alpha14_beta2_kernel[(M,)](
        router_logits.contiguous(), topk_sum, q_out, h_out,
        M, E, float(math.log2(B)),
        topk=topk, BLOCK_E=BLOCK_E, BLOCK_K=BLOCK_K,
        num_warps=4, num_stages=2,
    )
    return topk_sum.view(M, 1), q_out, h_out

@triton.jit
def fused_topk_sum_qh_alpha225_beta4_kernel(
    router_logits_ptr, topk_sum_ptr, q_out_ptr, h_out_ptr,
    M, E, log2B,
    topk: tl.constexpr, BLOCK_E: tl.constexpr, BLOCK_K: tl.constexpr,
):
    pid = tl.program_id(0)
    if pid >= M:
        return
    row_start = pid * E
    offs_e = tl.arange(0, BLOCK_E)
    mask_e = offs_e < E
    logits = tl.load(router_logits_ptr + row_start + offs_e, mask=mask_e, other=-float('inf')).to(tl.float32)
    row_max = tl.max(logits, axis=0)
    q = 2.25 * (logits - row_max)
    q = tl.where(q <= -4.0, -4.0, -tl.floor(-q))
    h = tl.exp2(q * log2B)
    tl.store(q_out_ptr + row_start + offs_e, q, mask=mask_e)
    tl.store(h_out_ptr + row_start + offs_e, h, mask=mask_e)
    exps = tl.exp(logits - row_max)
    denom = tl.sum(exps, axis=0)
    probs = exps / denom
    scratch = probs
    offs_k = tl.arange(0, BLOCK_K)
    top_vals = tl.zeros((BLOCK_K,), dtype=tl.float32)
    for i in range(topk):
        mx = tl.max(scratch, axis=0)
        idx = tl.argmax(scratch, axis=0).to(tl.int32)
        top_vals = tl.where(offs_k == i, mx, top_vals)
        scratch = tl.where(offs_e == idx, 0.0, scratch)
    pre_sum = tl.sum(tl.where(offs_k < topk, top_vals, 0.0), axis=0)
    tl.store(topk_sum_ptr + pid, pre_sum)

def fused_topk_sum_qh_alpha225_beta4(router_logits, topk, B):
    M, E = router_logits.shape
    device = router_logits.device
    topk_sum = torch.empty((M,), dtype=torch.float32, device=device)
    q_out    = torch.empty((M, E), dtype=torch.float32, device=device)
    h_out    = torch.empty((M, E), dtype=torch.float32, device=device)
    BLOCK_E = 1 << (E - 1).bit_length()
    BLOCK_K = 1 << (topk - 1).bit_length()
    fused_topk_sum_qh_alpha225_beta4_kernel[(M,)](
        router_logits.contiguous(), topk_sum, q_out, h_out,
        M, E, float(math.log2(B)),
        topk=topk, BLOCK_E=BLOCK_E, BLOCK_K=BLOCK_K,
        num_warps=4, num_stages=2,
    )
    return topk_sum.view(M, 1), q_out, h_out

def fused_topk_sum_qh_alpha3(router_logits, topk, B):
    M, E = router_logits.shape
    device = router_logits.device
    topk_sum = torch.empty((M,), dtype=torch.float32, device=device)
    q_out    = torch.empty((M, E), dtype=torch.float32, device=device)
    h_out    = torch.empty((M, E), dtype=torch.float32, device=device)
    BLOCK_E = 1 << (E - 1).bit_length()
    BLOCK_K = 1 << (topk - 1).bit_length()
    fused_topk_sum_qh_alpha3_kernel[(M,)](
        router_logits.contiguous(), topk_sum, q_out, h_out,
        M, E, float(math.log2(B)),
        topk=topk, BLOCK_E=BLOCK_E, BLOCK_K=BLOCK_K,
        num_warps=4, num_stages=2,
    )
    return topk_sum.view(M, 1), q_out, h_out

@triton.jit
def fused_topk_sum_qh_alpha07_kernel(
    router_logits_ptr, topk_sum_ptr, q_out_ptr, h_out_ptr,
    M, E, log2B,
    topk: tl.constexpr, BLOCK_E: tl.constexpr, BLOCK_K: tl.constexpr,
):
    pid = tl.program_id(0)
    if pid >= M:
        return
    row_start = pid * E
    offs_e = tl.arange(0, BLOCK_E)
    mask_e = offs_e < E
    logits = tl.load(router_logits_ptr + row_start + offs_e, mask=mask_e, other=-float('inf')).to(tl.float32)
    row_max = tl.max(logits, axis=0)
    q = 0.7 * (logits - row_max)
    h = tl.exp2(q * log2B)
    tl.store(q_out_ptr + row_start + offs_e, q, mask=mask_e)
    tl.store(h_out_ptr + row_start + offs_e, h, mask=mask_e)
    exps = tl.exp(logits - row_max)
    denom = tl.sum(exps, axis=0)
    probs = exps / denom
    scratch = probs
    offs_k = tl.arange(0, BLOCK_K)
    top_vals = tl.zeros((BLOCK_K,), dtype=tl.float32)
    for i in range(topk):
        mx = tl.max(scratch, axis=0)
        idx = tl.argmax(scratch, axis=0).to(tl.int32)
        top_vals = tl.where(offs_k == i, mx, top_vals)
        scratch = tl.where(offs_e == idx, 0.0, scratch)
    pre_sum = tl.sum(tl.where(offs_k < topk, top_vals, 0.0), axis=0)
    tl.store(topk_sum_ptr + pid, pre_sum)

def fused_topk_sum_qh_alpha07(router_logits, topk, B):
    M, E = router_logits.shape
    device = router_logits.device
    topk_sum = torch.empty((M,), dtype=torch.float32, device=device)
    q_out    = torch.empty((M, E), dtype=torch.float32, device=device)
    h_out    = torch.empty((M, E), dtype=torch.float32, device=device)
    BLOCK_E = 1 << (E - 1).bit_length()
    BLOCK_K = 1 << (topk - 1).bit_length()
    fused_topk_sum_qh_alpha07_kernel[(M,)](
        router_logits.contiguous(), topk_sum, q_out, h_out,
        M, E, float(math.log2(B)),
        topk=topk, BLOCK_E=BLOCK_E, BLOCK_K=BLOCK_K,
        num_warps=4, num_stages=2,
    )
    return topk_sum.view(M, 1), q_out, h_out

@triton.jit
def fused_topk_sum_qh_alpha1125_kernel(
    router_logits_ptr, topk_sum_ptr, q_out_ptr, h_out_ptr,
    M, E, log2B,
    topk: tl.constexpr, BLOCK_E: tl.constexpr, BLOCK_K: tl.constexpr,
):
    pid = tl.program_id(0)
    if pid >= M:
        return
    row_start = pid * E
    offs_e = tl.arange(0, BLOCK_E)
    mask_e = offs_e < E
    logits = tl.load(router_logits_ptr + row_start + offs_e, mask=mask_e, other=-float('inf')).to(tl.float32)
    row_max = tl.max(logits, axis=0)
    q = 1.125 * (logits - row_max)
    h = tl.exp2(q * log2B)
    tl.store(q_out_ptr + row_start + offs_e, q, mask=mask_e)
    tl.store(h_out_ptr + row_start + offs_e, h, mask=mask_e)
    exps = tl.exp(logits - row_max)
    denom = tl.sum(exps, axis=0)
    probs = exps / denom
    scratch = probs
    offs_k = tl.arange(0, BLOCK_K)
    top_vals = tl.zeros((BLOCK_K,), dtype=tl.float32)
    for i in range(topk):
        mx = tl.max(scratch, axis=0)
        idx = tl.argmax(scratch, axis=0).to(tl.int32)
        top_vals = tl.where(offs_k == i, mx, top_vals)
        scratch = tl.where(offs_e == idx, 0.0, scratch)
    pre_sum = tl.sum(tl.where(offs_k < topk, top_vals, 0.0), axis=0)
    tl.store(topk_sum_ptr + pid, pre_sum)

def fused_topk_sum_qh_alpha1125(router_logits, topk, B):
    M, E = router_logits.shape
    device = router_logits.device
    topk_sum = torch.empty((M,), dtype=torch.float32, device=device)
    q_out    = torch.empty((M, E), dtype=torch.float32, device=device)
    h_out    = torch.empty((M, E), dtype=torch.float32, device=device)
    BLOCK_E = 1 << (E - 1).bit_length()
    BLOCK_K = 1 << (topk - 1).bit_length()
    fused_topk_sum_qh_alpha1125_kernel[(M,)](
        router_logits.contiguous(), topk_sum, q_out, h_out,
        M, E, float(math.log2(B)),
        topk=topk, BLOCK_E=BLOCK_E, BLOCK_K=BLOCK_K,
        num_warps=4, num_stages=2,
    )
    return topk_sum.view(M, 1), q_out, h_out

@triton.jit
def fused_topk_sum_qh_alpha4_kernel(
    router_logits_ptr, topk_sum_ptr, q_out_ptr, h_out_ptr,
    M, E, log2B,
    topk: tl.constexpr, BLOCK_E: tl.constexpr, BLOCK_K: tl.constexpr,
):
    pid = tl.program_id(0)
    if pid >= M:
        return
    row_start = pid * E
    offs_e = tl.arange(0, BLOCK_E)
    mask_e = offs_e < E
    logits = tl.load(router_logits_ptr + row_start + offs_e, mask=mask_e, other=-float('inf')).to(tl.float32)
    row_max = tl.max(logits, axis=0)
    q = 4 * (logits - row_max)
    h = tl.exp2(q * log2B)
    tl.store(q_out_ptr + row_start + offs_e, q, mask=mask_e)
    tl.store(h_out_ptr + row_start + offs_e, h, mask=mask_e)
    exps = tl.exp(logits - row_max)
    denom = tl.sum(exps, axis=0)
    probs = exps / denom
    scratch = probs
    offs_k = tl.arange(0, BLOCK_K)
    top_vals = tl.zeros((BLOCK_K,), dtype=tl.float32)
    for i in range(topk):
        mx = tl.max(scratch, axis=0)
        idx = tl.argmax(scratch, axis=0).to(tl.int32)
        top_vals = tl.where(offs_k == i, mx, top_vals)
        scratch = tl.where(offs_e == idx, 0.0, scratch)
    pre_sum = tl.sum(tl.where(offs_k < topk, top_vals, 0.0), axis=0)
    tl.store(topk_sum_ptr + pid, pre_sum)

def fused_topk_sum_qh_alpha4(router_logits, topk, B):
    M, E = router_logits.shape
    device = router_logits.device
    topk_sum = torch.empty((M,), dtype=torch.float32, device=device)
    q_out    = torch.empty((M, E), dtype=torch.float32, device=device)
    h_out    = torch.empty((M, E), dtype=torch.float32, device=device)
    BLOCK_E = 1 << (E - 1).bit_length()
    BLOCK_K = 1 << (topk - 1).bit_length()
    fused_topk_sum_qh_alpha4_kernel[(M,)](
        router_logits.contiguous(), topk_sum, q_out, h_out,
        M, E, float(math.log2(B)),
        topk=topk, BLOCK_E=BLOCK_E, BLOCK_K=BLOCK_K,
        num_warps=4, num_stages=2,
    )
    return topk_sum.view(M, 1), q_out, h_out

@triton.jit
def fused_topk_sum_qh_alpha2_kernel(
    router_logits_ptr, topk_sum_ptr, q_out_ptr, h_out_ptr,
    M, E, log2B,
    topk: tl.constexpr, BLOCK_E: tl.constexpr, BLOCK_K: tl.constexpr,
):
    pid = tl.program_id(0)
    if pid >= M:
        return
    row_start = pid * E
    offs_e = tl.arange(0, BLOCK_E)
    mask_e = offs_e < E
    logits = tl.load(router_logits_ptr + row_start + offs_e, mask=mask_e, other=-float('inf')).to(tl.float32)
    row_max = tl.max(logits, axis=0)
    q = 2 * (logits - row_max)
    h = tl.exp2(q * log2B)
    tl.store(q_out_ptr + row_start + offs_e, q, mask=mask_e)
    tl.store(h_out_ptr + row_start + offs_e, h, mask=mask_e)
    exps = tl.exp(logits - row_max)
    denom = tl.sum(exps, axis=0)
    probs = exps / denom
    scratch = probs
    offs_k = tl.arange(0, BLOCK_K)
    top_vals = tl.zeros((BLOCK_K,), dtype=tl.float32)
    for i in range(topk):
        mx = tl.max(scratch, axis=0)
        idx = tl.argmax(scratch, axis=0).to(tl.int32)
        top_vals = tl.where(offs_k == i, mx, top_vals)
        scratch = tl.where(offs_e == idx, 0.0, scratch)
    pre_sum = tl.sum(tl.where(offs_k < topk, top_vals, 0.0), axis=0)
    tl.store(topk_sum_ptr + pid, pre_sum)

def fused_topk_sum_qh_alpha2(router_logits, topk, B):
    M, E = router_logits.shape
    device = router_logits.device
    topk_sum = torch.empty((M,), dtype=torch.float32, device=device)
    q_out    = torch.empty((M, E), dtype=torch.float32, device=device)
    h_out    = torch.empty((M, E), dtype=torch.float32, device=device)
    BLOCK_E = 1 << (E - 1).bit_length()
    BLOCK_K = 1 << (topk - 1).bit_length()
    fused_topk_sum_qh_alpha2_kernel[(M,)](
        router_logits.contiguous(), topk_sum, q_out, h_out,
        M, E, float(math.log2(B)),
        topk=topk, BLOCK_E=BLOCK_E, BLOCK_K=BLOCK_K,
        num_warps=4, num_stages=2,
    )
    return topk_sum.view(M, 1), q_out, h_out

@triton.jit
def fused_topk_sum_qh_alpha5_kernel(
    router_logits_ptr, topk_sum_ptr, q_out_ptr, h_out_ptr,
    M, E, log2B,
    topk: tl.constexpr, BLOCK_E: tl.constexpr, BLOCK_K: tl.constexpr,
):
    pid = tl.program_id(0)
    if pid >= M:
        return
    row_start = pid * E
    offs_e = tl.arange(0, BLOCK_E)
    mask_e = offs_e < E
    logits = tl.load(router_logits_ptr + row_start + offs_e, mask=mask_e, other=-float('inf')).to(tl.float32)
    row_max = tl.max(logits, axis=0)
    q = 5 * (logits - row_max)
    h = tl.exp2(q * log2B)
    tl.store(q_out_ptr + row_start + offs_e, q, mask=mask_e)
    tl.store(h_out_ptr + row_start + offs_e, h, mask=mask_e)
    exps = tl.exp(logits - row_max)
    denom = tl.sum(exps, axis=0)
    probs = exps / denom
    scratch = probs
    offs_k = tl.arange(0, BLOCK_K)
    top_vals = tl.zeros((BLOCK_K,), dtype=tl.float32)
    for i in range(topk):
        mx = tl.max(scratch, axis=0)
        idx = tl.argmax(scratch, axis=0).to(tl.int32)
        top_vals = tl.where(offs_k == i, mx, top_vals)
        scratch = tl.where(offs_e == idx, 0.0, scratch)
    pre_sum = tl.sum(tl.where(offs_k < topk, top_vals, 0.0), axis=0)
    tl.store(topk_sum_ptr + pid, pre_sum)

@triton.jit
def fused_topk_sum_qh_alpha8_kernel(
    router_logits_ptr, topk_sum_ptr, q_out_ptr, h_out_ptr,
    M, E, log2B,
    topk: tl.constexpr, BLOCK_E: tl.constexpr, BLOCK_K: tl.constexpr,
):
    pid = tl.program_id(0)
    if pid >= M:
        return
    row_start = pid * E
    offs_e = tl.arange(0, BLOCK_E)
    mask_e = offs_e < E
    logits = tl.load(router_logits_ptr + row_start + offs_e, mask=mask_e, other=-float('inf')).to(tl.float32)
    row_max = tl.max(logits, axis=0)
    q = 8 * (logits - row_max)
    h = tl.exp2(q * log2B)
    tl.store(q_out_ptr + row_start + offs_e, q, mask=mask_e)
    tl.store(h_out_ptr + row_start + offs_e, h, mask=mask_e)
    exps = tl.exp(logits - row_max)
    denom = tl.sum(exps, axis=0)
    probs = exps / denom
    scratch = probs
    offs_k = tl.arange(0, BLOCK_K)
    top_vals = tl.zeros((BLOCK_K,), dtype=tl.float32)
    for i in range(topk):
        mx = tl.max(scratch, axis=0)
        idx = tl.argmax(scratch, axis=0).to(tl.int32)
        top_vals = tl.where(offs_k == i, mx, top_vals)
        scratch = tl.where(offs_e == idx, 0.0, scratch)
    pre_sum = tl.sum(tl.where(offs_k < topk, top_vals, 0.0), axis=0)
    tl.store(topk_sum_ptr + pid, pre_sum)

@triton.jit
def fused_topk_sum_qh_alpha16_kernel(
    router_logits_ptr, topk_sum_ptr, q_out_ptr, h_out_ptr,
    M, E, log2B,
    topk: tl.constexpr, BLOCK_E: tl.constexpr, BLOCK_K: tl.constexpr,
):
    pid = tl.program_id(0)
    if pid >= M:
        return
    row_start = pid * E
    offs_e = tl.arange(0, BLOCK_E)
    mask_e = offs_e < E
    logits = tl.load(router_logits_ptr + row_start + offs_e, mask=mask_e, other=-float('inf')).to(tl.float32)
    row_max = tl.max(logits, axis=0)
    q = 16 * (logits - row_max)
    h = tl.exp2(q * log2B)
    tl.store(q_out_ptr + row_start + offs_e, q, mask=mask_e)
    tl.store(h_out_ptr + row_start + offs_e, h, mask=mask_e)
    exps = tl.exp(logits - row_max)
    denom = tl.sum(exps, axis=0)
    probs = exps / denom
    scratch = probs
    offs_k = tl.arange(0, BLOCK_K)
    top_vals = tl.zeros((BLOCK_K,), dtype=tl.float32)
    for i in range(topk):
        mx = tl.max(scratch, axis=0)
        idx = tl.argmax(scratch, axis=0).to(tl.int32)
        top_vals = tl.where(offs_k == i, mx, top_vals)
        scratch = tl.where(offs_e == idx, 0.0, scratch)
    pre_sum = tl.sum(tl.where(offs_k < topk, top_vals, 0.0), axis=0)
    tl.store(topk_sum_ptr + pid, pre_sum)

def fused_topk_sum_qh_alpha5(router_logits, topk, B):
    M, E = router_logits.shape
    device = router_logits.device
    topk_sum = torch.empty((M,), dtype=torch.float32, device=device)
    q_out    = torch.empty((M, E), dtype=torch.float32, device=device)
    h_out    = torch.empty((M, E), dtype=torch.float32, device=device)
    BLOCK_E = 1 << (E - 1).bit_length()
    BLOCK_K = 1 << (topk - 1).bit_length()
    fused_topk_sum_qh_alpha5_kernel[(M,)](
        router_logits.contiguous(), topk_sum, q_out, h_out,
        M, E, float(math.log2(B)),
        topk=topk, BLOCK_E=BLOCK_E, BLOCK_K=BLOCK_K,
        num_warps=4, num_stages=2,
    )
    return topk_sum.view(M, 1), q_out, h_out

def fused_topk_sum_qh_alpha8(router_logits, topk, B):
    M, E = router_logits.shape
    device = router_logits.device
    topk_sum = torch.empty((M,), dtype=torch.float32, device=device)
    q_out    = torch.empty((M, E), dtype=torch.float32, device=device)
    h_out    = torch.empty((M, E), dtype=torch.float32, device=device)
    BLOCK_E = 1 << (E - 1).bit_length()
    BLOCK_K = 1 << (topk - 1).bit_length()
    fused_topk_sum_qh_alpha8_kernel[(M,)](
        router_logits.contiguous(), topk_sum, q_out, h_out,
        M, E, float(math.log2(B)),
        topk=topk, BLOCK_E=BLOCK_E, BLOCK_K=BLOCK_K,
        num_warps=4, num_stages=2,
    )
    return topk_sum.view(M, 1), q_out, h_out

def fused_topk_sum_qh_alpha16(router_logits, topk, B):
    M, E = router_logits.shape
    device = router_logits.device
    topk_sum = torch.empty((M,), dtype=torch.float32, device=device)
    q_out    = torch.empty((M, E), dtype=torch.float32, device=device)
    h_out    = torch.empty((M, E), dtype=torch.float32, device=device)
    BLOCK_E = 1 << (E - 1).bit_length()
    BLOCK_K = 1 << (topk - 1).bit_length()
    fused_topk_sum_qh_alpha16_kernel[(M,)](
        router_logits.contiguous(), topk_sum, q_out, h_out,
        M, E, float(math.log2(B)),
        topk=topk, BLOCK_E=BLOCK_E, BLOCK_K=BLOCK_K,
        num_warps=4, num_stages=2,
    )
    return topk_sum.view(M, 1), q_out, h_out

@triton.jit
def fused_counts_kernel(
    q_ptr,               # *f32 [M, E]
    h_ptr,               # *f32 [M, E]
    score_ptr,           # *f32 [E]
    counts_ptr,          # *i32 [E]
    M, E,
    invB,                # f32
    topk: tl.constexpr,
    BLOCK_E: tl.constexpr,
    BLOCK_K: tl.constexpr,
):
    pid = tl.program_id(0)
    if pid >= M:
        return

    row_start = pid * E
    offs_e = tl.arange(0, BLOCK_E)
    mask_e = offs_e < E

    q     = tl.load(q_ptr     + row_start + offs_e, mask=mask_e, other=0.0)
    h     = tl.load(h_ptr     + row_start + offs_e, mask=mask_e, other=0.0)
    score = tl.load(score_ptr + offs_e,               mask=mask_e, other=0.0)

    logits = q + (score - h) * invB
    logits = tl.where(mask_e, logits, -float("inf"))

    scratch = logits
    for _ in range(topk):
        idx = tl.argmax(scratch, axis=0).to(tl.int32)
        tl.atomic_add(counts_ptr + idx, 1)
        scratch = tl.where(mask_e & (offs_e == idx), -float("inf"), scratch)


def fused_counts(quantized_logits_f32, h_f32, common_expert_score_f32, B, topk):
    M, E = quantized_logits_f32.shape
    device = quantized_logits_f32.device

    counts = torch.zeros(E, dtype=torch.int32, device=device)

    BLOCK_E = 1 << (E - 1).bit_length()
    BLOCK_K = 1 << (topk - 1).bit_length()

    fused_counts_kernel[(M,)](
        quantized_logits_f32.contiguous(),
        h_f32.contiguous(),
        common_expert_score_f32.view(-1).contiguous(),
        counts,
        M, E,
        float(1.0 / B),
        topk=topk,
        BLOCK_E=BLOCK_E, BLOCK_K=BLOCK_K,
        num_warps=4, num_stages=2,
    )
    return counts


@triton.jit
def masked_topk_kernel(
    router_logits_ptr,   # *any [M, E] (bf16/f16/f32)
    counts_ptr,          # *i32 [E]
    orig_sum_ptr,        # *f32 [M]
    out_weights_ptr,     # *f32 [M, topk]
    out_ids_ptr,         # *i32 [M, topk]
    M, E,
    topk: tl.constexpr,
    renormalize: tl.constexpr,
    BLOCK_E: tl.constexpr,
    BLOCK_K: tl.constexpr,
):
    pid = tl.program_id(0)
    if pid >= M:
        return

    row_start = pid * E
    offs_e = tl.arange(0, BLOCK_E)
    mask_e = offs_e < E

    logits = tl.load(router_logits_ptr + row_start + offs_e,
                     mask=mask_e, other=-float("inf")).to(tl.float32)
    cnt = tl.load(counts_ptr + offs_e, mask=mask_e, other=0)
    valid = (cnt > 0)
    logits = tl.where(mask_e & valid, logits, -float("inf"))

    # stable softmax
    row_max = tl.max(logits, axis=0)
    exps = tl.exp(logits - row_max)
    denom = tl.sum(exps, axis=0)
    probs = exps / denom

    # top-k
    scratch = probs
    offs_k = tl.arange(0, BLOCK_K)
    top_vals = tl.zeros((BLOCK_K,), dtype=tl.float32)
    top_idx  = tl.zeros((BLOCK_K,), dtype=tl.int32)
    for i in range(topk):
        mx  = tl.max(scratch, axis=0)
        idx = tl.argmax(scratch, axis=0).to(tl.int32)
        top_vals = tl.where(offs_k == i, mx, top_vals)
        top_idx  = tl.where(offs_k == i, idx, top_idx)
        scratch  = tl.where(mask_e & (offs_e == idx), 0.0, scratch)

    sel = offs_k < topk
    pre_sum = tl.sum(tl.where(sel, top_vals, 0.0), axis=0)

    if renormalize:
        s = tl.where(pre_sum > 0, pre_sum, 1.0)
        top_vals = tl.where(sel, top_vals / s, top_vals)
    else:
        orig  = tl.load(orig_sum_ptr + pid)
        scale = tl.where(pre_sum > 0, orig / pre_sum, 1.0)
        top_vals = tl.where(sel, top_vals * scale, top_vals)

    row_out = pid * topk
    tl.store(out_weights_ptr + row_out + offs_k, top_vals, mask=sel)
    tl.store(out_ids_ptr    + row_out + offs_k, top_idx,  mask=sel)


def masked_topk_triton(router_logits, counts, topk, renormalize, orig_topk_weights_sum):
    M, E = router_logits.shape
    device = router_logits.device

    out_w = torch.empty((M, topk), dtype=torch.float32, device=device)
    out_i = torch.empty((M, topk), dtype=torch.int32,   device=device)

    BLOCK_E = 1 << (E - 1).bit_length()
    BLOCK_K = 1 << (topk - 1).bit_length()

    masked_topk_kernel[(M,)](
        router_logits.contiguous(),                     # no host cast
        counts.contiguous(),
        orig_topk_weights_sum.view(-1).contiguous(),    # already f32
        out_w, out_i,
        M, E,
        topk=topk,
        renormalize=renormalize,
        BLOCK_E=BLOCK_E, BLOCK_K=BLOCK_K,
        num_warps=4, num_stages=2,
    )
    return out_w, out_i



def quant_policy(hidden_states, router_logits, topk, renormalize, alpha: float):
    B, E = router_logits.size()
    dtype = router_logits.dtype
    device = router_logits.device

    # # token-wise topk
    # topk_weights_initial, _ = fused_topk_triton(
    #     hidden_states, router_logits, topk, renormalize=renormalize
    # )
    # orig_topk_weights_sum = topk_weights_initial.sum(dim=1, keepdim=True) # token-wise
    # max_logits, _ = torch.max(router_logits, dim=1, keepdim=True) # token-wise

    # quantized_logits = torch.ceil(alpha * (router_logits - max_logits)) # element-wise
    # h = torch.exp(quantized_logits.to(torch.float32) * math.log(B)).to(dtype) # element-wise

    orig_topk_weights_sum, quantized_logits_f32, h_f32 = fused_topk_sum_qh(
        router_logits, topk, alpha, B
    )

    common_expert_score = torch.sum(h_f32, dim=0, keepdim=True, dtype=torch.float32) # expert-wise    

    # common_expert_weight = (common_expert_score - h) / B # element-wise
    # modified_router_logits = (quantized_logits + common_expert_weight).contiguous() # element-wise
    
    # _, topk_ids_chosen = fused_topk_triton(hidden_states, modified_router_logits, topk, renormalize) # token-wise

    # flat_ids = topk_ids_chosen.reshape(-1)

    # # element-wise mask
    # counts = torch.zeros(E, dtype=torch.int32, device=device)
    # ones = torch.ones_like(flat_ids, dtype=torch.int32)
    # counts.scatter_add_(0, flat_ids.to(torch.int64), ones)

    counts = fused_counts(quantized_logits_f32, h_f32, common_expert_score, B, topk)

    # experts_to_keep_mask = counts > 0

    # # element-wise mask
    # ninf = torch.finfo(dtype).min
    # router_logits.masked_fill_((~experts_to_keep_mask).unsqueeze(0), ninf)


    # # token-wise topk
    # topk_weights, topk_ids = fused_topk_triton(
    #     hidden_states, router_logits, topk, renormalize=renormalize
    # )

    # # token-wise
    # final_topk_weights_sum = topk_weights.sum(dim=1, keepdim=True) 
    # if not renormalize:
    #     topk_weights = topk_weights * (orig_topk_weights_sum / final_topk_weights_sum)
    

    topk_weights, topk_ids = masked_topk_triton(
        router_logits, counts, topk, renormalize, orig_topk_weights_sum
    )
    return topk_weights, topk_ids

def quant_policy_beta(hidden_states, router_logits, topk, renormalize, alpha: float, beta: float):
    B, E = router_logits.size()
    dtype = router_logits.dtype
    device = router_logits.device

    # # token-wise topk
    # topk_weights_initial, _ = fused_topk_triton(
    #     hidden_states, router_logits, topk, renormalize=renormalize
    # )
    # orig_topk_weights_sum = topk_weights_initial.sum(dim=1, keepdim=True) # token-wise
    # max_logits, _ = torch.max(router_logits, dim=1, keepdim=True) # token-wise

    # quantized_logits = torch.ceil(alpha * (router_logits - max_logits)) # element-wise
    # h = torch.exp(quantized_logits.to(torch.float32) * math.log(B)).to(dtype) # element-wise

    orig_topk_weights_sum, quantized_logits_f32, h_f32 = fused_topk_sum_qh_beta(
        router_logits, topk, alpha, beta, B
    )

    common_expert_score = torch.sum(h_f32, dim=0, keepdim=True, dtype=torch.float32) # expert-wise    

    # common_expert_weight = (common_expert_score - h) / B # element-wise
    # modified_router_logits = (quantized_logits + common_expert_weight).contiguous() # element-wise
    
    # _, topk_ids_chosen = fused_topk_triton(hidden_states, modified_router_logits, topk, renormalize) # token-wise

    # flat_ids = topk_ids_chosen.reshape(-1)

    # # element-wise mask
    # counts = torch.zeros(E, dtype=torch.int32, device=device)
    # ones = torch.ones_like(flat_ids, dtype=torch.int32)
    # counts.scatter_add_(0, flat_ids.to(torch.int64), ones)

    counts = fused_counts(quantized_logits_f32, h_f32, common_expert_score, B, topk)

    # experts_to_keep_mask = counts > 0

    # # element-wise mask
    # ninf = torch.finfo(dtype).min
    # router_logits.masked_fill_((~experts_to_keep_mask).unsqueeze(0), ninf)


    # # token-wise topk
    # topk_weights, topk_ids = fused_topk_triton(
    #     hidden_states, router_logits, topk, renormalize=renormalize
    # )

    # # token-wise
    # final_topk_weights_sum = topk_weights.sum(dim=1, keepdim=True) 
    # if not renormalize:
    #     topk_weights = topk_weights * (orig_topk_weights_sum / final_topk_weights_sum)
    

    topk_weights, topk_ids = masked_topk_triton(
        router_logits, counts, topk, renormalize, orig_topk_weights_sum
    )
    return topk_weights, topk_ids

def quant_policy_qwen2_57b(hidden_states, router_logits, topk, renormalize):
    B, E = router_logits.size()
    dtype = router_logits.dtype
    device = router_logits.device

    orig_topk_weights_sum, quantized_logits_f32, h_f32 = fused_topk_sum_qh_alpha3_beta4(
        router_logits, topk, B
    )

    common_expert_score = torch.sum(h_f32, dim=0, keepdim=True, dtype=torch.float32) # expert-wise    

    counts = fused_counts(quantized_logits_f32, h_f32, common_expert_score, B, topk)

    topk_weights, topk_ids = masked_topk_triton(
        router_logits, counts, topk, renormalize, orig_topk_weights_sum
    )
    return topk_weights, topk_ids

def quant_policy_alpha3_beta2(hidden_states, router_logits, topk, renormalize):
    """Specialized policy for alpha=3, beta=2 (used by Qwen3-30B)."""
    B, E = router_logits.size()
    orig_topk_weights_sum, quantized_logits_f32, h_f32 = fused_topk_sum_qh_alpha3_beta2(
        router_logits, topk, B
    )
    common_expert_score = torch.sum(h_f32, dim=0, keepdim=True, dtype=torch.float32)
    counts = fused_counts(quantized_logits_f32, h_f32, common_expert_score, B, topk)
    topk_weights, topk_ids = masked_topk_triton(
        router_logits, counts, topk, renormalize, orig_topk_weights_sum
    )
    return topk_weights, topk_ids

def quant_policy_deepseekv2lite(hidden_states, router_logits, topk, renormalize, alpha: float):
    B, E = router_logits.size()
    dtype = router_logits.dtype
    device = router_logits.device

    orig_topk_weights_sum, quantized_logits_f32, h_f32 = fused_topk_sum_qh_beta1(
        router_logits, topk, alpha, B
    )

    common_expert_score = torch.sum(h_f32, dim=0, keepdim=True, dtype=torch.float32) # expert-wise    

    counts = fused_counts(quantized_logits_f32, h_f32, common_expert_score, B, topk)
    
    topk_weights, topk_ids = masked_topk_triton(
        router_logits, counts, topk, renormalize, orig_topk_weights_sum
    )
    return topk_weights, topk_ids

def quant_policy_mixtral8x7b(hidden_states, router_logits, topk, renormalize):
    B, E = router_logits.size()
    dtype = router_logits.dtype
    device = router_logits.device

    orig_topk_weights_sum, quantized_logits_f32, h_f32 = fused_topk_sum_qh_alpha70_beta1(
        router_logits, topk, B
    )

    common_expert_score = torch.sum(h_f32, dim=0, keepdim=True, dtype=torch.float32) # expert-wise    

    counts = fused_counts(quantized_logits_f32, h_f32, common_expert_score, B, topk)
    
    topk_weights, topk_ids = masked_topk_triton(
        router_logits, counts, topk, renormalize, orig_topk_weights_sum
    )
    return topk_weights, topk_ids

def quant_policy_alpha3_beta6(hidden_states, router_logits, topk, renormalize):
    B, E = router_logits.size()
    orig_topk_weights_sum, quantized_logits_f32, h_f32 = fused_topk_sum_qh_alpha3_beta6(
        router_logits, topk, B
    )
    common_expert_score = torch.sum(h_f32, dim=0, keepdim=True, dtype=torch.float32)
    counts = fused_counts(quantized_logits_f32, h_f32, common_expert_score, B, topk)
    topk_weights, topk_ids = masked_topk_triton(
        router_logits, counts, topk, renormalize, orig_topk_weights_sum
    )
    return topk_weights, topk_ids

def quant_policy_alpha4_beta5(hidden_states, router_logits, topk, renormalize):
    B, E = router_logits.size()
    orig_topk_weights_sum, quantized_logits_f32, h_f32 = fused_topk_sum_qh_alpha4_beta5(
        router_logits, topk, B
    )
    common_expert_score = torch.sum(h_f32, dim=0, keepdim=True, dtype=torch.float32)
    counts = fused_counts(quantized_logits_f32, h_f32, common_expert_score, B, topk)
    topk_weights, topk_ids = masked_topk_triton(
        router_logits, counts, topk, renormalize, orig_topk_weights_sum
    )
    return topk_weights, topk_ids

def quant_policy_alpha4_beta7(hidden_states, router_logits, topk, renormalize):
    B, E = router_logits.size()
    orig_topk_weights_sum, quantized_logits_f32, h_f32 = fused_topk_sum_qh_alpha4_beta7(
        router_logits, topk, B
    )
    common_expert_score = torch.sum(h_f32, dim=0, keepdim=True, dtype=torch.float32)
    counts = fused_counts(quantized_logits_f32, h_f32, common_expert_score, B, topk)
    topk_weights, topk_ids = masked_topk_triton(
        router_logits, counts, topk, renormalize, orig_topk_weights_sum
    )
    return topk_weights, topk_ids

def quant_policy_alpha1_beta2(hidden_states, router_logits, topk, renormalize):
    B, E = router_logits.size()
    orig_topk_weights_sum, quantized_logits_f32, h_f32 = fused_topk_sum_qh_alpha1_beta2(
        router_logits, topk, B
    )
    common_expert_score = torch.sum(h_f32, dim=0, keepdim=True, dtype=torch.float32)
    counts = fused_counts(quantized_logits_f32, h_f32, common_expert_score, B, topk)
    topk_weights, topk_ids = masked_topk_triton(
        router_logits, counts, topk, renormalize, orig_topk_weights_sum
    )
    return topk_weights, topk_ids

def quant_policy_alpha2_beta2(hidden_states, router_logits, topk, renormalize):
    B, E = router_logits.size()
    orig_topk_weights_sum, quantized_logits_f32, h_f32 = fused_topk_sum_qh_alpha2_beta2(
        router_logits, topk, B
    )
    common_expert_score = torch.sum(h_f32, dim=0, keepdim=True, dtype=torch.float32)
    counts = fused_counts(quantized_logits_f32, h_f32, common_expert_score, B, topk)
    topk_weights, topk_ids = masked_topk_triton(
        router_logits, counts, topk, renormalize, orig_topk_weights_sum
    )
    return topk_weights, topk_ids

def quant_policy_alpha2_beta3(hidden_states, router_logits, topk, renormalize):
    B, E = router_logits.size()
    orig_topk_weights_sum, quantized_logits_f32, h_f32 = fused_topk_sum_qh_alpha2_beta3(
        router_logits, topk, B
    )
    common_expert_score = torch.sum(h_f32, dim=0, keepdim=True, dtype=torch.float32)
    counts = fused_counts(quantized_logits_f32, h_f32, common_expert_score, B, topk)
    topk_weights, topk_ids = masked_topk_triton(
        router_logits, counts, topk, renormalize, orig_topk_weights_sum
    )
    return topk_weights, topk_ids

def quant_policy_alpha2_beta4(hidden_states, router_logits, topk, renormalize):
    B, E = router_logits.size()
    orig_topk_weights_sum, quantized_logits_f32, h_f32 = fused_topk_sum_qh_alpha2_beta4(
        router_logits, topk, B
    )
    common_expert_score = torch.sum(h_f32, dim=0, keepdim=True, dtype=torch.float32)
    counts = fused_counts(quantized_logits_f32, h_f32, common_expert_score, B, topk)
    topk_weights, topk_ids = masked_topk_triton(
        router_logits, counts, topk, renormalize, orig_topk_weights_sum
    )
    return topk_weights, topk_ids

def quant_policy_alpha15_beta2(hidden_states, router_logits, topk, renormalize):
    B, E = router_logits.size()
    orig_topk_weights_sum, quantized_logits_f32, h_f32 = fused_topk_sum_qh_alpha15_beta2(
        router_logits, topk, B
    )
    common_expert_score = torch.sum(h_f32, dim=0, keepdim=True, dtype=torch.float32)
    counts = fused_counts(quantized_logits_f32, h_f32, common_expert_score, B, topk)
    topk_weights, topk_ids = masked_topk_triton(
        router_logits, counts, topk, renormalize, orig_topk_weights_sum
    )
    return topk_weights, topk_ids

def quant_policy_alpha15_beta3(hidden_states, router_logits, topk, renormalize):
    B, E = router_logits.size()
    orig_topk_weights_sum, quantized_logits_f32, h_f32 = fused_topk_sum_qh_alpha15_beta3(
        router_logits, topk, B
    )
    common_expert_score = torch.sum(h_f32, dim=0, keepdim=True, dtype=torch.float32)
    counts = fused_counts(quantized_logits_f32, h_f32, common_expert_score, B, topk)
    topk_weights, topk_ids = masked_topk_triton(
        router_logits, counts, topk, renormalize, orig_topk_weights_sum
    )
    return topk_weights, topk_ids

def quant_policy_alpha1125_beta2(hidden_states, router_logits, topk, renormalize):
    B, E = router_logits.size()
    orig_topk_weights_sum, quantized_logits_f32, h_f32 = fused_topk_sum_qh_alpha1125_beta2(
        router_logits, topk, B
    )
    common_expert_score = torch.sum(h_f32, dim=0, keepdim=True, dtype=torch.float32)
    counts = fused_counts(quantized_logits_f32, h_f32, common_expert_score, B, topk)
    topk_weights, topk_ids = masked_topk_triton(
        router_logits, counts, topk, renormalize, orig_topk_weights_sum
    )
    return topk_weights, topk_ids

def quant_policy_alpha6_beta8(hidden_states, router_logits, topk, renormalize):
    B, E = router_logits.size()
    orig_topk_weights_sum, quantized_logits_f32, h_f32 = fused_topk_sum_qh_alpha6_beta8(
        router_logits, topk, B
    )
    common_expert_score = torch.sum(h_f32, dim=0, keepdim=True, dtype=torch.float32)
    counts = fused_counts(quantized_logits_f32, h_f32, common_expert_score, B, topk)
    topk_weights, topk_ids = masked_topk_triton(
        router_logits, counts, topk, renormalize, orig_topk_weights_sum
    )
    return topk_weights, topk_ids

def quant_policy_alpha3_beta3(hidden_states, router_logits, topk, renormalize):
    B, E = router_logits.size()
    orig_topk_weights_sum, quantized_logits_f32, h_f32 = fused_topk_sum_qh_alpha3_beta3(
        router_logits, topk, B
    )
    common_expert_score = torch.sum(h_f32, dim=0, keepdim=True, dtype=torch.float32)
    counts = fused_counts(quantized_logits_f32, h_f32, common_expert_score, B, topk)
    topk_weights, topk_ids = masked_topk_triton(
        router_logits, counts, topk, renormalize, orig_topk_weights_sum
    )
    return topk_weights, topk_ids

def quant_policy_alpha14_beta2(hidden_states, router_logits, topk, renormalize):
    B, E = router_logits.size()
    orig_topk_weights_sum, quantized_logits_f32, h_f32 = fused_topk_sum_qh_alpha14_beta2(
        router_logits, topk, B
    )
    common_expert_score = torch.sum(h_f32, dim=0, keepdim=True, dtype=torch.float32)
    counts = fused_counts(quantized_logits_f32, h_f32, common_expert_score, B, topk)
    topk_weights, topk_ids = masked_topk_triton(
        router_logits, counts, topk, renormalize, orig_topk_weights_sum
    )
    return topk_weights, topk_ids

def quant_policy_alpha225_beta4(hidden_states, router_logits, topk, renormalize):
    B, E = router_logits.size()
    orig_topk_weights_sum, quantized_logits_f32, h_f32 = fused_topk_sum_qh_alpha225_beta4(
        router_logits, topk, B
    )
    common_expert_score = torch.sum(h_f32, dim=0, keepdim=True, dtype=torch.float32)
    counts = fused_counts(quantized_logits_f32, h_f32, common_expert_score, B, topk)
    topk_weights, topk_ids = masked_topk_triton(
        router_logits, counts, topk, renormalize, orig_topk_weights_sum
    )
    return topk_weights, topk_ids

def quant_policy_alpha3(hidden_states, router_logits, topk, renormalize):
    B, E = router_logits.size()
    orig_topk_weights_sum, quantized_logits_f32, h_f32 = fused_topk_sum_qh_alpha3(
        router_logits, topk, B
    )
    common_expert_score = torch.sum(h_f32, dim=0, keepdim=True, dtype=torch.float32)
    counts = fused_counts(quantized_logits_f32, h_f32, common_expert_score, B, topk)
    topk_weights, topk_ids = masked_topk_triton(
        router_logits, counts, topk, renormalize, orig_topk_weights_sum
    )
    return topk_weights, topk_ids

def quant_policy_alpha07(hidden_states, router_logits, topk, renormalize):
    B, E = router_logits.size()
    orig_topk_weights_sum, quantized_logits_f32, h_f32 = fused_topk_sum_qh_alpha07(
        router_logits, topk, B
    )
    common_expert_score = torch.sum(h_f32, dim=0, keepdim=True, dtype=torch.float32)
    counts = fused_counts(quantized_logits_f32, h_f32, common_expert_score, B, topk)
    topk_weights, topk_ids = masked_topk_triton(
        router_logits, counts, topk, renormalize, orig_topk_weights_sum
    )
    return topk_weights, topk_ids

def quant_policy_alpha1125(hidden_states, router_logits, topk, renormalize):
    B, E = router_logits.size()
    orig_topk_weights_sum, quantized_logits_f32, h_f32 = fused_topk_sum_qh_alpha1125(
        router_logits, topk, B
    )
    common_expert_score = torch.sum(h_f32, dim=0, keepdim=True, dtype=torch.float32)
    counts = fused_counts(quantized_logits_f32, h_f32, common_expert_score, B, topk)
    topk_weights, topk_ids = masked_topk_triton(
        router_logits, counts, topk, renormalize, orig_topk_weights_sum
    )
    return topk_weights, topk_ids

def quant_policy_alpha4(hidden_states, router_logits, topk, renormalize):
    B, E = router_logits.size()
    orig_topk_weights_sum, quantized_logits_f32, h_f32 = fused_topk_sum_qh_alpha4(
        router_logits, topk, B
    )
    common_expert_score = torch.sum(h_f32, dim=0, keepdim=True, dtype=torch.float32)
    counts = fused_counts(quantized_logits_f32, h_f32, common_expert_score, B, topk)
    topk_weights, topk_ids = masked_topk_triton(
        router_logits, counts, topk, renormalize, orig_topk_weights_sum
    )
    return topk_weights, topk_ids

def quant_policy_alpha2(hidden_states, router_logits, topk, renormalize):
    B, E = router_logits.size()
    orig_topk_weights_sum, quantized_logits_f32, h_f32 = fused_topk_sum_qh_alpha2(
        router_logits, topk, B
    )
    common_expert_score = torch.sum(h_f32, dim=0, keepdim=True, dtype=torch.float32)
    counts = fused_counts(quantized_logits_f32, h_f32, common_expert_score, B, topk)
    topk_weights, topk_ids = masked_topk_triton(
        router_logits, counts, topk, renormalize, orig_topk_weights_sum
    )
    return topk_weights, topk_ids

def quant_policy_alpha5(hidden_states, router_logits, topk, renormalize):
    B, E = router_logits.size()
    orig_topk_weights_sum, quantized_logits_f32, h_f32 = fused_topk_sum_qh_alpha5(
        router_logits, topk, B
    )
    common_expert_score = torch.sum(h_f32, dim=0, keepdim=True, dtype=torch.float32)
    counts = fused_counts(quantized_logits_f32, h_f32, common_expert_score, B, topk)
    topk_weights, topk_ids = masked_topk_triton(
        router_logits, counts, topk, renormalize, orig_topk_weights_sum
    )
    return topk_weights, topk_ids

def quant_policy_alpha8(hidden_states, router_logits, topk, renormalize):
    B, E = router_logits.size()
    orig_topk_weights_sum, quantized_logits_f32, h_f32 = fused_topk_sum_qh_alpha8(
        router_logits, topk, B
    )
    common_expert_score = torch.sum(h_f32, dim=0, keepdim=True, dtype=torch.float32)
    counts = fused_counts(quantized_logits_f32, h_f32, common_expert_score, B, topk)
    topk_weights, topk_ids = masked_topk_triton(
        router_logits, counts, topk, renormalize, orig_topk_weights_sum
    )
    return topk_weights, topk_ids

def quant_policy_alpha16(hidden_states, router_logits, topk, renormalize):
    B, E = router_logits.size()
    orig_topk_weights_sum, quantized_logits_f32, h_f32 = fused_topk_sum_qh_alpha16(
        router_logits, topk, B
    )
    common_expert_score = torch.sum(h_f32, dim=0, keepdim=True, dtype=torch.float32)
    counts = fused_counts(quantized_logits_f32, h_f32, common_expert_score, B, topk)
    topk_weights, topk_ids = masked_topk_triton(
        router_logits, counts, topk, renormalize, orig_topk_weights_sum
    )
    return topk_weights, topk_ids

@torch.compile
def custom_routing_function_grouped_topk(
    gating_output: torch.Tensor,
    scores: torch.Tensor,
    topk: int,
    renormalize: bool):

    logit_store = LynxState.get_instance()
    if logit_store is None:
        topk_weights, topk_ids = torch.topk(scores, k=topk, dim=-1, sorted=False)
        return topk_weights, topk_ids
    is_prefill = logit_store.is_prefill
    profile_complete = logit_store.profile_complete
    policy = logit_store.policy

    # rank1 = logit_store.rank1
    # rank2 = logit_store.rank2
    ########################################
    if policy == "do-nothing":
        topk_weights, topk_ids = torch.topk(scores, k=topk, dim=-1, sorted=False)

        return topk_weights, topk_ids
        

    # Option 1: Only apply expert reduction in decode phase (original behavior)
    # if profile_complete and not is_prefill:
    
    # Option 2: Apply expert reduction after profiling regardless of prefill/decode
    # (Better for systems with continuous mixed batches where pure decode is rare)
    if profile_complete and not is_prefill:
        if policy == "quant" or policy.startswith("quant_alpha") and policy.endswith("_optimized"):
            alpha = logit_store.alpha
            beta = logit_store.beta if logit_store.beta > 0 else alpha + 1
            B, E = gating_output.size()

            dtype = gating_output.dtype
            device = gating_output.device

            orig_router_scores, orig_router_indices = torch.topk(scores, k=topk, dim=-1, sorted=False)

            orig_topk_scores_sum = orig_router_scores.sum(dim=-1, keepdim=True)

            max_logits, _ = torch.max(gating_output, dim=1, keepdim=True)
            quantized_logits = torch.clamp(torch.ceil(alpha * (gating_output - max_logits)), min=-beta)

            h = torch.exp(quantized_logits.to(torch.float32) * math.log(B)).to(dtype)
            common_expert_weight = (torch.sum(h, dim=0, keepdim=True) - h) / B
            modified_router_logits = (quantized_logits + common_expert_weight).contiguous()
            _, topk_ids_chosen = _fast_topk(modified_router_logits, topk, dim=-1)

            flat_ids = topk_ids_chosen.reshape(-1)
            counts = torch.zeros(E, dtype=torch.int32, device=device)
            ones = torch.ones_like(flat_ids, dtype=torch.int32)
            counts.scatter_add_(0, flat_ids.to(torch.int64), ones)
            experts_to_keep_mask = counts > 0
            ninf = torch.finfo(dtype).min
            scores.masked_fill_((~experts_to_keep_mask).unsqueeze(0), ninf)

            topk_weights, topk_ids = torch.topk(scores, k=topk, dim=-1, sorted=False)
            final_topk_scores_sum = topk_weights.sum(dim=-1, keepdim=True)

            if not renormalize:
                topk_weights = topk_weights * (orig_topk_scores_sum / final_topk_scores_sum)

            return topk_weights, topk_ids

    # Fallback: no policy matched or not in decode phase yet
    topk_weights, topk_ids = torch.topk(scores, k=topk, dim=-1, sorted=False)
    return topk_weights, topk_ids

def _fast_topk(values: torch.Tensor, topk: int, dim:int):
    if topk == 1:
        return torch.max(values, dim=dim, keepdim=True)
    else:
        return torch.topk(values, topk, dim=dim)

@torch.compile
def custom_routing_function_sigmoid(hidden_states: torch.Tensor,
                            gating_output: torch.Tensor,
                            topk: int,
                            renormalize: bool):
    logit_store = LynxState.get_instance()
    if logit_store is None:
        router_scores, router_indices = _fast_topk(gating_output, topk, dim=-1)
        router_scores = torch.sigmoid(router_scores.float())
        return (router_scores, router_indices.to(torch.int32))
    is_prefill = logit_store.is_prefill
    profile_complete = logit_store.profile_complete
    policy = logit_store.policy


    if policy == "do-nothing":
        router_scores, router_indices = _fast_topk(gating_output, topk, dim=-1)
        router_scores = torch.sigmoid(router_scores.float())
        return (router_scores, router_indices.to(torch.int32))
        

    if profile_complete and not is_prefill:       
  
        # First pass top-k
        orig_router_scores, orig_router_indices = _fast_topk(gating_output, topk, dim=-1)
        orig_router_scores = torch.sigmoid(orig_router_scores.float())

        if policy == "quant":
            orig_topk_scores_sum = orig_router_scores.sum(dim=-1, keepdim=True)

            alpha = logit_store.alpha
            B, E = gating_output.size()

            dtype = gating_output.dtype
            device = gating_output.device

            router_logits = gating_output
            
            max_logits, _ = torch.max(router_logits, dim=1, keepdim=True)
            quantized_logits = torch.ceil(alpha * (router_logits - max_logits))

            h = torch.exp(quantized_logits.to(torch.float32) * math.log(B)).to(dtype)
            common_expert_weight = (torch.sum(h, dim=0, keepdim=True) - h) / B
            modified_router_logits = (quantized_logits + common_expert_weight).contiguous()
            _, topk_ids_chosen = _fast_topk(modified_router_logits, topk, dim=-1)

            flat_ids = topk_ids_chosen.reshape(-1)
            counts = torch.zeros(E, dtype=torch.int32, device=device)
            ones = torch.ones_like(flat_ids, dtype=torch.int32)
            counts.scatter_add_(0, flat_ids.to(torch.int64), ones)
            experts_to_keep_mask = counts > 0
            ninf = torch.finfo(dtype).min
            gating_output.masked_fill_((~experts_to_keep_mask).unsqueeze(0), ninf)

            router_scores, router_indices = _fast_topk(gating_output, topk, dim=-1)
            router_scores = torch.sigmoid(router_scores.float())

            final_topk_scores_sum = router_scores.sum(dim=-1, keepdim=True)
            if not renormalize:
                router_scores = router_scores * (orig_topk_scores_sum / final_topk_scores_sum)
     
    else:
        router_scores, router_indices = _fast_topk(gating_output, topk, dim=-1)
        router_scores = torch.sigmoid(router_scores.float())
        
    return (router_scores, router_indices.to(torch.int32))


def custom_routing_function(hidden_states: torch.Tensor,
                            gating_output: torch.Tensor,
                            topk: int,
                            renormalize: bool):
    # Internal alias retained from the v0.10.1-era source.
    router_logits = gating_output
    logit_store = LynxState.get_instance()
    if logit_store is None:
        topk_weights, topk_ids, _ = fused_topk(
            hidden_states, router_logits, topk, renormalize
        )
        return topk_weights, topk_ids
    metric_store = MetricStore.get_instance()
    is_prefill = logit_store.is_prefill
    profile_complete = logit_store.profile_complete
    num_total_experts = logit_store.num_total_experts
    num_experts_per_token = logit_store.num_experts_per_token
    num_experts_to_keep = logit_store.num_experts_to_keep
    min_experts = logit_store.min_experts
    policy = logit_store.policy
    alpha = logit_store.alpha
    beta = logit_store.beta
    threshold_percentile = logit_store.threshold_percentile

    # DEBUG: Log entry into custom routing function
    
    # Debug logging (uncomment if needed)
    ## Params for ablation studies ##########
    count_of_topk = logit_store.count_of_topk
    # rank1 = logit_store.rank1
    # rank2 = logit_store.rank2
    ########################################
    if policy == "do-nothing":
        # task_name = "DECODE_GATING_COMPUTATION_WITHOUT_POLICY"
        # with TaskLoggingContextManagerGPU(task_type=task_name):
        topk_weights, topk_ids, _ = fused_topk(
            hidden_states, router_logits, topk, renormalize
        )
        num_experts_to_keep = logit_store.num_total_experts
        return topk_weights, topk_ids
        

    # Option 1: Only apply expert reduction in decode phase (original behavior)
    # if profile_complete and not is_prefill:
    
    # Option 2: Apply expert reduction after profiling regardless of prefill/decode
    # (Better for systems with continuous mixed batches where pure decode is rare)
    if profile_complete and not is_prefill:

        if policy == "quant_optimized":
            topk_weights, topk_ids = quant_policy_beta(
                hidden_states, router_logits, topk, renormalize=renormalize, alpha=alpha, beta=beta
            )
            return topk_weights, topk_ids
                
        elif policy == "quant_alpha1_optimized":
            topk_weights, topk_ids = quant_policy(
                hidden_states, router_logits, topk, renormalize=renormalize, alpha=1.0
            )
            return topk_weights, topk_ids

        elif policy == "quant_alpha1_beta1_optimized":
            topk_weights, topk_ids = quant_policy_deepseekv2lite(
                hidden_states, router_logits, topk, renormalize=renormalize, alpha=1.0
            )
            return topk_weights, topk_ids


        elif policy == "quant_alpha0.7_beta1_optimized":
            topk_weights, topk_ids = quant_policy_mixtral8x7b(
                hidden_states, router_logits, topk, renormalize=renormalize
            )
            return topk_weights, topk_ids


        elif policy == "quant_alpha3_beta4_optimized":
            topk_weights, topk_ids = quant_policy_qwen2_57b(
                hidden_states, router_logits, topk, renormalize=renormalize,
            )
            return topk_weights, topk_ids

        elif policy == "quant_alpha3_beta2_optimized":
            topk_weights, topk_ids = quant_policy_alpha3_beta2(
                hidden_states, router_logits, topk, renormalize=renormalize,
            )
            return topk_weights, topk_ids

        elif policy == "quant_alpha3_beta6_optimized":
            topk_weights, topk_ids = quant_policy_alpha3_beta6(
                hidden_states, router_logits, topk, renormalize=renormalize,
            )
            return topk_weights, topk_ids

        elif policy == "quant_alpha4_beta5_optimized":
            topk_weights, topk_ids = quant_policy_alpha4_beta5(
                hidden_states, router_logits, topk, renormalize=renormalize,
            )
            return topk_weights, topk_ids

        elif policy == "quant_alpha4_beta7_optimized":
            topk_weights, topk_ids = quant_policy_alpha4_beta7(
                hidden_states, router_logits, topk, renormalize=renormalize,
            )
            return topk_weights, topk_ids

        elif policy == "quant_alpha1_beta2_optimized":
            topk_weights, topk_ids = quant_policy_alpha1_beta2(
                hidden_states, router_logits, topk, renormalize=renormalize,
            )
            return topk_weights, topk_ids

        elif policy == "quant_alpha2_beta2_optimized":
            topk_weights, topk_ids = quant_policy_alpha2_beta2(
                hidden_states, router_logits, topk, renormalize=renormalize,
            )
            return topk_weights, topk_ids

        elif policy == "quant_alpha2_beta3_optimized":
            topk_weights, topk_ids = quant_policy_alpha2_beta3(
                hidden_states, router_logits, topk, renormalize=renormalize,
            )
            return topk_weights, topk_ids

        elif policy == "quant_alpha2_beta4_optimized":
            topk_weights, topk_ids = quant_policy_alpha2_beta4(
                hidden_states, router_logits, topk, renormalize=renormalize,
            )
            return topk_weights, topk_ids

        elif policy == "quant_alpha1.5_beta2_optimized":
            topk_weights, topk_ids = quant_policy_alpha15_beta2(
                hidden_states, router_logits, topk, renormalize=renormalize,
            )
            return topk_weights, topk_ids

        elif policy == "quant_alpha1.5_beta3_optimized":
            topk_weights, topk_ids = quant_policy_alpha15_beta3(
                hidden_states, router_logits, topk, renormalize=renormalize,
            )
            return topk_weights, topk_ids

        elif policy == "quant_alpha1.125_beta2_optimized":
            topk_weights, topk_ids = quant_policy_alpha1125_beta2(
                hidden_states, router_logits, topk, renormalize=renormalize,
            )
            return topk_weights, topk_ids

        elif policy == "quant_alpha6_beta8_optimized":
            topk_weights, topk_ids = quant_policy_alpha6_beta8(
                hidden_states, router_logits, topk, renormalize=renormalize,
            )
            return topk_weights, topk_ids

        elif policy == "quant_alpha3_beta3_optimized":
            topk_weights, topk_ids = quant_policy_alpha3_beta3(
                hidden_states, router_logits, topk, renormalize=renormalize,
            )
            return topk_weights, topk_ids

        elif policy == "quant_alpha1.4_beta2_optimized":
            topk_weights, topk_ids = quant_policy_alpha14_beta2(
                hidden_states, router_logits, topk, renormalize=renormalize,
            )
            return topk_weights, topk_ids

        elif policy == "quant_alpha2.25_beta4_optimized":
            topk_weights, topk_ids = quant_policy_alpha225_beta4(
                hidden_states, router_logits, topk, renormalize=renormalize,
            )
            return topk_weights, topk_ids

        elif policy == "quant_alpha3_optimized":
            topk_weights, topk_ids = quant_policy_alpha3(
                hidden_states, router_logits, topk, renormalize=renormalize,
            )
            return topk_weights, topk_ids

        elif policy == "quant_alpha0.7_optimized":
            topk_weights, topk_ids = quant_policy_alpha07(
                hidden_states, router_logits, topk, renormalize=renormalize,
            )
            return topk_weights, topk_ids

        elif policy == "quant_alpha1.125_optimized":
            topk_weights, topk_ids = quant_policy_alpha1125(
                hidden_states, router_logits, topk, renormalize=renormalize,
            )
            return topk_weights, topk_ids

        elif policy == "quant_alpha4_optimized":
            topk_weights, topk_ids = quant_policy_alpha4(
                hidden_states, router_logits, topk, renormalize=renormalize,
            )
            return topk_weights, topk_ids

        elif policy == "quant_alpha2_optimized":
            topk_weights, topk_ids = quant_policy_alpha2(
                hidden_states, router_logits, topk, renormalize=renormalize,
            )
            return topk_weights, topk_ids

        elif policy == "quant_alpha5_optimized":
            topk_weights, topk_ids = quant_policy_alpha5(
                hidden_states, router_logits, topk, renormalize=renormalize,
            )
            return topk_weights, topk_ids

        elif policy == "quant_alpha8_optimized":
            topk_weights, topk_ids = quant_policy_alpha8(
                hidden_states, router_logits, topk, renormalize=renormalize,
            )
            return topk_weights, topk_ids

        elif policy == "quant_alpha16_optimized":
            topk_weights, topk_ids = quant_policy_alpha16(
                hidden_states, router_logits, topk, renormalize=renormalize,
            )
            return topk_weights, topk_ids



        # task_name = "DECODE_GATING_COMPUTATION_WITH_POLICY"
        # with TaskLoggingContextManagerGPU(task_type=task_name):
        # masked_logits = router_logits.clone()
        
        masked_logits = router_logits

        # First pass top-k
        topk_weights_initial, topk_ids_initial, _ = fused_topk(
            hidden_states, masked_logits, topk, renormalize=renormalize
        )


        #### temporarily copying in outside
        # topk_weights, topk_ids = fused_topk(
        #     hidden_states, masked_logits, topk, renormalize=renormalize
        #         )
        
        # changed_mask = topk_ids_initial != topk_ids
        # num_changed = changed_mask.sum().item()
        # assert num_changed == 0, "The number of changed topk id entries should be zero."
        # # change in top k weights
        # diff = torch.abs(topk_weights_initial - topk_weights)
        # assert torch.allclose(topk_weights_initial, topk_weights, atol=1e-5), "The topk weights should be close."

        # # assert that router logits are not changed
        # assert torch.allclose(original_logits, masked_logits, atol=1e-5), "The router logits should be close."

        # # assert that both ways give the same topk ids
        # assert torch.allclose(topk_ids_initial, topk_ids_initial_topk, atol=1e-5), "The topk ids should be close."

        if policy == "simple":
            # task_name = "DECODE_GATING_COMPUTATION_WITH_SIMPLE_POLICY"
            # with TaskLoggingContextManagerGPU(task_type=task_name):
                
            # task_name = "NOOP"
            # with TaskLoggingContextManagerGPU(task_type=task_name):
            
            # num_experts_to_drop = 4  # or however many you want to drop
            num_experts_to_keep = logit_store.num_experts_to_keep
            num_experts_to_drop = num_total_experts - num_experts_to_keep
            # 1) Extract how often each expert was chosen (via topk_ids_initial),
            #    pick the experts to drop, and mask them in-place in masked_logits.
            # task_name = "EXTRACT_VOTING_AND_MASK"
            # with TaskLoggingContextManagerGPU(task_type=task_name):
            
            # NOTE: extract_voting_and_mask is not yet ported to new-vllm
            # For now, don't mask any experts
            # masked_logits, drop_indices = extract_voting_and_mask(
            #     topk_ids_initial,  # shape [batch_size, topk]
            #     masked_logits,     # shape [batch_size, num_experts], fp16
            #     num_experts_to_drop=num_experts_to_drop
            # )

            # 2) Now compute your topk outputs (for MoE gating usage).
            # task_name = "FUSED_TOPK"
            # with TaskLoggingContextManagerGPU(task_type=task_name):
            topk_weights, topk_ids, _ = fused_topk(
                hidden_states, 
                masked_logits, 
                topk, 
                renormalize=renormalize
            )
            # 3) Optionally, if you need to know how many experts remain “unmasked”                

        elif policy == "advanced":
            # task_name = "DECODE_GATING_COMPUTATION_WITH_ADVANCED_POLICY"
            # with TaskLoggingContextManagerGPU(task_type=task_name):
            
            # NOTE: optimize_expert_selection is not yet ported to new-vllm
            # The code below is commented out because the function doesn't exist
            # masked_logits, num_experts_to_keep, num_unique_experts, importance_mask_sum, min, max, mean, var, top_0_1, top_1_2, top_2_3, top_3_4, top_4_5, top_5_6 = optimize_expert_selection(
            #     topk_ids=topk_ids_initial,
            #     topk_weights=topk_weights_initial,
            #     gating_output=masked_logits,
            #     num_experts=num_total_experts,
            #     threshold_percentile=threshold_percentile,
            #     min_experts=min_experts,  # or set based on config
            #     topk=num_experts_per_token,
            # )
            
            # For now, just use all experts
            num_experts_to_keep = num_total_experts
            # Metrics logging disabled since optimize_expert_selection is not available
            # importance_mask_percentage = (importance_mask_sum / topk_ids_initial.shape[0]) * 100
            # metric_store.put_metric(TaskType.CONF_MIN, min)
            # metric_store.put_metric(TaskType.CONF_MAX, max)
            # metric_store.put_metric(TaskType.CONF_MEAN, mean)
            # metric_store.put_metric(TaskType.CONF_VAR, var)
            # metric_store.put_metric(TaskType.PERCENTAGE_CONFIDENCE_TOKENS, importance_mask_percentage)
            # metric_store.put_metric(TaskType.EXPERTS_TO_KEEP, num_experts_to_keep)
            # metric_store.put_metric(TaskType.UNIQUE_EXPERTS, num_unique_experts)
            # metric_store.put_metric(TaskType.TOP_0_1, top_0_1)
            # metric_store.put_metric(TaskType.TOP_1_2, top_1_2)
            # metric_store.put_metric(TaskType.TOP_2_3, top_2_3)
            # metric_store.put_metric(TaskType.TOP_3_4, top_3_4)
            # metric_store.put_metric(TaskType.TOP_4_5, top_4_5)
            # metric_store.put_metric(TaskType.TOP_5_6, top_5_6)


            # num_experts_to_keep = 3
            
            topk_weights, topk_ids, _ = fused_topk(
                hidden_states, masked_logits, topk, renormalize=renormalize
                    )
            # changed_mask = topk_ids_initial != topk_ids
            # num_changed = changed_mask.sum().item()
            # assert num_changed == 0, "The number of changed topk id entries should be zero."
            # # change in top k weights

            # diff = torch.abs(topk_weights_initial - topk_weights)
            # assert torch.allclose(topk_weights_initial, topk_weights, atol=1e-5), "The topk weights should be close."
                # Optional logging
                # if not logit_store.disable:
                #     layer_idx = logit_store.get_layer_idx()
                #     logit_store.log_router_logits(router_logits, masked_logits, layer_idx)

        elif policy == "advanced_parametrized":
            beta = logit_store.beta
            alpha = logit_store.alpha
            
            # Check masked_logits before optimization
            
            num_experts_to_keep = optimize_expert_selection_parameterized(
                topk_ids=topk_ids_initial,
                topk_weights=topk_weights_initial,
                gating_output=masked_logits,
                num_experts=num_total_experts,
                min_experts=min_experts,  # or set based on config
                topk=num_experts_per_token,
                beta=beta,
                alpha=alpha,
            )
            
            # Debug after optimization
        
            # num_experts_to_keep, num_unique_experts = optimize_expert_selection_fast(
            #     topk_ids=topk_ids_initial,
            #     topk_weights=topk_weights_initial,
            #     gating_output=masked_logits,
            #     num_experts=num_total_experts,
            #     min_experts=min_experts,  # or set based on config
            #     beta=0,
            #     alpha_val=2,
            # )

            # metric_store.put_metric(TaskType.EXPERTS_TO_KEEP, num_experts_to_keep)
            # metric_store.put_metric(TaskType.UNIQUE_EXPERTS, num_unique_experts)
            num_experts_to_keep = 3
            topk_weights, topk_ids, _ = fused_topk(
                hidden_states, masked_logits, topk, renormalize=renormalize
                    )

        elif policy == "quant":
            orig_topk_weights_sum = topk_weights_initial.sum(dim=1, keepdim=True)

            alpha = logit_store.alpha
            B, E = router_logits.size()

            # torch.set_printoptions(profile="full")
            #     if torch.cuda.current_device() == 0:
            # if B <= 16:
            #     torch.set_printoptions(profile="full")
            
            #     n_unique = torch.unique(topk_ids_initial).numel()



            dtype = router_logits.dtype
            device = router_logits.device

            base_logits = router_logits
            
            max_logits, _ = torch.max(base_logits, dim=1, keepdim=True)
            quantized_logits = torch.ceil(alpha * (base_logits - max_logits))



            # if B <= 16:



            h = torch.exp(quantized_logits.to(torch.float32) * math.log(B)).to(dtype)
            common_expert_weight = (torch.sum(h, dim=0, keepdim=True) - h) / B
            modified_router_logits = (quantized_logits + common_expert_weight).contiguous()
            _, topk_ids_chosen, _ = fused_topk(
                hidden_states, modified_router_logits, topk, renormalize
            )



            # if B <= 16:



            flat_ids = topk_ids_chosen.reshape(-1)
            counts = torch.zeros(E, dtype=torch.int32, device=device)
            ones = torch.ones_like(flat_ids, dtype=torch.int32)
            counts.scatter_add_(0, flat_ids.to(torch.int64), ones)
            experts_to_keep_mask = counts > 0
            ninf = torch.finfo(dtype).min
            base_logits.masked_fill_((~experts_to_keep_mask).unsqueeze(0), ninf)


            # if B <= 16:



            topk_weights, topk_ids, _ = fused_topk(
                hidden_states, base_logits, topk, renormalize=renormalize
            )

            final_topk_weights_sum = topk_weights.sum(dim=1, keepdim=True)
            if not renormalize:
                topk_weights = topk_weights * (orig_topk_weights_sum / final_topk_weights_sum)


            # if B <= 16:

            #     n_unique_final = torch.unique(topk_ids).numel()

            
            #     add_num_selected_experts_policy(n_unique_final)
            #     add_num_selected_experts_base(n_unique)

            #     if n_unique < n_unique_final:

            # torch.set_printoptions(profile="default")

        elif policy == "quant_flatten":
            orig_topk_weights_sum = topk_weights_initial.sum(dim=1, keepdim=True)

            alpha = logit_store.alpha
            beta = logit_store.beta
            B, E = router_logits.size()

            dtype = router_logits.dtype
            device = router_logits.device

            base_logits = router_logits
            
            max_logits, _ = torch.max(base_logits, dim=1, keepdim=True)
            quantized_logits = torch.ceil(alpha * (base_logits - max_logits))


            h = torch.exp(quantized_logits.to(torch.float32) * math.log(B)).to(dtype)
            common_expert_weight = (torch.sum(h, dim=0, keepdim=True) - h) / B
            modified_router_logits = (quantized_logits + common_expert_weight).contiguous()
            _, topk_ids_chosen, _ = fused_topk(
                hidden_states, modified_router_logits, topk, renormalize
            )

            flat_ids = topk_ids_chosen.reshape(-1)
            counts = torch.zeros(E, dtype=torch.int32, device=device)
            ones = torch.ones_like(flat_ids, dtype=torch.int32)
            counts.scatter_add_(0, flat_ids.to(torch.int64), ones)
            experts_to_keep_mask = counts > 0
            ninf = torch.finfo(dtype).min
            base_logits.masked_fill_((~experts_to_keep_mask).unsqueeze(0), ninf)

            topk_weights, topk_ids, _ = fused_topk(
                hidden_states, base_logits, topk, renormalize=renormalize
            )

            final_topk_weights_sum = topk_weights.sum(dim=1, keepdim=True)
            if not renormalize:
                topk_weights = topk_weights * (orig_topk_weights_sum / final_topk_weights_sum)

        elif policy == "quant_flatten_remove":
            orig_topk_weights_sum = topk_weights_initial.sum(dim=1, keepdim=True)

            alpha = logit_store.alpha
            beta = logit_store.beta
            B, E = router_logits.size()

            dtype = router_logits.dtype
            device = router_logits.device

            base_logits = router_logits

            max_logits, _ = torch.max(base_logits, dim=1, keepdim=True)
            quantized_logits = torch.ceil(alpha * (base_logits - max_logits))

            # Ensure the minimum value is exactly -beta
            beta_t = torch.as_tensor(beta, dtype=dtype, device=device)
            quantized_logits = torch.clamp(quantized_logits, min=-beta_t)

            h = torch.exp(quantized_logits.to(torch.float32) * math.log(B)).to(dtype)
            common_expert_weight = (torch.sum(h, dim=0, keepdim=True) - h) / B
            modified_router_logits = (quantized_logits + common_expert_weight).contiguous()
            _, topk_ids_chosen, _ = fused_topk(
                hidden_states, modified_router_logits, topk, renormalize
            )

            flat_ids = topk_ids_chosen.reshape(-1)
            counts = torch.zeros(E, dtype=torch.int32, device=device)
            ones = torch.ones_like(flat_ids, dtype=torch.int32)
            counts.scatter_add_(0, flat_ids.to(torch.int64), ones)
            experts_to_keep_mask = counts > 0
            ninf = torch.finfo(dtype).min
            base_logits.masked_fill_((~experts_to_keep_mask).unsqueeze(0), ninf)

            topk_weights, topk_ids, _ = fused_topk(
                hidden_states, base_logits, topk, renormalize=renormalize
            )

            # ---- NEW: record per-row columns where quantized_logits == -beta,
            # and if those column indices appear in topk_ids at the same row,
            # replace them with the 0th value of that row in topk_ids. ----
            neg_beta_mask = quantized_logits.eq(-beta_t)                 # [B, E] boolean mask
            neg_beta_indices = torch.nonzero(neg_beta_mask, as_tuple=False)  # [N, 2] (row, col) pairs if you need them

            # Membership test of each topk id against the -beta columns, per row
            banned_in_topk = neg_beta_mask.gather(1, topk_ids.long())    # [B, topk] boolean

            # Replace banned ids with the first id in each row
            first_id_per_row = topk_ids[:, :1]                           # [B, 1]
            topk_ids = torch.where(banned_in_topk, first_id_per_row.expand_as(topk_ids), topk_ids)
            # ------------------------------------------------------------------

            final_topk_weights_sum = topk_weights.sum(dim=1, keepdim=True)
            if not renormalize:
                topk_weights = topk_weights * (orig_topk_weights_sum / final_topk_weights_sum)

        elif policy == "rotate":
            # rank1 = logit_store.rank1
            # rank2 = logit_store.rank2
            # rot_pref(masked_logits, rank1, rank2)  # Function not available
            topk_weights, topk_ids, _ = fused_topk(
            hidden_states, router_logits, topk, renormalize
        )
            num_experts_to_keep = logit_store.num_total_experts

        elif policy == "rotate_based_on_confidence":
            # confidence_policy = logit_store.confidence_policy
            # quartile = logit_store.quartile
            # rotate_based_on_confidence(masked_logits, topk_weights_initial ,threshold_percentile, confidence_policy, quartile)  # Function not available
            topk_weights, topk_ids, _ = fused_topk(
            hidden_states, router_logits, topk, renormalize
        )
            num_experts_to_keep = logit_store.num_total_experts
            
    else:
        # task_name = "PREFILL_GATING_COMPUTATION"  
        # with TaskLoggingContextManagerGPU(task_type=task_name):
        # Prefill case — no expert dropping
        topk_weights, topk_ids, _ = fused_topk(
            hidden_states, router_logits, topk, renormalize
        )
        num_experts_to_keep = logit_store.num_total_experts
    # metric_store.put_metric(TaskType.EXPERTS_TO_KEEP, num_experts_to_keep)
    
    
    return topk_weights, topk_ids
