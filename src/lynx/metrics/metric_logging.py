"""Lynx ablation telemetry — opt-in via ``--lynx-metrics``.

Captures per-task latencies (CUDA event-timed and CPU-timed) and dumps
distribution stats to ``$VLLM_LYNX_PROFILE_DIR`` (default
``./profiling_output/``). All public APIs no-op when
``MetricStore.disable`` is True. The module imports are kept light;
``ddsketch`` and ``wandb`` are only required when actually emitting
metrics.
"""

from __future__ import annotations

import logging
import os
import time
from datetime import datetime
from enum import Enum, auto
from typing import Any

import torch

from lynx.metrics.cdf_sketch import CDFSketch

logger = logging.getLogger(__name__)


def _profile_dir_root() -> str:
    return os.environ.get("VLLM_LYNX_PROFILE_DIR", "./profiling_output")


class TaskType(Enum):
    TOKEN_REROUTING = auto()
    SCHEDULE_ITERATION = auto()
    WORKER_ITERATION = auto()
    E2E_REQUEST = auto()
    TOTAL_ATTN_COMPUTATION = auto()
    LAYERNORM_AFTER_ATTN = auto()
    TOTAL_EXPERT_COMPUTATION = auto()
    TOTAL_MLP_COMPUTATION = auto()
    PREFILL_EXPERT_COMPUTATION = auto()
    DECODE_EXPERT_COMPUTATION = auto()
    DECODE_GATING_COMPUTATION_WITHOUT_POLICY = auto()
    DECODE_GATING_COMPUTATION_WITH_POLICY = auto()
    DECODE_GATING_COMPUTATION_WITH_SIMPLE_POLICY = auto()
    DECODE_GATING_COMPUTATION_WITH_ADVANCED_POLICY = auto()
    FUSED_MOE_KERNEL_CALL = auto()
    EXPERT_COMPUTATION = auto()
    EXTRACT_VOTING = auto()
    FIND_DROP_INDICES = auto()
    APPLY_MASK = auto()
    EXTRACT_VOTING_AND_MASK = auto()
    FUSED_TOPK = auto()
    NOOP = auto()
    BATCH_SIZE = auto()
    PREFILL_BATCH_SIZE = auto()
    DECODE_BATCH_SIZE = auto()
    EXPERTS_DROPPED = auto()
    EXPERTS_TO_KEEP = auto()
    PERCENTAGE_CONFIDENCE_TOKENS = auto()
    CONF_MIN = auto()
    CONF_MAX = auto()
    CONF_MEAN = auto()
    CONF_VAR = auto()
    TOP_0_1 = auto()
    TOP_1_2 = auto()
    TOP_2_3 = auto()
    TOP_3_4 = auto()
    TOP_4_5 = auto()
    TOP_5_6 = auto()
    UNIQUE_EXPERTS = auto()


def _if_enabled(func):
    def wrapper(self, *args, **kwargs):
        if not self.disable and self.profile_complete:
            return func(self, *args, **kwargs)
        return None
    return wrapper


class TaskLoggingContextManagerGPU:
    def __init__(self, task_type: TaskType) -> None:
        self.task_type = task_type
        self.start_event = torch.cuda.Event(enable_timing=True)
        self.end_event = torch.cuda.Event(enable_timing=True)

    def __enter__(self) -> "TaskLoggingContextManagerGPU":
        self.start_event.record()
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        self.end_event.record()
        store = MetricStore.get_instance()
        if store is not None:
            store.add_metric_gpu(self.task_type, self.start_event, self.end_event)


class TaskLoggingContextManagerCPU:
    def __init__(self, task_type: TaskType) -> None:
        self.task_type = task_type
        self.start_time: float | None = None
        self.end_time: float | None = None

    def __enter__(self) -> "TaskLoggingContextManagerCPU":
        self.start_time = time.perf_counter()
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        self.end_time = time.perf_counter()
        elapsed = (self.end_time or 0) - (self.start_time or 0)
        store = MetricStore.get_instance()
        if store is not None:
            store.add_metric_cpu(self.task_type, elapsed)


class MetricStore:
    _instance: "MetricStore | None" = None

    def __init__(self, hf_config: Any) -> None:
        self.profile_complete = False
        self.batch_idx = 0
        self.metrics: dict[TaskType, CDFSketch] = {}
        self.raw_metrics: dict[TaskType, list[tuple]] = {}
        self.num_experts = -1
        self.top_k = -1
        self.hidden_size = -1
        self.num_prefill_batches = 0
        self.num_decode_batches = 0
        self.disable = False

        ts = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
        self.full_path = os.path.join(_profile_dir_root(), f"plots_{ts}")
        os.makedirs(self.full_path, exist_ok=True)

        self.setting_params(hf_config)

    @classmethod
    def get_instance(cls) -> "MetricStore | None":
        return cls._instance

    @classmethod
    def get_or_create_instance(cls, hf_config: Any) -> "MetricStore":
        if cls._instance is None:
            cls._instance = cls(hf_config)
        return cls._instance

    @classmethod
    def reset(cls) -> None:
        cls._instance = None

    def setting_params(self, config: Any) -> None:
        """Read MoE topology from a HuggingFace config (or a vLLM
        ModelConfig wrapping one)."""
        hf_config = getattr(config, "hf_text_config", config)
        for name in (
            "num_local_experts",
            "num_experts",
            "moe_num_experts",
            "n_routed_experts",
        ):
            if hasattr(hf_config, name):
                self.num_experts = int(getattr(hf_config, name))
                break

    def mark_profiling_done(self) -> None:
        self.profile_complete = True
        logger.info("Lynx metrics: profiling complete")

    def on_batch_end_worker(self) -> None:
        self.batch_idx += 1
        self.calculate_elapsed_times()

    def on_batch_start_worker(self, scheduler_output: Any = None) -> None:
        return

    @_if_enabled
    def add_metric_gpu(
        self,
        task_type: TaskType,
        start_event: torch.cuda.Event,
        end_event: torch.cuda.Event,
    ) -> None:
        self.raw_metrics.setdefault(task_type, []).append(
            (start_event, end_event)
        )

    @_if_enabled
    def put_metric(self, task_type: TaskType, metric_value: float) -> None:
        if task_type not in self.metrics:
            self.metrics[task_type] = CDFSketch(
                metric_name=task_type.name, save_table_to_wandb=False
            )
        self.metrics[task_type].put(metric_value)

    @_if_enabled
    def add_metric_cpu(self, task_type: TaskType, elapsed_time: float) -> None:
        self.put_metric(task_type, elapsed_time)

    def get_metrics(self) -> dict[TaskType, CDFSketch]:
        return self.metrics

    @_if_enabled
    def calculate_elapsed_times(self) -> None:
        torch.cuda.synchronize()
        for task_type, measurements in list(self.raw_metrics.items()):
            if task_type == TaskType.SCHEDULE_ITERATION:
                continue
            for start_event, end_event in measurements:
                try:
                    elapsed = start_event.elapsed_time(end_event)
                    self.add_metric_cpu(task_type, elapsed)
                except (AttributeError, RuntimeError) as e:
                    logger.warning(
                        "Lynx metrics: skipping %s measurement: %s",
                        task_type, e,
                    )
            self.raw_metrics[task_type] = []

    def set_disable(self) -> None:
        self.disable = True

    def plot_metrics(self) -> None:
        if self.disable or not self.profile_complete:
            return
        with open(os.path.join(self.full_path, "metrics.txt"), "w") as f:
            for task_type, measurements in self.metrics.items():
                f.write(f"{task_type.name}: count={len(measurements)} "
                        f"mean={measurements.mean:.6f} "
                        f"median={measurements.median:.6f}\n")
        for task_type, measurements in self.metrics.items():
            label_map = {
                TaskType.BATCH_SIZE: "size",
                TaskType.PERCENTAGE_CONFIDENCE_TOKENS: "mask size",
                TaskType.EXPERTS_TO_KEEP: "num experts kept",
                TaskType.UNIQUE_EXPERTS: "num experts selected originally",
            }
            label = label_map.get(task_type, "time")
            try:
                measurements.plot_cdf(
                    self.full_path,
                    f"{task_type.name}_execution_time"
                    if task_type not in label_map else task_type.name,
                    label,
                )
            except Exception as e:  # pragma: no cover
                logger.warning(
                    "Lynx metrics: plot_cdf failed for %s: %s",
                    task_type, e,
                )
