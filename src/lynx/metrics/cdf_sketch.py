"""DDSketch-backed quantile distribution helper used by the Lynx
metrics path. Soft-imports ``ddsketch`` (and ``wandb``/``plotly``) so
that the rest of the Lynx feature works without these optional
dependencies installed.
"""

import logging
from typing import TYPE_CHECKING

logger = logging.getLogger(__name__)

try:
    from ddsketch.ddsketch import DDSketch
except ImportError:  # pragma: no cover
    DDSketch = None  # type: ignore[assignment]

try:
    import wandb  # type: ignore
except ImportError:  # pragma: no cover
    wandb = None  # type: ignore[assignment]

if TYPE_CHECKING:
    import pandas as pd


def _require_ddsketch() -> None:
    if DDSketch is None:
        raise ImportError(
            "Lynx metrics require the `ddsketch` package. Install with: "
            "`uv pip install ddsketch`."
        )


class CDFSketch:
    def __init__(
        self,
        metric_name: str,
        save_table_to_wandb: bool = True,
        relative_accuracy: float = 0.001,
        num_quantiles_in_df: int = 101,
    ) -> None:
        _require_ddsketch()
        self._sketch = DDSketch(relative_accuracy=relative_accuracy)
        self._metric_name = metric_name
        self._last_data = 0.0
        self._save_table_to_wandb = save_table_to_wandb
        self._num_quantiles_in_df = num_quantiles_in_df

    @property
    def mean(self) -> float:
        return self._sketch.avg

    @property
    def median(self) -> float:
        return self._sketch.get_quantile_value(0.5)

    @property
    def sum(self) -> float:
        return self._sketch.sum

    def __len__(self) -> int:
        return int(self._sketch.count)

    def merge(self, other: "CDFSketch") -> None:
        assert self._metric_name == other._metric_name
        self._sketch.merge(other._sketch)

    def put(self, data: float) -> None:
        self._last_data = data
        self._sketch.add(data)

    def put_pair(self, data_x: float, data_y: float) -> None:
        self._last_data = data_y
        self._sketch.add(data_y)

    def put_delta(self, delta: float) -> None:
        self.put(self._last_data + delta)

    def print_distribution_stats(self, plot_name: str) -> None:
        if self._sketch._count == 0:
            return
        logger.info(
            "%s: %s stats: min=%s max=%s mean=%s p25=%s p50=%s p75=%s "
            "p95=%s p99=%s p99.9=%s count=%s sum=%s",
            plot_name,
            self._metric_name,
            self._sketch._min,
            self._sketch._max,
            self._sketch.avg,
            self._sketch.get_quantile_value(0.25),
            self._sketch.get_quantile_value(0.5),
            self._sketch.get_quantile_value(0.75),
            self._sketch.get_quantile_value(0.95),
            self._sketch.get_quantile_value(0.99),
            self._sketch.get_quantile_value(0.999),
            self._sketch._count,
            self._sketch.sum,
        )
        if wandb is not None and wandb.run is not None:
            wandb.log({
                f"{plot_name}_min": self._sketch._min,
                f"{plot_name}_max": self._sketch._max,
                f"{plot_name}_mean": self._sketch.avg,
                f"{plot_name}_p25": self._sketch.get_quantile_value(0.25),
                f"{plot_name}_p50": self._sketch.get_quantile_value(0.5),
                f"{plot_name}_p75": self._sketch.get_quantile_value(0.75),
                f"{plot_name}_p95": self._sketch.get_quantile_value(0.95),
                f"{plot_name}_p99": self._sketch.get_quantile_value(0.99),
                f"{plot_name}_count": self._sketch.count,
                f"{plot_name}_sum": self._sketch.sum,
            })

    def _to_df(self) -> "pd.DataFrame":
        import numpy as np
        import pandas as pd

        quantiles = np.linspace(0, 0.99, self._num_quantiles_in_df)
        values = [self._sketch.get_quantile_value(q) for q in quantiles]
        return pd.DataFrame({"cdf": quantiles, self._metric_name: values})

    def plot_cdf(
        self,
        path: str,
        plot_name: str,
        x_axis_label: str | None = None,
    ) -> None:
        if self._sketch._count == 0:
            return
        if x_axis_label is None:
            x_axis_label = self._metric_name
        df = self._to_df()
        self.print_distribution_stats(plot_name)
        if wandb is not None and wandb.run is not None and self._save_table_to_wandb:
            wandb_df = df.rename(columns={self._metric_name: x_axis_label})
            wandb.log({
                f"{plot_name}_cdf": wandb.plot.line(
                    wandb.Table(dataframe=wandb_df),
                    "cdf",
                    x_axis_label,
                    title=plot_name,
                )
            })
