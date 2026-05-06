# lynx-vllm

Opt-in MoE expert-skipping for [vLLM](https://github.com/vllm-project/vllm).

On Qwen2-57B-A14B-Instruct (TP=2 + cudagraph) using the bundled
`quant_alpha3_beta4_optimized` policy, lynx delivers:

- **−37.8% median TPOT** (15.5 ms → 9.6 ms)
- **+41.1% output throughput** (786 → 1110 tok/s)
- GSM8K accuracy within 1σ of vanilla (greedy, 100-sample, 5-shot)

When the `VLLM_LYNX_ENABLED` environment variable is unset, the plugin
is a no-op — vllm's behaviour is byte-for-byte unchanged. So
`pip install lynx-vllm` is safe to add to any vLLM environment.

## Install

```bash
# 1. Install vLLM (any v0.20.x; we test against 0.20.1).
pip install vllm==0.20.1

# 2. Install lynx-vllm. (Until we publish to PyPI, install from GitHub.)
pip install git+https://github.com/VimaGupta345/lynx.git@main

# Optional: ablation telemetry CSVs (pulls in ddsketch).
pip install 'git+https://github.com/VimaGupta345/lynx.git@main#egg=lynx-vllm[metrics]'
```

`lynx-vllm` ships a pure-Python wheel — no CUDA toolkit, `nvcc`, or
`cmake` is needed. The Triton kernels in `lynx.routing` are JIT-compiled
on first launch and cached in `~/.triton/cache/`.

### vLLM install troubleshooting

If `pip install vllm==0.20.1` pulls a wheel that doesn't match your CUDA
version (you'll see `ImportError: libcudart.so.13: cannot open shared
object file` or similar at runtime), build vLLM from source against your
local torch:

```bash
git clone --depth 1 --branch v0.20.1 https://github.com/vllm-project/vllm.git
cd vllm
VLLM_USE_PRECOMPILED=1 pip install -e . --torch-backend=auto
```

`VLLM_USE_PRECOMPILED=1` tells vLLM to download the prebuilt CUDA wheel
that matches your installed torch's CUDA version, instead of always
pulling the latest. (See vLLM's docs for the full list of install
options.)

## Usage

```bash
# Auto-resolve a bundled config from the model name:
VLLM_LYNX_ENABLED=1 vllm serve Qwen/Qwen2-57B-A14B-Instruct \
    --tensor-parallel-size 2

# Override with an explicit policy file:
VLLM_LYNX_ENABLED=1 \
VLLM_LYNX_CONFIG_FILE=/path/to/your/policy.json \
    vllm serve <model>

# Enable ablation telemetry (CSV dumps to $VLLM_LYNX_PROFILE_DIR):
VLLM_LYNX_ENABLED=1 VLLM_LYNX_METRICS=1 \
    vllm serve <model>
```

`vllm serve` works exactly as it normally does. The only addition is the
`VLLM_LYNX_*` environment variables.

## Adding a new model

The bundled registry maps model-name substrings to bundled policy files.
To add support for a model that isn't covered out-of-the-box, register a
config path before launching the server:

```python
import lynx
lynx.register_model("my-org/my-7b-moe", "/etc/lynx/my-policy.json")
```

Or set `VLLM_LYNX_CONFIG_FILE=/path/to/policy.json` to override the
registry for one launch.

A policy file is a small JSON:

```json
{
    "policy": "quant_alpha3_beta4_optimized",
    "num_experts_per_tok": 8,
    "num_local_experts": 64,
    "alpha": 1,
    "beta": 0,
    "min_experts": 0,
    "threshold_percentile": 0,
    "count_of_topk": 0
}
```

`policy` selects the routing kernel (most production policies bake α/β
into their name; the JSON's `alpha`/`beta` fields are used by the
parametric `quant` policy). Lower β/α ratio = more aggressive expert
pruning = larger TPOT speedup at potentially some quality cost.

## Bundled defaults

| Model substring | Default policy |
|---|---|
| `qwen3-235b-a22b` | `qwen3_235b/quant_alpha1_beta2_optimized.json` |
| `qwen3-30b-a3b` | `qwen3_30b/quant_alpha3_beta2_optimized.json` |
| `qwen2-57b-a14b` | `qwen2/quant_alpha3_beta4_optimized.json` |
| `mixtral-8x7b`, `mixtral-8x22b` | `mixtral/quant_alpha0.7_beta1_optimized.json` |

Architectures wired for lynx via the plugin: Mixtral, Qwen2-MoE,
Qwen3-MoE, DeepSeek-V2 (grouped routing), DBRX, OLMoE, Llama4 (sigmoid
routing), GptOss. Any other MoE model can use lynx by passing its model
name to `register_model()`.

## Environment variables

| Var | Default | Notes |
|---|---|---|
| `VLLM_LYNX_ENABLED` | unset | `1`/`true` to activate the plugin. When unset, vllm runs unchanged. |
| `VLLM_LYNX_CONFIG_FILE` | unset | Override the registry lookup with a custom JSON policy. |
| `VLLM_LYNX_METRICS` | unset | `1`/`true` to dump ablation CSVs. Requires `lynx-vllm[metrics]`. |
| `VLLM_LYNX_PROFILE_DIR` | `./profiling_output/` | Output dir for metrics CSVs. |

## How it works

`LynxState` is a per-worker singleton holding the active policy and
lifecycle flags (`profile_complete`, `is_prefill`). The plugin
monkey-patches five vllm hooks:

1. `FusedMoE.__init__` — injects the right Lynx routing variant
   (softmax / grouped / sigmoid) based on the model's routing style.
2. `ModelConfig.__post_init__` — loads the JSON policy and writes its
   keys onto `hf_config`.
3. `Worker.__init__` — creates the per-worker `LynxState` singleton.
4. `Worker.initialize_from_config` — flips `profile_complete=True` after
   `kernel_warmup`.
5. `GPUModelRunner.execute_model` + `_dummy_run` — per-batch prefill
   detection and a forced `is_prefill=False` during cudagraph capture
   (essential — without it, captured graphs bypass lynx entirely).

When `VLLM_LYNX_ENABLED` is unset, `install()` returns immediately and
none of these patches are applied.

## Caveats

- Compatible with vllm v0.20.1; later versions may need a matching lynx
  release (the `pyproject.toml` requires `vllm>=0.20.1`, no upper bound,
  but our patches are validated only against 0.20.1).
- TTFT regresses by ~5–15 ms when the plugin is active because prefill
  batches are forced to eager (cudagraph mode `NONE`). Lynx is a decode
  optimization; for very-short generations the net win may be neutral.
- EPLB (Expert Parallelism Load Balancing) is forced off when lynx is
  on; the two have not been validated together.

## License

Apache 2.0 — same as vLLM.

## Reference / archive

The original in-tree port (vllm v0.20.1 fork with lynx integrated) lives
on the [`archive/vllm-fork-v0.20.1` branch](https://github.com/VimaGupta345/lynx/tree/archive/vllm-fork-v0.20.1)
of this repo, for historical and regression reference.
