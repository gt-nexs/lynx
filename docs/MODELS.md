# Supported models

Lynx works with any MoE model that uses vLLM's `FusedMoE` layer. The package ships pre-tuned policies for the model families below; for any of these, `VLLM_LYNX_ENABLED=1 vllm serve <model>` is zero-config.

| Model |
|---|
| `Qwen/Qwen2-57B-A14B-Instruct` |
| `Qwen/Qwen3-30B-A3B-Instruct-2507` |
| `Qwen/Qwen3-235B-A22B-Instruct-2507` |
| `Qwen/Qwen3-235B-A22B-Thinking-2507` |
| `mistralai/Mixtral-8x7B-Instruct-v0.1` |
| `mistralai/Mixtral-8x22B-Instruct-v0.1` |
| `deepseek-ai/DeepSeek-Coder-V2-Instruct` |
| `openai/gpt-oss-120b` |
| `meta-llama/Llama-4-*` (Scout / Maverick) |

## Adding a new model

Register a config path before launching the server:

```python
import lynx
lynx.register_model("my-org/my-7b-moe", "/path/to/policy.json")
```

Or pass the config directly via env var for one launch:

```bash
VLLM_LYNX_CONFIG_FILE=/path/to/policy.json \
    VLLM_LYNX_ENABLED=1 vllm serve my-org/my-7b-moe
```

## Policy file format

A policy file is a small JSON specifying the routing kernel and the model's expert topology:

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

- `num_experts_per_tok` and `num_local_experts` describe the model architecture.
- `policy` selects the routing kernel; most production policies bake α/β into their name (`quant_alpha3_beta4_optimized` ⇒ α=3, β=4 hardcoded).
- The runtime `alpha`/`beta` fields apply to the parametric `quant` policy.

Examples for each supported family ship under [`src/lynx/configs/`](../src/lynx/configs/) — copy the closest one and edit `num_experts_per_tok` / `num_local_experts` for your model.
