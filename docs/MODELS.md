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

Lynx is built on **affinity binning**: a routing technique that groups experts into bins based on each token's routing affinity, then skips bins below a threshold. The binning configuration depends only on the model's architecture (the number of experts and routing top-k) — it is not workload-dependent and does not need to be retuned per request distribution.

A policy file describes one such configuration:

```json
{
    "policy": "quant_alpha3_beta4_optimized",
    "num_experts_per_tok": 8,
    "num_local_experts": 64
}
```

The two architectural fields (`num_experts_per_tok`, `num_local_experts`) must match your model. The `policy` field selects a pre-validated binning tuned for that architecture. Bundled policy examples for every supported model family live under [`src/lynx/configs/`](../src/lynx/configs/); copy the closest one, update the architectural fields, and the same policy will apply.
