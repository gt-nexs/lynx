# lynx-vllm

**Lynx is a workload-agnostic expert remapping technique that improves throughput by 2.0× while maintaining accuracy.**

This package ships Lynx as a drop-in plugin for [vLLM](https://github.com/vllm-project/vllm). Install it, set one environment variable, and any supported MoE model serves faster — no fork, no kernel rebuild, no application changes.

## Install

```bash
pip install vllm==0.20.1
pip install git+https://github.com/VimaGupta345/lynx.git@main
```

Pure-Python wheel. No CUDA toolkit, `nvcc`, or `cmake` required.

## Quickstart

```bash
VLLM_LYNX_ENABLED=1 vllm serve Qwen/Qwen2-57B-A14B-Instruct --tensor-parallel-size 2
```

Hit `http://localhost:8000/v1/completions` like any vLLM server. When `VLLM_LYNX_ENABLED` is unset, the plugin is a no-op and vLLM is unchanged.

Lynx works with any MoE model that uses vLLM's `FusedMoE` layer; bundled defaults cover popular Qwen, Mixtral, DeepSeek, and GPT-OSS variants. See [`docs/MODELS.md`](docs/MODELS.md) for the supported-model list and how to register your own.

## Configuration

| Variable | Notes |
|---|---|
| `VLLM_LYNX_ENABLED=1` | Activate the plugin. |
| `VLLM_LYNX_CONFIG_FILE=path.json` | Override the bundled policy for any model. |

## License

Apache 2.0.
