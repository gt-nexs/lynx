# lynx-vllm

**Lynx is a workload-agnostic expert remapping technique that improves throughput by up to 2.0× — across reasoning and multi-modal MoE models — while maintaining accuracy.**

This package ships Lynx as a drop-in plugin for [vLLM](https://github.com/vllm-project/vllm). Activation requires a single environment variable; the plugin requires no source modifications, kernel rebuilds, or application-level changes.

## Install

```bash
pip install vllm==0.20.1
pip install git+https://github.com/VimaGupta345/lynx.git@main
```

The package distributes as a pure-Python wheel and does not require a CUDA toolkit, host compiler, or CMake build environment at install time. Triton kernels are JIT-compiled on first invocation.

## Quickstart

```bash
VLLM_LYNX_ENABLED=1 vllm serve Qwen/Qwen2-57B-A14B-Instruct --tensor-parallel-size 2
```

Send requests to `http://localhost:8000/v1/completions` as you would with any vLLM server. With `VLLM_LYNX_ENABLED` unset, the plugin is inert and the server's behaviour is identical to vanilla vLLM.

Lynx supports any MoE architecture built on vLLM's `FusedMoE` layer. Bundled policies cover all popular model families, including Qwen, Mixtral, DeepSeek, GPT-OSS, and Llama 4. See [`docs/MODELS.md`](docs/MODELS.md) for the complete supported-model list and instructions for registering custom models.

## Configuration

| Variable | Description |
|---|---|
| `VLLM_LYNX_ENABLED=1` | Activates the plugin. |
| `VLLM_LYNX_CONFIG_FILE=<path>` | Overrides the bundled policy for the served model. |

## License

Apache 2.0.
