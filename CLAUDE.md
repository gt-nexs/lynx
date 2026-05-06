# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## What this repo is

`lynx-vllm` is a standalone PyPI package (currently distributed via this
GitHub URL) that activates Lynx — a workload-agnostic MoE expert
remapping technique — inside an unmodified vLLM install. The user gets
the speedup by adding two pip installs and setting one env var
(`VLLM_LYNX_ENABLED=1`); no vLLM fork is required.

The original in-tree port (a vLLM v0.20.1 fork with Lynx integrated
directly into the model and worker files) is preserved on the
`archive/vllm-fork-v0.20.1` branch of this repo for regression
reference. **Do not develop on that branch.** All forward work happens
on `main` against the standalone package.

## Architecture (the part that requires reading several files)

The package hooks into vLLM via the `vllm.general_plugins` entry point
declared in `pyproject.toml`. vLLM calls our `install()` function once
per process — API server, EngineCore, and every TP worker. From there:

1. **`src/lynx/_plugin.py:install()`** is the plugin entry point. It
   gates on `VLLM_LYNX_ENABLED` (via `_env.is_enabled()`) and, when on,
   delegates to `_patches.install_all()`. When off, it returns
   immediately and the rest of vLLM is untouched (verified by
   `tests/test_no_op_when_off.py`).
2. **`src/lynx/_patches.py`** applies four idempotent monkey-patches.
   Read this file first — every behavioural decision lives here:
   - `FusedMoE.__init__`: injects the right Lynx routing variant
     (softmax / grouped / sigmoid) into vLLM's existing
     `custom_routing_function` parameter.
   - `Worker.__init__`: resolves the policy file (registry lookup or
     `VLLM_LYNX_CONFIG_FILE`), `setattr`s its keys onto
     `model_config.hf_config`, and creates the per-worker `LynxState`
     singleton. *Policy injection lives here, not in
     `ModelConfig.__post_init__`, because Pydantic re-binds
     `__post_init__` at class creation and a patch on it is silently
     ineffective.*
   - `Worker.initialize_from_config`: flips
     `LynxState.profile_complete = True` after `kernel_warmup`. This
     is what turns the routing kernel on.
   - `GPUModelRunner.execute_model` + `_dummy_run`: per-batch prefill
     detection and a forced `is_prefill=False` during cudagraph
     capture. **The `_dummy_run` flip is mandatory.** Without it,
     captured graphs trace through the vanilla `fused_topk` fallback
     and runtime cudagraph replays silently bypass Lynx entirely.
3. **`src/lynx/state.py:LynxState`** is the per-worker singleton with
   the active policy + `profile_complete` / `is_prefill` lifecycle
   flags. The routing kernels in `routing.py` consult it on the hot
   path.
4. **`src/lynx/routing.py`** is ~3000 LOC of Triton kernels lifted
   verbatim from the in-tree fork. Keep it that way unless you have a
   specific reason to touch a kernel — the JIT-compile cost is paid
   once per (kernel, GPU) and cached by triton.
5. **`src/lynx/registry.py:LYNX_MODEL_REGISTRY`** is the
   model-substring → bundled-policy table. Order matters: more
   specific keys must be listed first because `lookup()` returns on
   the first substring match. The `qwen3-235b-a22b-thinking` entry
   must precede `qwen3-235b-a22b` for that reason.

The metric-logging path (`src/lynx/metrics/`) is opt-in via
`VLLM_LYNX_METRICS=1` and not on by default; it lazily imports
`ddsketch`. Off-by-default no-op is enforced and tested.

## Commands

All commands assume you are in this repo's root. The test venv I use
locally is `/data/vgupta345/prowl_related_data/prowl-open-source/vanilla-v0.20.1/.venv`
(it has stock vllm 0.20.1 + an editable install of this package).

```bash
# Install lynx-vllm editable into a venv that already has vllm 0.20.1:
VIRTUAL_ENV=/path/to/venv uv pip install -e ".[test]"

# Run the unit tests (fast, no GPU required):
VANILLA_VENV=/data/vgupta345/prowl_related_data/prowl-open-source/vanilla-v0.20.1/.venv
$VANILLA_VENV/bin/pytest tests/ -v -k "not integration"

# Run a single test:
$VANILLA_VENV/bin/pytest tests/test_registry.py::test_lookup_qwen3_235b_thinking_vs_instruct -v

# Run the integration test (boots a real vllm serve, requires 2x H100, ~3 min):
LYNX_RUN_INTEGRATION=1 \
CUDA_VISIBLE_DEVICES=0,1 \
$VANILLA_VENV/bin/pytest tests/test_integration_qwen2.py -v -s

# Cut stale workers between runs (per memory rule):
pkill -9 -u vgupta345 -f "vllm|VLLM|EngineCore|Worker_TP" 2>/dev/null
```

A `.venv-test` is also present in the repo from earlier experiments;
prefer the vanilla v0.20.1 venv since it has clean stock vllm.

## Conventions

- **Adding a new model family**: copy the closest bundled JSON in
  `src/lynx/configs/`, drop it into a new subdir, add an entry to
  `LYNX_MODEL_REGISTRY` in `registry.py` (more-specific substring keys
  first), and add a test in `tests/test_registry.py`. The
  `test_all_registry_paths_resolve_to_real_files` coverage test will
  catch entries that point at non-existent files at the wheel level.
- **Lynx state markers**: `Worker.__init__` and
  `Worker.initialize_from_config` patches both emit `print(...)` to
  stderr (in addition to `logger.info`) because vLLM's logging config
  silences third-party loggers. Keep these prints; they're how the
  integration test asserts the patches fired.
- **Idempotency**: every patch checks for a `_lynx_patched` sentinel
  attribute and returns early if already applied. `install_all()` may
  be called multiple times safely.
- **No upper-bound vLLM pin**: `pyproject.toml` requires
  `vllm>=0.20.1` only. Patches are validated against 0.20.1 and may
  break on later vLLM majors; bump the pin (and cut a Lynx release)
  when validating against a new vLLM line.
- **Public API surface**: only `lynx.LynxState`,
  `lynx.LYNX_MODEL_REGISTRY`, `lynx.lookup`, `lynx.register_model`.
  Everything in `_plugin`, `_patches`, `_env`, `_expert_count` is
  internal and may change without notice.

## What's left for v0.2

- Bump `version` in `pyproject.toml` and tag releases (`v0.1.1`,
  `v0.2.0`, …) so users can pin against tags instead of `@main`.
- Publish to PyPI (`twine upload`). README still says "until we publish
  to PyPI"; remove that line once it's live.
- CI: a GitHub Actions workflow that runs the unit-test suite on every
  push (no GPU needed; integration test stays gated).
- Re-validate the four monkey-patches against the next vLLM minor when
  it ships, then decide whether to widen the pin or cut a v0.2 against
  the new vLLM line.
