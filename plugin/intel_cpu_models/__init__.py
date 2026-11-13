"""Intel CPU model-enablement plugin (throughput-thesis leg).

External model package for SGLang. Point SGLang at it with:

    export SGLANG_EXTERNAL_MODEL_PACKAGE=intel_cpu_models

SGLang's ModelRegistry imports every module here that defines ``EntryClass`` and,
because the external package is registered with ``overwrite=True``, a class whose
``__name__`` matches a built-in architecture (e.g. ``Olmo2ForCausalLM``) REPLACES
the built-in with this CPU-optimized version. No fork of ``sglang/`` is required;
a human reviews the diff in this package and upstreams it as a PR when ready.

Each model module carries a ``CPU_ENABLEMENT`` manifest (op -> existing kernel ->
donor model) that the enablement-certificate tool records as provenance.
"""

import logging
import os

logger = logging.getLogger(__name__)
logger.info(
    "intel_cpu_models plugin loaded — CPU-optimized model overrides active "
    "(SGLANG_EXTERNAL_MODEL_PACKAGE)."
)

# Neutralize pin_memory on CPU FIRST — newer sglang allocates pin_memory=True tensors
# (DSA backend / KV pool / scheduler) that crash on a CPU-only host.
try:
    from intel_cpu_models._cpu_pin_memory import install_cpu_pin_memory_shim

    install_cpu_pin_memory_shim()
except Exception as _e:  # never block loading on this shim
    logger.warning("intel_cpu_models: pin_memory shim not installed: %s", _e)

# Install CPU arg-resolution guards at package-import time — BEFORE sglang's
# resolve_once() runs the DSA-family device probes (GLM-5.3 Flash routes there).
try:
    from intel_cpu_models._glm5_cpu_infra import install_cpu_resolution_guards

    install_cpu_resolution_guards()
except Exception as _e:  # never block loading other models on this guard
    logger.warning("intel_cpu_models: CPU resolution guard not installed: %s", _e)

# CONTAINMENT (reviewer-recommended): the CPU DeepseekV4 DSA/MLA compressed-state path does not
# support prefix-cache HITS (compressor/overlap/indexer state is replaced, not restored, on resume),
# which corrupts a reused prefix and flips confident tokens (net gsm8k 0.9329 radix-on -> 0.9397
# radix-off). Force disable_radix_cache for DeepseekV4 on CPU, BEFORE arg resolution builds the radix
# cache. Override (unsafe) with INTEL_CPU_DSV4_ALLOW_PREFIX_CACHE=1.
try:
    from intel_cpu_models._dsv4_cpu_infra import (
        install_dsv4_prefix_cache_guard,
        register_dsv4_cpu_radix_override,
    )

    register_dsv4_cpu_radix_override()  # layer 1: config-consistent disable at resolution
    install_dsv4_prefix_cache_guard()  # layer 2: fail-closed backstop at build time
except Exception as _e:  # never block loading other models on this guard
    logger.warning("intel_cpu_models: DSV4 prefix-cache guard not installed: %s", _e)

# Deterministic dummy weights (gated DETERMINISTIC_DUMMY=1) so a CPU dummy run is
# bit-identical to the GPU dummy reference -> the per-layer parity diff is valid.
try:
    from _dummy_determinism import install_deterministic_dummy

    install_deterministic_dummy()
except Exception as _e:  # noqa: BLE001
    if os.environ.get("DETERMINISTIC_DUMMY") == "1":
        raise
    logger.warning("intel_cpu_models: deterministic dummy not installed: %s", _e)

# Per-layer hidden-state capture (gated INTEL_CPU_DSV4_LAYER_CAPTURE=<file>): the DEBUGGER that
# localizes a CPU/GPU (or CPU-C/CPU-eager) divergence to the first divergent LAYER. Hardware-agnostic
# (same package the GPU run loads via SGLANG_EXTERNAL_MODEL_PACKAGE); inert unless the env is set.
try:
    import dsv4_layer_capture  # noqa: F401  (import side-effect installs the gated patch)
except Exception as _e:  # noqa: BLE001
    logger.warning("intel_cpu_models: layer-capture not installed: %s", _e)
