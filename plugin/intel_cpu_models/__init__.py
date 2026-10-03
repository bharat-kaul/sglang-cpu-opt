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

logger = logging.getLogger(__name__)
logger.info(
    "intel_cpu_models plugin loaded — CPU-optimized model overrides active "
    "(SGLANG_EXTERNAL_MODEL_PACKAGE)."
)

# Install CPU arg-resolution guards at package-import time — BEFORE sglang's
# resolve_once() runs the DSA-family device probes (GLM-5.3 Flash routes there).
try:
    from intel_cpu_models._glm5_cpu_infra import install_cpu_resolution_guards

    install_cpu_resolution_guards()
except Exception as _e:  # never block loading other models on this guard
    logger.warning("intel_cpu_models: CPU resolution guard not installed: %s", _e)

# Deterministic dummy weights (gated DETERMINISTIC_DUMMY=1) so a CPU dummy run is
# bit-identical to the GPU dummy reference -> the per-layer parity diff is valid.
try:
    from _dummy_determinism import install_deterministic_dummy

    install_deterministic_dummy()
except Exception as _e:  # noqa: BLE001
    logger.warning("intel_cpu_models: deterministic dummy not installed: %s", _e)
