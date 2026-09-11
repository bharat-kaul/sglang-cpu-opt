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
