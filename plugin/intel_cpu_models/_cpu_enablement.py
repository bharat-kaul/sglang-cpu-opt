"""Generic CPU-enablement mixin — model-agnostic.

Workflow-GENERATED per-model modules subclass their upstream SGLang model class
together with ``CpuEnablementMixin``. The mixin provides the AMX-readiness probe,
the load-time prepack verification, and the ``CPU_ENABLEMENT`` provenance plumbing
that the enablement-certificate reads. NO per-model logic lives here — this is part
of the reusable framework, not a generated artifact.
"""

from __future__ import annotations

import logging

from sglang.srt.utils.common import (
    cpu_has_amx_support,
    is_cpu,
    use_intel_amx_backend,
)

logger = logging.getLogger(__name__)


class CpuEnablementMixin:
    """Shared CPU/AMX wiring checks for generated model classes."""

    # Overridden by the generated subclass: op -> {kernel, donor}. Provenance only.
    CPU_ENABLEMENT: dict = {}

    def _init_cpu_enablement(self) -> None:
        # cpu_has_amx_support() is the exact gate the unquantized CPU linear method
        # uses to decide whether to AMX-prepack weights (unquant.py).
        self._cpu_amx = is_cpu() and cpu_has_amx_support()
        if is_cpu() and not self._cpu_amx:
            logger.warning(
                "%s on CPU without the Intel AMX backend — kernels fall back to "
                "AVX-512; the peer-relative roofline gate will flag the gap.",
                type(self).__name__,
            )

    def verify_amx_wired(self, sample_linear) -> bool:
        """Confirm a linear landed on the intel_amx path.

        Call AFTER the model loader runs ``process_weights_after_loading`` (which
        sets the per-layer flag) — NOT inside ``load_weights``, where it is still
        unset. A miss when ``_cpu_amx`` is True is a real wiring bug (would show up
        as a peer-relative shortfall).
        """
        wired = use_intel_amx_backend(sample_linear)
        if getattr(self, "_cpu_amx", False) and not wired:
            logger.warning(
                "%s: linears not on the intel_amx path; expect a peer-relative "
                "roofline shortfall. Check --device cpu + AMX caps.",
                type(self).__name__,
            )
        return wired
