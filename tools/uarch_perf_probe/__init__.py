"""uArch Performance Probe (uPP).

A standalone, workload-agnostic microbenchmark suite that characterizes a CPU
microarchitecture and emits a `machine_constants.json` consumed by kernel-authoring
and roofline workflows. The durable asset is the *harness that regenerates the
constants on real silicon*, not the constants themselves — so it ports to an unseen
uarch (e.g. a GNR follow-on) with no priors, and it also refreshes/verifies priors
on known parts (GNR) to catch drift.

No dependency on any serving stack or model — any agentic workflow can call it.
"""

__version__ = "0.1"
