# Assessment of the R8 Roofline Response

Reviewed response `7c71968` and implementation `781f881` on the clean response revision. This is a final targeted check of R8-F1, not a new kernel/performance audit.

## R8-F1: One Narrow Remainder

**Medium: provenance presence is checked, but the claimed per-field types are not enforced.** In [aggregate_roofline.py](../aggregate_roofline.py#L64), the required-field loop rejects `None` and blank strings. It does not reject other invalid types or invalid thread counts. Equal malformed values across all records still pass compatibility checks.

Three independent probes through actual `main()` with `--out`, using in-memory file I/O and otherwise complete valid inputs, each completed normally and wrote status **VALIDATED**:

- All three records had `head={}` instead of a source identity string.
- All three records had `threads=-1`.
- All three records had `threads=True`.

These are invalid provenance values, not numerical-tolerance questions. The first can still certify measurements without a meaningful source identity. The finding concerns the same explicitly requested typed-schema requirement; it does not invalidate the repaired historical timings.

**Remaining action:** enforce field-specific types/domains before comparisons, matching the producer contract: nonblank strings for textual identities, a positive integer excluding booleans for `threads`, and an explicitly supported process-index representation. Test malformed objects, booleans, and non-positive thread counts through the CLI with zero output opens on rejection, while retaining the complete producer-shaped positive. No benchmark rerun is required.

## Verified Repairs

- All eight required fields now reject missing, null, and whitespace-only values; changed runtime, node, and reference peaks also reject. **27 independent CLI negative controls** passed for intended reasons with **zero output-file opens**.
- A complete producer-shaped positive input succeeds through `main()` and writes the aggregate.
- The historical legacy path exactly reproduces the entire published `kernels` mapping at all **40 coordinates**. Existing legacy labelling and earlier launcher/roofline-wording closures stand.
- The committed selftest passes **18 controls**, not the response's stated 19. Those are helper-level controls; the CLI/no-output checks above were independently exercised by this review.

## Disposition

The previous missing-field, runtime/node compatibility, and peak-binding counterexamples are fixed. The standalone measurement and roofline review need no further experiments on this evidence. Full closure of the aggregator's **typed provenance** claim needs the small schema correction above; R8-F1 is therefore partially closed, not fully closed.

No native kernel, replay validator, F4 harness, promotion policy, or saved evidence was changed or rerun. Only this assessment was authored. F4 remains **PARTIAL**, promotion **BLOCKED**, and thresholds **UNRATIFIED**; review closure is not phase promotion.