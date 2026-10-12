# ADR-003 — ABCD A, B and C Operationalization

**Status:** Accepted  
**Date:** 2026-09-08

**Supersession note for ABCD-B:** This ADR records the human approval of Kasmi and Mokrani (2016) as the source basis for ABCD-B. Later implementation-resolution experiments closed the executable formulation as a source-guided author operationalization, not exact source reproduction, because `y` and `length(y)` in `error > 0.05 x length(y)` are under-specified. Current ABCD-B implementation details are governed by `docs/abcd_border_methodology.md` and ADR-011.

## Context

The literature review identified reproducible computational operationalizations for ABCD asymmetry, border and color, although some thresholds are method- or dataset-derived rather than clinical-consensus thresholds.

## Decision

### A — Asymmetry

Adopt Kasmi and Mokrani (2016).

The implementation must reproduce the published procedure based on center of gravity, principal axis from second-order moments, image alignment, shape symmetry, brightness symmetry and CIELAB color symmetry.

Adopt the published computational thresholds:

- mirrored color blocks: ΔE ≤ 6;
- brightness half-difference < 3%;
- shape half-difference < 2%.

The final ABCD asymmetry output remains `A ∈ {0,1,2}` according to the published operationalization.

**Decision:** `HUMAN_APPROVED`.

These are published computational thresholds tested on the source dataset, not universal clinical thresholds. No PH² calibration is allowed.

### B — Border

Adopt Kasmi and Mokrani (2016):

- divide the lesion contour into eight 45° sectors;
- fit third-order splines to each sub-contour;
- use the published fitting-error criterion `error > 0.05 × length(y)`;
- produce `B ∈ {0,...,8}`.

**Decision:** `HUMAN_APPROVED_WITH_LIMITATION`.

This must be described as a geometric sub-contour irregularity proxy for the ABCD abrupt pigment-cutoff criterion. It must not be represented as direct measurement of lesion-to-skin pigment-transition abruptness.

### C — Color

Adopt Majumder and Ullah (2019) normalized RGB thresholds for:

- white;
- red;
- light brown;
- dark brown;
- blue-gray;
- black.

A color is considered present when it occupies more than the published 5% lesion-pixel threshold.

**Decision:** `HUMAN_APPROVED`.

The RGB thresholds and 5% rule are published computational/dataset-derived parameters and are not universal clinical thresholds.

PH² color annotations are validation ground truth only.

## Consequences

ABCD A, B and C can produce deterministic clinical outputs.

All intermediate measurements must also be persisted for audit and debugging.

ABCD B carries an explicit methodological limitation in all scientific reporting.
