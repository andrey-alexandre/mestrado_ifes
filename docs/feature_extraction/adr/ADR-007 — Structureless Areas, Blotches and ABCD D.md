# ADR-007 — Structureless Areas, Blotches and ABCD D

**Status:** Accepted  
**Date:** 2026-09-08

## Structureless Areas and Blotches

### Decision

Adopt Stoecker et al. (2005) as the primary computational reference for structureless/blotch detection.

`STRUCTURELESS-BLOTCH-001 = HUMAN_APPROVED`

Preserve raw outputs such as:

- candidate mask;
- total blotch area fraction;
- largest blotch fraction;
- component count;
- centroid displacement;
- shape irregularity;
- color descriptors;
- spatial distribution.

The clinical criterion that a structureless/blotch region occupies approximately at least 10% of lesion area is approved as `CLINICAL_CONSENSUS`.

The exact color segmentation thresholds used by the adopted computational method must be recovered from the primary source before implementation. They must not be reconstructed by PH² tuning.

### Algorithm mappings

**ABCD structureless area:** approved conditional on faithful reproduction of the detector and application of the supported 10% criterion.

**7PCL irregular pigmentation:** `UNRESOLVED`. Raw pigmentation/blotch descriptors may be supplied, but no deterministic irregular-pigmentation decision is approved.

**Menzies color non-uniformity:** blotch evidence is `AUXILIARY_ONLY`; blotch presence must not be treated as equivalent to the complete Menzies color criterion.

---

## ABCD D

ABCD D consists of the relevant differential dermoscopic structures, including pigment network, structureless areas, branched streaks, dots and globules.

Only some of these currently have approved deterministic mappings.

Therefore the project must not compute a partial D and present it as a valid complete ABCD D score.

Until all required structure categories have approved deterministic mappings:

`ABCD_D = null`

Approved raw and partial structure evidence must still be preserved and supplied where appropriate.

A value such as `D=2` must never mean “two of the measurable structures were present” when other required structures could not be determined.