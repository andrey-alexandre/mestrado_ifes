# ADR-006 — Radial Structures and Dots/Globules

**Status:** Accepted  
**Date:** 2026-09-08

## Radial Structures

### Decision

Adopt Sadeghi et al. (2013) as the primary shared streak/radial-structure method.

`STREAK-DETECT-001 = HUMAN_APPROVED`

Preserve raw evidence including, when supported by the reproduced method:

- candidate mask;
- count;
- length;
- width;
- orientation;
- radiality;
- angular distribution/coverage;
- branching;
- terminal morphology.

Implementation is blocked until the complete primary source provides sufficient methodological detail for faithful reproduction.

The published regular/irregular streak classifier may be used with the explicit limitation that it is a classifier-derived computational operationalization.

### Algorithm mappings

**7PCL irregular streaks:** `HUMAN_APPROVED_WITH_LIMITATION`.

**ABCD branched streaks:** `UNRESOLVED`.

**Menzies radial streaming:** `UNRESOLVED`.

**Menzies pseudopods:** `UNRESOLVED`.

No arbitrary branch-count, terminal-width, radiality or angular thresholds may be introduced.

PH² streak annotations may validate presence/absence only; they cannot validate regular/irregular or radial-streaming/pseudopod subtype decisions.

---

## Dots and Globules

### Decision

Adopt Maglogiannis and Delibasis (2015) as the primary computational reference for dark dot/globule detection.

`BLOB-DETECT-001 = HUMAN_APPROVED_PENDING_SOURCE_DETAIL`

Implementation remains blocked until the primary source provides sufficient parameters for reproducible detection/segmentation.

The clinical approximately 0.1 mm dot/globule size boundary is accepted as a clinical definition.

However, no conversion from 0.1 mm to PH² pixels is approved until a scientifically supported physical image scale or source-specific operationalization is established.

### Algorithm mappings

The following remain `UNRESOLVED`:

- ABCD dots;
- ABCD globules;
- 7PCL irregular dots/globules;
- Menzies blue-gray dots;
- Menzies peripheral black dots/globules;
- Menzies multiple brown dots.

Raw blob measurements are approved as auxiliary VLM evidence.

No arbitrary definition of “multiple”, “peripheral”, “irregular”, “black”, “blue-gray”, or dot/globule pixel size may be introduced.

PH² absent/typical/atypical dots/globules annotations may be used for post-hoc validation only.