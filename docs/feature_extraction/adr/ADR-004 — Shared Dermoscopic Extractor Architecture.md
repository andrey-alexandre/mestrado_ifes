# ADR-004 — Shared Dermoscopic Extractor Architecture

**Status:** Accepted  
**Date:** 2026-09-08

## Context

ABCD, reduced Seven-Point Checklist and reduced Menzies repeatedly refer to related dermoscopic phenomena:

- pigment network;
- blue-white veil;
- radial/streak structures;
- dots/globules;
- structureless/blotch regions.

Implementing independent image-processing pipelines for each algorithm would duplicate code and incorrectly imply that visually related low-level structures require independent detection.

However, a shared low-level detector does not imply that different clinical algorithms use the same definition or decision threshold.

## Decision

Implement reusable low-level extractor families for:

1. `pigment_network`
2. `blue_white_veil`
3. `radial_structures`
4. `dots_globules`
5. `structureless_blotch`

Low-level detection, masks and quantitative measurements may be shared.

Algorithm-specific clinical interpretations and thresholds must remain separate.

Therefore:

`shared detector ≠ shared clinical criterion`

and:

`shared raw evidence ≠ shared final threshold`.

Each extractor must preserve its intermediate outputs and expose them independently of the VLM pipeline.

## Consequences

ABCD, 7PCL-r and Menzies-r may consume the same raw measurements while reaching different clinical interpretations.

The architecture should favor reusable feature modules followed by algorithm-specific mapping layers.

A final threshold may only be shared when literature explicitly supports equivalence.