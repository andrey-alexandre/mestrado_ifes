# ADR-001 — Scientific Evidence and No-Invention Policy

**Status:** Accepted  
**Date:** 2026-09-08  
**Scope:** Feature Extraction methodology and implementation

## Context

The dissertation introduces deterministic dermoscopic Feature Extraction (FE) as an auxiliary evidence stage for the existing multi-agent VLM diagnostic pipeline.

Several dermoscopic criteria have well-established clinical definitions but do not have equally well-established deterministic computational operationalizations. In particular, the literature may define a visual phenomenon without providing a reproducible formula, parameter set, threshold, or mapping from quantitative measurements to a binary clinical decision.

Creating missing thresholds or formulas during implementation would introduce undocumented methodological decisions and weaken scientific traceability.

## Decision

No scientific decision may originate during implementation.

Every formula, threshold, preprocessing operation, clinical mapping, or algorithm-specific decision used by FE must be supported by an approved scientific specification or an explicit human methodological decision.

Threshold origins must be classified as one of:

- `CLINICAL_CONSENSUS`
- `PUBLISHED_COMPUTATIONAL`
- `DATASET_DERIVED`
- `CLASSIFIER_DERIVED`
- `AUTHOR_DEFINED`
- `UNRESOLVED`

Published computational or classifier-derived thresholds may be adopted when explicitly approved, but they must not be described as universal clinical thresholds.

If implementation encounters a missing scientific parameter, the implementation must stop for that decision and report a specification gap. It must not estimate, tune, infer, optimize, or invent a replacement.

## Evidence hierarchy

Preference order:

1. Original clinical algorithm publications or authoritative dermoscopy references.
2. Primary computational papers implementing the specific dermoscopic phenomenon.
3. Peer-reviewed methodological studies.
4. Reviews for discovery and contextualization.
5. Secondary sources only when primary evidence cannot be recovered.

Source code repositories may aid reproduction but do not independently establish scientific validity.

## Consequences

The FE implementation may intentionally contain `null` clinical decisions when the literature does not support a deterministic mapping.

An incomplete but traceable implementation is preferred to a complete implementation containing invented scientific rules.

Scientific ambiguity discovered during coding becomes a specification issue, not an implementation decision.