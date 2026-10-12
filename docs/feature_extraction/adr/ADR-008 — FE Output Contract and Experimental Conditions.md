# ADR-008 — FE Output Contract and Experimental Conditions

**Status:** Accepted  
**Date:** 2026-09-08

**Supersession notice:** Current implementation readiness and output semantics are governed by ADR-009 and ADR-010. Historical language below that describes 7PCL-r blue-whitish veil, irregular streaks, or Menzies-r blue-white veil as deterministic/approved must not be interpreted as authorization to emit lesion-level binary values. ADR-010 approves only the Celebi pixel-level veil detector, `veil_mask`, and `S1` / veil-area fraction as implementation-ready raw evidence; 7PCL-r and Menzies-r blue-white-veil binary mappings remain `UNRESOLVED`.

## Context

The literature review supports complete deterministic clinical outputs for some criteria, quantitative evidence without a justified final clinical threshold for others, and no valid measurement in some situations.

Using a single nullable value would make these situations ambiguous.

The dissertation compares the original multi-agent VLM pipeline with a version augmented by deterministic dermoscopic evidence.

## Decision

FE outputs are divided conceptually into three states:

### `DETERMINISTIC_CLINICAL`

A scientifically approved computational operationalization can produce the algorithm-specific clinical value.

### `RAW_EVIDENCE`

Scientifically supported quantitative measurements exist, but no approved deterministic mapping to the algorithm-specific clinical criterion exists.

### `UNAVAILABLE`

The measurement could not be calculated for the image or required information is unavailable.

`null` must never silently mean “criterion absent”.

Each output must distinguish absence (`false`/`0`) from unresolved interpretation and extraction failure.

Raw intermediate measurements must be persisted even when a deterministic clinical decision exists.

## Algorithm contracts

### ABCD

Deterministic:

- A;
- B, with documented proxy limitation;
- C.

D remains `null` until all required structure mappings are defensibly operationalized.

Raw D-related structure evidence is still supplied.

### 7PCL-r

Currently deterministic/approved:

- blue-whitish veil;
- irregular streaks, with classifier-derived limitation.

Other unresolved criteria receive raw evidence where available and no fabricated binary value.

### Menzies-r

Blue-white veil and any separately approved color rules may produce deterministic outputs.

Unresolved network, dot/globule and radial-structure criteria receive raw evidence rather than invented binary decisions.

Algorithm-specific color definitions must remain distinct from ABCD even when low-level color extraction code is shared.

## Experimental conditions

Three conditions are retained:

1. **Image only** — original VLM input without FE.
2. **Image + FE** — original image plus structured FE evidence.
3. **FE only** — structured FE evidence without the dermoscopic image.

FE-only means the scientifically supported structured evidence package, not necessarily complete deterministic clinical scores.

In Image + FE, FE is auxiliary evidence. The VLM may disagree with deterministic FE.

Raw evidence must be explicitly identified as measurement rather than established clinical classification.

VLM-versus-FE conflicts must be persisted for qualitative analysis.

## Pipeline constraints

FE executes before the VLM.

Each diagnostic agent receives only evidence relevant to its algorithm.

The existing ABCD, 7PCL-r, Menzies-r and summarizer logic should otherwise remain minimally changed.

The critical reviewer is removed because the planned RAG validation stage will not be implemented.

Temperature remains zero and the complete experiment is rerun rather than reusing qualification predictions.

F1 is the primary quantitative metric due to class imbalance, alongside sensitivity/recall and the existing evaluation metrics.

Predictions and raw responses are persisted for later stratified-bootstrap analysis.
