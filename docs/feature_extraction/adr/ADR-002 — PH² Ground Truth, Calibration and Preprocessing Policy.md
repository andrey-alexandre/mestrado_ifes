# ADR-002 — PH² Ground Truth, Calibration and Preprocessing Policy

**Status:** Accepted  
**Date:** 2026-09-08

## Context

PH² provides dermoscopic RGB images, expert lesion masks, diagnostic labels, and several expert dermoscopic annotations.

Using expert annotations to construct FE outputs would leak ground-truth information into the diagnostic pipeline. Likewise, tuning FE thresholds against the same PH² labels later used for final evaluation would produce optimistic estimates.

The project also requires deterministic FE to be evaluated separately from segmentation performance.

## Decision

PH² ground-truth lesion masks are permitted as FE inputs.

Other PH² expert annotations — including diagnosis, asymmetry, colors, pigment network, blue-white veil, streaks, dots/globules and related annotations — must not be used:

- as FE inputs;
- to select thresholds;
- to calibrate thresholds;
- to train FE classifiers;
- to optimize FE parameters.

They may be used exclusively as post-hoc validation ground truth where an appropriate annotation exists.

Published parameters and thresholds will be transferred without PH² calibration and their external origin and limitations must be documented.

The original PH² RGB resolution is preserved for FE unless a primary adopted method explicitly requires another transformation.

No mandatory global color normalization is adopted. Blanket DullRazor/hair removal is rejected because preprocessing may remove diagnostically relevant network or streak structures.

Conservative artifact detection, exclusion masks and quality-control flags are allowed when scientifically supported.

## Consequences

The study evaluates transferability of published computational methods rather than optimizing them to PH².

Ground-truth segmentation performance is outside the FE experiment.

Any failure caused by transferring published thresholds to PH² is a methodological limitation and potentially an experimental finding, not a reason to silently recalibrate the method.