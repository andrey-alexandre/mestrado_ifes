"""Stage 3 of the reconstruction: CNN baseline (EfficientNetV2-S) data
preparation. Methodologically distinct from ``src.data_audit`` (structural
auditing) and ``src.data_audit.sampling`` (HAM10000 experimental sample,
reserved for final external evaluation only) -- this package prepares the
BCN20000 development split. No training/threshold/Grad-CAM code lives here
yet; this stage starts with data preparation only.
"""
