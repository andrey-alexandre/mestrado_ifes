"""Classifies a single saved pipeline result event into an explicit
diagnostic-output status. Shared by ``src.evaluation.evaluate`` (single-run,
single-system convenience metrics) and the stage-6 evaluator
(``src.evaluation.stage6_metrics``), so both apply the exact same rule.

Five explicit statuses -- never collapsed into each other:
  - ``"positivo"``        -- validation diagnostico matches "suspeita de melanoma".
  - ``"negativo"``        -- validation diagnostico matches "lesão benigna".
  - ``"indeterminado"``   -- validation diagnostico matches "lesão indeterminada".
  - ``"ausente"``         -- no ``validation`` at all (summarizer did not run
                              or produced nothing) -- the system gave NO output.
  - ``"nao_reconhecido"`` -- ``validation`` is present (the system DID
                              respond) but its text does not match any of the
                              three known categories -- distinct from
                              ``"ausente"``: this is a malformed/unexpected
                              response, not a missing one.

Researcher decision (2026-11-08): "indeterminado" is a VALID output (counts
toward the valid-output rate) but is EXCLUDED from the binary confusion
matrix used for accuracy/precision/sensitivity/specificity/F1 -- see
``src.evaluation.stage6_metrics`` for where that distinction is applied.
This module only classifies; it does not decide what each status means for
any particular metric.

"indeterminado-as-positivo" (the previous, now-removed silent convention in
``evaluate.py``) is NOT implemented here. It would only be relevant to a
SEPARATE, not-yet-requested "referral/triage" analysis (treat any uncertain
case as flagged for human review) -- out of scope for this module and for
the stage-6 primary metrics.
"""
from __future__ import annotations

from typing import Any

POSITIVO = "positivo"
NEGATIVO = "negativo"
INDETERMINADO = "indeterminado"
AUSENTE = "ausente"
NAO_RECONHECIDO = "nao_reconhecido"

ALL_STATUSES = (POSITIVO, NEGATIVO, INDETERMINADO, AUSENTE, NAO_RECONHECIDO)

# Order does not matter for correctness (the three phrases are mutually
# exclusive substrings), but is kept stable for readability.
_KEYWORD_TO_STATUS = (
    ("suspeita de melanoma", POSITIVO),
    ("lesão indeterminada", INDETERMINADO),
    ("lesao indeterminada", INDETERMINADO),  # tolerate missing accent (free-text models)
    ("lesão benigna", NEGATIVO),
    ("lesao benigna", NEGATIVO),
)


def classify_case_status(event: dict[str, Any] | None) -> str:
    """Classifies one saved result event (one lesion, one system/condition).

    Args:
        event: the saved dict for one image/lesion, as written by
            ``src.pipeline.pipeline.run_pipeline`` (has a ``"validation"``
            key that is a dict with a ``"diagnostico"`` field, a free-text
            string, or ``None``/absent).

    Returns:
        One of :data:`ALL_STATUSES`.
    """
    if not event:
        return AUSENTE

    validation = event.get("validation")
    if validation is None:
        return AUSENTE

    if isinstance(validation, dict):
        diagnostico = validation.get("diagnostico")
        if not diagnostico:
            return NAO_RECONHECIDO
        text = str(diagnostico).lower()
    else:
        text = str(validation).lower()

    if not text.strip():
        return NAO_RECONHECIDO

    for keyword, status in _KEYWORD_TO_STATUS:
        if keyword in text:
            return status
    return NAO_RECONHECIDO
