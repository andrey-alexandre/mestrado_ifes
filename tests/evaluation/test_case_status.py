from __future__ import annotations

import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT))

from src.evaluation.case_status import (  # noqa: E402
    AUSENTE,
    INDETERMINADO,
    NAO_RECONHECIDO,
    NEGATIVO,
    POSITIVO,
    classify_case_status,
)


def test_positivo_from_dict():
    assert classify_case_status({"validation": {"diagnostico": "Suspeita de Melanoma"}}) == POSITIVO


def test_negativo_from_dict():
    assert classify_case_status({"validation": {"diagnostico": "Lesão Benigna"}}) == NEGATIVO


def test_indeterminado_is_not_positivo():
    assert classify_case_status({"validation": {"diagnostico": "Lesão Indeterminada"}}) == INDETERMINADO


def test_indeterminado_without_accent():
    assert classify_case_status({"validation": {"diagnostico": "lesao indeterminada"}}) == INDETERMINADO


def test_ausente_when_validation_is_none():
    assert classify_case_status({"validation": None}) == AUSENTE


def test_ausente_when_validation_key_missing():
    assert classify_case_status({}) == AUSENTE


def test_ausente_when_event_is_none():
    assert classify_case_status(None) == AUSENTE


def test_nao_reconhecido_free_text_outside_pattern():
    assert classify_case_status({"validation": "resposta livre sem nenhuma das 3 categorias"}) == NAO_RECONHECIDO


def test_nao_reconhecido_dict_without_diagnostico_field():
    assert classify_case_status({"validation": {"justificativa": "sem campo diagnostico"}}) == NAO_RECONHECIDO


def test_free_text_string_validation_matches_keyword():
    # Modo llava:13b -- validation e uma string livre, nao um dict.
    assert classify_case_status({"validation": "Após análise: Suspeita de Melanoma, recomenda-se biópsia."}) == POSITIVO
