"""Testes para _resolve_retriever() em scripts/run_pipeline.py e
scripts/run_train.py (correção A36 na camada de entry points, 2026-11-08).

STATUS: implementados, aguardando execução do pesquisador (pytest). Não
executados pelo assistente/fork que os escreveu.

Confirmam, com build_rag/prepare_medical_texts monkey-patchados (sem
construir nenhum índice/embedding real), que:
  - sem build_rag_flag (caminho oficial): retorna None e NÃO chama
    build_rag nem prepare_medical_texts.
  - com build_rag_flag (caminho histórico, preservado): chama build_rag
    (e, em run_pipeline, prepare_medical_texts quando skip_preprocess=False)
    e retorna o valor que build_rag produzir.
"""
from __future__ import annotations

import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT))
sys.path.insert(0, str(REPO_ROOT / "scripts"))

import run_pipeline  # noqa: E402
import run_train  # noqa: E402

FAKE_CFG = {
    "paths": {
        "rag_zip": "fake_rag.zip",
        "extracted_pdfs": "fake_extracted/",
        "medical_texts": "fake_medical_texts.txt",
    },
    "rag": {
        "embedding_model": "fake-embedding-model",
        "chunk_size": 500,
        "chunk_overlap": 50,
    },
}

_SENTINEL = object()


def _forbid(*args, **kwargs):
    raise AssertionError("não deveria ter sido chamado no caminho oficial")


def _fake_build_rag(*args, **kwargs):
    return _SENTINEL


def _fake_prepare(*args, **kwargs):
    return "fake_medical_texts.txt"


# ──────────────────────────────────────────────
# run_pipeline._resolve_retriever
# ──────────────────────────────────────────────


def test_run_pipeline_official_path_returns_none_without_building_rag(monkeypatch):
    monkeypatch.setattr(run_pipeline, "build_rag", _forbid)
    monkeypatch.setattr(run_pipeline, "prepare_medical_texts", _forbid)
    result = run_pipeline._resolve_retriever(
        build_rag_flag=False, skip_preprocess=False, cfg=FAKE_CFG
    )
    assert result is None


def test_run_pipeline_historical_path_calls_build_rag_and_preprocess(monkeypatch):
    calls = []
    monkeypatch.setattr(run_pipeline, "build_rag", lambda **kw: (calls.append("build_rag"), _SENTINEL)[1])
    monkeypatch.setattr(run_pipeline, "prepare_medical_texts", lambda **kw: calls.append("prepare"))
    result = run_pipeline._resolve_retriever(
        build_rag_flag=True, skip_preprocess=False, cfg=FAKE_CFG
    )
    assert calls == ["prepare", "build_rag"]
    assert result is _SENTINEL


def test_run_pipeline_historical_path_skips_preprocess_when_requested(monkeypatch):
    calls = []
    monkeypatch.setattr(run_pipeline, "build_rag", lambda **kw: (calls.append("build_rag"), _SENTINEL)[1])
    monkeypatch.setattr(run_pipeline, "prepare_medical_texts", _forbid)
    result = run_pipeline._resolve_retriever(
        build_rag_flag=True, skip_preprocess=True, cfg=FAKE_CFG
    )
    assert calls == ["build_rag"]
    assert result is _SENTINEL


# ──────────────────────────────────────────────
# run_train._resolve_retriever
# ──────────────────────────────────────────────


def test_run_train_official_path_returns_none_without_building_rag(monkeypatch):
    monkeypatch.setattr(run_train, "build_rag", _forbid)
    result = run_train._resolve_retriever(build_rag_flag=False, cfg=FAKE_CFG)
    assert result is None


def test_run_train_historical_path_calls_build_rag(monkeypatch):
    calls = []
    monkeypatch.setattr(run_train, "build_rag", lambda **kw: (calls.append("build_rag"), _SENTINEL)[1])
    result = run_train._resolve_retriever(build_rag_flag=True, cfg=FAKE_CFG)
    assert calls == ["build_rag"]
    assert result is _SENTINEL
