"""Seleção pura (sem I/O) das 40 lesões para a galeria Grad-CAM: 10 por
célula da matriz de confusão (TP/TN/FP/FN) das predições S0 já
produzidas pela CNN congelada, usando o limiar já fixado (nunca
reajustado aqui). Nenhuma seleção por confiança do modelo ou qualidade
do mapa -- isso só é conhecido DEPOIS de gerar o mapa, que é uma etapa
posterior e separada (``src.cnn_baseline.gradcam``).
"""
from __future__ import annotations

import random
from dataclasses import dataclass, field

STATUS_OK = "ok"

GROUP_TP = "TP"
GROUP_TN = "TN"
GROUP_FP = "FP"
GROUP_FN = "FN"
ALL_GROUPS = (GROUP_TP, GROUP_TN, GROUP_FP, GROUP_FN)

DEFAULT_SEED = 42
DEFAULT_N_PER_CELL = 10


def _confusion_group(true_binary_target: str, predicted_binary_target: str) -> str:
    is_real_melanoma = true_binary_target == "melanoma"
    is_pred_melanoma = predicted_binary_target == "melanoma"
    if is_real_melanoma and is_pred_melanoma:
        return GROUP_TP
    if (not is_real_melanoma) and (not is_pred_melanoma):
        return GROUP_TN
    if (not is_real_melanoma) and is_pred_melanoma:
        return GROUP_FP
    return GROUP_FN  # is_real_melanoma and not is_pred_melanoma


@dataclass
class GradCamSampleResult:
    selected_by_group: dict[str, list[dict]] = field(default_factory=dict)
    n_eligible_by_group: dict[str, int] = field(default_factory=dict)
    n_selected_by_group: dict[str, int] = field(default_factory=dict)
    shortfall_groups: dict[str, str] = field(default_factory=dict)  # group -> nota registrada
    seed: int = DEFAULT_SEED
    n_per_cell_requested: int = DEFAULT_N_PER_CELL


def select_gradcam_sample(
    cnn_score_rows: list[dict], seed: int = DEFAULT_SEED, n_per_cell: int = DEFAULT_N_PER_CELL,
) -> GradCamSampleResult:
    """``cnn_score_rows``: linhas de ``s0_cnn_scores.csv`` (dicts com
    ``lesion_id``/``image_id``/``status``/``true_binary_target``/
    ``predicted_binary_target``/``predicted_score``/``dx``). Só linhas
    com ``status == "ok"`` são elegíveis (uma predição real existe) --
    falhas são excluídas da seleção, nunca substituídas por outro caso.

    Dentro de cada célula, os candidatos elegíveis são ordenados por
    ``lesion_id`` (string, ascendente) ANTES do sorteio -- garante que o
    mesmo ``random.Random(seed).sample(...)`` produza exatamente a mesma
    escolha em qualquer execução, independente da ordem de leitura do
    CSV. Se uma célula tiver menos de ``n_per_cell`` elegíveis, usa todos
    e registra o atalho em ``shortfall_groups`` -- nunca busca substitutos
    em outra célula."""
    by_group: dict[str, list[dict]] = {g: [] for g in ALL_GROUPS}
    for row in cnn_score_rows:
        if row.get("status") != STATUS_OK:
            continue
        group = _confusion_group(row["true_binary_target"], row["predicted_binary_target"])
        by_group[group].append(row)

    result = GradCamSampleResult(seed=seed, n_per_cell_requested=n_per_cell)
    for group in ALL_GROUPS:
        candidates = sorted(by_group[group], key=lambda r: r["lesion_id"])
        result.n_eligible_by_group[group] = len(candidates)
        rng = random.Random(seed)  # reseed por grupo -- cada celula tem sua propria amostra independente, determinada so pela semente global e pelo conjunto de candidatos daquela celula
        k = min(n_per_cell, len(candidates))
        selected = rng.sample(candidates, k=k) if candidates else []
        selected_sorted = sorted(selected, key=lambda r: r["lesion_id"])  # ordem de saida estavel, nao a ordem interna do sample
        result.selected_by_group[group] = selected_sorted
        result.n_selected_by_group[group] = len(selected_sorted)
        if len(candidates) < n_per_cell:
            result.shortfall_groups[group] = (
                f"apenas {len(candidates)} caso(s) elegivel(is) na celula {group} "
                f"(esperado {n_per_cell}) -- todos usados, nenhum substituto buscado"
            )
    return result
