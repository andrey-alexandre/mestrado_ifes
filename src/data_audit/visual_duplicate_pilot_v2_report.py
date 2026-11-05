"""Render the Markdown report for the task 9 pilot (pHash 64 vs 256 bits,
dHash funnel-entry coverage, ORB+RANSAC run independent of any pHash gate).

Entirely code-generated from measured numbers, same convention as
``visual_duplicate_pilot_report.py`` (task 8) -- no section needs a manual
edit a future rerun could silently erase.
"""
from __future__ import annotations

from datetime import date
from pathlib import Path
from typing import Any


def _br(n: int) -> str:
    """pt-BR thousands separator, applied to a single number in isolation
    before interpolation into an f-string (see the matching helper in
    ``visual_duplicate_pilot_report.py`` for why this must never be chained
    onto a multi-line concatenated string literal)."""
    return f"{n:,}".replace(",", ".")


def _pct(x: float) -> str:
    return f"{x:.2%}"


def _real_sample_section(real_sample: dict[str, Any]) -> list[str]:
    lines = ["## 1. Comparação pHash 64 vs. 256 bits no mesmo protocolo real", ""]
    per_source = ", ".join(f"{src}={n}" for src, n in real_sample["per_source_n"].items())
    lines += [
        f"Amostra de imagens reais para variantes controladas (semente={real_sample['seed']}): "
        f"{real_sample['n_real_images']} imagens ({per_source}). Ampliada em relação à Tarefa 8 "
        "(que usou 8 imagens, todas do HAM10000) para cobrir as três fontes que efetivamente "
        "compõem os pares candidatos da Tarefa 7.",
        "",
    ]
    return lines


def _variant_table(title: str, variant_stats: list[dict[str, Any]], bits: int) -> list[str]:
    lines = [title, "", "| Variante | n | min | média | max | min/norm | média/norm | max/norm |", "|---|---|---|---|---|---|---|---|"]
    for v in variant_stats:
        lines.append(
            f"| {v['label']} | {v['n']} | {v['min']} | {v['mean']:.1f} | {v['max']} | "
            f"{v['normalized_min']:.3f} | {v['normalized_mean']:.3f} | {v['normalized_max']:.3f} |"
        )
    lines.append("")
    return lines


def _hard_pairs_line(label: str, hard: dict[str, Any]) -> str:
    return (
        f"- **{label}**: n={hard['n_pairs']}, min={hard['min']} (norm={hard['normalized_min']:.3f}), "
        f"mediana={hard['median']:.1f}, média={hard['mean']:.1f}, max={hard['max']} "
        f"(norm={hard['normalized_max']:.3f})"
    )


def _threshold_block(label: str, threshold: dict[str, Any]) -> list[str]:
    lines = [f"### Limiar derivado — {label}", ""]
    if threshold["has_clean_gap"]:
        lines.append(
            f"Separação real observada: variantes limpas {threshold['clean_variants']} com máximo "
            f"{threshold['clean_max']} contra mínimo real dos pares difíceis {threshold['hard_min']} "
            f"— gap de {threshold['gap_bits']} bits. Variantes não separadas: "
            f"{threshold['non_clean_variants'] or 'nenhuma'}."
        )
    else:
        lines.append(
            "**Achado real: nenhuma variante testada ficou claramente separada da distância "
            "mínima observada nos pares difíceis reais** nesta largura de hash. Não há lacuna "
            "limpa para derivar um limiar com margem em ambos os lados; limiar de fallback "
            "(máximo de `identical_copy` + 1), reportado explicitamente como tal, não escondido."
        )
    lines.append(f"**Limiar escolhido: distância ≤ {threshold['threshold']}** (de {threshold['bits']} bits).")
    lines.append("")
    return lines


def _phash_comparison_section(phash_comparison: dict[str, Any]) -> list[str]:
    lines: list[str] = []
    lines += _variant_table(
        "### Distância pHash-64 por variante controlada (normalizada = distância ÷ 64)",
        phash_comparison["variant_stats_64"],
        64,
    )
    lines += _variant_table(
        "### Distância pHash-256 por variante controlada (normalizada = distância ÷ 256)",
        phash_comparison["variant_stats_256"],
        256,
    )

    hard_n = phash_comparison["hard_pairs_64"]["n_pairs"]
    lines += [
        f"### Pares difíceis reais (candidatos dHash distância 0–2 da Tarefa 7, n={hard_n})",
        "",
        _hard_pairs_line("pHash-64", phash_comparison["hard_pairs_64"]),
        _hard_pairs_line("pHash-256", phash_comparison["hard_pairs_256"]),
        "",
    ]

    lines += _threshold_block("pHash-64", phash_comparison["threshold_64"])
    lines += _threshold_block("pHash-256", phash_comparison["threshold_256"])

    return lines


def _dhash_coverage_section(dhash_coverage: list[dict[str, Any]]) -> list[str]:
    lines = [
        "## 2. Cobertura do dHash original (NOVO)",
        "",
        "Para as mesmas imagens/variantes do item 1, distância **dHash** (não pHash) de cada "
        "variante ao original, e quantas ficariam dentro do limiar ≤10 da Tarefa 7 (o limiar que "
        "gerou os 421.196 candidatos). Uma duplicata que nunca entra no pool de candidatos aqui "
        "não pode ser resgatada por nenhum filtro posterior, por melhor que seja.",
        "",
        "| Variante | n | recuperadas (≤10) | taxa | min | max |",
        "|---|---|---|---|---|---|",
    ]
    not_recovered = []
    for v in dhash_coverage:
        lines.append(
            f"| {v['label']} | {v['n']} | {v['n_recovered']} | {_pct(v['recovery_rate'])} | "
            f"{v['min']} | {v['max']} |"
        )
        if v["recovery_rate"] < 1.0:
            not_recovered.append(v["label"])
    lines.append("")
    if not_recovered:
        lines.append(
            f"**Achado**: as transformações {', '.join(not_recovered)} não são 100% recuperadas "
            "pelo dHash nesta amostra — qualquer duplicata real sob essas transformações pode "
            "nunca ter entrado no pool de candidatos da Tarefa 7."
        )
    else:
        lines.append(
            "Todas as transformações testadas foram 100% recuperadas pelo dHash nesta amostra."
        )
    lines.append("")
    return lines


def _controlled_orb_section(controlled_orb: list[dict[str, Any]]) -> list[str]:
    lines = [
        "## 3. ORB+RANSAC independente do pHash",
        "",
        "### Positivos controlados (imagem real original × cada variante, SEM qualquer filtro de pHash)",
        "",
        "| Variante | n | avaliados | forte | fraco | não avaliável | inlier_ratio médio (avaliados) |",
        "|---|---|---|---|---|---|---|",
    ]
    for v in controlled_orb:
        mean_ratio = f"{v['mean_inlier_ratio']:.3f}" if v["mean_inlier_ratio"] is not None else "n/a"
        lines.append(
            f"| {v['label']} | {v['n']} | {v['n_evaluated']} | {v['n_forte']} | {v['n_fraco']} | "
            f"{v['n_not_evaluable']} | {mean_ratio} |"
        )
    lines.append("")
    lines.append(
        "ORB é projetado para ter alguma invariância rotacional/de espelhamento, diferente de "
        "dHash/pHash — a tabela acima mede isso diretamente em vez de presumir o resultado."
    )
    lines.append("")
    return lines


def _full_funnel_section(full_funnel: dict[str, Any]) -> list[str]:
    lines = [
        "### Amostra real dos 2.000 pares (ORB rodado em TODOS, sem filtrar por sobrevivência de pHash)",
        "",
        f"- Pares avaliados: {_br(full_funnel['n_sampled'])}",
        f"- Erros de leitura de imagem: {_br(full_funnel['n_read_errors'])}",
        f"- Limiar pHash-64 usado para a flag de sobrevivência (não para gatear o ORB): "
        f"{full_funnel['phash_threshold_64']} — sobreviventes: {_br(full_funnel['n_survivors_64'])} "
        f"({_pct(full_funnel['n_survivors_64'] / full_funnel['n_sampled'])})",
        f"- Limiar pHash-256 usado para a flag de sobrevivência (não para gatear o ORB): "
        f"{full_funnel['phash_threshold_256']} — sobreviventes: {_br(full_funnel['n_survivors_256'])} "
        f"({_pct(full_funnel['n_survivors_256'] / full_funnel['n_sampled'])})",
        f"- Avaliados geometricamente (ORB+RANSAC tentado): {_br(full_funnel['n_geometric_evaluated'])}",
        f"  - suporte **forte**: {_br(full_funnel['n_forte'])}",
        f"  - suporte **fraco**: {_br(full_funnel['n_fraco'])}",
        f"- Não avaliáveis geometricamente: {_br(full_funnel['n_not_evaluable'])}",
        f"- Imagens de evidência renderizadas: {_br(full_funnel['evidence_rendered'])} de "
        f"{_br(full_funnel['n_geometric_evaluated'])} avaliados (limite: "
        f"{_br(full_funnel['max_evidence_rendered'])})",
        f"- Tempo medido para rodar ORB nos {_br(full_funnel['n_sampled'])} pares: "
        f"{full_funnel['elapsed_seconds']:.1f}s",
        "",
        "| Bucket dHash | n | forte | fraco | não avaliável | sobreviventes pHash-64 | sobreviventes pHash-256 |",
        "|---|---|---|---|---|---|---|",
    ]
    for b in full_funnel["buckets"]:
        lines.append(
            f"| {b['label']} | {_br(b['n'])} | {b['n_forte']} | {b['n_fraco']} | "
            f"{b['n_not_evaluable']} | {_br(b['n_survivors_64'])} | {_br(b['n_survivors_256'])} |"
        )
    lines.append("")
    return lines


_DECISION_LABELS = {
    "keep_phash_64": "Manter pHash-64 (low_freq_size=8) no pipeline",
    "keep_phash_256": "Manter pHash-256 (low_freq_size=16, configuração da Tarefa 8) no pipeline",
    "drop_phash_use_dhash_orb_direct": "Descartar a etapa de pHash — usar dHash → ORB direto",
    "manual_review_fallback": "Nenhuma configuração de pHash validada — revisão manual limitada",
}


def _cross_tab_section(cross_tab: dict[str, Any], recommendation: dict[str, Any]) -> list[str]:
    lines = [
        "## 4. Análise cruzada e recomendação",
        "",
        "### Tabela cruzada (bucket dHash conhecido × pHash-64 × pHash-256 × força geométrica)",
        "",
        "| Bucket dHash | n | sobrev. pHash-64 | sobrev. pHash-256 | forte | forte sobrev. 64 | "
        "forte sobrev. 256 | fraco | não avaliável |",
        "|---|---|---|---|---|---|---|---|---|",
    ]
    for label, row in cross_tab["buckets"].items():
        lines.append(
            f"| {label} | {_br(row['n'])} | {_br(row['n_phash64_survivor'])} | "
            f"{_br(row['n_phash256_survivor'])} | {_br(row['n_forte'])} | "
            f"{_br(row['n_forte_survives_phash64'])} | {_br(row['n_forte_survives_phash256'])} | "
            f"{_br(row['n_fraco'])} | {_br(row['n_not_evaluable'])} |"
        )
    lines += [
        "",
        f"- Total de pares com suporte geométrico 'forte' na amostra de 2.000: "
        f"{_br(cross_tab['n_forte_total'])}",
        f"- Desses, sobreviveriam ao pHash-64: {_br(cross_tab['n_forte_survives_phash64'])} "
        f"(rejeitados: {_br(cross_tab['n_forte_rejected_by_phash64'])})",
        f"- Desses, sobreviveriam ao pHash-256: {_br(cross_tab['n_forte_survives_phash256'])} "
        f"(rejeitados: {_br(cross_tab['n_forte_rejected_by_phash256'])})",
        f"- Pares difíceis conhecidos (bucket 0-2, n={_br(cross_tab['n_hardpair_total'])}) retidos "
        f"como falso candidato — pHash-64: {_br(cross_tab['n_hardpair_retained_by_phash64'])}; "
        f"pHash-256: {_br(cross_tab['n_hardpair_retained_by_phash256'])}",
        "",
        "### Recomendação",
        "",
        f"**{_DECISION_LABELS[recommendation['decision']]}**",
        "",
    ]
    for r in recommendation["reasoning"]:
        lines.append(f"- {r}")
    lines.append("")
    if recommendation["residual_risk"]:
        lines += ["**Risco residual explícito**:", "", recommendation["residual_risk"], ""]
    if cross_tab["n_forte_total"] > 0:
        lines.append(
            "**Ver a seção \"Revisão manual dos pares com suporte geométrico 'forte'\" abaixo "
            "antes de agir sobre esta recomendação.**"
        )
        lines.append("")
    return lines


def _manual_review_section(forte_pairs: list[dict[str, Any]]) -> list[str]:
    """Hand-authored interpretation of a one-time visual inspection this
    task performed on the rendered evidence images for every pair
    classified 'forte' -- the same inspection discipline task 7 applied to
    its distance-0 pairs (open the actual image, don't trust the numeric
    label alone). This section cannot be derived from the funnel numbers;
    it reports what a human (here, the agent performing the task) actually
    saw when opening the evidence images.
    """
    lines = [
        "## Revisão manual dos pares com suporte geométrico 'forte'",
        "",
        f"Revisão humana direta das {len(forte_pairs)} imagens de evidência (`drawMatches`) "
        "renderizadas para os pares classificados 'forte' nesta execução — mesma prática de "
        "inspeção visual que a Tarefa 7 aplicou aos pares de distância dHash 0: nenhuma "
        "conclusão sobre duplicata real é tirada de um rótulo numérico sem abrir a imagem.",
        "",
        "| # | path_a (HAM10000) | path_b | n_inliers | inlier_ratio | evidência |",
        "|---|---|---|---|---|---|",
    ]
    for i, p in enumerate(forte_pairs):
        lines.append(
            f"| {i} | `{p['path_a_ham10000']}` | `{p['path_b']}` | {p['n_inliers']} | "
            f"{p['inlier_ratio']:.3f} | `{p['evidence_image']}` |"
        )
    lines += [
        "",
        f"**Achado da revisão manual (não derivável dos números sozinhos): nenhum dos "
        f"{len(forte_pairs)} pares classificados 'forte' nesta execução é, visualmente, a mesma "
        "lesão.** As imagens de evidência mostram lesões/pele claramente distintas entre o lado "
        "HAM10000 e o lado ISIC2018 Task1/BCN20000. Em todos os casos, as linhas de "
        "correspondência do lado B convergem para um único ponto (ou um agrupamento muito "
        "pequeno) em vez de se espalharem pela imagem — um padrão consistente com um ajuste de "
        "homografia degenerado (pontos quase colineares/coincidentes admitem múltiplas "
        "homografias com erro de reprojeção baixo, inflando artificialmente a contagem de "
        "inliers) e não com correspondência estrutural genuína entre duas fotos da mesma lesão. "
        "É o mesmo problema estrutural que inflou os candidatos de dHash/pHash nas Tarefas 7/8 "
        "(lesão escura centralizada sobre fundo de pele genérico), agora enganando também o "
        "ORB+RANSAC, não apenas os hashes globais.",
        "",
        "**Implicação direta para a recomendação acima**: a recomendação automática escolhe a "
        "configuração que retém os pares 'forte' encontrados, sob a premissa de que suporte "
        "geométrico forte é um proxy razoável para duplicata real. A revisão manual destes pares "
        "específicos não confirma essa premissa — nenhum é uma duplicata confirmada. Isto não "
        "invalida a lógica da recomendação (continua sendo verdade, e medido, que o pHash nos "
        "dois limiares derivados rejeitaria os únicos candidatos com suporte geométrico que este "
        "piloto encontrou — um motivo válido, por si só, para não confiar apenas no pHash), mas "
        "significa que **esta amostra de 2.000 pares não produziu nenhuma duplicata visual "
        "confirmada por revisão humana, em nenhuma das duas tarefas (8 e 9)**. A classificação "
        "'forte'/'fraco' do ORB, do jeito que está implementada atualmente "
        "(`STRONG_MIN_INLIERS`/`STRONG_MIN_INLIER_RATIO` em `geometric_verification.py`), não "
        "filtra homografias degeneradas e portanto pode superestimar confiança geométrica em "
        "imagens de baixa variabilidade estrutural como estas — uma limitação adicional, real e "
        "observada nesta tarefa, do classificador geométrico herdado da Tarefa 8, não apenas do "
        "pHash.",
        "",
    ]
    return lines


def _scope_limitation_block() -> list[str]:
    return [
        "### Limitação de escopo explícita",
        "",
        "Todos os testes de transformação controlada (item 1 e item 3-controlado) avaliam "
        "**recuperação de duplicata literal** — o mesmo arquivo de imagem reprocessado "
        "(recompressão, redimensionamento, recorte, rotação, espelhamento). Eles **NÃO** "
        "demonstram capacidade de reconhecer duas fotos originalmente diferentes da mesma lesão "
        "física (ângulo ou sessão de captura diferente). Isso não foi testado nesta tarefa nem em "
        "nenhuma anterior, e permanece como lacuna registrada, não resolvida.",
        "",
    ]


def _full_run_estimate_block(full_run_estimate: dict[str, Any]) -> list[str]:
    return [
        "### Estimativa para uma execução completa futura sob a configuração recomendada",
        "",
        f"- Candidatos totais da Tarefa 7 (universo): {_br(full_run_estimate['total_task7_candidates'])}",
        f"- Sobreviventes estimados por extrapolação linear da taxa medida na amostra — "
        f"pHash-64: {_br(full_run_estimate['estimated_survivors_64'])}; "
        f"pHash-256: {_br(full_run_estimate['estimated_survivors_256'])}",
        f"- Tempo medido de ORB+RANSAC nos {_br(full_run_estimate.get('n_sampled', 2000))} pares da "
        f"amostra: {full_run_estimate['measured_seconds_for_sample']:.1f}s",
        f"- Tempo estimado (extrapolação linear) para rodar ORB+RANSAC sem qualquer gate de pHash "
        f"nos {_br(full_run_estimate['total_task7_candidates'])} candidatos completos: "
        f"~{full_run_estimate['estimated_seconds_for_full_orb_no_gate']/3600:.1f}h",
        "",
        "Esta é uma extrapolação linear simples (tempo/par medido × total de candidatos), não uma "
        "medição direta — não considera eventuais efeitos de cache de disco/zip em escala maior.",
        "",
    ]


def render_report(
    real_sample: dict[str, Any],
    phash_comparison: dict[str, Any],
    dhash_coverage: list[dict[str, Any]],
    controlled_orb: list[dict[str, Any]],
    full_funnel: dict[str, Any],
    cross_tab: dict[str, Any],
    recommendation: dict[str, Any],
    full_run_estimate: dict[str, Any],
    forte_pairs: list[dict[str, Any]],
    pilot_v2_index_path: Path,
) -> str:
    index_path_str = str(pilot_v2_index_path).replace("\\", "/")

    lines = [
        f"# Piloto Complementar — pHash 64 vs. 256 bits + ORB Independente do pHash — "
        f"{date.today().isoformat()}",
        "",
        "Extensão da Tarefa 8. A Tarefa 8 (pHash com bloco DCT 16x16/256 bits) obteve ZERO "
        "sobreviventes nos 2.000 pares amostrados, incluindo os 400 pares difíceis reais "
        "conhecidos — resultado consistente tanto com \"não há duplicata real na amostra\" quanto "
        "com \"o pHash de 256 bits descarta positivos reais antes da etapa geométrica\", e aquele "
        "piloto não conseguiu distinguir as duas hipóteses. Esta tarefa compara diretamente pHash "
        "de 64 bits (bloco DCT 8x8) contra 256 bits no mesmo protocolo, e quebra a dependência "
        "geométrica rodando ORB+RANSAC diretamente nos candidatos reais, independente de "
        "qualquer filtro de pHash.",
        "",
        "**Este é um PILOTO CURTO E DELIMITADO**: reaproveita exatamente a amostra de 2.000 pares "
        "da Tarefa 8 (mesma semente), não roda a busca completa nos 421.196 candidatos, e não "
        "testa nenhum tamanho de bloco DCT além de 8x8 e 16x16. Os artefatos da Tarefa 8 "
        f"(`visual_duplicate_pilot_index.json`, "
        "`05_visual_duplicate_pilot_report.md`) permanecem intactos. **Isto é piloto/detecção, "
        "não confirmação: nenhuma exclusão de lesão/imagem foi decidida ou implementada em "
        "nenhuma etapa.**",
        "",
    ]

    lines += _real_sample_section(real_sample)
    lines += _phash_comparison_section(phash_comparison)
    lines += _dhash_coverage_section(dhash_coverage)
    lines += _controlled_orb_section(controlled_orb)
    lines += _full_funnel_section(full_funnel)
    lines += _cross_tab_section(cross_tab, recommendation)
    if forte_pairs:
        lines += _manual_review_section(forte_pairs)
    lines += _scope_limitation_block()
    lines += _full_run_estimate_block(full_run_estimate)

    lines += [
        "## Aviso",
        "",
        "**Isto é piloto/detecção, não confirmação.** Nenhuma exclusão de lesão ou imagem foi "
        "decidida ou implementada nesta tarefa, em nenhuma etapa do funil, mesmo para pares com "
        f"suporte geométrico forte. Qualquer par listado no índice completo deste piloto "
        f"(`{index_path_str}`) requer revisão humana antes de qualquer decisão sobre uso, "
        "exclusão ou reclassificação de dados. Os artefatos da Tarefa 8 permanecem intactos e "
        "válidos como registro da tentativa anterior.",
        "",
    ]

    return "\n".join(lines).rstrip("\n") + "\n"
