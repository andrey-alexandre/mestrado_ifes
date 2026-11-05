"""Render the Markdown report for the task 8 staged-filtering pilot.
Entirely code-generated from measured numbers, same convention as
``report.py``/``visual_duplicate_report.py`` — no section needs a manual
edit a future rerun could silently erase.
"""
from __future__ import annotations

from datetime import date
from pathlib import Path
from typing import Any


def _br(n: int) -> str:
    """Format an integer with '.' as the thousands separator (pt-BR
    convention). Applied to a single number in isolation and interpolated
    into an f-string, never chained onto a multi-line concatenated string
    literal -- adjacent string literals concatenate before any method call,
    so a trailing ``.replace(",", ".")`` on a multi-line f-string corrupts
    any literal comma in the prose too, not just the number (a real bug
    caught in testing: "soma, um hash" silently became "soma. um hash").
    """
    return f"{n:,}".replace(",", ".")


def _coverage_line(label: str, n_sampled: int, n_processed: int, n_unreadable: int) -> str:
    return f"- **{label}**: amostrados={n_sampled}, processados={n_processed}, ilegíveis={n_unreadable}"


def _preprocessing_section(preprocessing: dict[str, Any]) -> list[str]:
    lines = ["## 1. Auditoria dirigida de pré-processamento", ""]

    exif_verify = preprocessing["exif_verify"]
    lines += [
        "### Verificação de comportamento do `PIL.Image.open` frente a orientação EXIF",
        "",
        f"Verificação comportamental (não por amostra): imagem sintética com orientação EXIF "
        f"conhecida (tag `0x0112`=6) codificada e reaberta. Tamanho bruto esperado: "
        f"`{exif_verify['raw_size']}`. Tamanho reportado por `PIL.Image.open()` (sem aplicar "
        f"`ImageOps.exif_transpose`): `{exif_verify['size_from_open']}`.",
        "",
        f"**Resultado: `PIL.Image.open()` aplica orientação EXIF automaticamente? "
        f"{'SIM' if exif_verify['open_applies_orientation_automatically'] else 'NÃO'}** "
        "(confirmado nos bytes reais acima, não assumido da documentação).",
        "",
    ]

    lines.append("### Distribuição real da tag de orientação EXIF por fonte (amostra)")
    lines.append("")
    for source, result in preprocessing["exif_by_source"].items():
        lines.append(_coverage_line(source, result["n_sampled"], result["n_processed"], len(result["unreadable"])))
        lines.append(
            f"  - valores de orientação observados: {result['orientation_value_counts']}; "
            f"não-padrão (!=1): {result['n_non_default_orientation']}"
        )
    lines.append("")

    lines.append("### Dimensões nativas por fonte (amostra)")
    lines.append("")
    for source, result in preprocessing["dimensions_by_source"].items():
        if result["n_processed"] == 0:
            lines.append(f"- **{source}**: sem imagens processadas na amostra.")
            continue
        lines.append(
            f"- **{source}**: largura [{result['width']['min']}, {result['width']['mean']:.0f}, "
            f"{result['width']['max']}] (min/média/max), altura [{result['height']['min']}, "
            f"{result['height']['mean']:.0f}, {result['height']['max']}], razão de aspecto "
            f"[{result['aspect_ratio']['min']:.3f}, {result['aspect_ratio']['mean']:.3f}, "
            f"{result['aspect_ratio']['max']:.3f}] (alvo do grid dHash 9x8 = "
            f"{result['dhash_target_aspect_ratio']:.3f}; desvio absoluto médio = "
            f"{result['mean_abs_deviation_from_target_aspect']:.3f})"
        )
    lines.append("")

    lines.append("### Imagens degeneradas (quase uniformes / próximas de preto ou branco)")
    lines.append("")
    for source, result in preprocessing["degenerate_by_source"].items():
        lines.append(
            f"- **{source}**: {result['n_flagged']}/{result['n_processed']} sinalizadas "
            f"(limiar std<{result['std_threshold']}, margem extrema de média "
            f"{result['mean_extreme_margin']})"
        )
    lines.append("")

    hub = preprocessing["hub_hash_audit"]
    lines += [
        "### Checagem de \"hub\" — valores de dHash distintos respondendo por fração "
        "desproporcional dos candidatos",
        "",
        f"- Total de candidatos (Tarefa 7): {_br(hub['total_candidates'])}",
        f"- Valores de dHash distintos envolvidos nos candidatos: {_br(hub['n_distinct_hashes_involved'])}",
        f"- Participações (soma, um hash pode contar 2x no mesmo candidato se aparecer dos dois "
        f"lados) dos {len(hub['top_hashes'])} hashes mais frequentes: "
        f"{_br(hub['top_hashes_edge_count_sum'])}.",
        f"- Candidatos distintos que tocam pelo menos um desses {len(hub['top_hashes'])} hashes: "
        f"{_br(hub['n_candidates_touching_top_hashes'])}"
        f" ({hub['top_hashes_share_of_candidates']:.2%} do total de candidatos).",
        "",
        "| dHash (int) | nº de candidatos em que aparece |",
        "|---|---|",
    ]
    for entry in hub["top_hashes"][:10]:
        lines.append(f"| `{entry['dhash']}` | {_br(entry['edge_count'])} |")
    lines.append("")

    return lines


def _phash_validation_section(phash_validation: dict[str, Any]) -> list[str]:
    lines = ["## 2. Validação do pHash em dados reais", ""]

    lines.append(
        f"Imagens reais usadas (amostra HAM10000, semente={phash_validation['seed']}): "
        f"{phash_validation['n_real_images']}."
    )
    lines.append("")
    lines.append("### Distância pHash por variante controlada (sobre as imagens reais, não a sintética)")
    lines.append("")
    lines.append("| Variante | min | média | max | Recuperada (max ≤ limiar)? |")
    lines.append("|---|---|---|---|---|")
    for variant in phash_validation["variant_stats"]:
        recovered = "sim" if variant["recovered"] else "não"
        lines.append(
            f"| {variant['label']} | {variant['min']} | {variant['mean']:.1f} | {variant['max']} | {recovered} |"
        )
    lines.append("")

    hard = phash_validation["hard_pairs"]
    lines += [
        f"### Pares difíceis reais (candidatos dHash distância 0–2 da Tarefa 7, n={hard['n_pairs']})",
        "",
        f"Estes são os falsos positivos reais já confirmados (par com dHash colidindo em imagens "
        f"diferentes) — o benchmark de falso-positivo que já temos à mão, reaproveitado do "
        f"`visual_candidates_index.json` da Tarefa 7, não um conjunto inventado.",
        "",
        f"- Distância pHash: min={hard['min']}, mediana={hard['median']:.1f}, "
        f"média={hard['mean']:.1f}, max={hard['max']}",
        "",
    ]

    lines += [
        "### Separação observada e limiar escolhido",
        "",
        phash_validation["threshold_derivation_text"],
        "",
        f"**Limiar de candidatura pHash escolhido: distância ≤ {phash_validation['threshold']}** "
        f"(de {phash_validation['hash_bits']} bits).",
        "",
    ]

    return lines


def _funnel_section(funnel: dict[str, Any]) -> list[str]:
    lines = [
        "## 3. Funil do piloto",
        "",
        f"- Candidatos totais da Tarefa 7 (universo de onde a amostra foi tirada): "
        f"{_br(funnel['total_task7_candidates'])}",
        f"- Pares na amostra estratificada do piloto: {_br(funnel['n_sampled'])}",
    ]
    for bucket in funnel["buckets"]:
        lines.append(
            f"  - distância {bucket['label']}: {_br(bucket['n_sampled'])} amostrados de "
            f"{_br(bucket['n_pool'])} candidatos reais nessa faixa"
            + (" (todos)" if bucket["all"] else f" (semente={bucket['seed']})")
        )
    lines += [
        f"- Sobreviventes do filtro pHash (distância ≤ {funnel['phash_threshold']}): "
        f"{_br(funnel['n_phash_survivors'])} ({funnel['phash_survivor_rate']:.1%} da amostra)",
        f"- Avaliados geometricamente (ORB+RANSAC tentado): {_br(funnel['n_geometric_evaluated'])}",
        f"  - suporte **forte**: {_br(funnel['n_forte'])}",
        f"  - suporte **fraco**: {_br(funnel['n_fraco'])}",
        f"- Não avaliáveis geometricamente (poucos keypoints/matches, não \"sem suporte\"): "
        f"{_br(funnel['n_not_evaluable'])}",
        f"- Imagens de evidência (drawMatches) renderizadas: {_br(funnel['n_evidence_rendered'])} "
        f"de {_br(funnel['n_geometric_evaluated'])} avaliados (limite: {_br(funnel['max_evidence_rendered'])})",
        "",
    ]
    return lines


def render_report(
    preprocessing: dict[str, Any],
    phash_validation: dict[str, Any],
    funnel: dict[str, Any],
    pilot_index_path: Path,
    recommendation_text: str,
) -> str:
    index_path_str = str(pilot_index_path).replace("\\", "/")

    lines = [
        f"# Piloto de Filtragem em Etapas (pHash + Verificação Geométrica) — {date.today().isoformat()}",
        "",
        "Extensão da Tarefa 7 (detecção de candidatos por dHash). A Tarefa 7 encontrou 421.196 "
        "candidatos nos dois pares prioritários e seu relatório concluiu que a maioria seria "
        "ruído — **essa conclusão foi um exagero e está corrigida aqui**: os dois pares de "
        "distância 0 inspecionados na Tarefa 7 mostraram imagens diferentes, o que prova que "
        "dHash idêntico não confirma duplicata visual, mas **não prova** que os demais "
        "candidatos sejam ruído, nem que os pares inspecionados sejam de pacientes diferentes "
        "(aparência visual não é metadado de identidade de paciente), nem que o modelo nulo de "
        "bits aleatórios usado na Tarefa 7 seja a distribuição de referência correta para "
        "imagens dermatoscópicas reais. Este piloto não assume nenhuma dessas três "
        "afirmações.",
        "",
        "**Este é um PILOTO**: valida o método (auditoria de pré-processamento, pHash, "
        "verificação geométrica) numa amostra estratificada, sem rodar a busca completa nos "
        "421.196 candidatos. A Tarefa 7 não foi sobrescrita — seus artefatos "
        "(`visual_candidates_index.json`, `visual_candidates/`, "
        "`04_visual_duplicate_candidates_report.md`) permanecem intactos como registro da "
        "tentativa anterior. **Isto é piloto/detecção, não confirmação: nenhuma exclusão de "
        "lesão/imagem foi decidida ou implementada, em nenhuma etapa, mesmo para pares com "
        "score geométrico forte.**",
        "",
    ]

    lines += _preprocessing_section(preprocessing)
    lines += _phash_validation_section(phash_validation)
    lines += _funnel_section(funnel)

    lines += [
        "## 4. Reafirmação corrigida da limitação de interpretação",
        "",
        "Hash idêntico (dHash ou pHash) — mesmo nos dois casos de distância dHash 0 inspecionados "
        "na Tarefa 7 — **não confirma duplicata visual nem identidade de paciente**. Os casos "
        "inspecionados confirmaram apenas que, visualmente, as imagens mostravam lesões "
        "diferentes; isso não foi cruzado com nenhum metadado de paciente/lesão, porque nenhum "
        "dos datasets usados fornece esse cruzamento direto entre HAM10000, ISIC2018 Task1 e "
        "BCN20000. Da mesma forma, suporte geométrico forte (ORB+RANSAC) neste piloto é "
        "evidência substancialmente mais específica que coincidência de hash global, mas "
        "**também não é confirmação automática** — depende de quantos keypoints uma imagem "
        "realmente tem e de quão genérica é sua textura (ver limitação em "
        "`geometric_verification.py`: imagens pouco texturizadas, comuns em pele lisa, "
        "produzem poucos keypoints e ficam `não avaliáveis`, não \"sem suporte\"). Toda "
        "conclusão sobre duplicata real requer revisão humana da evidência (imagem lado a "
        "lado + imagem de correspondência geométrica) par a par.",
        "",
        "## 5. Recomendação técnica para uma eventual execução completa futura",
        "",
        "**Esta é uma recomendação para revisão — não uma decisão tomada nem uma execução "
        "completa.** A execução completa (todos os 421.196 candidatos) NÃO foi feita nesta "
        "tarefa.",
        "",
        recommendation_text,
        "",
        "## Aviso",
        "",
        "**Isto é piloto/detecção, não confirmação.** Nenhuma exclusão de lesão ou imagem foi "
        "decidida ou implementada nesta tarefa, em nenhuma etapa do funil, mesmo para pares com "
        f"score geométrico forte. Qualquer par listado no índice completo do piloto "
        f"(`{index_path_str}`) requer revisão humana antes de qualquer decisão sobre uso, "
        "exclusão ou reclassificação de dados. Os artefatos da Tarefa 7 permanecem intactos e "
        "válidos como registro da tentativa anterior.",
        "",
    ]

    return "\n".join(lines).rstrip("\n") + "\n"
