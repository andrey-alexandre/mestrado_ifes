"""Render the Markdown report for the visual-duplicate candidate audit
(task 7). Entirely code-generated — including the controlled-test numbers
and the threshold-derivation prose — so no section needs a manual edit that
a future rerun could silently erase (the fragility task 5 had to fix twice
for the PH2 note, see ``report.py``/``_render_ph2`` for that precedent).
"""
from __future__ import annotations

from datetime import date
from pathlib import Path
from typing import Any

from . import visual_duplicates as vd

_VARIANT_LABELS = {
    "identical_copy": "Cópia idêntica",
    "jpeg_recompress_q40": "Recompressão JPEG (qualidade 40)",
    "resize_50pct": "Redimensionamento (50%)",
    "crop_5pct_border": "Recorte pequeno (5% da borda)",
    "rotate_90": "Rotação 90°",
    "mirror_horizontal": "Espelhamento horizontal",
    "crop_30pct_border": "Recorte significativo (30% da borda)",
}

def _coverage_line(label: str, dhash_result: dict[str, Any]) -> str:
    return (
        f"- **{label}**: expected={dhash_result['expected']}, "
        f"processed={dhash_result['processed']}, "
        f"unreadable={len(dhash_result['unreadable'])}"
    )


def _controlled_variants_table(controlled_variants: list[dict[str, Any]], threshold: int) -> list[str]:
    lines = [
        "| Variante | Distância de Hamming observada | Candidato (distância ≤ "
        f"{threshold})? |",
        "|---|---|---|",
    ]
    by_name = {v["variant"]: v for v in controlled_variants}
    ordered_names = [
        "identical_copy",
        "jpeg_recompress_q40",
        "resize_50pct",
        "crop_5pct_border",
        "rotate_90",
        "mirror_horizontal",
        "crop_30pct_border",
    ]
    for name in ordered_names:
        v = by_name[name]
        label = _VARIANT_LABELS[name]
        candidato = "sim" if v["is_candidate"] else "não"
        lines.append(f"| {label} | {v['distance']} | {candidato} |")
    lines.append("")
    return lines


def _pair_section(result: dict[str, Any], index_path: str) -> list[str]:
    n_a, n_b = result["n_a"], result["n_b"]
    expected_random = vd.expected_random_candidate_count(n_a, n_b, threshold=result["threshold"])
    lines = [
        f"## {result['pair']}",
        "",
        f"- Imagens comparadas: {n_a} x {n_b} ({n_a * n_b:,} pares possíveis)".replace(",", "."),
        f"- Limiar de distância usado: {result['threshold']}",
        f"- **Total de candidatos encontrados: {result['total_candidates']}**",
        f"- Contagem esperada sob um modelo nulo de hashes uniformemente aleatórios e "
        f"independentes (ou seja, SE as imagens não compartilhassem nenhuma estrutura): "
        f"~{expected_random:.2f}",
        f"- Imagens lado a lado renderizadas: {result['rendered_count']} "
        f"(as de menor distância primeiro; limite por par: {result['max_rendered_per_pair']})",
        f"- Pasta das imagens lado a lado: `{result['candidates_dir']}`",
        f"- Índice completo (todos os {result['total_candidates']} candidatos, "
        f"renderizados ou não): `{index_path}`",
        "",
    ]
    if result["total_candidates"] > result["max_rendered_per_pair"]:
        lines.append(
            f"**Nota**: {result['total_candidates']} candidatos encontrados excede o limite de "
            f"renderização ({result['max_rendered_per_pair']}); apenas os "
            f"{result['rendered_count']} de menor distância foram renderizados como imagem — a "
            "contagem total real está acima, não escondida pela amostra, e está disponível "
            "completa no índice."
        )
        lines.append("")
    lines.append(
        f"O total observado ({result['total_candidates']}) é muitas ordens de grandeza maior "
        f"do que o esperado sob o modelo nulo (~{expected_random:.2f}) — ver seção "
        "\"Achado real\" abaixo para a interpretação disso."
    )
    lines.append("")
    return lines


def render_report(
    threshold: int,
    controlled_variants: list[dict[str, Any]],
    ham_dhash: dict[str, Any],
    isic_dhash: dict[str, Any],
    bcn_dhash: dict[str, Any],
    pair_results: list[dict[str, Any]],
    index_path: Path,
) -> str:
    index_path_str = str(index_path).replace("\\", "/")

    by_name = {v["variant"]: v["distance"] for v in controlled_variants}
    reprocessing_names = ["identical_copy", "jpeg_recompress_q40", "resize_50pct", "crop_5pct_border"]
    structural_names = ["rotate_90", "mirror_horizontal", "crop_30pct_border"]
    reprocessing_max = max(by_name[n] for n in reprocessing_names)
    structural_min = min(by_name[n] for n in structural_names)

    lines = [
        f"# Candidatos a Duplicata Visual entre Datasets — {date.today().isoformat()}",
        "",
        "Extensão da detecção de sobreposição da Tarefa 6: SHA-256 só encontra arquivos "
        "byte-idênticos, então a mesma foto de lesão reenviada com resolução, recorte, "
        "rotação, espelhamento ou compressão diferentes não é detectada por hash "
        "criptográfico. Esta tarefa implementa a detecção de **candidatos** a essa "
        "duplicata visual via hash perceptual (dHash). "
        "**Isto é detecção, não confirmação: nenhuma exclusão de lesão/imagem foi decidida "
        "ou implementada aqui. Todo candidato listado requer revisão humana antes de "
        "qualquer decisão sobre uso dos dados.**",
        "",
        "## Método",
        "",
        "**dHash (difference hash)**: a imagem é decodificada, convertida para escala de "
        "cinza e redimensionada para 9×8 pixels; cada bit do hash de 64 bits compara um "
        "pixel com seu vizinho horizontal (bit 1 se o pixel atual é mais claro que o "
        "próximo). Esse método foi escolhido porque não exige treinar nem carregar nenhum "
        "modelo (usa só Pillow+numpy, já disponíveis no ambiente), é determinístico, e é "
        "tolerante a exatamente o tipo de reprocessamento esperado entre releases do mesmo "
        "dataset (recompressão, redimensionamento, recorte pequeno) — porque codifica "
        "gradientes grosseiros de brilho, não valores exatos de pixel. Em contrapartida, "
        "não é tolerante a rotação, espelhamento ou recorte significativo, porque essas "
        "transformações mudam quais gradientes ficam adjacentes a quais — ver seção de "
        "limitações abaixo, com números medidos, não apenas teoria.",
        "",
        f"**Limiar de candidatura escolhido: distância de Hamming ≤ {threshold}** (de 64 "
        "bits). Derivado dos testes controlados abaixo: variantes que são genuinamente a "
        "mesma imagem reprocessada (cópia idêntica, recompressão JPEG, redimensionamento, "
        f"recorte pequeno de 5% da borda) mediram distância máxima {reprocessing_max}; "
        "variantes com alteração estrutural (rotação 90°, espelhamento horizontal, recorte "
        f"significativo de 30% da borda) mediram distância mínima {structural_min}. O limiar "
        f"escolhido ({threshold}) fica "
        + (
            "estritamente entre os dois agrupamentos medidos, com margem de "
            f"{threshold - reprocessing_max} abaixo do caso real mais alto e margem de "
            f"{structural_min - threshold} acima do caso estrutural mais baixo"
            if reprocessing_max < threshold < structural_min
            else "FORA do intervalo estritamente entre os dois agrupamentos medidos "
            f"(reprocessamento real até {reprocessing_max}, alteração estrutural a partir de "
            f"{structural_min}) — reavaliar este limiar antes de confiar nos resultados"
        )
        + " — não foi escolhido arbitrariamente.",
        "",
        "## Testes controlados (exemplos sintéticos, não imagens de paciente)",
        "",
        "Base: uma imagem sintética (gradiente + formas geométricas, gerada via "
        "`PIL.ImageDraw`) e um conjunto fixo de variantes dela. Números medidos pela mesma "
        "função usada pelos testes automatizados "
        "(`src/data_audit/visual_duplicates.py::measure_controlled_variant_distances`), não "
        "copiados manualmente:",
        "",
    ]
    lines += _controlled_variants_table(controlled_variants, threshold)

    lines += [
        "## Limitações explícitas (confirmadas pelos testes acima, não apenas em teoria)",
        "",
        "- **Sensível a rotação**: uma rotação de 90° mediu distância "
        f"{next(v['distance'] for v in controlled_variants if v['variant'] == 'rotate_90')} "
        f"(acima do limiar {threshold}) — **não detectada** como candidato por este método.",
        "- **Sensível a espelhamento**: um espelhamento horizontal mediu distância "
        f"{next(v['distance'] for v in controlled_variants if v['variant'] == 'mirror_horizontal')} "
        f"(acima do limiar {threshold}) — **não detectada** como candidato por este método.",
        "- **Sensível a recorte significativo**: um recorte de 30% da borda mediu distância "
        f"{next(v['distance'] for v in controlled_variants if v['variant'] == 'crop_30pct_border')} "
        f"(acima do limiar {threshold}) — **não detectada** como candidato por este método.",
        "- Um candidato com distância baixa não é uma duplicata confirmada: imagens "
        "diferentes, mas visualmente simples/parecidas (ex.: lesão redonda centrada em pele "
        "lisa) podem coincidir por acaso. Cada candidato requer inspeção humana da imagem "
        "lado a lado antes de qualquer conclusão.",
        "",
        "## Cobertura por dataset (dHash)",
        "",
        _coverage_line("HAM10000", ham_dhash),
        _coverage_line("ISIC2018 Task1", isic_dhash),
        _coverage_line("BCN20000 (treino+teste)", bcn_dhash),
        "",
    ]

    for result in pair_results:
        lines += _pair_section(result, index_path_str)

    lines += [
        "## Achado real: alta taxa de falso-positivo em imagens dermatoscópicas",
        "",
        "A contagem observada de candidatos em ambos os pares é muitas ordens de grandeza "
        "maior do que a esperada sob um modelo nulo de hashes uniformemente aleatórios (ver "
        "`Contagem esperada sob um modelo nulo` em cada seção acima). Isso não indica um bug "
        "— indica que imagens dermatoscópicas de datasets diferentes **não são visualmente "
        "aleatórias entre si**: elas compartilham fortemente a mesma convenção de aquisição "
        "(lesão pigmentada centralizada, fundo de pele razoavelmente uniforme, iluminação e "
        "enquadramento do dermatoscópio, por vezes régua/pelos), o que já basta para produzir "
        "um dHash (9×8, grosseiro) muito parecido entre imagens de **pacientes e lesões "
        "completamente diferentes**.",
        "",
        "Inspeção manual direta (local, nunca publicada — ver restrições desta tarefa) dos "
        "candidatos de menor distância possível (distância 0, o caso mais forte que este "
        "método pode produzir) confirmou isso: em "
        "`experiments/data_audit/visual_candidates/ham10000_x_isic2018_task1/00000.png` "
        "(`ISIC_0026029.jpg` x "
        "`ISIC2018_Task1-2_Training_Input/ISIC_0010220.jpg`) e em "
        "`experiments/data_audit/visual_candidates/ham10000_x_bcn20000/00000.png` "
        "(`ISIC_0025653.jpg` x `bcn_20k_test/BCN_0000005436.jpg`), as duas imagens de cada par "
        "mostram claramente lesões diferentes — não a mesma imagem reenviada. A semelhança é "
        "de gênero fotográfico, não de conteúdo.",
        "",
        "**Implicação prática**: o volume de candidatos (incluindo os de distância 0) não deve "
        "ser lido como \"dezenas de milhares de duplicatas prováveis\". É a taxa de "
        "falso-positivo esperada deste método de baixa resolução aplicado a um domínio "
        "visualmente homogêneo — e reforça, em vez de enfraquecer, a exigência de que cada "
        "candidato seja revisado individualmente por um humano antes de qualquer conclusão "
        "sobre duplicata real.",
        "",
        "## Aviso",
        "",
        "**Isto é detecção, não confirmação.** Nenhuma exclusão de lesão ou imagem foi "
        "decidida ou implementada nesta tarefa, para nenhum candidato, mesmo os de "
        "distância muito baixa. Qualquer candidato listado no índice completo "
        f"(`{index_path_str}`) requer revisão humana — da imagem lado a lado e do contexto "
        "de cada dataset — antes de qualquer decisão sobre uso, exclusão ou reclassificação "
        "de dados.",
        "",
    ]

    return "\n".join(lines).rstrip("\n") + "\n"
