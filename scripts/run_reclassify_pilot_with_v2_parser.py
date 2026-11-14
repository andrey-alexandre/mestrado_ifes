"""Reclassificação OFFLINE de um piloto local de VLM já executado, usando
o classificador NOVO e separadamente versionado
`src.evaluation.case_status_v2.classify_case_status_v2`, lado a lado com
o classificador v1 histórico (`src.evaluation.case_status.classify_case_status`,
intocado).

Motivação: ver `src/evaluation/case_status_v2.py` e
`docs/reconstruction/20_local_vlm_pilot.md`. Resumo: o piloto real de
`qwen3-vl:8b-instruct-q4_K_M` (Estágio 1) produziu uma resposta de síntese
para o cenário `um_parecer_ausente` que o v1 classificou
`"nao_reconhecido"` porque o modelo nunca escreveu a frase literal "lesão
benigna" (usou "Lesão Cutânea Benigna" no meio do texto e "BENIGNO"
isolado no veredito final). O pesquisador pediu um classificador novo,
NUNCA uma modificação do v1 (usado em avaliações históricas/stage6), e
pediu que esse novo classificador fosse reaplicado às saídas BRUTAS já
coletadas -- SEM REPETIR CHAMADAS AO MODELO. Este script faz exatamente
isso.

REGRAS DURAS (herdadas de `feedback_no_autonomous_execution` e da
instrução explícita do pesquisador "mantenha as avaliações históricas
intactas"):
  - NUNCA faz uma nova chamada ao modelo/Ollama -- lê só o
    `call_log.jsonl` (e, opcionalmente, o `report.md`) já gravados por
    uma execução anterior de `scripts/run_local_vlm_pilot.py`.
  - NUNCA sobrescreve/muta `call_log.jsonl`, `call_log_matriz_principal.jsonl`,
    `summary.json` ou `report.md` -- escreve SEMPRE em arquivos novos
    (`reclassification_v2_report.md` / `reclassification_v2_results.json`
    no mesmo `--output-dir`).
  - Este script também não é executado pelo assistente que o escreveu --
    só o pesquisador roda, na própria máquina, depois de ter completado
    pelo menos o Estágio 1 (controles) de um modelo (basta existir
    `call_log.jsonl`; a matriz principal, `call_log_matriz_principal.jsonl`,
    é opcional e lida também se existir).

Uso:
    python scripts/run_reclassify_pilot_with_v2_parser.py \
        --output-dir experiments/multiagent/local_vlm_pilot/qwen3-vl_8b-instruct-q4_K_M

ENTREGA 2026-11-14 (correção de bug real achado na primeira execução real
do pesquisador, ver docs/reconstruction/20_local_vlm_pilot.md §9b e a
docstring de `reclassify_record` abaixo): este script, antes desta
correção, chamava `classify_case_status`/`classify_case_status_v2` em
CIMA do `raw_response_text` de QUALQUER registro, inclusive os 12
registros de CONTROLE DE ENVIO DE IMAGEM (`role` começando com
`"controle_imagem_"`) -- que nunca têm `classified_status` aplicável por
design (são controles de quadrante/cor do caminho multimodal, não
pareceres diagnósticos). Isso produzia `"nao_reconhecido"` espúrio para
os 12 e uma falsa contagem de "v1 recomputado difere do histórico"
(histórico real = `None` por design, não um bug de `case_status.py`).
Corrigido: estes 12 registros são agora excluídos da reclassificação e
reportados com uma nota dedicada, nunca contados nos contadores de
divergência. Este script também foi reaplicado (parser v2 atualizado,
`PARSER_VERSION` novo -- ver `src/evaluation/case_status_v2.py`) para a
extensão do rótulo POSITIVO ("suspeita ... melanoma" com
intensificadores) -- nenhuma mudança de código NESTE arquivo foi
necessária para essa parte: o script já lê `PARSER_VERSION`/
`classify_case_status_v2_detailed` por import, então a nova versão do
parser é usada automaticamente na próxima execução.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

_PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

from src.evaluation.case_status import classify_case_status  # noqa: E402
from src.evaluation.case_status_v2 import (  # noqa: E402
    PARSER_VERSION,
    classify_case_status_v2_detailed,
)

CALL_LOG_FILENAMES = ("call_log.jsonl", "call_log_matriz_principal.jsonl")


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    with open(path, "r", encoding="utf-8") as f:
        for line_no, line in enumerate(f, start=1):
            line = line.strip()
            if not line:
                continue
            try:
                records.append(json.loads(line))
            except json.JSONDecodeError as exc:
                raise ValueError(f"{path}:{line_no}: linha JSONL inválida -- {exc}") from exc
    return records


def _load_all_call_log_records(output_dir: Path) -> list[tuple[str, dict[str, Any]]]:
    """Devolve ``(nome_do_arquivo_fonte, record)`` para cada linha lida --
    NUNCA muta/relê os arquivos como fonte de verdade para nada além desta
    leitura; nenhuma escrita acontece nestes arquivos."""
    found: list[tuple[str, dict[str, Any]]] = []
    any_found = False
    for filename in CALL_LOG_FILENAMES:
        path = output_dir / filename
        if not path.exists():
            continue
        any_found = True
        for record in _read_jsonl(path):
            found.append((filename, record))
    if not any_found:
        raise FileNotFoundError(
            f"Nenhum de {CALL_LOG_FILENAMES} encontrado em {output_dir} -- "
            "rode primeiro scripts/run_local_vlm_pilot.py (Estágio 1, pelo "
            "menos) para este modelo antes de reclassificar."
        )
    return found


def reclassify_record(source_filename: str, record: dict[str, Any]) -> dict[str, Any]:
    """Recomputa AMBAS as classificações (v1 e v2) a partir do
    ``raw_response_text`` já salvo -- nunca faz uma nova chamada. Para
    registros sem ``raw_response_text`` (ex.: ``nao_executado``,
    controles de envio de imagem que não têm ``classified_status``
    aplicável), ambas as classificações são reportadas como ``None`` com
    uma nota explícita, nunca inventadas.

    BUG REAL CORRIGIDO (2026-11-14, achado na primeira execução real do
    pesquisador): registros de CONTROLE DE ENVIO DE IMAGEM (``role``
    começando com ``"controle_imagem_"`` -- ``controle_imagem_vlm_direto``/
    ``controle_imagem_diagnostic_abcd``/``controle_imagem_diagnostic_menzies``/
    ``controle_imagem_diagnostic_spcl``) TÊM ``raw_response_text`` (ex.:
    ``"superior-esquerdo, vermelho"`` ou a resposta do controle "sem
    imagem") mas são controles de QUADRANTE/COR do caminho multimodal,
    nunca pareceres diagnósticos -- confirmado por leitura de
    `src/multiagent/vlm_direct.py::run_image_sending_control_call`, que
    grava ``classified_status=None`` EXPLICITAMENTE para estes registros
    (nunca chama `classify_case_status`/`classify_case_status_v2`).
    A versão anterior deste script ignorava o ``role`` e chamava os dois
    classificadores em CIMA do texto bruto de TODOS os registros com
    ``raw_response_text`` -- produzindo um ``"nao_reconhecido"`` espúrio
    para os 12 controles de imagem e, pior, comparando esse
    ``"nao_reconhecido"`` recomputado contra o histórico real ``None``,
    reportando uma falsa "v1 recomputado difere do histórico" para os 12
    (quando na verdade v1 nunca deveria ter sido recomputado para eles --
    não é evidência de que `case_status.py` mudou). Corrigido: estes
    registros são explicitamente excluídos de ``applicable`` abaixo,
    marcados com uma nota dedicada, e NUNCA contam no
    `n_v1_recomputed_differs_from_historical_log` nem no
    `n_v1_v2_disagree`."""
    raw_text = record.get("raw_response_text")
    v1_recorded = record.get("classified_status")  # já estava salvo no log original (histórico, intocado)
    role = record.get("role") or ""
    is_image_sending_control = role.startswith("controle_imagem_")

    applicable = (
        raw_text is not None
        and record.get("call_status") != "nao_executado"
        and not is_image_sending_control
    )

    if not applicable:
        v1_recomputed = None
        v2_result = None
        v2_status = None
        if is_image_sending_control:
            note = (
                f"não aplicável -- controle de envio de imagem (role='{role}'), "
                "nunca classificado por A34: run_image_sending_control_call grava "
                "classified_status=None por design (controle de quadrante/cor do "
                "caminho multimodal, não um parecer diagnóstico). Nenhuma "
                "reclassificação v1/v2 tentada; excluído de ambos os contadores "
                "de divergência do relatório."
            )
        else:
            note = "sem raw_response_text aplicável (ex.: nao_executado, ou papel sem campo de diagnóstico classificável) -- nenhuma reclassificação tentada."
    else:
        v1_recomputed = classify_case_status({"validation": raw_text})
        v2_result = classify_case_status_v2_detailed({"validation": raw_text})
        v2_status = v2_result.status
        note = ""

    v1_matches_historical = (v1_recomputed == v1_recorded) if applicable else None

    return {
        "source_file": source_filename,
        "role": record.get("role"),
        "condition": record.get("condition"),
        "lesion_id": record.get("lesion_id"),
        "image_id": record.get("image_id"),
        "call_status": record.get("call_status"),
        "raw_response_text": raw_text,
        "v1_classified_status_historical": v1_recorded,
        "v1_classified_status_recomputed": v1_recomputed,
        "v1_recomputed_matches_historical_log": v1_matches_historical,
        "v2_classified_status": v2_status,
        "v2_parser_version": PARSER_VERSION,
        "v2_matched_variant": v2_result.matched_variant if v2_result is not None else None,
        "v2_matched_category": v2_result.matched_category if v2_result is not None else None,
        "v2_extraction_fallback_used": (
            v2_result.extraction.extraction_fallback_used
            if v2_result is not None and v2_result.extraction is not None else None
        ),
        "v2_extraction_region_text": (
            v2_result.extraction.region_text
            if v2_result is not None and v2_result.extraction is not None else None
        ),
        "v1_v2_agree": (v1_recomputed == v2_status) if applicable else None,
        "note": note,
    }


def build_reclassification_report(output_dir: Path) -> dict[str, Any]:
    source_records = _load_all_call_log_records(output_dir)
    reclassified = [reclassify_record(fname, rec) for fname, rec in source_records]

    n_applicable = sum(1 for r in reclassified if r["v2_classified_status"] is not None or r["v1_classified_status_recomputed"] is not None)
    n_disagree = sum(1 for r in reclassified if r["v1_v2_agree"] is False)
    n_historical_mismatch = sum(1 for r in reclassified if r["v1_recomputed_matches_historical_log"] is False)
    # Correção 2026-11-14 (ver docstring de `reclassify_record`): contado
    # separadamente para que o pesquisador veja explicitamente quantos
    # registros foram excluídos por serem controle de envio de imagem --
    # nunca silenciosamente miscontados como 0 por coincidência.
    n_image_sending_control_records = sum(
        1 for r in reclassified if (r.get("role") or "").startswith("controle_imagem_")
    )

    # Confirmação explícita pedida pelo pesquisador: o cenário real
    # `um_parecer_ausente` deste piloto deve mudar de nao_reconhecido (v1)
    # para negativo (v2), mantendo v1 inalterado.
    um_parecer_ausente_records = [
        r for r in reclassified
        if r.get("image_id") == "controle_um_parecer_ausente" and r.get("role") == "summary"
    ]

    return {
        "output_dir": str(output_dir),
        "parser_version_v2": PARSER_VERSION,
        "n_records_total": len(reclassified),
        "n_records_applicable": n_applicable,
        "n_image_sending_control_records_excluded": n_image_sending_control_records,
        "n_v1_v2_disagree": n_disagree,
        "n_v1_recomputed_differs_from_historical_log": n_historical_mismatch,
        "um_parecer_ausente_confirmation": um_parecer_ausente_records,
        "records": reclassified,
    }


def _format_record_markdown(r: dict[str, Any]) -> str:
    lines = [
        f"### {r['role']} / {r['condition']} / lesion={r['lesion_id']} / image={r['image_id']}",
        "",
        f"- Arquivo fonte: `{r['source_file']}`",
        f"- `call_status`: `{r['call_status']}`",
        f"- v1 histórico (já salvo no log original): `{r['v1_classified_status_historical']}`",
        f"- v1 recomputado agora (deve ser idêntico ao histórico -- confirma que v1 não foi alterado): `{r['v1_classified_status_recomputed']}`",
        f"- v1 recomputado == histórico? **{r['v1_recomputed_matches_historical_log']}**",
        f"- v2 (`{r['v2_parser_version']}`): `{r['v2_classified_status']}`",
        f"- v2 variante casada: `{r['v2_matched_variant']}` (categoria `{r['v2_matched_category']}`)",
        f"- v2 heurística de extração usada: `{r['v2_extraction_fallback_used']}`",
        f"- v1 e v2 concordam? **{r['v1_v2_agree']}**",
    ]
    if r["note"]:
        lines.append(f"- nota: {r['note']}")
    if r["v2_extraction_region_text"]:
        lines += ["", "Região de conclusão final extraída pelo v2 (texto literal):", "", "```", r["v2_extraction_region_text"], "```"]
    if r["raw_response_text"]:
        lines += ["", "<details><summary>Texto bruto completo (literal, do log original)</summary>", "", "```", r["raw_response_text"], "```", "</details>"]
    lines.append("")
    return "\n".join(lines)


def write_reclassification_report(output_dir: Path, report: dict[str, Any]) -> tuple[Path, Path]:
    json_path = output_dir / "reclassification_v2_results.json"
    md_path = output_dir / "reclassification_v2_report.md"

    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(report, f, ensure_ascii=False, indent=2, default=str)

    lines = [
        f"# Reclassificação offline com parser v2 -- `{report['output_dir']}`",
        "",
        "NUNCA repetiu chamadas ao modelo -- reprocessamento puramente "
        "offline do `call_log.jsonl`/`call_log_matriz_principal.jsonl` já "
        "existentes. `call_log.jsonl`, `call_log_matriz_principal.jsonl`, "
        "`summary.json` e `report.md` originais NÃO foram tocados por este "
        "script.",
        "",
        f"- Versão do parser v2: `{report['parser_version_v2']}`",
        f"- Registros totais lidos: {report['n_records_total']}",
        f"- Registros com classificação aplicável (v1 e/ou v2): {report['n_records_applicable']}",
        f"- Registros de controle de envio de imagem excluídos da classificação (nunca têm `classified_status` aplicável -- ver `run_image_sending_control_call`): {report['n_image_sending_control_records_excluded']}",
        f"- Divergências v1 vs v2 (só entre registros aplicáveis): {report['n_v1_v2_disagree']}",
        f"- v1 recomputado difere do histórico salvo (deveria ser SEMPRE 0 -- confirma que v1/case_status.py não foi alterado; controles de envio de imagem EXCLUÍDOS deste contador -- correção 2026-11-14, ver `reclassify_record`): {report['n_v1_recomputed_differs_from_historical_log']}",
        "",
        "## Confirmação explícita -- cenário real `um_parecer_ausente`",
        "",
    ]
    if report["um_parecer_ausente_confirmation"]:
        for r in report["um_parecer_ausente_confirmation"]:
            lines.append(_format_record_markdown(r))
    else:
        lines.append("(nenhum registro `controle_um_parecer_ausente`/`summary` encontrado neste `--output-dir`.)")
        lines.append("")

    lines += ["## Todos os registros reclassificados", ""]
    for r in report["records"]:
        lines.append(_format_record_markdown(r))

    with open(md_path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))

    return json_path, md_path


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output-dir", required=True,
        help="Diretório de um modelo já executado por run_local_vlm_pilot.py "
             "(ex.: experiments/multiagent/local_vlm_pilot/qwen3-vl_8b-instruct-q4_K_M)",
    )
    args = parser.parse_args()
    output_dir = Path(args.output_dir)
    if not output_dir.exists():
        raise FileNotFoundError(f"--output-dir não existe: {output_dir}")

    report = build_reclassification_report(output_dir)
    json_path, md_path = write_reclassification_report(output_dir, report)

    print(f"Registros lidos: {report['n_records_total']}")
    print(f"Registros de controle de envio de imagem excluídos da classificação: {report['n_image_sending_control_records_excluded']}")
    print(f"Divergências v1 vs v2 (só entre aplicáveis): {report['n_v1_v2_disagree']}")
    print(f"v1 recomputado != histórico (deveria ser 0, controles de imagem excluídos): {report['n_v1_recomputed_differs_from_historical_log']}")
    print(f"Relatório JSON: {json_path}")
    print(f"Relatório Markdown: {md_path}")


if __name__ == "__main__":
    main()
