"""Monta o material de revisão qualitativa das sínteses multiagente (S2,
as 300 lesões já congeladas para os modelos locais -- nenhum novo
sorteio). Gera, por lesão: uma página HTML estática reunindo a imagem
S2, os 3 pareceres (ABCD/Menzies/SPCL) e a síntese de cada um dos 4
modelos (Gemini/GPT/Qwen3-VL/MedGemma, sob identificadores neutros), com
indicação explícita de agente ausente/consolidação parcial; e linhas
pendentes em um arquivo de revisão CSV único (``review_template.csv``,
formato longo: uma linha por lesão × modelo × critério).

Esta é uma análise dos resultados, não um critério de aprovação de
modelo. Nenhuma nova chamada a LLM é feita aqui -- só leitura dos
registros já existentes e heurísticas de destaque textual (negrito/
marcação), nunca preenchimento automático de veredito.

Reusa, sem modificação, ``src.evaluation.stage6_sources`` (seleção da
tentativa oficial, reprocessamento oficial do Gemini) e
``src.evaluation.response_template_extraction`` (extração de texto já
validada).

Usage:
    python scripts/prepare_qualitative_review_materials.py
"""
from __future__ import annotations

import csv
import html
import json
import re
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

from src.evaluation.qualitative_review_schema import (  # noqa: E402
    REVIEW_CSV_FIELDNAMES,
    build_neutral_model_id_mapping,
    build_pending_review_rows,
)
from src.evaluation.response_template_extraction import extract_final_answer  # noqa: E402
from src.evaluation.stage6_sources import (  # noqa: E402
    load_call_log_rows,
    load_gemini_recovered_text_by_full_key,
    load_manifest_rows,
    load_s2_availability,
    merge_gemini_reprocessed_decisao_final,
    select_official_call_records,
)

MANIFEST_300 = REPO_ROOT / "experiments" / "multiagent" / "local_300_s2_subset" / "local_300_s2_subset_manifest.csv"
S0_S1_S2_MANIFEST = REPO_ROOT / "experiments" / "multiagent" / "s0_s1_s2_preparation_manifest.csv"
S2_IMAGES_DIR = REPO_ROOT / "experiments" / "multiagent" / "s0_s1_s2_images" / "s2"

GEMINI_MODEL = "gemini-3.1-flash-lite"
GEMINI_CALL_LOG = REPO_ROOT / "experiments" / "multiagent" / "full_scale_matrix" / GEMINI_MODEL / "call_log_main.jsonl"
GEMINI_REPROCESSED = (
    REPO_ROOT / "experiments" / "multiagent" / "full_scale_matrix" / GEMINI_MODEL / "reprocessing"
    / "call_log_main.decisao_final_reprocessed.jsonl"
)
GPT_MODEL = "gpt-5.4-nano"
GPT_CALL_LOG = REPO_ROOT / "experiments" / "multiagent" / "full_scale_matrix" / GPT_MODEL / "call_log_main.jsonl"
MEDGEMMA_MODEL = "medgemma1.5:4b-it-q4_K_M"
MEDGEMMA_CALL_LOG = (
    REPO_ROOT / "experiments" / "multiagent" / "local_300_s2_subset_matrix" / "medgemma1.5_4b-it-q4_K_M"
    / "call_log_main.jsonl"
)
QWEN_MODEL = "qwen3-vl:8b-instruct-q4_K_M"
QWEN_CALL_LOG = (
    REPO_ROOT / "experiments" / "multiagent" / "local_300_s2_subset_matrix" / "qwen3-vl_8b-instruct-q4_K_M"
    / "call_log_main.jsonl"
)

CONDITION = "s2"
ROLE_SUMMARY = "summary"
ROLES_DIAGNOSTIC = ("diagnostic_abcd", "diagnostic_menzies", "diagnostic_spcl")
ROLE_LABELS = {"diagnostic_abcd": "ABCD", "diagnostic_menzies": "Menzies", "diagnostic_spcl": "SPCL"}

OUTPUT_DIR = REPO_ROOT / "experiments" / "evaluation" / "qualitative_review"
PAGES_DIR = OUTPUT_DIR / "pages"
REVIEW_TEMPLATE_CSV = OUTPUT_DIR / "review_template.csv"
MODEL_ID_KEY_JSON = OUTPUT_DIR / "model_id_key_CONFIDENCIAL.json"  # separado do material, nao linkado do HTML
ANSWER_KEY_CSV = OUTPUT_DIR / "answer_key_CONFIDENCIAL.csv"  # idem

# Heuristica leve de destaque (NUNCA preenche veredito, so marca span) --
# vocabulario fechado, documentado, mesma familia de heuristicas lexicais
# ja existentes no projeto (case_status_v2/vlm_direct).
_HYPOTHESIS_MARKERS = (
    "poderia ser", "poderia indicar", "poderia sugerir", "possivelmente",
    "é possível que", "pode haver", "sugere a possibilidade", "hipotese", "hipótese",
)


def _highlight_html(text: str) -> str:
    escaped = html.escape(text)
    for role, label in ROLE_LABELS.items():
        escaped = re.sub(rf"\b{label}\b", f"<mark class='agent'>{label}</mark>", escaped)
    for marker in _HYPOTHESIS_MARKERS:
        escaped = re.sub(re.escape(html.escape(marker)), lambda m: f"<mark class='hypothesis'>{m.group(0)}</mark>", escaped, flags=re.IGNORECASE)
    return escaped.replace("\n", "<br>")


def _display_text(raw_text, recovered_text: str | None) -> tuple[str, str]:
    """Devolve ``(texto_exibido, origem)``. Prioriza o texto recuperado
    oficial (Gemini) quando existe; senão aplica a extração de
    template já validada (``extract_final_answer``) sobre o texto bruto."""
    if recovered_text:
        return recovered_text, "gemini_reprocessado_oficial"
    if not raw_text:
        return "", "ausente"
    extraction = extract_final_answer(raw_text)
    if extraction.final_answer_text:
        return extraction.final_answer_text, f"extract_final_answer:{extraction.status}"
    return str(raw_text), f"extract_final_answer:{extraction.status}(sem_texto_final-exibindo_bruto)"


def _load_model_rows(model_name: str, call_log_path: Path, reprocessed_path: Path | None):
    rows = load_call_log_rows(call_log_path)
    recovered_by_key = {}
    if reprocessed_path is not None and reprocessed_path.exists():
        recovered_by_key = load_gemini_recovered_text_by_full_key(reprocessed_path)
        rows = merge_gemini_reprocessed_decisao_final(rows, reprocessed_path)
    selection = select_official_call_records(rows, model_name)
    return selection, recovered_by_key


def _recovered_text_for(selection_record, recovered_by_key: dict) -> str | None:
    if not selection_record:
        return None
    full_key = (
        selection_record["role"], selection_record["condition"], selection_record["lesion_id"],
        selection_record["image_id"], selection_record["attempt_timestamp_utc"],
    )
    return recovered_by_key.get(full_key)


def main() -> None:
    manifest_rows = load_manifest_rows(MANIFEST_300)
    s2_availability = load_s2_availability(S0_S1_S2_MANIFEST)
    print(f"{len(manifest_rows)} lesões carregadas de {MANIFEST_300}.")

    neutral_mapping = build_neutral_model_id_mapping()
    PAGES_DIR.mkdir(parents=True, exist_ok=True)

    models = {
        "Gemini": (GEMINI_MODEL, GEMINI_CALL_LOG, GEMINI_REPROCESSED),
        "GPT": (GPT_MODEL, GPT_CALL_LOG, None),
        "Qwen3-VL": (QWEN_MODEL, QWEN_CALL_LOG, None),
        "MedGemma": (MEDGEMMA_MODEL, MEDGEMMA_CALL_LOG, None),
    }

    loaded = {}
    for real_label, (model_name, call_log_path, reprocessed_path) in models.items():
        if not call_log_path.exists():
            print(f"AVISO: call log ausente para {real_label}: {call_log_path} -- todas as sínteses deste modelo ficarão marcadas como ausência de execução.")
            loaded[real_label] = (None, {})
            continue
        loaded[real_label] = _load_model_rows(model_name, call_log_path, reprocessed_path)
        print(f"{real_label}: {len(loaded[real_label][0].record_by_key)} registros oficiais carregados.")

    all_review_rows: list[dict] = []
    answer_key_rows: list[dict] = []
    n_text_available_by_model = {m: 0 for m in models}
    n_text_absent_by_model = {m: 0 for m in models}

    for row in manifest_rows:
        lesion_id, image_id = row["lesion_id"], row["image_id"]
        status_s2 = s2_availability.status_by_lesion.get(lesion_id, "")
        reason_s2 = s2_availability.reason_by_lesion.get(lesion_id, "")
        answer_key_rows.append({"lesion_id": lesion_id, "image_id": image_id, "dx": row["dx"], "binary_target": row["binary_target"]})

        model_sections = []
        for real_label in models:
            neutral_id = neutral_mapping[real_label]
            selection, recovered_by_key = loaded[real_label]

            summary_record = None
            diag_records = {r: None for r in ROLES_DIAGNOSTIC}
            if selection is not None:
                summary_record = selection.record_by_key.get((ROLE_SUMMARY, CONDITION, lesion_id, image_id))
                for diag_role in ROLES_DIAGNOSTIC:
                    diag_records[diag_role] = selection.record_by_key.get((diag_role, CONDITION, lesion_id, image_id))

            summary_raw = summary_record.get("raw_response_text") if summary_record else None
            summary_recovered = _recovered_text_for(summary_record, recovered_by_key) if summary_record else None
            synthesis_text, synthesis_origin = _display_text(summary_raw, summary_recovered)
            text_available = bool(synthesis_text.strip())

            if status_s2 != "disponivel":
                availability_note = f"S2 estruturalmente indisponível para esta lesão (reason_s2={reason_s2 or status_s2}) -- nenhuma chamada foi planejada."
                text_available = False
            elif summary_record is None:
                availability_note = "Síntese ausente: nenhum registro oficial encontrado (ausência de execução/falha não reaproveitável)."
            else:
                availability_note = f"call_status={summary_record.get('call_status')}; decision_status={summary_record.get('decision_status')}; decisao_final={summary_record.get('decisao_final')}."

            missing_diag_agents = [ROLE_LABELS[r] for r, rec in diag_records.items() if rec is None or rec.get("call_status") != "ok"]

            pareceres_html = []
            for diag_role in ROLES_DIAGNOSTIC:
                rec = diag_records.get(diag_role)
                label = ROLE_LABELS[diag_role]
                if rec is None:
                    pareceres_html.append(f"<div class='parecer ausente'><h4>{label} (ausente)</h4><p><em>Nenhum registro oficial.</em></p></div>")
                    continue
                raw = rec.get("raw_response_text")
                text, origin = _display_text(raw, None)
                status_note = f"call_status={rec.get('call_status')}"
                pareceres_html.append(
                    f"<div class='parecer'><h4>{label}</h4><p class='meta'>{html.escape(status_note)} (origem do texto: {origin})</p>"
                    f"<p>{_highlight_html(text) if text else '<em>sem texto</em>'}</p></div>"
                )

            if text_available:
                n_text_available_by_model[real_label] += 1
            else:
                n_text_absent_by_model[real_label] += 1

            model_sections.append(
                f"<section class='model-block'><h3>{neutral_id}</h3>"
                f"<p class='meta'>{html.escape(availability_note)}</p>"
                + (f"<p class='meta missing'>Pareceres ausentes/não-ok: {', '.join(missing_diag_agents)}</p>" if missing_diag_agents else "")
                + f"<h4>Síntese</h4><p class='meta'>origem do texto: {synthesis_origin}</p>"
                + f"<p>{_highlight_html(synthesis_text) if synthesis_text else '<em>sem texto analisável</em>'}</p>"
                + "<details><summary>Pareceres de entrada (ABCD/Menzies/SPCL)</summary>" + "".join(pareceres_html) + "</details>"
                + f"<p class='meta'>Arquivo de revisão: <code>review_template.csv</code> (filtrar por lesion_id='{lesion_id}' e model_label_neutral='{neutral_id}')</p>"
                + "</section>"
            )

            all_review_rows.extend(build_pending_review_rows(lesion_id, neutral_id, text_available))

        image_rel = f"../../../multiagent/s0_s1_s2_images/s2/{image_id}_s2.png"
        page_html = (
            "<!doctype html><html lang='pt-br'><head><meta charset='utf-8'>"
            f"<title>Revisão qualitativa -- {lesion_id}</title>"
            "<style>body{font-family:sans-serif;margin:2em;max-width:900px;}"
            ".model-block{border:1px solid #ccc;padding:1em;margin-bottom:1em;}"
            ".meta{color:#666;font-size:0.9em;} .missing{color:#c80;}"
            "mark.agent{background:#dde;} mark.hypothesis{background:#fde;}"
            ".parecer{border-left:3px solid #ddd;padding-left:0.5em;margin:0.5em 0;}"
            ".parecer.ausente{border-left-color:#c33;}"
            "</style></head><body>"
            f"<h1>Lesão {lesion_id}</h1>"
            f"<img src='{image_rel}' alt='S2' style='max-width:400px;'>"
            "<p><strong>Aviso:</strong> análise qualitativa de fidelidade da síntese, não validação clínica. "
            "O diagnóstico real não é mostrado nesta página (ver nota de cegamento no roteiro)."
            " Esta revisão não é um critério de aprovação de modelo nem autoriza excluir casos.</p>"
            + "".join(model_sections)
            + "</body></html>"
        )
        (PAGES_DIR / f"{lesion_id}.html").write_text(page_html, encoding="utf-8")

    with open(REVIEW_TEMPLATE_CSV, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(REVIEW_CSV_FIELDNAMES))
        writer.writeheader()
        writer.writerows(all_review_rows)

    with open(MODEL_ID_KEY_JSON, "w", encoding="utf-8") as f:
        json.dump({"neutral_to_real": {v: k for k, v in neutral_mapping.items()}, "real_to_neutral": neutral_mapping,
                   "note": "NAO e cegamento garantido -- estilo de escrita pode revelar o modelo. Manter separado do material de revisao."},
                  f, ensure_ascii=False, indent=2)

    with open(ANSWER_KEY_CSV, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=["lesion_id", "image_id", "dx", "binary_target"])
        writer.writeheader()
        writer.writerows(answer_key_rows)

    print(f"\n{len(manifest_rows)} páginas HTML em {PAGES_DIR}")
    print(f"Arquivo de revisão (pendente): {REVIEW_TEMPLATE_CSV} ({len(all_review_rows)} linhas)")
    print(f"Chave de modelos (NÃO distribuir com o material): {MODEL_ID_KEY_JSON}")
    print(f"Gabarito (NÃO distribuir com o material): {ANSWER_KEY_CSV}")
    for m in models:
        print(f"  {m}: {n_text_available_by_model[m]} sínteses com texto analisável, {n_text_absent_by_model[m]} sem texto (marcadas nao_avaliavel automaticamente).")


if __name__ == "__main__":
    main()
