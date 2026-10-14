"""Auditoria de equivalência de configuração/entrada e composição documental do ISIC 2019.

Somente leitura: calcula impressões digitais do código, extrai fatos verificáveis do
código-fonte (o que realmente entra nos prompts, modo de segmentação, regra de
interpretação) e confronta rótulos. Nada é inferido de conversas anteriores.
"""
import collections
import os
import re
import subprocess
import zipfile
from pathlib import Path
from typing import Dict, List, Optional

from src.dataset_validation import labels as L
from src.dataset_validation.config import ROOT
from src.dataset_validation.io_utils import read_json, sha256_file

TRACKED_SOURCES = [
    "src/agents/agents.py", "src/agents/schemas.py", "src/pipeline/pipeline.py",
    "src/models/model.py", "src/data/dataset.py", "src/evaluation/evaluate.py",
    "scripts/run_pipeline.py",
]


def code_fingerprints() -> Dict[str, Optional[str]]:
    return {p: (sha256_file(ROOT / p) if (ROOT / p).exists() else None) for p in TRACKED_SOURCES}


def _git_lines(*args: str) -> List[str]:
    try:
        out = subprocess.run(["git", *args], capture_output=True, text=True, timeout=15, cwd=str(ROOT))
        return [l for l in out.stdout.splitlines() if l.strip()]
    except Exception:  # noqa: BLE001
        return []


def source_facts() -> Dict:
    """Fatos extraídos do código-fonte (não de memória): o que cada agente recebe etc."""
    src = (ROOT / "src/agents/agents.py").read_text(encoding="utf-8")
    model_src = (ROOT / "src/models/model.py").read_text(encoding="utf-8")
    pipe_src = (ROOT / "src/pipeline/pipeline.py").read_text(encoding="utf-8")
    ev_src = (ROOT / "src/evaluation/evaluate.py").read_text(encoding="utf-8")
    run_src = (ROOT / "scripts/run_pipeline.py").read_text(encoding="utf-8")
    return {
        "agents_use_only_segmented_image": {
            "image_message_calls_with_seg_image_data": len(re.findall(r"_image_message\(state\.seg_image_data", src)),
            "references_to_state_image_data_in_agents": len(re.findall(r"state\.image_data\b", src)),
            "interpretation": "As 4 chamadas LLM (ABCD, Menzies, SPCL, sumarizador) recebem APENAS "
                              "a imagem mascarada; a imagem original (image_data) nunca é enviada.",
        },
        "lesion_size_interpolated_in_prompts": bool(re.search(r"\{[^}]*lesion_size[^}]*\}", src)),
        "legacy_mask_application": "pixels fora da lesão tornam-se pretos (original * máscara>128), "
                                   "resolução nativa, JPEG (qualidade padrão do PIL)"
        if "original_array * mask_array" in model_src else "desconhecido",
        "model_mode_segmentation": {
            "random_resize_range": re.findall(r"resize_range: Tuple\[int, int\] = \((\d+), (\d+)\)", model_src),
            "target_width": 256 if "int(256 * aspect_ratio)" in model_src else None,
            "note": "Modo 'model' (U-Net em runtime) reduz a imagem para ~256 px de largura e usa máscara "
                    "PREVISTA; modo 'precomputed' mantém a resolução nativa e a máscara de referência.",
        },
        "interpretation_rule_in_evaluate_py": {
            "keys": re.findall(r'"([^"]+)":\s*([01]),', ev_src.split("_DIAGNOSTICO_TO_LABEL")[1].split("}")[0]),
            "unrecognized_policy": "excluído do cálculo (warning)" if "continue" in ev_src else "desconhecido",
        },
        "pipeline_writes_keys": sorted(set(re.findall(r'event\["(\w+)"\]', pipe_src))),
        "run_pipeline_script_default_segmentation_mode_source": "config segmentation.mode"
        if "cfg[\"segmentation\"][\"mode\"]" in run_src else "desconhecido",
        "build_llm_uses_literal_api_key_strings": bool(re.search(r'openai_api_key="OPENAI_API_KEY"', src)),
    }


def legacy_output_compatibility(legacy_json_path: str) -> Dict:
    """Compara o conjunto de campos do JSON legado com o que `run_pipeline` grava hoje."""
    data = read_json(legacy_json_path)
    first = next(iter(data.values()))
    expected = {"image_path", "lesion_size", "seg_path", "diagnosis_abcd", "diagnosis_menzies",
                "diagnosis_spcl", "validation", "final_report", "real_value"}
    return {"file": legacy_json_path, "n": len(data), "keys": sorted(first.keys()),
            "matches_current_pipeline_schema": set(first.keys()) == expected,
            "note": "Compatibilidade de formato NÃO prova mesma versão de prompts/modelo/parâmetros."}


def isic_composition(cfg: Dict, ham_meta: List[Dict]) -> Dict:
    """Sobreposição verificável em disco entre ISIC 2019, HAM10000 e ISIC 2018 Task 1."""
    c = cfg["datasets"]["isic2019"]
    out: Dict = {"sources": {}, "findings": {}}
    ids19 = None
    d = c.get("image_dir")
    if d and Path(d).is_dir():
        files = os.listdir(d)
        ids19 = {f[:-4] for f in files if f.lower().endswith(".jpg")}
        out["sources"]["isic2019_dir"] = {"path": d, "n_jpg": len(ids19),
                                          "other_files": [f for f in files if not f.lower().endswith(".jpg")]}
        attr = Path(d) / "ATTRIBUTION.txt"
        if attr.exists():
            txt = attr.read_text(encoding="utf-8", errors="replace")
            out["sources"]["isic2019_attribution_sources"] = re.findall(r"^\s+(\S[^:]+Dataset):", txt, flags=re.M)
    else:
        out["sources"]["isic2019_dir"] = {"path": d, "present": False}
    ham_ids = {r["image_id"] for r in ham_meta}
    if ids19 is not None:
        out["findings"]["ham10000_images_in_isic2019"] = len(ham_ids & ids19)
        out["findings"]["ham10000_images_total"] = len(ham_ids)
        out["findings"]["isic2019_not_in_ham10000"] = len(ids19 - ham_ids)
        out["findings"]["isic2019_total"] = len(ids19)
        out["findings"]["ham10000_share_of_isic2019"] = len(ham_ids & ids19) / max(1, len(ids19))
    z = c.get("isic2018_task1_zip")
    if z and Path(z).exists():
        with zipfile.ZipFile(z) as zf:
            ids18 = {Path(n).stem for n in zf.namelist() if n.lower().endswith(".jpg")}
        out["sources"]["isic2018_task1_zip"] = {"path": z, "n_jpg": len(ids18)}
        out["findings"]["isic2018_task1_total"] = len(ids18)
        out["findings"]["isic2018_task1_in_ham10000"] = len(ids18 & ham_ids)
        if ids19 is not None:
            out["findings"]["isic2018_task1_in_isic2019"] = len(ids18 & ids19)
            out["findings"]["isic2018_task1_not_in_isic2019"] = len(ids18 - ids19)
    else:
        out["sources"]["isic2018_task1_zip"] = {"path": z, "present": False}
    for key in ("ground_truth_csv", "metadata_csv"):
        p = c.get(key)
        out["sources"][key] = {"path": p, "present": bool(p and Path(p).exists())}
    out["findings"]["note"] = (
        "Rótulos do ISIC 2019 (ground truth/metadata) não estão em disco quando 'present' é falso: "
        "composição por classe e por fonte (BCN_20000/MSK) NÃO é verificável localmente."
    )
    return out


def ham_source_class_table(ham_meta: List[Dict], cfg_ham: Dict) -> Dict:
    """Fonte de aquisição × classe (mel/nv) e tipo de confirmação no HAM10000 completo (pool)."""
    t: Dict[str, Dict[str, int]] = collections.defaultdict(lambda: collections.defaultdict(int))
    d: Dict[str, Dict[str, int]] = collections.defaultdict(lambda: collections.defaultdict(int))
    for r in ham_meta:
        c = L.ham_class(r, cfg_ham)
        if not c:
            continue
        t[r["dataset"]][c] += 1
        d[r["dx_type"]][c] += 1
    return {"acquisition_source_x_class": {k: dict(v) for k, v in sorted(t.items())},
            "dx_type_x_class": {k: dict(v) for k, v in sorted(d.items())}}


def build_audit(cfg: Dict, ph2_records: Dict[str, Dict], ham_meta: List[Dict],
                legacy_json: Optional[str] = None) -> Dict:
    audit: Dict = {
        "code_fingerprints": code_fingerprints(),
        "agents_py_commits": _git_lines("log", "--format=%h %s", "--", "src/agents/agents.py"),
        "head_commit": (_git_lines("rev-parse", "HEAD") or [None])[0],
        "branch": (_git_lines("rev-parse", "--abbrev-ref", "HEAD") or [None])[0],
        "source_facts": source_facts(),
        "ph2_label_audit": L.ph2_label_audit(ph2_records, cfg["datasets"]["ph2"]),
        "ham10000_pool_composition": ham_source_class_table(ham_meta, cfg["datasets"]["ham10000"]),
        "isic_composition": isic_composition(cfg, ham_meta),
        "llm_config": {"config_yaml_llm": cfg["_config_sections"]["llm"],
                       "pilot_models": cfg["inference"]["models"],
                       "pilot_temperature": cfg["inference"]["temperature"]},
    }
    if legacy_json and Path(legacy_json).exists():
        audit["legacy_output_compatibility"] = legacy_output_compatibility(legacy_json)
    return audit
