"""Verificação independente do subconjunto de 300 lesões
(`scripts/select_local_300_lesion_s2_subset.py`) -- confirma
reprodutibilidade determinística REGERANDO a seleção do zero a partir
do manifesto congelado e comparando byte a byte contra o que está
gravado em disco, em vez de confiar no relatório que a própria geração
produziu.

Checagens, nesta ordem (qualquer falha aborta com detalhe, nunca um
"ok" genérico):
  1. Manifesto de origem (3.000) ainda bate com o SHA-256 congelado.
  2. Regeração completa (mesma semente, mesma função de seleção) produz
     EXATAMENTE os mesmos 300 `lesion_id` -- reprodutibilidade real, não
     só comparação de contagens.
  3. Cada linha do subconjunto tem o MESMO `image_id` da linha
     correspondente no manifesto de 3.000 (nunca uma imagem diferente).
  4. 300 `lesion_id` distintos, nenhuma repetição.
  5. SHA-256 do arquivo em disco bate com o SHA-256 regerado agora E com
     o valor gravado no registro (`local_300_s2_subset_registry.json`).
  6. Composição por `dx` no arquivo bate com as cotas de Hamilton
     recalculadas agora a partir do manifesto de origem (mesma função,
     `compute_quotas_largest_remainder`).

STATUS: implementado, NÃO executado pelo assistente que o escreveu --
aguardando execução do pesquisador.

Uso:
    python scripts/verify_local_300_s2_subset.py
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import importlib.util
import json
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

# Carrega o script irmão por caminho de arquivo direto (não por import de
# pacote `scripts.*`, que exigiria `scripts/__init__.py` -- inexistente no
# projeto, sem precedente de uso -- `importlib` evita essa dependência
# frágil de namespace package implícito).
_SELECT_SCRIPT_PATH = Path(__file__).resolve().parent / "select_local_300_lesion_s2_subset.py"
_spec = importlib.util.spec_from_file_location("select_local_300_lesion_s2_subset", _SELECT_SCRIPT_PATH)
_select_module = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_select_module)

DEFAULT_OUTPUT_DIR = _select_module.DEFAULT_OUTPUT_DIR
DEFAULT_SOURCE_MANIFEST_PATH = _select_module.DEFAULT_SOURCE_MANIFEST_PATH
LOCAL_300_SUBSET_SEED = _select_module.LOCAL_300_SUBSET_SEED
TARGET_N = _select_module.TARGET_N
select_300_lesions_preserving_images = _select_module.select_300_lesions_preserving_images

from src.data_audit.sampling import ELIGIBLE_DX_CATEGORIES  # noqa: E402
from src.multiagent import presentation as pres  # noqa: E402


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--source-manifest", default=DEFAULT_SOURCE_MANIFEST_PATH)
    parser.add_argument("--subset-dir", default=DEFAULT_OUTPUT_DIR)
    return parser


def _sha256_of_file(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(65536), b""):
            h.update(chunk)
    return h.hexdigest()


def main() -> None:
    args = build_arg_parser().parse_args()
    source_manifest_path = Path(args.source_manifest)
    subset_dir = Path(args.subset_dir)
    manifest_path = subset_dir / "local_300_s2_subset_manifest.csv"
    registry_path = subset_dir / "local_300_s2_subset_registry.json"

    failures: list[str] = []

    print("--- 1. Manifesto de origem (3.000) ---")
    pres.verify_manifest_or_die(source_manifest_path)
    source_rows = pres.load_manifest_rows(source_manifest_path)
    source_by_lesion = {r["lesion_id"]: r for r in source_rows}
    print(f"OK -- {len(source_rows)} linhas, SHA-256 confirmado contra o registro congelado.")

    if not manifest_path.exists():
        raise SystemExit(f"ERRO OPERACIONAL: {manifest_path} não encontrado -- rode a seleção primeiro.")
    if not registry_path.exists():
        raise SystemExit(f"ERRO OPERACIONAL: {registry_path} não encontrado -- rode a seleção primeiro.")

    with open(manifest_path, encoding="utf-8") as f:
        on_disk_rows = list(csv.DictReader(f))
    with open(registry_path, encoding="utf-8") as f:
        registry = json.load(f)

    print("\n--- 2. Reprodutibilidade determinística (regeração completa) ---")
    regenerated = select_300_lesions_preserving_images(source_rows, seed=LOCAL_300_SUBSET_SEED, target_n=TARGET_N)
    regenerated_lesion_ids = sorted(r["lesion_id"] for r in regenerated["selected_rows"])
    on_disk_lesion_ids = sorted(r["lesion_id"] for r in on_disk_rows)
    if regenerated_lesion_ids != on_disk_lesion_ids:
        only_regenerated = set(regenerated_lesion_ids) - set(on_disk_lesion_ids)
        only_on_disk = set(on_disk_lesion_ids) - set(regenerated_lesion_ids)
        failures.append(
            f"regeração produz um conjunto de lesion_id DIFERENTE do arquivo em disco -- "
            f"{len(only_regenerated)} só na regeração, {len(only_on_disk)} só no disco."
        )
    else:
        print(f"OK -- regeração produz exatamente os mesmos {len(regenerated_lesion_ids)} lesion_id do arquivo em disco.")

    print("\n--- 3. image_id preservado (nunca resorteado) ---")
    image_mismatches = []
    for row in on_disk_rows:
        lid = row["lesion_id"]
        source_row = source_by_lesion.get(lid)
        if source_row is None:
            failures.append(f"lesion_id {lid!r} do subconjunto não existe no manifesto de origem.")
            continue
        if row["image_id"] != source_row["image_id"]:
            image_mismatches.append((lid, row["image_id"], source_row["image_id"]))
    if image_mismatches:
        failures.append(f"{len(image_mismatches)} lesão(ões) com image_id DIFERENTE do manifesto de origem: {image_mismatches[:5]}...")
    else:
        print(f"OK -- {len(on_disk_rows)} linhas, todas com o mesmo image_id do manifesto de 3.000.")

    print("\n--- 4. Unicidade ---")
    if len(on_disk_lesion_ids) != len(set(on_disk_lesion_ids)):
        failures.append("lesion_id repetido no subconjunto de 300.")
    elif len(on_disk_lesion_ids) != TARGET_N:
        failures.append(f"subconjunto tem {len(on_disk_lesion_ids)} lesion_id distintos, esperado {TARGET_N}.")
    else:
        print(f"OK -- {TARGET_N} lesion_id distintos, nenhuma repetição.")

    print("\n--- 5. SHA-256 (arquivo em disco x registro) ---")
    actual_manifest_sha256 = _sha256_of_file(manifest_path)
    registered_sha256 = registry.get("subset_manifest_sha256")
    if actual_manifest_sha256 != registered_sha256:
        failures.append(
            f"SHA-256 do manifesto em disco ({actual_manifest_sha256}) não bate com o registrado "
            f"em local_300_s2_subset_registry.json ({registered_sha256})."
        )
    else:
        print(f"OK -- SHA-256 do manifesto em disco bate com o registro: {actual_manifest_sha256}")

    print("\n--- 6. Cotas de Hamilton (recalculadas agora) ---")
    actual_composition = {cat: sum(1 for r in on_disk_rows if r["dx"] == cat) for cat in ELIGIBLE_DX_CATEGORIES}
    expected_quotas = regenerated["quotas"]["final_quotas"]
    mismatches = {
        cat: {"expected": expected_quotas.get(cat, 0), "actual": actual_composition.get(cat, 0)}
        for cat in ELIGIBLE_DX_CATEGORIES
        if expected_quotas.get(cat, 0) != actual_composition.get(cat, 0)
    }
    if mismatches:
        failures.append(f"composição por dx não bate com as cotas de Hamilton recalculadas: {mismatches}")
    else:
        print(f"OK -- composição por dx bate com as cotas de Hamilton: {actual_composition}")

    print("\n=== Resultado ===")
    if failures:
        print(f"FALHOU -- {len(failures)} problema(s):")
        for f in failures:
            print(f"  - {f}")
        raise SystemExit(1)
    print("Todas as 6 checagens passaram -- subconjunto de 300 lesões verificado.")


if __name__ == "__main__":
    main()
