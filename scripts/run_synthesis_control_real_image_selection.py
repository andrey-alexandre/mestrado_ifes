"""Etapa 5 -- tarefa adicional (2026-11-14): seleciona UMA imagem
dermatoscópica REAL do HAM10000, verificada como externa à amostra
congelada de 3.000 lesões (`experiments/sampling/sample_manifest.csv`,
SHA-256 `32bbd1b318dccca048abeb25de64e6b1f2db6184d589286606a73271d6392799`,
ver `docs/reconstruction/10_frozen_sample_registry.md`) por `lesion_id` E
`image_id`, para substituir o placeholder cinza (`_fixed_control_image()`)
usado pelos 4 controles de síntese de `src/multiagent/vlm_direct.py`.

Motivação real (não hipotética): na 3ª execução real do Estágio 1 do
piloto local de VLM (`qwen3-vl:8b-instruct-q4_K_M`), o cenário de controle
`um_parecer_ausente` falhou porque o modelo notou, corretamente, que a
imagem placeholder (64x64 cinza sólido) não mostra nenhuma lesão -- "a
imagem fornecida é completamente cinza e vazia... nenhum dos três
algoritmos pode ser aplicado". O sumarizador REAL
(`SummaryAgent.summarize`, `src/agents/agents.py`) sempre envia uma
imagem dermatoscópica REAL junto dos 3 pareceres; este script resolve a
parte "imagem real" dessa fidelidade -- ver
`docs/reconstruction/20_local_vlm_pilot.md` §9e para o relato completo.

Reusa integralmente as primitivas de `src/data_audit/sampling.py` e
`src/multiagent/presentation.py` já usadas por
`scripts/run_pilot12_sample.py` -- MESMO padrão de exclusão (por AMBAS as
chaves) e de verificação de hash. Esta seleção é INDEPENDENTE do piloto de
12 lesões (`experiments/multiagent/pilot12_manifest.csv`) -- não lê, não
escreve, nem reusa esse arquivo; são pools de exclusão/seleção distintos,
por instrução explícita do pesquisador ("a matriz das 12 lesões permanece
fora desta execução").

Semente NOVA desta tarefa (`SYNTHESIS_CONTROL_IMAGE_SEED = 20270901`) --
deliberadamente distinta de TODAS as sementes já documentadas neste
projeto: `20261106` (amostra congelada de 3.000,
`docs/reconstruction/10_frozen_sample_registry.md`), `20270203`
(`PILOT_SEED`, piloto de 12 lesões, `scripts/run_pilot12_sample.py`) e
`20270204` (`FOURTH_SLOT_SEED`, mesmo script). Governa SÓ a seleção da
imagem -- nunca o determinismo das respostas do VLM.

ESCOLHA DE DESIGN (julgamento deste assistente, NÃO ditada verbatim pelo
pesquisador -- sinalizada explicitamente, não assumida como já aprovada):
UMA ÚNICA imagem real é escolhida e compartilhada pelos 4 cenários de
síntese, em vez de 4 imagens distintas (uma por cenário). Razão: o
controle de síntese testa fidelidade de SÍNTESE TEXTUAL (os pareceres
ABCD/Menzies/SPCL variam por cenário, não a imagem) -- a imagem só
precisa deixar de ser um placeholder sem lesão; variar a imagem por
cenário não acrescenta sinal relevante para o que este controle mede e
amplia sem necessidade a superfície de verificação (4 pares
lesion_id/image_id em vez de 1). Ver campo `design_note_shared_image` no
registry gravado por este script.

Se a separação de uma imagem não puder ser verificada (hash do manifesto
não confere, pool residual vazio, sobreposição detectada por construção,
ou arquivo de imagem ilegível/ausente), este script NUNCA escolhe uma
alternativa silenciosamente -- grava `verification_status: "pendente"`
com o motivo exato no registry e nada mais.

STATUS: implementado, aguardando execução do pesquisador. Nada deste
script foi executado pelo assistente que o escreveu
(`feedback_no_autonomous_execution`) -- nenhum `python`/`pytest`/`git` foi
chamado.

Uso:
    # 1. Seleção (grava o registry + relatório -- UMA vez; reexecutar com
    #    o mesmo --seed produz o mesmo resultado, já que read_sample_manifest/
    #    build_lesion_table/apply_exclusions e os dois random.Random(seed)
    #    usados aqui são determinísticos):
    python scripts/run_synthesis_control_real_image_selection.py \\
        --ham10000-images-dir D:/Andrey/Mestrado/dataverse_files/HAM10000

    # 2. Verificação SEM INFERÊNCIA (antes de autorizar as 8 chamadas reais
    #    de scripts/run_synthesis_controls_real_image.py) -- relê o
    #    registry já gravado e reconfirma hash/exclusão/carregamento da
    #    imagem, sem sortear nada de novo:
    python scripts/run_synthesis_control_real_image_selection.py --verify-only \\
        --ham10000-images-dir D:/Andrey/Mestrado/dataverse_files/HAM10000
"""
from __future__ import annotations

import argparse
import json
import random
import sys
from pathlib import Path
from typing import Any, Optional

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

from src.data_audit import sampling  # noqa: E402
from src.multiagent.presentation import (  # noqa: E402
    load_original_image,
    resolve_original_image_path,
    sha256_of_file,
)

REGISTERED_FROZEN_MANIFEST_SHA256 = (
    "32bbd1b318dccca048abeb25de64e6b1f2db6184d589286606a73271d6392799"
)
DEFAULT_FROZEN_MANIFEST = str(REPO_ROOT / "experiments" / "sampling" / "sample_manifest.csv")
DEFAULT_METADATA_CSV = "D:/Andrey/Mestrado/dataverse_files/HAM10000_metadata"
DEFAULT_HAM10000_IMAGES_DIR = "D:/Andrey/Mestrado/dataverse_files/HAM10000"
DEFAULT_OUTPUT_DIR = str(REPO_ROOT / "experiments" / "multiagent")

# Semente NOVA desta tarefa -- ver docstring do módulo (distinta de
# 20261106/20270203/20270204, todas já documentadas neste projeto).
SYNTHESIS_CONTROL_IMAGE_SEED = 20270901

REGISTRY_FILENAME = "synthesis_control_real_image_registry.json"
REGISTRY_REPORT_FILENAME = "synthesis_control_real_image_registry.md"
VERIFY_ONLY_REPORT_FILENAME = "synthesis_control_real_image_verify_only_report.md"

STATUS_OK = "ok"
STATUS_PENDENTE = "pendente"


def _verify_frozen_manifest_or_die(path: Path) -> str:
    if not path.exists():
        raise SystemExit(f"ERRO: manifesto congelado não encontrado em {path}")
    actual = sha256_of_file(path)
    if actual != REGISTERED_FROZEN_MANIFEST_SHA256:
        raise SystemExit(
            f"ERRO FATAL: SHA-256 do manifesto congelado NÃO corresponde ao registrado.\n"
            f"  arquivo: {path}\n  esperado: {REGISTERED_FROZEN_MANIFEST_SHA256}\n"
            f"  encontrado: {actual}\nExecução abortada -- não usar um manifesto não "
            "verificado para decidir exclusão."
        )
    return actual


def select_real_image(
    metadata_csv: str,
    frozen_manifest_path: Path,
    images_dir: Path,
    seed: int = SYNTHESIS_CONTROL_IMAGE_SEED,
) -> dict[str, Any]:
    """Seleciona 1 lesão/imagem real do HAM10000, fora da amostra congelada
    de 3.000 lesões por `lesion_id` E `image_id` (exclusão por AMBAS as
    chaves, mesmo padrão de `scripts/run_pilot12_sample.py`). NUNCA cai de
    volta a uma imagem alternativa silenciosamente se qualquer verificação
    falhar -- devolve ``verification_status: "pendente"`` com o motivo
    explícito, em vez de prosseguir.
    """
    manifest_sha256 = _verify_frozen_manifest_or_die(frozen_manifest_path)
    frozen_rows = sampling.read_sample_manifest(frozen_manifest_path)
    frozen_lesion_ids = {r["lesion_id"] for r in frozen_rows}
    frozen_image_ids = {r["image_id"] for r in frozen_rows}

    rows = sampling.load_ham10000_metadata(metadata_csv)
    population_check = sampling.check_lesion_population(rows)
    if not population_check["ok"]:
        return {
            "verification_status": STATUS_PENDENTE,
            "reason": "populacao_de_metadados_invalida",
            "population_check": population_check,
            "frozen_manifest_sha256_verified": manifest_sha256,
        }

    lesion_table = sampling.build_lesion_table(rows)
    # Exclusão por AMBAS as chaves -- mesma lógica de run_pilot12_sample.py:
    # lesion_id presente no congelado é removido; além disso, qualquer
    # lesão cuja LISTA DE IMAGENS contenha um image_id do congelado também
    # é removida, mesmo que seu lesion_id em si não esteja no congelado.
    lesion_ids_with_frozen_image = {
        lid for lid, entry in lesion_table.items() if frozen_image_ids & set(entry["image_ids"])
    }
    exclusion_ids = sorted(frozen_lesion_ids | lesion_ids_with_frozen_image)
    exclusion_result = sampling.apply_exclusions(lesion_table, exclusion_ids)
    filtered_table = exclusion_result["filtered_lesion_table"]

    if not filtered_table:
        return {
            "verification_status": STATUS_PENDENTE,
            "reason": "pool_residual_vazio_apos_exclusao",
            "frozen_manifest_sha256_verified": manifest_sha256,
        }

    # Sorteio determinístico: 1 lesão, depois 1 imagem dessa lesão -- dois
    # random.Random(seed) independentes com a MESMA semente (mesmo padrão
    # de src.data_audit.sampling.sample_lesions: trocar a ordem de
    # chamadas de um sorteio não perturba o outro).
    pool_lesion_ids = sorted(filtered_table.keys())
    rng_lesion = random.Random(seed)
    chosen_lesion_id = rng_lesion.sample(pool_lesion_ids, 1)[0]
    rng_image = random.Random(seed)
    chosen_image_ids_pool = sorted(filtered_table[chosen_lesion_id]["image_ids"])
    chosen_image_id = rng_image.sample(chosen_image_ids_pool, 1)[0]
    chosen_dx = filtered_table[chosen_lesion_id]["dx"]

    overlap_lesion = chosen_lesion_id in frozen_lesion_ids
    overlap_image = chosen_image_id in frozen_image_ids
    if overlap_lesion or overlap_image:
        # NUNCA deveria ocorrer por construção (exclusão já removeu estes
        # casos do pool antes do sorteio) -- se ocorrer mesmo assim, é um
        # defeito a sinalizar explicitamente, nunca a ignorar ou escolher
        # outra imagem silenciosamente.
        return {
            "verification_status": STATUS_PENDENTE,
            "reason": "defeito_sobreposicao_detectada_apos_exclusao",
            "chosen_lesion_id": chosen_lesion_id,
            "chosen_image_id": chosen_image_id,
            "overlap_lesion_id": overlap_lesion,
            "overlap_image_id": overlap_image,
            "frozen_manifest_sha256_verified": manifest_sha256,
        }

    image_path = resolve_original_image_path(images_dir, chosen_image_id)
    image = load_original_image(image_path)
    if image is None:
        return {
            "verification_status": STATUS_PENDENTE,
            "reason": "imagem_ilegivel_ou_ausente",
            "chosen_lesion_id": chosen_lesion_id,
            "chosen_image_id": chosen_image_id,
            "image_path": str(image_path),
            "overlap_lesion_id": overlap_lesion,
            "overlap_image_id": overlap_image,
            "frozen_manifest_sha256_verified": manifest_sha256,
        }

    image_sha256 = sha256_of_file(image_path)
    return {
        "verification_status": STATUS_OK,
        "source": "HAM10000",
        "metadata_csv": str(metadata_csv),
        "images_dir": str(images_dir),
        "seed": seed,
        "seed_note": (
            "Semente NOVA desta tarefa -- distinta de 20261106 (amostra "
            "congelada), 20270203 (PILOT_SEED, piloto de 12) e 20270204 "
            "(FOURTH_SLOT_SEED, mesmo piloto). Governa só a seleção da "
            "imagem -- não garante determinismo das respostas do VLM."
        ),
        "chosen_lesion_id": chosen_lesion_id,
        "chosen_image_id": chosen_image_id,
        "chosen_dx": chosen_dx,
        "image_path": str(image_path),
        "image_sha256": image_sha256,
        "image_width": image.width,
        "image_height": image.height,
        "frozen_manifest_sha256_verified": manifest_sha256,
        "frozen_manifest_path": str(frozen_manifest_path),
        "n_eligible_pool_after_exclusion": len(filtered_table),
        "overlap_lesion_id": overlap_lesion,
        "overlap_image_id": overlap_image,
        "overlap_ok": (not overlap_lesion) and (not overlap_image),
        "design_note_shared_image": (
            "Decisão de design desta tarefa (julgamento deste assistente, "
            "NÃO ditada verbatim pelo pesquisador): 1 única imagem real "
            "compartilhada pelos 4 cenários de síntese (em vez de 4 "
            "imagens distintas), porque o controle testa fidelidade de "
            "SÍNTESE TEXTUAL (os pareceres variam por cenário, não a "
            "imagem) -- a imagem só precisa deixar de ser um placeholder "
            "cinza sem lesão. Sinalizado aqui como escolha deste "
            "assistente, não assumido como já aprovado pelo pesquisador."
        ),
        "scope_note": (
            "Esta seleção é INDEPENDENTE da matriz de 12 lesões "
            "(experiments/multiagent/pilot12_manifest.csv) -- não lê, não "
            "escreve, nem reusa esse arquivo; o critério de >=95% desse "
            "piloto não é estimado a partir desta seleção/dos 4 controles "
            "de síntese, por instrução explícita do pesquisador."
        ),
    }


def verify_existing_registry(
    registry_path: Path, frozen_manifest_path: Path, images_dir: Optional[Path] = None
) -> dict[str, Any]:
    """Checagem SEM INFERÊNCIA (nenhuma chamada a modelo) -- re-verifica, a
    partir de um registry já gravado por :func:`select_real_image`: (1) o
    SHA-256 do manifesto congelado ainda confere, (2) a exclusão por AMBAS
    as chaves (lesion_id/image_id) do item já escolhido continua válida,
    e (3) o arquivo de imagem ainda carrega (e seu SHA-256 ainda confere
    com o registrado). NÃO sorteia nada de novo -- só confirma que o que
    já foi registrado continua correto. Chamar ANTES de autorizar qualquer
    chamada real de `scripts/run_synthesis_controls_real_image.py`.
    """
    if not registry_path.exists():
        return {"ok": False, "reason": f"registry não encontrado: {registry_path}"}
    with open(registry_path, encoding="utf-8") as f:
        registry = json.load(f)

    if registry.get("verification_status") != STATUS_OK:
        return {
            "ok": False,
            "reason": (
                f"registry não está em status '{STATUS_OK}' (está "
                f"'{registry.get('verification_status')}') -- nada a "
                "reverificar além do que já está registrado como pendente."
            ),
            "registry": registry,
        }

    manifest_sha256 = _verify_frozen_manifest_or_die(frozen_manifest_path)
    hash_matches = manifest_sha256 == registry.get("frozen_manifest_sha256_verified")

    frozen_rows = sampling.read_sample_manifest(frozen_manifest_path)
    frozen_lesion_ids = {r["lesion_id"] for r in frozen_rows}
    frozen_image_ids = {r["image_id"] for r in frozen_rows}
    chosen_lesion_id = registry["chosen_lesion_id"]
    chosen_image_id = registry["chosen_image_id"]
    overlap_lesion = chosen_lesion_id in frozen_lesion_ids
    overlap_image = chosen_image_id in frozen_image_ids

    image_path = Path(registry["image_path"])
    if images_dir is not None:
        image_path = resolve_original_image_path(images_dir, chosen_image_id)
    image = load_original_image(image_path)
    image_loads = image is not None
    image_sha256_matches: Optional[bool] = None
    if image_loads:
        image_sha256_matches = sha256_of_file(image_path) == registry.get("image_sha256")

    ok = (
        hash_matches
        and not overlap_lesion
        and not overlap_image
        and image_loads
        and (image_sha256_matches is not False)
    )
    return {
        "ok": ok,
        "manifest_sha256_matches_registered": hash_matches,
        "overlap_lesion_id": overlap_lesion,
        "overlap_image_id": overlap_image,
        "image_loads": image_loads,
        "image_sha256_matches_registry": image_sha256_matches,
        "image_path_checked": str(image_path),
        "registry_path": str(registry_path),
        "registry": registry,
    }


def build_registry_markdown(result: dict[str, Any]) -> str:
    lines = [
        "# Registro da imagem real dos controles de síntese -- HAM10000",
        "",
        f"- Status de verificação: **{result['verification_status']}**",
        "",
    ]
    if result["verification_status"] != STATUS_OK:
        lines += [
            "## PENDENTE -- nenhuma imagem foi escolhida como alternativa silenciosa",
            "",
            f"- Motivo: `{result.get('reason')}`",
            f"```json\n{json.dumps(result, ensure_ascii=False, indent=2, default=str)}\n```",
            "",
        ]
        return "\n".join(lines)

    lines += [
        "## Imagem selecionada",
        "",
        f"- Fonte: {result['source']}",
        f"- `lesion_id`: `{result['chosen_lesion_id']}`",
        f"- `image_id`: `{result['chosen_image_id']}`",
        f"- `dx`: `{result['chosen_dx']}`",
        f"- Caminho: `{result['image_path']}`",
        f"- SHA-256 do arquivo de imagem: `{result['image_sha256']}`",
        f"- Dimensões: {result['image_width']}x{result['image_height']}",
        "",
        "## Verificação de separação da amostra congelada (3.000 lesões)",
        "",
        f"- SHA-256 do manifesto congelado verificado: `{result['frozen_manifest_sha256_verified']}`",
        f"- Manifesto congelado: `{result['frozen_manifest_path']}`",
        f"- Pool elegível após exclusão (lesion_id OU image_id no congelado): {result['n_eligible_pool_after_exclusion']}",
        f"- Sobreposição por `lesion_id`: {result['overlap_lesion_id']} (esperado False)",
        f"- Sobreposição por `image_id`: {result['overlap_image_id']} (esperado False)",
        f"- Critério atendido (zero sobreposição por ambas as chaves): **{result['overlap_ok']}**",
        "",
        "## Semente",
        "",
        f"- `SYNTHESIS_CONTROL_IMAGE_SEED = {result['seed']}`",
        f"- {result['seed_note']}",
        "",
        "## Escolha de design (sinalizada, não assumida como aprovada)",
        "",
        result["design_note_shared_image"],
        "",
        "## Escopo",
        "",
        result["scope_note"],
        "",
    ]
    return "\n".join(lines)


def build_verify_only_markdown(result: dict[str, Any]) -> str:
    registry = result.get("registry", {})
    lines = [
        "# Pré-checagem SEM INFERÊNCIA -- imagem real dos controles de síntese",
        "",
        f"- Resultado geral: **{'OK' if result.get('ok') else 'FALHOU'}**",
        "",
    ]
    if "reason" in result and not result.get("ok", True):
        lines += [f"- Motivo: {result['reason']}", ""]
        if "registry" not in result:
            return "\n".join(lines)
    lines += [
        "## Checagens",
        "",
        f"- Hash do manifesto congelado confere com o registrado no registry: {result.get('manifest_sha256_matches_registered')}",
        f"- Sobreposição por `lesion_id` (deve ser False): {result.get('overlap_lesion_id')}",
        f"- Sobreposição por `image_id` (deve ser False): {result.get('overlap_image_id')}",
        f"- Imagem carrega: {result.get('image_loads')}",
        f"- SHA-256 da imagem confere com o registry: {result.get('image_sha256_matches_registry')}",
        f"- Caminho de imagem checado: `{result.get('image_path_checked')}`",
        "",
        "## Registry re-verificado",
        "",
        f"```json\n{json.dumps(registry, ensure_ascii=False, indent=2, default=str)}\n```",
        "",
    ]
    return "\n".join(lines)


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--metadata-csv", default=DEFAULT_METADATA_CSV)
    parser.add_argument("--frozen-manifest", default=DEFAULT_FROZEN_MANIFEST)
    parser.add_argument("--ham10000-images-dir", default=DEFAULT_HAM10000_IMAGES_DIR)
    parser.add_argument("--output-dir", default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--seed", type=int, default=SYNTHESIS_CONTROL_IMAGE_SEED)
    parser.add_argument(
        "--verify-only",
        action="store_true",
        help="NENHUMA seleção nova -- só reverifica um registry já gravado (sem inferência).",
    )
    parser.add_argument(
        "--registry-path",
        default=None,
        help=f"Usado só com --verify-only; default: <output-dir>/{REGISTRY_FILENAME}",
    )
    return parser


def main() -> None:
    args = build_arg_parser().parse_args()
    output_dir = Path(args.output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    frozen_manifest_path = Path(args.frozen_manifest)

    if args.verify_only:
        registry_path = Path(args.registry_path) if args.registry_path else output_dir / REGISTRY_FILENAME
        result = verify_existing_registry(
            registry_path, frozen_manifest_path, images_dir=Path(args.ham10000_images_dir)
        )
        report_path = output_dir / VERIFY_ONLY_REPORT_FILENAME
        with open(report_path, "w", encoding="utf-8") as f:
            f.write(build_verify_only_markdown(result))
        print(json.dumps({k: v for k, v in result.items() if k != "registry"}, ensure_ascii=False, indent=2, default=str))
        print(f"relatório de verificação (sem inferência) gravado: {report_path}")
        if not result["ok"]:
            print("RESULTADO: FALHOU -- não autorizar as 8 chamadas reais até corrigir isto.")
        else:
            print("RESULTADO: OK -- seguro prosseguir para scripts/run_synthesis_controls_real_image.py.")
        return

    result = select_real_image(
        metadata_csv=args.metadata_csv,
        frozen_manifest_path=frozen_manifest_path,
        images_dir=Path(args.ham10000_images_dir),
        seed=args.seed,
    )
    registry_path = output_dir / REGISTRY_FILENAME
    with open(registry_path, "w", encoding="utf-8") as f:
        json.dump(result, f, ensure_ascii=False, indent=2, default=str)
    report_path = output_dir / REGISTRY_REPORT_FILENAME
    with open(report_path, "w", encoding="utf-8") as f:
        f.write(build_registry_markdown(result))

    print(f"status: {result['verification_status']}")
    if result["verification_status"] != STATUS_OK:
        print(
            f"PENDENTE -- motivo: {result.get('reason')}. Nenhuma imagem foi "
            "escolhida silenciosamente como alternativa."
        )
    else:
        print(
            f"imagem real selecionada: lesion_id={result['chosen_lesion_id']} "
            f"image_id={result['chosen_image_id']} dx={result['chosen_dx']}"
        )
    print(f"registry gravado: {registry_path}")
    print(f"relatório gravado: {report_path}")


if __name__ == "__main__":
    main()
