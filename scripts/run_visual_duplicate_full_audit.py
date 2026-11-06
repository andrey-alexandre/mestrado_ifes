"""Task 11: full-scale run of task 8/9/10's ORB+RANSAC + degeneracy-guard
method over ALL 421,196 dHash candidates from task 7
(``experiments/data_audit/visual_candidates_index.json``), with NO pHash
gate, then a dimensioned (not exhaustive) manual-review sample.

Explicit researcher decision this script implements literally (not a
default this script invented):
  candidatos dHash (421.196, Tarefa 7, já existentes)
    -> ORB+RANSAC (verify_geometric_match, Tarefa 8/9) em TODOS, sem gate de pHash
    -> classify_with_degeneracy_guard (Tarefa 10, limiar 0.061005) por prioridade
    -> fila de revisão dimensionada (não todos os 421.196 revisados manualmente)

No pHash filter is used anywhere in this run -- any pHash gate would
discard candidates before the geometric stage runs, exactly the dependency
tasks 8/9 showed is unsafe to rely on without evidence of what it discards.

Expected runtime: ~7.6 hours (linear extrapolation from task 9's measured
~129s/2,000 pairs). This is expected, not a failure. The run is resumable:
rerunning this script with the same --checkpoint path skips every
candidate already recorded there and continues from where it stopped --
see src/data_audit/visual_duplicate_full_run.py::run_full_pass. To resume
manually after an interruption (e.g. the process was killed, the machine
restarted), just run this exact command again; no flag is needed, presence
of the checkpoint file is itself the resume signal.

Nothing is excluded here: every one of the 421,196 candidates ends up in
the checkpoint regardless of its classification. The degeneracy guard
changes manual-review PRIORITY ordering/sampling only.
"""
from __future__ import annotations

import argparse
import json
import sys
import zipfile
from pathlib import Path
from typing import Any

import yaml
from PIL import Image

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

from src.data_audit import geometric_verification as gv  # noqa: E402
from src.data_audit import visual_duplicate_full_run as full_run  # noqa: E402
from src.data_audit import visual_duplicate_pilot_v2 as v2  # noqa: E402

MANIFEST_DIR = REPO_ROOT / "experiments" / "data_audit"
CANDIDATES_INDEX_PATH = MANIFEST_DIR / "visual_candidates_index.json"
CHECKPOINT_PATH = MANIFEST_DIR / "full_audit_checkpoint.jsonl"
POPULATION_SUMMARY_PATH = MANIFEST_DIR / "full_audit_population_summary.json"
REVIEW_QUEUE_PATH = MANIFEST_DIR / "full_audit_review_queue.json"
REVIEW_EVIDENCE_DIR = MANIFEST_DIR / "full_audit_review_evidence"


class ImageSourceReader:
    """Dispatches byte reads across disk (HAM10000) and the already-
    materialized zips (ISIC2018 Task1, BCN20000 train/test) -- the exact
    same files tasks 6/7/8/9/10 used; nothing is re-extracted or re-listed
    here. Deliberately a standalone copy of tasks 9/10's identical class
    (script-level plumbing, not analysis logic) rather than importing
    across sibling script modules, same convention as those scripts."""

    def __init__(self, ham_dir: Path, isic_zip_path: Path, bcn_train_zip_path: Path, bcn_test_zip_path: Path):
        self.ham_dir = ham_dir
        self._isic_zf = zipfile.ZipFile(isic_zip_path)
        self._bcn_train_zf = zipfile.ZipFile(bcn_train_zip_path)
        self._bcn_test_zf = zipfile.ZipFile(bcn_test_zip_path)

    def read_ham10000(self, path: str) -> bytes:
        return (self.ham_dir / path).read_bytes()

    def read_other(self, path: str) -> bytes:
        if path.startswith("bcn_20k_train/"):
            return self._bcn_train_zf.read(path)
        if path.startswith("bcn_20k_test/"):
            return self._bcn_test_zf.read(path)
        return self._isic_zf.read(path)

    def close(self) -> None:
        self._isic_zf.close()
        self._bcn_train_zf.close()
        self._bcn_test_zf.close()


def _load_json(path: Path) -> dict[str, Any]:
    with open(path, encoding="utf-8") as f:
        return json.load(f)


def _save_json(path: Path, data: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)


def _render_evidence_for_selected(reader: ImageSourceReader, selected_rows: list[dict[str, Any]]) -> None:
    """Render one evidence image per selected review-queue row, mutating
    each row in place with its ``evidence_image`` path. ``"evaluated"``
    rows (forte/forte_degenerada/fraco) get the ORB inlier-match drawing
    (``draw_match_evidence``); ``"not_evaluable"`` rows get a plain
    side-by-side (there is nothing to draw matches for) so they can still
    be opened and judged. A read/draw failure on an already-selected row is
    recorded, not raised -- the review step must not lose the rest of the
    queue over one bad file.
    """
    REVIEW_EVIDENCE_DIR.mkdir(parents=True, exist_ok=True)
    for i, row in enumerate(selected_rows):
        out_path = REVIEW_EVIDENCE_DIR / f"{i:05d}.png"
        try:
            data_a = reader.read_ham10000(row["path_a_ham10000"])
            data_b = reader.read_other(row["path_b"])
            array_a = v2.decode_grayscale_array(data_a)
            array_b = v2.decode_grayscale_array(data_b)
            if row["status"] == "evaluated":
                result = gv.verify_geometric_match(array_a, array_b)
                evidence_bgr = gv.draw_match_evidence(array_a, array_b, result)
                Image.fromarray(evidence_bgr[:, :, ::-1]).save(out_path)
            else:
                image_a = Image.fromarray(array_a).convert("RGB")
                image_b = Image.fromarray(array_b).convert("RGB")
                width = image_a.width + image_b.width
                height = max(image_a.height, image_b.height)
                canvas = Image.new("RGB", (width, height), (255, 255, 255))
                canvas.paste(image_a, (0, 0))
                canvas.paste(image_b, (image_a.width, 0))
                canvas.save(out_path)
            row["evidence_image"] = str(out_path.relative_to(REPO_ROOT)).replace("\\", "/")
            row["evidence_render_error"] = None
        except Exception as exc:  # noqa: BLE001 -- must not abort the batch over one bad file
            row["evidence_image"] = None
            row["evidence_render_error"] = str(exc)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default=str(REPO_ROOT / "config" / "dataset_inventory.yaml"))
    parser.add_argument("--checkpoint", default=str(CHECKPOINT_PATH))
    parser.add_argument(
        "--progress-every", type=int, default=1000, help="Imprime progresso a cada N pares processados NESTA execução."
    )
    parser.add_argument(
        "--skip-review-queue",
        action="store_true",
        help="Só executa a passada ORB+guarda (item 1); não gera a fila de revisão (item 2) nem renderiza evidências.",
    )
    args = parser.parse_args()
    checkpoint_path = Path(args.checkpoint)

    with open(args.config, encoding="utf-8") as f:
        config = yaml.safe_load(f)

    ham_dir = Path(config["ham10000"]["images_dir"])
    isic_zip = Path(config["isic2018_task1"]["images_zip"])
    bcn_content = _load_json(MANIFEST_DIR / "bcn20000_content_manifest.json")
    bcn_train_zip = Path(bcn_content["train"]["materialization"]["dest_path"])
    bcn_test_zip = Path(bcn_content["test"]["materialization"]["dest_path"])

    candidates_index = _load_json(CANDIDATES_INDEX_PATH)
    candidates = candidates_index["candidates"]
    print(f"candidatos totais (Tarefa 7, reaproveitados sem regenerar): {len(candidates)}")

    reader = ImageSourceReader(ham_dir, isic_zip, bcn_train_zip, bcn_test_zip)
    try:
        def process_fn(cand: dict[str, Any]) -> dict[str, Any]:
            return full_run.process_candidate(reader, cand, full_run.DEGENERACY_DISPERSION_THRESHOLD)

        run_summary = full_run.run_full_pass(
            candidates, checkpoint_path, process_fn, progress_every=args.progress_every
        )
    finally:
        reader.close()

    print(
        f"item 1 concluído: {run_summary['n_processed_this_run']} processados nesta execução, "
        f"total acumulado {run_summary['n_already_done_at_start'] + run_summary['n_processed_this_run']}/"
        f"{run_summary['n_total']}, completo={run_summary['complete']} "
        f"({run_summary['elapsed_seconds_this_run'] / 3600:.2f}h nesta execução)"
    )
    if not run_summary["complete"]:
        print("EXECUÇÃO INCOMPLETA -- rode este mesmo comando novamente para retomar do checkpoint.")
        _save_json(POPULATION_SUMMARY_PATH, {"run_summary": run_summary, "complete": False})
        return

    all_rows_by_key, n_skipped = full_run.load_checkpoint(checkpoint_path)
    all_rows = list(all_rows_by_key.values())
    population = full_run.summarize_population(all_rows)
    population["run_summary"] = run_summary
    population["n_skipped_malformed_checkpoint_lines_on_final_load"] = n_skipped
    _save_json(POPULATION_SUMMARY_PATH, population)
    print(f"composição completa da população: {population['counts_by_final_strength']}")
    print(f"resumo de população gravado: {POPULATION_SUMMARY_PATH}")

    if args.skip_review_queue:
        print("--skip-review-queue: fila de revisão não gerada nesta chamada.")
        return

    queue = full_run.build_review_queue(all_rows)
    print(
        f"fila de revisão: forte={len(queue['forte']['selected'])}/{queue['population_by_status']['forte']} "
        f"forte_degenerada={len(queue['forte_degenerada']['selected'])}/{queue['population_by_status']['forte_degenerada']} "
        f"fraco={queue['fraco']['n_selected']}/{queue['population_by_status']['fraco']} "
        f"not_evaluable={len(queue['not_evaluable']['selected'])}/{queue['population_by_status']['not_evaluable']}"
    )

    selected_rows = [r for r in queue["rows"] if r["selected_for_review"]]
    reader = ImageSourceReader(ham_dir, isic_zip, bcn_train_zip, bcn_test_zip)
    try:
        _render_evidence_for_selected(reader, selected_rows)
    finally:
        reader.close()
    n_render_errors = sum(1 for r in selected_rows if r.get("evidence_render_error"))
    print(f"evidências renderizadas: {len(selected_rows) - n_render_errors} ({n_render_errors} erros)")

    _save_json(REVIEW_QUEUE_PATH, queue)
    print(f"fila de revisão gravada: {REVIEW_QUEUE_PATH}")
    print(f"evidências em: {REVIEW_EVIDENCE_DIR.relative_to(REPO_ROOT)}")


if __name__ == "__main__":
    main()
