"""Smoke da CLI: pipeline completo com dados sintéticos e erros de configuração claros."""
import subprocess
import sys
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parents[2]
SCRIPT = ROOT / "scripts" / "run_segmentation_experiment.py"


def _run(args, cwd):
    # O pacote externo da U-Net é simulado via sitecustomize-like preload (ver `-c` abaixo).
    code = (
        "import sys, runpy; sys.path.insert(0, %r); sys.path.insert(0, %r);"
        "import conftest; sys.argv = ['run_segmentation_experiment.py'] + %r;"
        "runpy.run_path(%r, run_name='__main__')"
    ) % (str(ROOT), str(Path(__file__).parent), [str(a) for a in args], str(SCRIPT))
    return subprocess.run([sys.executable, "-c", code], cwd=cwd, capture_output=True, text=True, timeout=900)


def _write_cfg(tmp_path, cfg_factory, isic, ph2):
    cfg = cfg_factory(isic=isic, ph2=ph2)
    keep = {k: cfg[k] for k in ("datasets", "segmentation", "transfer_learning", "evaluation", "multiagent",
                                "output")}
    keep["multiagent"]["feature_extractor"] = "modulo.inexistente:fn"
    path = tmp_path / "cfg.yaml"
    path.write_text(yaml.safe_dump({"segmentation_experiment": keep,
                                    "llm": {"model_name": "fake", "temperature": 0}}), encoding="utf-8")
    return path


def test_cli_missing_required_paths_exit_code_and_message(tmp_path):
    cfg = tmp_path / "c.yaml"
    cfg.write_text("paths: {}\n", encoding="utf-8")
    r = _run(["train-isic", "--config", cfg], tmp_path)
    assert r.returncode == 2
    assert "datasets.isic2018.image_dir" in r.stderr


def test_cli_invalid_value_is_rejected(tmp_path):
    cfg = tmp_path / "c.yaml"
    cfg.write_text("segmentation_experiment: {segmentation: {optimizer: rmsprop}}\n", encoding="utf-8")
    r = _run(["train-isic", "--config", cfg], tmp_path)
    assert r.returncode == 2 and "optimizer" in r.stderr


def test_cli_all_stages_with_synthetic_data(world, cfg_factory, tmp_path):
    cfg, isic, ph2 = world
    path = _write_cfg(tmp_path, cfg_factory, isic, ph2)
    r = _run(["all", "--config", path], tmp_path)
    assert r.returncode == 0, r.stderr[-3000:]
    out = Path(cfg["output"]["root_dir"])
    assert (out / "reports" / "report.md").exists()
    assert (out / "checkpoints" / "final" / "final_segmenter.pt").exists()
    assert (out / "results" / "multiagent" / "summary.json").exists()

    # segunda execução: tudo compatível -> etapas puladas, sem erro
    r2 = _run(["all", "--config", path], tmp_path)
    assert r2.returncode == 0, r2.stderr[-3000:]
    assert "etapa pulada" in r2.stderr

    # mudar a configuração sem --force deve ser recusado
    cfgd = yaml.safe_load(path.read_text(encoding="utf-8"))
    cfgd["segmentation_experiment"]["segmentation"]["learning_rate"] = 0.01
    path.write_text(yaml.safe_dump(cfgd), encoding="utf-8")
    r3 = _run(["train-isic", "--config", path], tmp_path)
    assert r3.returncode == 2 and "--force" in r3.stderr
