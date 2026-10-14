"""Entry point CLI do experimento de segmentação (ISIC 2018 -> PH²).

Cada etapa pode ser executada separadamente; `all` encadeia todas. Nada é
executado automaticamente: treinos e chamadas a VLMs só ocorrem sob o subcomando
correspondente (VLMs exigem `--run-vlm`). Etapas concluídas são puladas quando os
artefatos são compatíveis com a configuração (use --force para refazer).

Uso:
    python scripts/run_segmentation_experiment.py train-isic
    python scripts/run_segmentation_experiment.py make-folds
    python scripts/run_segmentation_experiment.py direct-transfer
    python scripts/run_segmentation_experiment.py finetune
    python scripts/run_segmentation_experiment.py compare
    python scripts/run_segmentation_experiment.py multiagent              # sem VLMs
    python scripts/run_segmentation_experiment.py multiagent --run-vlm    # chama os VLMs
    python scripts/run_segmentation_experiment.py final-model
    python scripts/run_segmentation_experiment.py report
    python scripts/run_segmentation_experiment.py all --config config/config.yaml

    # Reaproveitando um checkpoint já treinado no ISIC 2018 (state_dict legado .pkl)
    python scripts/run_segmentation_experiment.py direct-transfer --checkpoint U_Net-100-0.0005-70-0.6667.pkl
"""
import argparse
import logging
import sys
from pathlib import Path

import yaml

# Garante que o diretório raiz do projeto está no sys.path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.segmentation import config as cfgmod
from src.segmentation.artifacts import StageStore

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s — %(message)s",
)
logger = logging.getLogger("segmentation_experiment")

COMMANDS = ["train-isic", "make-folds", "direct-transfer", "finetune", "compare",
            "multiagent", "final-model", "report", "all"]


def parse_args():
    p = argparse.ArgumentParser(description="Pipeline experimental de segmentação pós-qualificação.")
    p.add_argument("command", choices=COMMANDS, help="Etapa a executar.")
    p.add_argument("--config", default="config/config.yaml",
                   help="Caminho para o arquivo de configuração YAML. (padrão: config/config.yaml)")
    p.add_argument("--force", action="store_true",
                   help="Refaz etapas mesmo com artefatos existentes (sobrescreve).")
    p.add_argument("--checkpoint", default=None,
                   help="Checkpoint inicial (ISIC 2018) para transferência/fine-tuning. "
                        "Padrão: {checkpoints_dir}/isic2018/best.pt.")
    p.add_argument("--strategy", choices=["direct_transfer", "finetune"], default=None,
                   help="Estratégia para 'multiagent'/'final-model' (padrão: seleção da etapa 'compare').")
    p.add_argument("--run-vlm", action="store_true",
                   help="Em 'multiagent': executa os VLMs (custo de API). Padrão: não executa.")
    p.add_argument("--isic-limit", type=int, default=None,
                   help="Limita nº de imagens do ISIC (teste rápido; altera o fingerprint).")
    # Overrides pontuais (padrão do repositório: CLI sobrescreve o YAML)
    p.add_argument("--isic-image-dir", default=None)
    p.add_argument("--isic-mask-dir", default=None)
    p.add_argument("--ph2-dataset-dir", default=None)
    p.add_argument("--ph2-metadata-txt", default=None)
    p.add_argument("--output-root", default=None, help="Sobrescreve output.root_dir.")
    p.add_argument("--device", default=None, help="cpu | cuda | mps | auto.")
    p.add_argument("--max-epochs", type=int, default=None)
    p.add_argument("--seed", type=int, default=None)
    return p.parse_args()


def load_cfg(args) -> dict:
    with open(args.config, "r", encoding="utf-8") as f:
        full = yaml.safe_load(f) or {}
    sec = full.setdefault(cfgmod.SECTION, {})
    paths = full.setdefault("paths", {})

    def put(path, value):
        if value is None:
            return
        cur = sec
        for key in path[:-1]:
            cur = cur.setdefault(key, {})
        cur[path[-1]] = value

    put(["datasets", "isic2018", "image_dir"], args.isic_image_dir)
    put(["datasets", "isic2018", "mask_dir"], args.isic_mask_dir)
    if args.ph2_dataset_dir:
        paths["ph2_dataset_dir"] = args.ph2_dataset_dir
    if args.ph2_metadata_txt:
        paths["ph2_metadata_txt"] = args.ph2_metadata_txt
    put(["output", "root_dir"], args.output_root)
    put(["segmentation", "device"], args.device)
    put(["segmentation", "max_epochs"], args.max_epochs)
    put(["segmentation", "seed"], args.seed)
    if args.run_vlm:
        put(["multiagent", "run_vlm"], True)
    return cfgmod.build_config(full)


def main():
    args = parse_args()
    cfg = load_cfg(args)
    store = StageStore(cfg["output"]["results_dir"], force=args.force)
    cmd = args.command

    # Importações pesadas só após validar a configuração
    from src.segmentation import stages
    from src.segmentation.comparison import run_comparison
    from src.segmentation.multiagent import run_multiagent
    from src.segmentation.report import build_report

    steps = {
        "train-isic": lambda: stages.run_train_isic(cfg, store, args.isic_limit),
        "make-folds": lambda: stages.run_make_folds(cfg, store),
        "direct-transfer": lambda: stages.run_direct_transfer(cfg, store, args.checkpoint),
        "finetune": lambda: stages.run_finetune(cfg, store, args.checkpoint),
        "compare": lambda: run_comparison(cfg, store),
        "multiagent": lambda: run_multiagent(cfg, store, args.strategy,
                                             True if args.run_vlm else None),
        "final-model": lambda: stages.prepare_final_model(cfg, store, args.strategy, args.checkpoint),
        "report": lambda: build_report(cfg),
    }
    order = ["train-isic", "make-folds", "direct-transfer", "finetune", "compare",
             "multiagent", "final-model", "report"]
    to_run = order if cmd == "all" else [cmd]
    if cmd == "all" and args.checkpoint:
        to_run = [s for s in order if s != "train-isic"]  # checkpoint externo: sem retreino
    for step in to_run:
        logger.info("=== Etapa: %s ===", step)
        steps[step]()
    logger.info("Concluído. Resultados em: %s | Relatórios em: %s",
                cfg["output"]["results_dir"], cfg["output"]["reports_dir"])


if __name__ == "__main__":
    try:
        main()
    except (cfgmod.ConfigError, ValueError, FileNotFoundError, RuntimeError) as exc:
        logger.error("%s", exc)
        sys.exit(2)
