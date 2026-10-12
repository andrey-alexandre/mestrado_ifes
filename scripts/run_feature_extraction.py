"""Generate frozen Feature Extraction JSON packages without LLM inference."""
import argparse
import json
import sys
from pathlib import Path

import yaml

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.data.dataset import PH2Dataset
from src.features.extractor import build_feature_package


def parse_args():
    parser = argparse.ArgumentParser(description="Executa somente a Feature Extraction congelada para PH2.")
    parser.add_argument("--config", default="config/config.yaml")
    parser.add_argument("--max-images", type=int, default=0)
    parser.add_argument("--balanced", action="store_true")
    parser.add_argument("--output", default=None)
    return parser.parse_args()


def main():
    args = parse_args()
    with open(args.config, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)
    fe_cfg = cfg.get("feature_extraction", {})
    dataset = PH2Dataset(
        dataset_dir=cfg["paths"]["ph2_dataset_dir"],
        random_seed=cfg["pipeline"]["random_seed"],
        metadata_txt=cfg["paths"].get("ph2_metadata_txt"),
    )
    entries = dataset.get_image_paths(max_images=args.max_images, balanced=args.balanced)
    results = {}
    for image_path, _, _, mask_path in entries:
        package = build_feature_package(
            image_path=image_path,
            mask_path=mask_path,
            experiment_condition="feature_extraction_only",
            config=fe_cfg,
        )
        results[image_path] = package.model_dump()
    output_path = Path(args.output or Path(cfg["paths"].get("feature_extraction_output_dir", "results/feature_extraction")) / "feature_extraction_only.json")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=4, ensure_ascii=False)
    print(output_path)


if __name__ == "__main__":
    main()
