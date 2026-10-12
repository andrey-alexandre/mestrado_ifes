"""Generate Feature Extraction debug visualizations for selected PH2 images."""
import argparse
import sys
from pathlib import Path

import yaml

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.data.dataset import PH2Dataset
from src.features.visualization import save_feature_debug_visualization


def parse_args():
    parser = argparse.ArgumentParser(description="Gera visualizacoes PNG de depuracao da Feature Extraction.")
    parser.add_argument("--config", default="config/config.yaml")
    parser.add_argument("--image-ids", nargs="*", default=None, help="IDs PH2 especificos. Vazio usa --max-images.")
    parser.add_argument("--max-images", type=int, default=5)
    parser.add_argument("--output-dir", default=None)
    return parser.parse_args()


def main():
    args = parse_args()
    with open(args.config, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)
    dataset = PH2Dataset(
        dataset_dir=cfg["paths"]["ph2_dataset_dir"],
        random_seed=cfg["pipeline"]["random_seed"],
        metadata_txt=cfg["paths"].get("ph2_metadata_txt"),
    )
    entries = dataset.get_image_paths(max_images=0, balanced=False)
    selected = []
    requested = set(args.image_ids or [])
    for image_path, _, _, mask_path in entries:
        image_id = Path(image_path).stem
        if requested and image_id not in requested:
            continue
        selected.append((image_id, image_path, mask_path))
        if not requested and len(selected) >= args.max_images:
            break
    output_dir = args.output_dir or cfg["paths"].get("feature_debug_dir", "results/feature_debug")
    for image_id, image_path, mask_path in selected:
        print(save_feature_debug_visualization(image_path, mask_path, output_dir, image_id=image_id))


if __name__ == "__main__":
    main()
