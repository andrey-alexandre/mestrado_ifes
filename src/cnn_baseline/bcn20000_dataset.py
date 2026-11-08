"""PyTorch Dataset for the approved BCN20000 train/val split (CNN
baseline, stage 3). Reads the split manifest
(experiments/cnn_baseline/bcn20000_split_manifest.csv, produced by
scripts/run_bcn20000_split.py: columns lesion_id, bcn_filename,
source_partition, diagnosis, binary_target, split) and loads images
directly from the already-materialized zip
D:\\Andrey\\Mestrado\\_extracted\\BCN_20k_train.zip (task 6), never
extracting individual files to disk -- consistent with this project's
established policy against mass zip extraction.

HAM10000 is never read or referenced here: it is reserved exclusively
for the final external evaluation.
"""
from __future__ import annotations

import csv
import io
import zipfile
from pathlib import Path
from typing import Any, Callable, Sequence

from PIL import Image
from torch.utils.data import Dataset

# melanoma=1 is the chosen convention (matches src.data_audit.sampling's
# 'melanoma'/'nao_melanoma' naming for HAM10000 -- kept consistent across
# the two datasets even though HAM10000 never participates here).
BINARY_TARGET_TO_LABEL = {"melanoma": 1, "nao_melanoma": 0}
LABEL_TO_BINARY_TARGET = {v: k for k, v in BINARY_TARGET_TO_LABEL.items()}

ZIP_ENTRY_PREFIX = "bcn_20k_train/"


def read_split_manifest(manifest_csv: str | Path) -> list[dict[str, Any]]:
    """Plain local-file CSV read (not zip-embedded, unlike most other CSVs
    in this project), so a direct csv.DictReader is simplest here."""
    with open(manifest_csv, encoding="utf-8") as f:
        return list(csv.DictReader(f))


def filter_rows_by_split(rows: Sequence[dict[str, Any]], split: str) -> list[dict[str, Any]]:
    return [r for r in rows if r["split"] == split]


class BCN20000ZipDataset(Dataset):
    """Loads images directly from the materialized BCN20000 train zip.

    Opens its own ``zipfile.ZipFile`` handle lazily, inside ``__getitem__``
    (never at construction time, never shared), because an open ZipFile
    handle is not guaranteed safe/picklable across DataLoader
    multiprocessing workers. On Windows (this project's platform),
    ``DataLoader(num_workers>0)`` uses the 'spawn' start method, which
    pickles the Dataset object to send to each worker process; keeping
    ``self._zf = None`` until first access means nothing unpicklable is
    ever sent, and each worker transparently opens (and keeps) its own
    handle on first use.
    """

    def __init__(
        self,
        rows: Sequence[dict[str, Any]],
        zip_path: str | Path,
        transform: Callable | None = None,
        augment: Callable | None = None,
    ) -> None:
        self.rows = list(rows)
        self.zip_path = str(zip_path)
        self.transform = transform
        self.augment = augment
        self._zf: zipfile.ZipFile | None = None

    def __len__(self) -> int:
        return len(self.rows)

    def _get_zip(self) -> zipfile.ZipFile:
        if self._zf is None:
            self._zf = zipfile.ZipFile(self.zip_path)
        return self._zf

    def __getitem__(self, idx: int):
        row = self.rows[idx]
        entry_name = f"{ZIP_ENTRY_PREFIX}{row['bcn_filename']}"
        data = self._get_zip().read(entry_name)
        image = Image.open(io.BytesIO(data)).convert("RGB")

        if self.augment is not None:
            image = self.augment(image)
        if self.transform is not None:
            image = self.transform(image)

        label = BINARY_TARGET_TO_LABEL[row["binary_target"]]
        return image, label, row["lesion_id"], row["bcn_filename"]
