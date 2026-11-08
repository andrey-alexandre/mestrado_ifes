"""PyTorch Dataset for ISIC2018 Task1 (image + ground-truth mask), read
directly from the two never-extracted zips (stage 4, segmentation
baseline development). HAM10000 is never read or referenced here: it is
reserved exclusively for the final external evaluation of segmentation.

Resolution choice: 256x256. Rationale, not left implicit:
  - Must be divisible by 16 (2**4) for ``src.segmentation.unet_model.UNet``'s
    4 pooling/upsampling stages to produce exactly matching skip-connection
    shapes on both encoder and decoder sides -- 256 satisfies this exactly
    (256/16 = 16), whereas the native ISIC2018 image sizes do not (seen
    ranging from 576x542 up to 6708x4461 in earlier inspection).
  - 256x256 is a common, modest baseline resolution for U-Net segmentation
    pilots -- small enough to keep this pilot (and later full folds
    training) tractable on CPU, without pretending this is a final,
    tuned choice for whatever architecture comparison happens later.
  - Both image and mask are resized with the SAME target size, but
    different interpolation: bilinear for the RGB image (smooth), nearest
    for the binary mask (never invent intermediate "half-lesion" pixel
    values at mask boundaries).
"""
from __future__ import annotations

import io
import zipfile
from pathlib import Path
from typing import Any, Callable, Sequence

import numpy as np
from PIL import Image
from torch.utils.data import Dataset
from torchvision import transforms as T

TARGET_SIZE = 256  # divisible by 16; see module docstring

IMAGES_ZIP_ENTRY_PREFIX = "ISIC2018_Task1-2_Training_Input/"
MASKS_ZIP_ENTRY_PREFIX = "ISIC2018_Task1_Training_GroundTruth/"
MASK_SUFFIX = "_segmentation.png"


def image_id_from_entry_name(entry_name: str) -> str:
    """'ISIC2018_Task1-2_Training_Input/ISIC_0000000.jpg' -> 'ISIC_0000000'."""
    return Path(entry_name).stem


def build_image_transform() -> Callable:
    return T.Compose(
        [
            T.Resize((TARGET_SIZE, TARGET_SIZE), interpolation=T.InterpolationMode.BILINEAR),
            T.ToTensor(),  # -> float32 in [0, 1], shape (3, H, W)
        ]
    )


def _resize_mask_nearest(mask: Image.Image) -> np.ndarray:
    resized = mask.resize((TARGET_SIZE, TARGET_SIZE), resample=Image.Resampling.NEAREST)
    arr = np.array(resized, dtype=np.uint8)
    return (arr > 127).astype(np.float32)  # binary: lesion=1.0, background=0.0


def load_original_mask(masks_zip: str | Path, image_id: str) -> np.ndarray:
    """Reads the ground-truth mask at its NATIVE resolution -- no resize at
    all, in either direction. This is the reference array used for metric
    computation (``src.segmentation.metrics.compute_image_metrics``), never
    the 256x256 training-resolution mask returned by
    ``ISIC2018Task1Dataset.__getitem__``. No continuous interpolation is
    ever applied to a reference mask anywhere in this pipeline: only the
    same binarization threshold (> 127) already used by
    :func:`_resize_mask_nearest`, applied at the original pixel grid.
    """
    mask_entry = f"{MASKS_ZIP_ENTRY_PREFIX}{image_id}{MASK_SUFFIX}"
    with zipfile.ZipFile(masks_zip) as zf:
        mask_bytes = zf.read(mask_entry)
    mask = Image.open(io.BytesIO(mask_bytes)).convert("L")
    arr = np.array(mask, dtype=np.uint8)
    return (arr > 127).astype(np.uint8)  # binary: lesion=1, background=0


class ISIC2018Task1Dataset(Dataset):
    """Loads (image, mask) pairs directly from the two ISIC2018 Task1 zips.

    Opens its own ``zipfile.ZipFile`` handles lazily, inside
    ``__getitem__`` (never at construction time, never shared) -- same
    multiprocessing-safety pattern already used by
    ``src.cnn_baseline.bcn20000_dataset.BCN20000ZipDataset`` (an open
    ``ZipFile`` is not guaranteed safe/picklable across Windows
    ``DataLoader`` worker processes using 'spawn').
    """

    def __init__(
        self,
        image_ids: Sequence[str],
        images_zip: str | Path,
        masks_zip: str | Path,
        image_transform: Callable | None = None,
    ) -> None:
        self.image_ids = list(image_ids)
        self.images_zip = str(images_zip)
        self.masks_zip = str(masks_zip)
        self.image_transform = image_transform or build_image_transform()
        self._images_zf: zipfile.ZipFile | None = None
        self._masks_zf: zipfile.ZipFile | None = None

    def __len__(self) -> int:
        return len(self.image_ids)

    def _get_images_zip(self) -> zipfile.ZipFile:
        if self._images_zf is None:
            self._images_zf = zipfile.ZipFile(self.images_zip)
        return self._images_zf

    def _get_masks_zip(self) -> zipfile.ZipFile:
        if self._masks_zf is None:
            self._masks_zf = zipfile.ZipFile(self.masks_zip)
        return self._masks_zf

    def __getitem__(self, idx: int):
        image_id = self.image_ids[idx]
        image_entry = f"{IMAGES_ZIP_ENTRY_PREFIX}{image_id}.jpg"
        mask_entry = f"{MASKS_ZIP_ENTRY_PREFIX}{image_id}{MASK_SUFFIX}"

        image_bytes = self._get_images_zip().read(image_entry)
        mask_bytes = self._get_masks_zip().read(mask_entry)

        image = Image.open(io.BytesIO(image_bytes)).convert("RGB")
        mask = Image.open(io.BytesIO(mask_bytes)).convert("L")

        image_tensor = self.image_transform(image)
        mask_array = _resize_mask_nearest(mask)
        import torch

        mask_tensor = torch.from_numpy(mask_array).unsqueeze(0)  # (1, H, W)

        return image_tensor, mask_tensor, image_id


def list_available_image_ids(images_zip: str | Path) -> list[str]:
    """All image ids present in the images zip (from its central
    directory only -- no extraction), reusing
    ``core.zip_entry_manifest``-equivalent logic directly here to avoid
    importing the whole ``data_audit`` package just for one function.
    Entries other than ``.jpg`` (``ATTRIBUTION.txt``, ``LICENSE.txt``,
    the bare directory entry) are skipped.
    """
    with zipfile.ZipFile(images_zip) as zf:
        names = [i.filename for i in zf.infolist() if i.filename.lower().endswith(".jpg")]
    return sorted(image_id_from_entry_name(n) for n in names)


def list_available_mask_ids(masks_zip: str | Path) -> list[str]:
    """Same as :func:`list_available_image_ids`, for the ground-truth
    masks zip. Entry names end in ``{image_id}_segmentation.png``, so the
    suffix (not just the extension) must be stripped to recover the id.
    """
    with zipfile.ZipFile(masks_zip) as zf:
        names = [i.filename for i in zf.infolist() if i.filename.lower().endswith(".png")]
    ids = []
    for name in names:
        stem = Path(name).stem  # '..._GroundTruth/ISIC_0000000_segmentation.png' -> 'ISIC_0000000_segmentation'
        if stem.endswith("_segmentation"):
            ids.append(stem[: -len("_segmentation")])
    return sorted(ids)


def verify_image_mask_correspondence(
    images_zip: str | Path, masks_zip: str | Path
) -> dict[str, Any]:
    """Explicit image<->mask id correspondence check (not just an
    incidental side effect of a later KeyError if something is missing).

    Does NOT reuse the Task 6 audit's byte-level manifests on disk
    (``experiments/data_audit/isic2018_task1_*_manifest.json``) because
    those are gitignored/regenerable outputs that may or may not be
    present on a given machine -- this recomputes the much cheaper
    id-set comparison directly from each zip's central directory (no
    extraction, no hashing), which is self-sufficient and just as
    correct for this purpose. A prior task already confirmed a perfect
    2,594/2,594 byte-level pairing; this is a lighter, independent
    re-check appropriate for a pilot, not a re-run of that full audit.
    """
    image_ids = set(list_available_image_ids(images_zip))
    mask_ids = set(list_available_mask_ids(masks_zip))
    only_images = sorted(image_ids - mask_ids)
    only_masks = sorted(mask_ids - image_ids)
    return {
        "n_images": len(image_ids),
        "n_masks": len(mask_ids),
        "n_matched": len(image_ids & mask_ids),
        "n_only_in_images": len(only_images),
        "n_only_in_masks": len(only_masks),
        "only_in_images": only_images,
        "only_in_masks": only_masks,
        "ok": not only_images and not only_masks,
    }
