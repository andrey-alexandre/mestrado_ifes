"""U-Net segmentation model loading and inference utilities."""
import base64
import logging
import random
from io import BytesIO
from typing import Tuple

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from torchvision import transforms as T

from Image_Segmentation.code.network import U_Net, R2U_Net, AttU_Net, R2AttU_Net

logger = logging.getLogger(__name__)

MODEL_CLASSES = {
    "U_Net": U_Net,
    "R2U_Net": R2U_Net,
    "AttU_Net": AttU_Net,
    "R2AttU_Net": R2AttU_Net,
}


def load_segmentation_model(
    model_path: str,
    model_name: str = "U_Net",
    img_ch: int = 3,
    output_ch: int = 1,
    device: str = "cpu",
) -> torch.nn.Module:
    """Loads a pre-trained segmentation model from a .pkl checkpoint.

    Args:
        model_path: Path to the .pkl state dict file.
        model_name: Architecture name (U_Net, R2U_Net, AttU_Net, R2AttU_Net).
        img_ch: Number of input image channels.
        output_ch: Number of output channels.
        device: Target device ('cpu', 'cuda', 'mps').

    Returns:
        Loaded PyTorch model set to eval mode.
    """
    model_cls = MODEL_CLASSES.get(model_name)
    if model_cls is None:
        raise ValueError(
            f"Unknown model name: '{model_name}'. "
            f"Choose from: {list(MODEL_CLASSES.keys())}"
        )

    model = model_cls(img_ch=img_ch, output_ch=output_ch)
    model.load_state_dict(
        torch.load(model_path, map_location=torch.device(device))
    )
    model.eval()
    logger.info("Loaded %s from %s on device '%s'", model_name, model_path, device)
    return model


def convert_to_base64(pil_image: Image.Image) -> str:
    """Converts a PIL image to a Base64-encoded JPEG string.

    Args:
        pil_image: PIL image to encode.

    Returns:
        Base64 string.
    """
    pil_image = pil_image.convert("RGB")
    buffered = BytesIO()
    pil_image.save(buffered, format="JPEG")
    img_str = base64.b64encode(buffered.getvalue()).decode("utf-8")
    return img_str


def segment_image(
    image_path: str,
    model: torch.nn.Module,
    threshold: float = 0.4,
    resize_range: Tuple[int, int] = (300, 320),
) -> Tuple[str, str]:
    """Segments a skin lesion image using the U-Net model.

    Applies random resize augmentation, runs the segmentation model, and returns
    both original and segmented images as Base64-encoded JPEG strings.

    Args:
        image_path: Path to the input JPEG image.
        model: Pre-loaded segmentation model (eval mode expected).
        threshold: Sigmoid threshold for binary mask application.
        resize_range: (min, max) pixel range for random resize augmentation.

    Returns:
        Tuple of (original_base64, segmented_base64) strings.
    """
    img = Image.open(image_path)
    encoded_string = convert_to_base64(img)

    aspect_ratio = img.size[1] / img.size[0]

    Transform = []
    ResizeRange = random.randint(resize_range[0], resize_range[1])
    Transform.append(T.Resize((int(ResizeRange * aspect_ratio), ResizeRange)))
    Transform.append(
        T.Resize((int(256 * aspect_ratio) - int(256 * aspect_ratio) % 16, 256))
    )
    Transform.append(T.ToTensor())
    Transform = T.Compose(Transform)

    img_tensor = Transform(img)

    SR = model(img_tensor.unsqueeze(0))
    SR_s = F.sigmoid(SR)

    SR_image = (SR_s > threshold) * img_tensor
    im = T.ToPILImage()(SR_image[0])
    seg_encoded_string = convert_to_base64(im)

    return encoded_string, seg_encoded_string


def apply_precomputed_mask(image_path: str, mask_path: str) -> Tuple[str, str]:
    """Applies a pre-computed binary segmentation mask to the original image.

    Usado para o dataset PH2, onde as máscaras de ground-truth já estão disponíveis.
    A máscara é um BMP em escala de cinza: branco (>128) = região da lesão.
    O resultado replicates o comportamento do SegmentationAgent: a imagem segmentada
    contém apenas os pixels dentro da lesão; o fundo se torna preto.

    Args:
        image_path: Caminho para a imagem dermoscópica original (.bmp ou .jpg).
        mask_path: Caminho para a máscara binária pré-computada (.bmp).

    Returns:
        Tuple de (original_base64, masked_image_base64) em formato JPEG.
    """
    original = Image.open(image_path).convert("RGB")
    mask = Image.open(mask_path).convert("L")

    # Redimensionar máscara se necessário (garante mesmas dimensões que a original)
    if mask.size != original.size:
        mask = mask.resize(original.size, Image.NEAREST)

    original_array = np.array(original, dtype=np.uint8)
    # Binarizar: pixels > 128 pertencem à lesão
    mask_array = (np.array(mask) > 128).astype(np.uint8)

    # Aplicar máscara: pixels fora da lesão tornam-se pretos
    seg_array = original_array * mask_array[:, :, np.newaxis]
    seg_image = Image.fromarray(seg_array)

    return convert_to_base64(original), convert_to_base64(seg_image)
