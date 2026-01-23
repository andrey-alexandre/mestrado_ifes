import torch
from app.segmentation.models import U_Net
from app.config import settings
import logging
import os

logger = logging.getLogger(__name__)

def load_model(weights_path: str = None):
    model = U_Net(img_ch=3, output_ch=1)

    path = weights_path or settings.SEGMENTATION_WEIGHTS_PATH

    if os.path.exists(path):
        try:
            model.load_state_dict(torch.load(path, map_location=torch.device(settings.DEVICE)))
            logger.info(f"Loaded segmentation model from {path}")
        except Exception as e:
            logger.error(f"Failed to load weights from {path}: {e}")
            logger.warning("Using initialized model with random weights.")
    else:
        logger.warning(f"Weights file not found at {path}. Using initialized model with random weights.")

    model.eval()
    return model
