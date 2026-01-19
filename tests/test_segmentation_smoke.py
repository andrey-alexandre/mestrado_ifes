import pytest
from app.segmentation.inference import segment_image
from app.segmentation.models import U_Net
from PIL import Image
import torch

def test_segmentation_smoke(tmp_path):
    # Setup dummy model
    model = U_Net(img_ch=3, output_ch=1)
    model.eval()

    # Setup dummy image
    img_path = tmp_path / "test.jpg"
    Image.new('RGB', (100, 100)).save(img_path)

    result = segment_image(model, str(img_path))

    assert "seg_image_data" in result
    assert "image_data" in result
    assert isinstance(result["seg_image_data"], str)
