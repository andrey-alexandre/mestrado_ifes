import torch
import torch.nn.functional as F
from torchvision import transforms as T
from PIL import Image
import base64
from io import BytesIO
from app.segmentation.preprocess import preprocess_image

def convert_to_base64(pil_image):
    pil_image = pil_image.convert("RGB")
    buffered = BytesIO()
    pil_image.save(buffered, format="JPEG")
    return base64.b64encode(buffered.getvalue()).decode("utf-8")

def segment_image(model, image_path: str):
    img = Image.open(image_path)
    encoded_string = convert_to_base64(img)

    img_tensor = preprocess_image(img)

    with torch.no_grad():
        sr = model(img_tensor.unsqueeze(0))
        sr_s = torch.sigmoid(sr)

        # Apply threshold and mask
        sr_image = (sr_s > 0.4).float() * img_tensor

    im = T.ToPILImage()(sr_image[0])
    seg_encoded_string = convert_to_base64(im)

    return {"seg_image_data": seg_encoded_string, "image_data": encoded_string}
