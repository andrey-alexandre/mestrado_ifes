from torchvision import transforms as T
from PIL import Image
import random

def preprocess_image(img: Image.Image, image_size=224):
    """
    Preprocesses the image for the segmentation model.
    Note: The original notebook used random transforms which seems odd for inference,
    but I will adapt it to be more deterministic or configurable if needed.
    The original notebook logic:
    ResizeRange = random.randint(300,320)
    Transform.append(T.Resize((int(ResizeRange*aspect_ratio),ResizeRange)))
    Transform.append(T.Resize((int(256*aspect_ratio)-int(256*aspect_ratio)%16,256)))
    Transform.append(T.ToTensor())
    """
    aspect_ratio = img.size[1]/img.size[0]

    # Deterministic resize for inference
    ResizeRange = 310

    transform_pipeline = T.Compose([
        T.Resize((int(ResizeRange*aspect_ratio), ResizeRange)),
        T.Resize((int(256*aspect_ratio)-int(256*aspect_ratio)%16, 256)),
        T.ToTensor()
    ])

    return transform_pipeline(img)
