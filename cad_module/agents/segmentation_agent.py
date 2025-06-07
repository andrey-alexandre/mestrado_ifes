from io import BytesIO
import base64
from PIL import Image
import random
import torchvision.transforms as T
import torch.nn.functional as F

class SegmentationAgent:
  def __init__(self, model, threshold=.5, image_size=224, mode='train', augmentation_prob=0.4):
    self.model = model
    self.image_size = image_size
    self.mode = mode
    self.RotationDegree = [0,90,180,270]
    self.augmentation_prob = augmentation_prob

  def convert_to_base64(self, pil_image):
    """
    Convert PIL images to Base64 encoded strings

    :param pil_image: PIL image
    :return: Base64 string
    """
    pil_image = pil_image.convert("RGB")
    buffered = BytesIO()
    pil_image.save(buffered, format="JPEG")
    img_str = base64.b64encode(buffered.getvalue()).decode("utf-8")
    return img_str

  def segment_image(self, state):
    """Reads an image from a file and preprocesses it and returns."""
    img = Image.open(state.image_path)
    encoded_string = self.convert_to_base64(img)

    aspect_ratio = img.size[1]/img.size[0]

    Transform = []

    ResizeRange = random.randint(300,320)
    Transform.append(T.Resize((int(ResizeRange*aspect_ratio),ResizeRange)))
    p_transform = random.random()

    Transform.append(T.Resize((int(256*aspect_ratio)-int(256*aspect_ratio)%16,256)))
    Transform.append(T.ToTensor())
    Transform = T.Compose(Transform)
    img_ = img
    img = Transform(img)

    SR = model(img.unsqueeze(0))
    SR_s = F.sigmoid(SR)

    SR_image = (SR_s>.4)*img
    im = T.ToPILImage()(SR_image[0])
    seg_encoded_string = self.convert_to_base64(im)

    return {"seg_image_data": seg_encoded_string, "image_data": encoded_string}