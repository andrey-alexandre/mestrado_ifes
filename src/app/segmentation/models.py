import torch.nn as nn
import torch

# Dummy implementation since the original code is missing
class U_Net(nn.Module):
    def __init__(self, img_ch=3, output_ch=1):
        super(U_Net, self).__init__()
        self.conv = nn.Conv2d(img_ch, output_ch, 1)

    def forward(self, x):
        return self.conv(x)
