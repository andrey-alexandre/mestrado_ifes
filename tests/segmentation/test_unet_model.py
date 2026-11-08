import torch

from src.segmentation.unet_model import build_unet


def test_build_unet_output_shape_matches_input_spatial_size():
    model = build_unet(img_ch=3, output_ch=1)
    model.eval()
    x = torch.randn(2, 3, 256, 256)
    with torch.no_grad():
        out = model(x)
    assert out.shape == (2, 1, 256, 256)


def test_build_unet_has_no_internal_sigmoid():
    # Guards the BCEWithLogitsLoss contract: the model must return raw
    # logits, never apply sigmoid internally (that happens only at
    # inference time, in the caller, e.g. scripts/run_isic2018_unet_pilot.py).
    model = build_unet()
    assert not any(isinstance(m, torch.nn.Sigmoid) for m in model.modules())


def test_build_unet_accepts_other_divisible_by_16_sizes():
    model = build_unet()
    model.eval()
    x = torch.randn(1, 3, 128, 128)
    with torch.no_grad():
        out = model(x)
    assert out.shape == (1, 1, 128, 128)


def test_build_unet_backward_pass_updates_weights():
    model = build_unet()
    model.train()
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)
    criterion = torch.nn.BCEWithLogitsLoss()

    x = torch.randn(2, 3, 256, 256)
    y = (torch.rand(2, 1, 256, 256) > 0.5).float()

    before = model.out_conv.weight.detach().clone()
    optimizer.zero_grad()
    loss = criterion(model(x), y)
    loss.backward()
    optimizer.step()
    after = model.out_conv.weight.detach().clone()

    assert not torch.equal(before, after)
