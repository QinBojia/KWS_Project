from __future__ import annotations

import torch
import torch.nn as nn


class QuantizableWrapper(nn.Module):
    """Wraps a model with QuantStub/DeQuantStub for static INT8 quantization."""
    def __init__(self, float_model: nn.Module):
        super().__init__()
        self.quant = torch.ao.quantization.QuantStub()
        self.model = float_model
        self.dequant = torch.ao.quantization.DeQuantStub()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.quant(x)
        x = self.model(x)
        x = self.dequant(x)
        return x


def fuse_for_quant(model: nn.Module) -> nn.Module:
    """Fuse Conv+BN+ReLU operators for better quantization.

    Handles different model types: CustomDSCNN, MobileNet, TENet, LiCoNet, BCResNet.
    For models without explicit fusion support, returns the model as-is
    (FX graph mode quantization will handle fusion automatically).
    """
    from kws.models import CustomDSCNN, MobileNetStyleKWS, DSConvBlock

    model.eval()

    # Only manually fuse DSConvBlock-based models (CustomDSCNN, MobileNet)
    if isinstance(model, (CustomDSCNN, MobileNetStyleKWS)):
        torch.ao.quantization.fuse_modules(model.stem, ["0", "1", "2"], inplace=True)
        for blk in model.features:
            if isinstance(blk, DSConvBlock):
                torch.ao.quantization.fuse_modules(blk.dw, ["0", "1", "2"], inplace=True)
                torch.ao.quantization.fuse_modules(blk.pw, ["0", "1", "2"], inplace=True)

    # For TENet, LiCoNet, BCResNet: skip manual fusion.
    # FX graph mode quantization auto-detects fusible Conv+BN+ReLU patterns.
    return model


@torch.no_grad()
def ptq_int8_static(
    float_model: nn.Module,
    calibration_loader,
    device: str = "cpu",
    calibration_batches: int = 30,
) -> nn.Module:
    """INT8 static PTQ via FX Graph Mode with onednn backend."""
    import torch.ao.quantization.quantize_fx as qfx

    torch.backends.quantized.engine = "onednn"

    float_model = float_model.to("cpu").eval()

    qconfig = torch.ao.quantization.get_default_qconfig("onednn")
    qconfig_mapping = torch.ao.quantization.QConfigMapping().set_global(qconfig)

    example_x, _ = next(iter(calibration_loader))
    example_x = example_x[:1].to("cpu")

    prepared = qfx.prepare_fx(float_model, qconfig_mapping, example_inputs=(example_x,))

    n = 0
    for x, _ in calibration_loader:
        x = x.to("cpu")
        _ = prepared(x)
        n += 1
        if n >= calibration_batches:
            break

    quantized = qfx.convert_fx(prepared)
    quantized.eval()
    return quantized


def fp16_cast(model: nn.Module) -> nn.Module:
    """Cast model weights to FP16."""
    return model.half()
