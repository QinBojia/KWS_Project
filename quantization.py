from __future__ import annotations

from typing import Tuple, Dict

import torch
import torch.nn as nn


class QuantizableWrapper(nn.Module):
    """
    让卷积网络支持 PyTorch 的静态 INT8 量化（PTQ）：
    - QuantStub/DeQuantStub 会在前后插入量化/反量化节点
    """
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
    """
    尝试做算子融合（Conv+BN+ReLU），量化效果通常更好、更快。
    注意：因为我们的模型里 Conv/BN/ReLU 在 Sequential 里面，
    PyTorch 可以按模块路径 fuse。
    """
    # stem: Conv2d, BN, ReLU
    torch.ao.quantization.fuse_modules(model.stem, ["0", "1", "2"], inplace=True)

    # 每个 DSConvBlock 里有 dw 和 pw 两段 Sequential
    for blk in model.features:
        # dw: Conv, BN, ReLU
        torch.ao.quantization.fuse_modules(blk.dw, ["0", "1", "2"], inplace=True)
        # pw: Conv, BN, ReLU
        torch.ao.quantization.fuse_modules(blk.pw, ["0", "1", "2"], inplace=True)

    return model


@torch.no_grad()
def ptq_int8_static(
    float_model: nn.Module,
    calibration_loader,
    device: str = "cpu",
    calibration_batches: int = 30,
) -> nn.Module:
    """
    INT8 静态后训练量化（PTQ）
    - 一般在 CPU 上做（fbgemm 后端）
    - calibration：用一小部分训练数据跑一遍，收集激活范围
    """
    # 量化通常在 CPU 上进行
    torch.backends.quantized.engine = "onednn"

    float_model = float_model.to("cpu").eval()
    float_model = fuse_for_quant(float_model)

    qmodel = QuantizableWrapper(float_model)
    qmodel.eval()

    # 选择量化配置（默认就够用）
    qmodel.qconfig = torch.ao.quantization.get_default_qconfig("onednn")

    # 准备：插入观测器
    torch.ao.quantization.prepare(qmodel, inplace=True)

    # calibration：跑几批数据
    n = 0
    for x, _ in calibration_loader:
        x = x.to("cpu")
        _ = qmodel(x)
        n += 1
        if n >= calibration_batches:
            break

    # 转换：把观测器替换成真正 int8 算子
    torch.ao.quantization.convert(qmodel, inplace=True)
    return qmodel


def fp16_cast(model: nn.Module) -> nn.Module:
    """
    FP16：本质是把权重转 half。
    - GPU 上通常能加速（取决于显卡）
    - CPU 上一般不加速，但能减少模型权重内存占用
    """
    return model.half()
