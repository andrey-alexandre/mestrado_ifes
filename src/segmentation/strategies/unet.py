"""Estratégia U-Net: adapta a família `Image_Segmentation.code.network` já usada no projeto."""
import re
from typing import Dict, List

import torch.nn as nn

from src.segmentation.strategies.base import SegmentationStrategy

_ENCODER = re.compile(r"^(Conv|RRCNN)[1-5]$")
_DECODER = re.compile(r"^(Up\d|Up_conv\d|Up_RRCNN\d|Att\d|Conv_1x1)$")


class UNetStrategy(SegmentationStrategy):
    """U-Net (e variantes R2U/Att/R2Att) via o registro existente em `src.models.model`.

    Parâmetros (`segmentation.strategy_params`): `model_name`, `img_ch`, `output_ch`.
    A saída do modelo são logits; a sigmoide é aplicada na predição/perda.
    """

    name = "unet"
    size_multiple = 16  # quatro max-pools de fator 2

    def build_model(self) -> nn.Module:
        try:
            from src.models.model import MODEL_CLASSES
        except ModuleNotFoundError as exc:
            raise RuntimeError(
                "Não foi possível importar a arquitetura: o pacote externo "
                "'Image_Segmentation.code.network' (LeeJunHyun/Image_Segmentation) precisa estar "
                "no PYTHONPATH, como já exigido por src/models/model.py."
            ) from exc
        model_name = self.params.get("model_name", "U_Net")
        cls = MODEL_CLASSES.get(model_name)
        if cls is None:
            raise ValueError(f"Arquitetura desconhecida/indisponível: '{model_name}'. "
                             f"Opções: {sorted(MODEL_CLASSES)}")
        return cls(img_ch=int(self.params.get("img_ch", 3)),
                   output_ch=int(self.params.get("output_ch", 1)))

    def block_groups(self, model: nn.Module) -> Dict[str, List[str]]:
        names = [n for n, _ in model.named_children()]
        enc = [n for n in names if _ENCODER.match(n)]
        dec = [n for n in names if _DECODER.match(n)]
        if not enc or not dec:
            raise ValueError(
                "Não foi possível identificar blocos encoder/decoder na arquitetura. "
                f"Filhos do módulo: {names}"
            )
        return {"encoder": sorted(enc), "decoder": dec}
