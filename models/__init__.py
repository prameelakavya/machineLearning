import sys
sys.path.append("..")
from enum import Enum
import torch.nn as nn


class ModelEnum(Enum):
    DISTILBERT = "DistilBert"
    TWINBERT = "TwinBert"

def instantiate(config, process_params):
    if config['name'] == ModelEnum.TWINBERT.value:
        from .bertModels import TwinBert
        return TwinBert(process_params=process_params, config=config)