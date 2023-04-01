import sys
sys.path.append("..")
from enum import Enum
import torch
import math

class OptimizerEnum(Enum):
    ADAM = ["Adam", "adam"]
    SGD = ["Sgd", "sgd"]
    ADAMW = ["AdamW", "adamw"]


class WarmupLinearScheduler(torch.optim.lr_scheduler.LambdaLR):
    def __init__(self, optimizer, warmup, total, ratio, last_epoch=-1):
        self.warmup = warmup
        self.total = total
        self.ratio = ratio
        super(WarmupLinearScheduler, self).__init__(optimizer, self.lr_lambda, last_epoch=last_epoch)

    def lr_lambda(self, step):
        if step < self.warmup:
            return (1 - self.ratio) * step / float(max(1, self.warmup))
        return max(
            0.0,
            1.0 + (self.ratio - 1) * (step - self.warmup) / float(max(1.0, self.total - self.warmup)),
        )


class CosineScheduler(torch.optim.lr_scheduler.LambdaLR):
    def __init__(self, optimizer, warmup, total, ratio=0.1, last_epoch=-1):
        self.warmup = warmup
        self.total = total
        self.ratio = ratio
        super(CosineScheduler, self).__init__(optimizer, self.lr_lambda, last_epoch=last_epoch)

    def lr_lambda(self, step):
        if step < self.warmup:
            return float(step) / self.warmup
        s = float(step - self.warmup) / (self.total - self.warmup)
        return self.ratio + (1.0 - self.ratio) * math.cos(0.5 * math.pi * s)


def instantiate(config, param_groups, hyperparameters):
    if config['name'] in OptimizerEnum.ADAM.value:
        optimizer = torch.optim.Adam(params=param_groups,
                                lr=hyperparameters.get('lr', 1e-4),
                                betas=(hyperparameters.get('beta1', 0.9), 
                                       hyperparameters.get('beta2', 0.999)),
                                weight_decay=hyperparameters.get('weight_decay', 0))
    elif config['name'] in OptimizerEnum.SGD.value:
        optimizer = torch.optim.SGD(params=param_groups,
                               lr=hyperparameters.get('lr', 1e-4),
                               momentum=hyperparameters.get('momentum', 0.9))
    else:
        raise NotImplementedError
    scheduler_args = {
        "warmup": config['warmup_steps'],
        "total": config['total_steps'],
        "ratio": config['lr_min_ratio'],
    }
    if config['lr_schedular'] == "linear":
        scheduler_class = WarmupLinearScheduler
    elif config['lr_schedular'] == "cosine":
        scheduler_class = CosineScheduler
    else:
        raise ValueError
    scheduler = scheduler_class(optimizer, **scheduler_args)
    return optimizer, scheduler
