import sys
sys.path.append("..")
from .contrastiveLearning import ClipLoss, NceLoss, MaxMarginLoss, WebCELoss
from enum import Enum
import torch.nn as nn
import logging

logger = logging.getLogger(__name__)

class LossEnum(Enum):
    CLIPLOSS = "Clip"
    NCELOSS = "Nce"
    MAXMARGINLOSS = "MaxMargin"
    WEBCELOSS = "WebCE"


def instantiate(config, process_params):
    if config['name'] == LossEnum.CLIPLOSS.value:
        return ClipLoss(device=process_params['device'], local_rank=process_params['local_rank'],
                        global_rank=process_params['global_rank'],
                        negative_samples_cnt=config['negative_samples_cnt'], logit_scale=config['logit_scale'],
                        negative_sampling_strategy=config['negative_sampling_strategy'],
                        undersample_similar_queries=config['undersample_similar_queries'],
                        use_one_hot_label=config['use_one_hot_label'])
    elif config['name'] == LossEnum.NCELOSS.value:
        return NceLoss(device=process_params['device'], local_rank=process_params['local_rank'],
                       global_rank=process_params['global_rank'],
                       negative_samples_cnt=config['negative_samples_cnt'], logit_scale=config['logit_scale'],
                       negative_sampling_strategy=config['negative_sampling_strategy'],
                       undersample_similar_queries=config['undersample_similar_queries'],
                       use_one_hot_label=config['use_one_hot_label'])
    elif config['name'] == LossEnum.MAXMARGINLOSS.value:
        return MaxMarginLoss(device=process_params['device'], local_rank=process_params['local_rank'],
                             global_rank=process_params['global_rank'],
                             negative_samples_cnt=config['negative_samples_cnt'], margin=config['margin'],
                             reduction=config['reduction'])
    elif config['name'] == LossEnum.WEBCELOSS.value:
        return WebCELoss(device=process_params['device'], local_rank=process_params['local_rank'],
                         global_rank=process_params['global_rank'],
                         negative_samples_cnt=config['negative_samples_cnt'],
                         logit_scale=config['logit_scale'],
                         negative_sampling_strategy=config['negative_sampling_strategy'],
                         undersample_similar_queries=config['undersample_similar_queries'])
    else:
        raise Exception("loss not implemented error")

# class Instantiate(nn.Module):
#     def __init__(self, config):
#         super(Instantiate).__init__()
#         self.loss_name = config['name']
#         if config['name'] == Loss.CLIPLOSS.value:
#             from contrastiveLearning import ClipLoss
#             self.loss = ClipLoss(device=config['device'], local_rank=config['local_rank'],
#                                  global_rank=config['global_rank'],
#                                  negative_samples_cnt=config['negative_samples_cnt'], logit_scale=config['logit_scale'],
#                                  negative_sampling_strategy=config['negative_sampling_strategy'],
#                                  undersample_similar_queries=config['undersample_similar_queries'],
#                                  use_one_hot_label=config['use_one_hot_label'])
#         elif config['name'] == Loss.NCELOSS.value:
#             from contrastiveLearning import NceLoss
#             self.loss = NceLoss(device=config['device'], local_rank=config['local_rank'],
#                                 global_rank=config['global_rank'],
#                                 negative_samples_cnt=config['negative_samples_cnt'], logit_scale=config['logit_scale'],
#                                 negative_sampling_strategy=config['negative_sampling_strategy'],
#                                 undersample_similar_queries=config['undersample_similar_queries'],
#                                 use_one_hot_label=config['use_one_hot_label'])
#         elif config['name'] == Loss.MAXMARGINLOSS.value:
#             from contrastiveLearning import MaxMarginLoss
#             self.loss = MaxMarginLoss(device=config['device'], local_rank=config['local_rank'],
#                                       global_rank=config['global_rank'],
#                                       negative_samples_cnt=config['negative_samples_cnt'], margin=config['margin'],
#                                       reduction=config['reduction'])
#         elif config['name'] == Loss.WEBCELOSS.value:
#             from contrastiveLearning import WebCELoss
#             self.loss = WebCELoss(device=config['device'], local_rank=config['local_rank'],
#                                   global_rank=config['global_rank'],
#                                   negative_samples_cnt=config['negative_samples_cnt'],
#                                   logit_scale=config['logit_scale'],
#                                   negative_sampling_strategy=config['negative_sampling_strategy'],
#                                   undersample_similar_queries=config['undersample_similar_queries'])
#         else:
#             raise Exception("loss not implemented error")
#
#     def forward(self, model_output_batch, model_to_loss_arg_map):
#
#         if self.loss_name == Loss.CLIPLOSS.value:
#             try:
#                 if len(model_to_loss_arg_map) > 2:
#                     raise Exception("model sending in more args than expected, please check")
#                 return self.loss(model_output_batch[model_to_loss_arg_map[0]],
#                                  model_output_batch[model_to_loss_arg_map[1]])
#             except:
#                 print("model_to_loss_arg_map : ", model_to_loss_arg_map)
#                 raise Exception(self.loss_name + "expects key:qvec of shape:[batch_size, dim], key:kvec of shape:["
#                                                  "batch_size, dim]")
#         if self.loss_name == Loss.NCELOSS.value:
#             try:
#                 if len(model_to_loss_arg_map) > 2:
#                     raise Exception("model sending in more args than expected, please check")
#                 return self.loss(model_output_batch[model_to_loss_arg_map[0]],
#                                  model_output_batch[model_to_loss_arg_map[1]])
#             except:
#                 print("model_to_loss_arg_map : ", model_to_loss_arg_map)
#                 raise Exception(self.loss_name + "expects key:qvec of shape:[batch_size, dim], key:kvec of shape:["
#                                                  "batch_size, dim]")
#         if self.loss_name == Loss.MAXMARGINLOSS.value:
#             try:
#                 if len(model_to_loss_arg_map) > 2:
#                     raise Exception("model sending in more args than expected, please check")
#                 return self.loss(model_output_batch[model_to_loss_arg_map[0]],
#                                  model_output_batch[model_to_loss_arg_map[1]])
#             except:
#                 raise Exception(self.loss_name + "expects key:qvec of shape:[batch_size, dim], key:kvec of shape:["
#                                                  "batch_size, dim]")
#         if self.loss_name == Loss.WEBCELOSS.value:
#             try:
#                 if len(model_to_loss_arg_map) > 4:
#                     raise Exception("model sending in more args than expected, please check")
#                 return self.loss(model_output_batch[model_to_loss_arg_map[0]],
#                                  model_output_batch[model_to_loss_arg_map[1]],
#                                  model_output_batch[model_to_loss_arg_map[2]],
#                                  model_output_batch[model_to_loss_arg_map[3]])
#             except:
#                 raise Exception(self.loss_name + "expects key:qvec of shape:[batch_size, dim], key:kvec of shape["
#                                                  "batch_size, dim], key:wvec of shape[batch_size, dim], key:wvec of "
#                                                  "shape[batch_size, dim], key:inf_label of shape[batch_size]")
