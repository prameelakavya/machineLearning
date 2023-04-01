import sys
sys.path.append("..")
import os
import objectives
import metrics
import optimizers
import datasets
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import optim
from transformers import BertModel, BertConfig
import logging
from torchinfo import summary
import tqdm
import numpy as np

logger = logging.getLogger(__name__)

class ClsPooler(nn.Module):
    def __init__(self, config):
        super(ClsPooler, self).__init__()
        self.config = config

    def forward(self, cls_tensor, term_tensor, mask):
        return cls_tensor


class WeightPooler(nn.Module):
    def __init__(self, config):
        super(WeightPooler, self).__init__()
        self.config = config
        self.weighting = nn.Linear(self.config['bert']['hidden_size'], 1)

    def forward(self, term_tensor, mask):
        weights = self.weighting(term_tensor)
        weights = F.softmax(weights, dim=1)
        return (term_tensor * weights).sum(dim=1)


class EncoderOutputProcessor(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.pooler = globals()[config['pooler']](config)
        self.downscale = nn.Linear(config['bert']['hidden_size'], config['downscale'])
    
    def forward(self, term_tensor, mask):
        return self.downscale(self.pooler(term_tensor, mask))


class TwinBert(nn.Module):
    def __init__(self, config, process_params):
        super().__init__()
        self.config = config
        self.device = process_params['device']
        self.local_rank = process_params['local_rank']
        self.global_rank = process_params['global_rank']
        self.world_size = process_params['world_size']
        self.bertConfig = BertConfig() # bertConfig is required to initialize the bert models
        self.set_bert_config(self.config['bert'])
        logger.info('bert config' + str(self.bertConfig))
        self.qencoder = BertModel(self.bertConfig, add_pooling_layer=False)
        if self.config['use_separate_doc_encoder']:
            self.dencoder = BertModel(self.bertConfig, add_pooling_layer=False)
        else:
            self.dencoder = self.qencoder
        self.EncoderOutputProcessor = EncoderOutputProcessor(config)
        self.loss = objectives.instantiate(config['loss'], process_params=process_params)
        self.cuda(self.device)

    def forward(self, input_batch):
        qterm, _ = self.qencoder(input_ids = input_batch['queryseq'], 
                                        attention_mask = input_batch['querymask'],
                                        token_type_ids = input_batch['queryseg'] if 'queryseg' in input_batch else None,
                                        return_dict=False)
        dterm, _ = self.dencoder(input_ids = input_batch['docseq'], 
                                        attention_mask = input_batch['docmask'],
                                        token_type_ids = input_batch['docseg'] if 'docseg' in input_batch else None,
                                        return_dict=False)
        qvec = self.EncoderOutputProcessor(qterm, input_batch['querymask'])
        dvec = self.EncoderOutputProcessor(dterm, input_batch['docmask'])
        loss = self.loss(qvec, dvec)
        return qvec, dvec, loss

    def set_bert_config(self, bertconfig):
        for key in bertconfig:
            setattr(self.bertConfig, key, bertconfig[key])
    
    @torch.no_grad()
    def print_summary(self):
        bs = 256
        max_tokens_q = 5
        max_tokens_d = 25
        summary(self, 
                input_data=[{'queryseq': torch.randint(1, 30520, (bs, max_tokens_q)),
                    'querymask': torch.ones(bs, max_tokens_q, dtype=int),
                    'docseq': torch.randint(1, 30520, (bs, max_tokens_d)),
                    'docmask': torch.ones(bs, max_tokens_d, dtype=int)}],
                verbose=1,
                col_width=16,
                col_names=["kernel_size", "output_size", "num_params", "mult_adds", "trainable"],
                row_settings=["var_names"])

    @torch.enable_grad()
    def train_step(self, input_batch):
        _, _, loss = self(input_batch)
        return loss
    
    @torch.no_grad()
    def validate_step(self, input_batch):
        qvec, dvec, _ = self(input_batch)
        if 'label' in input_batch:
            return qvec, dvec, input_batch['label']
        return qvec, dvec

    @torch.no_grad()
    def compute_metrics(self, input_data):
        # this is based off of the above validaiton step
        qemb = torch.vstack([input_data[x][0] for x in range(len(input_data))])
        demb = torch.vstack([input_data[x][1] for x in range(len(input_data))])
        targets = torch.hstack([input_data[x][2] for x in range(len(input_data))])
        scores = F.cosine_similarity(qemb, demb)
        assert scores.shape == targets.shape
        return metrics.compute(self.config['metrics']['validation'], {'targets':targets, 'scores':scores}) 
    
    @torch.no_grad()
    def compute_embeddings(self, tb_summary_writer):
        tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")
        for embtype in self.config['compute_embeddings'].keys():
            embeddings = []
            labels = []
            with open(self.config['compute_embeddings'][embtype]['file'], 'r', encoding='utf-8') as fp:
                while True:
                    input_batch = []
                    for x in range(self.config['compute_embeddings'][embtype]['bs']):
                        line = fp.readline().strip('\n\r')
                        if line == '':
                            break
                        input_batch.append(line)
                    if len(input_batch)==0:
                        break
                    from transformers import AutoTokenizer
                    encoded_batch = tokenizer(input_batch, padding=True, truncation=True)
                    encoded_batch = self.totorchtensor(encoded_batch)
                    if embtype=='query':
                        term, _ = self.qencoder(**encoded_batch, return_dict=False)
                    elif embtype=='doc':
                        term, _ = self.dencoder(**encoded_batch, return_dict=False)       
                    else:
                        logger.error("unknown embedding type from bertmodels.py compute_embeddings function")      
                        raise NotImplementedError()       
                    vec, _ = self.EncoderOutputProcessor(term, encoded_batch['attention_mask'])
                    embeddings.append(vec.detach().cpu().numpy())
                    labels.append(np.asarray(input_batch))
            embeddings = np.stack(embeddings, axis=0)
            labels = np.stack(labels, axis=0)
            tb_summary_writer.add_embedding(embeddings, labels)
            if self.config['compute_embeddings']['write_to_dir']:
                with open(os.path.join(self.config['compute_embeddings']['write_to_dir'],embtype+'.tsv'), 'w', encoding='utf-8') as fp:
                    fp.writelines([k[1]+'\t'+' '.join(map(str,k[0].tolist())) for k in zip(labels, embeddings)])
        return(True)
    
    def configure_optimizer(self, total_steps_per_epoch):
        if self.config['fix_params'] is None or self.config['fix_params'] == '':
            fix_params = []
        else:
            fix_params = self.config['fix_params'].split(',')
        param_groups = [{'params': [], 'weight_decay': 0.01}, {'params': [], 'weight_decay': 0.0}]
        train_params = []
        train_params_cnt = 0
        fix_params_cnt = 0
        for name, p in self.named_parameters():
            if not any(n in name for n in fix_params):
                train_params_cnt += p.numel()
                if not any(i in name for i in ['LayerNorm.weight', 'LayerNorm.bias', 'res_bn.weight', 'res_bn.bias']):
                    param_groups[0]['params'].append(p)
                    train_params.append((name, p.shape))
                elif any(i in name for i in ['LayerNorm.weight', 'LayerNorm.bias', 'res_bn.weight', 'res_bn.bias']):
                    param_groups[1]['params'].append(p)
                    train_params.append((name, p.shape))
            else:
                fix_params_cnt += p.numel()
                p.requires_grad = False
        # if self.global_rank == 0:
        #     logger.info(str(self.global_rank) + ': ' + 'train params:\n' + ','.join([str(train_param) for train_param in train_params]))
        logger.info(f"total_trainable_params_cnt : {train_params_cnt}, fix_params_cnt : {fix_params_cnt}")
        hpdict = dict()
        for hp in self.config['optimizer']:
            if hp != "name":
                hpdict[hp] = self.config['optimizer'][hp]
        #logger.info(f"optimizer param groups: \n{param_groups}")
        logger.info(f"optimizer set hyper parameters: \n{hpdict}")
        self.config['optimizer']['total_steps'] = total_steps_per_epoch
        return optimizers.instantiate(self.config['optimizer'], param_groups, hpdict)
    
    def totorchtensor(self, x_batch, non_blocking=False):
        if len(self.device.split(':')) == 1: #cpu
            return x_batch
        ret = {}
        cuda_id = int(self.device.split(':')[1])
        #logger.info(f'moving batch to gpu {cuda_id}')
        for f in x_batch:
            if 'seq' in f:
                ret[f] = torch.LongTensor(x_batch[f])
            elif 'int' in str(x_batch[f].dtype):
                ret[f] = torch.IntTensor(x_batch[f])
            else:
                ret[f] = torch.FloatTensor(x_batch[f])
            ret[f] = ret[f].cuda(cuda_id, non_blocking=non_blocking) if cuda_id >= 0 else ret[f]
        return ret