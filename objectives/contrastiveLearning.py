import sys
sys.path.append("..")
import torch
import torch.nn as nn
import torch.nn.functional as F
import logging

logger = logging.getLogger(__name__)

def lossReduce(val, reduction='mean'):
    if reduction == 'mean':
        return val.mean()
    elif reduction == 'sum':
        return val.sum()


def contrastiveLoss(logits, labels=None):
    # print("contrastive loss")
    if type(labels) == type(None):
        return F.cross_entropy(logits, torch.arange(logits.shape[0], device=logits.device))
    else:
        if len(labels.shape) == 1:
            return F.cross_entropy(logits, labels)
        else:  # labels = one-hot encodings or class probabilities
            kl_loss = nn.KLDivLoss(reduction="batchmean")
            input = F.log_softmax(logits, dim=1)
            return kl_loss(input, labels)


# neg_sampling_strategy random/nonramdom
def cosineSimWithNegSampling(qvec, kvec, negative_samples_cnt, negative_sampling_strategy, undersample_similar_queries):
    y_batch_ = nn.CosineSimilarity(dim=1, eps=1e-06)(qvec, kvec).view(-1, 1)
    y_batch = [y_batch_]
    if negative_sampling_strategy == 'random':
        for i in range(negative_samples_cnt):
            batch_idx = torch.randperm(y_batch[0].shape[0]).to(y_batch[0].device)
            y_batch_ = nn.CosineSimilarity(dim=1, eps=1e-06)(qvec, kvec[batch_idx]).view(-1, 1)
            y_batch.append(y_batch_)
    else:
        t_encoded_q_norm = F.normalize(qvec)
        t_encoded_k_norm = F.normalize(kvec)
        t_cross_sim = torch.matmul(t_encoded_q_norm, t_encoded_k_norm.transpose(0, 1))
        t_cross_sim_masked = -10.0 * torch.eye(t_cross_sim.shape[0]).to(t_cross_sim.device) + t_cross_sim
        if undersample_similar_queries:
            t_cross_q_sim = torch.matmul(t_encoded_q_norm, t_encoded_q_norm.transpose(0, 1))
            t_cross_q_mask = torch.where(t_cross_q_sim > 0.9, -10.0 * torch.ones_like(t_cross_q_sim),
                                         torch.zeros_like(t_cross_q_sim))
            t_cross_sim_masked = t_cross_q_mask + t_cross_sim_masked
        t_cross_sim_masked = F.softmax(t_cross_sim_masked, dim=1)
        t_max_neg_idx = torch.multinomial(t_cross_sim_masked, negative_samples_cnt).transpose(0, 1)
        for i in range(t_max_neg_idx.shape[0]):
            t_neg_encoded_trg = torch.index_select(kvec, 0, t_max_neg_idx[i])
            y_batch_ = nn.CosineSimilarity(dim=1, eps=1e-06)(qvec, t_neg_encoded_trg).view(-1, 1)
            y_batch.append(y_batch_)
    y_batch = torch.stack(y_batch, dim=1).squeeze()  # [batch_size, neg_samples + 1]
    return y_batch


class ClipLoss(nn.Module):
    def __init__(self, device=0, local_rank=0, global_rank=0, negative_samples_cnt=0, logit_scale=None,
                 negative_sampling_strategy=None, undersample_similar_queries=False, use_one_hot_label=False):
        super(ClipLoss, self).__init__()
        self.device = device
        self.local_rank = local_rank
        self.global_rank = global_rank
        self.negative_samples_cnt = negative_samples_cnt
        if self.negative_samples_cnt > 0:
            self.nceLoss = NceLoss(device, local_rank, global_rank, negative_samples_cnt, logit_scale,
                                   negative_sampling_strategy, undersample_similar_queries, use_one_hot_label)
        if type(logit_scale) == type(None):
            self.logit_scale = nn.Parameter(torch.Tensor([0.7]))
        else:
            self.logit_scale = nn.Parameter(torch.Tensor([logit_scale]))
        self.cuda(self.device)

    def forward(self, qvec, kvec):
        if self.negative_samples_cnt == 0:
            qvec_norm = qvec / qvec.norm(dim=-1, keepdim=True)
            kvec_norm = kvec / kvec.norm(dim=-1, keepdim=True)
            qkcross = torch.matmul(qvec_norm, kvec_norm.t())
            logits_per_query = qkcross * self.logit_scale
            query_loss = contrastiveLoss(logits=logits_per_query)
            doc_loss = contrastiveLoss(logits=logits_per_query.T)
            loss = (query_loss+doc_loss)/2.0
            return loss
        else:
            query_loss = self.nceLoss(qvec, kvec)
            doc_loss = self.nceLoss(kvec, qvec)
            return (query_loss + doc_loss) / 2.0


class NceLoss(nn.Module):
    def __init__(self, device=0, local_rank=0, global_rank=0, negative_samples_cnt=10, logit_scale=None,
                 negative_sampling_strategy="random", undersample_similar_queries=False, use_one_hot_label=False):
        super(NceLoss, self).__init__()
        self.device = device
        self.local_rank = local_rank
        self.global_rank = global_rank
        if type(logit_scale) == type(None):
            self.logit_scale = 0.7  # static temperature scaling
        else:
            self.logit_scale = logit_scale
        self.negative_samples_cnt = negative_samples_cnt
        self.negative_sampling_strategy = negative_sampling_strategy
        self.undersample_similar_queries = undersample_similar_queries
        self.use_one_hot_label = use_one_hot_label

    def forward(self, qvec, kvec):
        if self.use_one_hot_label:
            labels = [torch.ones(qvec.shape[0])]
            labels = labels + [torch.zeros(qvec.shape[0]) for i in range(self.negative_samples_cnt)]
            labels = torch.stack(labels, dim=1)
        else:
            labels = torch.zeros(qvec.shape[0]).long()
        # simscore size = [batch_size, neg_samples + 1]
        simscore = cosineSimWithNegSampling(qvec, kvec, self.negative_samples_cnt, self.negative_sampling_strategy,
                                            self.undersample_similar_queries)
        simscore = simscore * self.logit_scale
        labels = labels.to(simscore.device)
        return contrastiveLoss(simscore, labels)


class MaxMarginLoss(nn.Module):
    def __init__(self, device=0, local_rank=0, global_rank=0, negative_samples_cnt=10, margin=0.3, reduction="mean"):
        super(MaxMarginLoss, self).__init__()
        self.device = device
        self.local_rank = local_rank
        self.global_rank = global_rank
        self.negative_samples_cnt = negative_samples_cnt
        self.margin = margin
        self.reduction = reduction

    def forward(self, qvec, kvec):
        qvec = F.normalize(qvec)
        kvec = F.normalize(kvec)
        pos_indices = torch.arange(qvec.shape[0]).to(qvec.device)
        target = F.one_hot(pos_indices, num_classes=kvec.shape[0]).to(qvec.device)
        sim = qvec @ kvec.T
        sim_p = sim.gather(1, pos_indices.unsqueeze(1))
        neg_sim = torch.where(target == 0, sim, torch.full_like(sim, -100))
        _, indices = torch.topk(neg_sim, largest=True, dim=1, k=self.negative_samples_cnt)
        sim_n = sim.gather(1, indices)
        loss = torch.max(torch.zeros_like(sim_p), sim_n - sim_p + self.margin)
        mask = torch.where(loss != 0, torch.ones_like(loss), torch.zeros_like(loss))
        prob = torch.softmax(sim_n * mask, dim=1)
        reduced_loss = lossReduce(loss * prob, reduction=self.reduction)
        return reduced_loss


class WebCELoss(nn.Module):
    def __init__(self, device=0, local_rank=0, global_rank=0, negative_samples_cnt=10, logit_scale=None,
                 negative_sampling_strategy="random", undersample_similar_queries=False):
        super(WebCELoss, self).__init__()
        self.device = device
        self.local_rank = local_rank
        self.global_rank = global_rank
        if type(logit_scale) == type(None):
            self.logit_scale = 0.7  # static temperature scaling
        else:
            self.logit_scale = logit_scale
        self.negative_samples_cnt = negative_samples_cnt
        self.negative_sampling_strategy = negative_sampling_strategy
        self.undersample_similar_queries = undersample_similar_queries

    def forward(self, qvec, kvec, wvec, inf_label):
        yw_batch = nn.CosineSimilarity(dim=1, eps=1e-06)(qvec, wvec).view(-1, 1)
        yk_batch, _ = self.cosineSimWithNegSampling(qvec, kvec, self.negative_samples_cnt,
                                                    self.negative_sampling_strategy, self.undersample_similar_queries)
        yw_batch = yw_batch * self.logit_scale
        yk_batch = yk_batch * self.logit_scale
        # doc side loss
        klabels = [inf_label] + [torch.zeros(qvec.shape[0]).to(qvec.device) for i in
                                 range(self.negative_samples_cnt)]
        klabels = torch.stack(klabels, dim=1).to(qvec.device)
        kloss = self.contrastive_loss(yk_batch, klabels)
        kloss = kloss / inf_label.sum()
        # web side loss
        yk_batch_req = torch.where(klabels == 1, torch.full_like(klabels, -1000), yk_batch)
        # yk_batch[:, 0] = yk_batch[:, 0] + inf_label * -1000
        wbatch = torch.cat([yw_batch, yk_batch_req], dim=1).to(qvec.device)
        wlabels = torch.zeros(wbatch.shape[0]).long().to(qvec.device)
        wloss = contrastiveLoss(wbatch, wlabels)
        # final loss
        loss = (wloss + kloss) / 2.0
        return loss
