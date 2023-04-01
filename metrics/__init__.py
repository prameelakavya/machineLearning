import sys
import torch
sys.path.append("..")
from sklearn.metrics import roc_auc_score, accuracy_score


def compute(metricsStr, params):
    metricsToCompute = metricsStr.split(',')
    metricsResults = {}
    if "accuracy" in metricsToCompute:
        targets = params['targets'].cpu() 
        scores = params['scores'].cpu()
        scores = (scores > 0.5).type(torch.int)  
        metricsResults['accuracy'] = accuracy_score(targets, scores)
    if "rocauc" in metricsToCompute:
        targets = params['targets'].cpu()
        scores = params['scores'].cpu()
        metricsResults['rocauc'] = roc_auc_score(targets, scores)
    return metricsResults