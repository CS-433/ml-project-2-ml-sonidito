import torch

def compute_accuracy(preds, labels):
    tp = (((preds==True) & (labels == True))).sum(axis = 1)
    tn = (((preds==False) & (labels == False))).sum(axis = 1)

    return (tp + tn)  / labels.shape[1]

def compute_precision(preds, labels):
    tp = (((preds==True) & (labels == True))).sum(axis = 1)
    fp = (((preds==True) & (labels == False))).sum(axis = 1)

    res = tp / (tp + fp)
    res[torch.isnan(res)] = 0

    return res


def compute_recall(preds, labels):
    tp = (((preds==True) & (labels == True))).sum(axis = 1)
    fn = (((preds==False) & (labels == True))).sum(axis = 1)

    res = tp / (tp + fn)
    res[torch.isnan(res)] = 0

    return res


def compute_f1(preds, labels):
    precision = compute_precision(preds, labels)
    recall = compute_recall(preds, labels)

    res = 2 * (precision * recall) / (precision + recall)
    res[torch.isnan(res)] = 0

    return res

def compute_kappa_score(preds, labels):
    tp = (((preds==True) & (labels == True))).sum(axis = 1)
    tn = (((preds==False) & (labels == False))).sum(axis = 1)
    fn = (((preds==False) & (labels == True))).sum(axis = 1)
    fp = (((preds==True) & (labels == False))).sum(axis = 1)

    res = (2 * (tp * tn - fn * fp)) / ((tp + fp) * (fp + tn) + (tp + fn) + (fn + tn))
    res[torch.isnan(res)] = 0

    return res
    