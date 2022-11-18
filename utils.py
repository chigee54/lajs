import os
import numpy as np
import random
import torch
import math
import json


stw_path = os.path.join(os.path.dirname(__file__), 'stopword.txt')


def load_stopwords():
    with open(stw_path, 'r', encoding='utf8') as g:
        words = g.readlines()
    stopwords = [i.strip() for i in words]
    stopwords.extend(['.','（','）','-','×'])
    return stopwords


def seed_everything(seed):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True


def _move_to_device(batch):
    for k, v in batch.items():
        if isinstance(v, torch.Tensor):
            batch[k] = v.cuda()
    return batch


def ndcg(ranks, K):
    dcg_value = 0.
    idcg_value = 0.
    sranks = sorted(ranks, reverse=True)
    for i in range(0,K):
        logi = math.log(i+2,2)
        dcg_value += ranks[i] / logi
        idcg_value += sranks[i] / logi
    if idcg_value == 0.0:
        idcg_value += 0.00000001
    return dcg_value/idcg_value


def cal_ndcg(all_preds, all_labels):
    ndcgs = []
    for qidx, pred_ids in all_preds.items():
        did2rel = all_labels[qidx]
        ranks = [did2rel[idx] if idx in did2rel else 0 for idx in pred_ids]
        ndcgs.append(ndcg(ranks, 30))
    return sum(ndcgs) / len(ndcgs)


def pad_seq(ids_list, types_list, batch_len):
    new_ids_list, new_types_list, new_masks_list = [], [], []
    for ids, types in zip(ids_list, types_list):
        masks = [1] * len(ids) + [0] * (batch_len - len(ids))
        types += [0] * (batch_len - len(ids))
        ids += [0] * (batch_len - len(ids))
        new_ids_list.append(ids)
        new_types_list.append(types)
        new_masks_list.append(masks)
    return new_ids_list, new_types_list, new_masks_list


def cosent(y_pred, y_true, device):
    y_pred = y_pred * 20
    y_pred = y_pred.unsqueeze(1) - y_pred.unsqueeze(0)  # 这里是算出所有位置 两两之间余弦的差值
    y_true = y_true.unsqueeze(1) < y_true.unsqueeze(0)  # 取出负例-正例的差值
    y_true = y_true.float()
    y_pred = y_pred - (1 - y_true) * 1e12
    y_pred = y_pred.view(-1)
    y_pred = torch.cat((torch.tensor([0]).float().to(device), y_pred), dim=0)  # 这里加0是因为e^0 = 1相当于在log中加了1
    return torch.logsumexp(y_pred, dim=0)


def load_labels(filename, embedding_npy, type_='train'):
    """加载标签
    """
    labels = np.zeros_like(embedding_npy[..., :3]) if type_ == 'train' else np.zeros_like(embedding_npy[..., :2])
    with open(filename, encoding='utf-8') as f:
        i, j = 0, 0
        for line in json.load(f):
            labels[i, j] = line['labels']
            if j < 99:
                j += 1
            else:
                i += 1
                j = 0
    return labels



