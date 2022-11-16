import datetime
import json, os, math
from os.path import join
import argparse
import numpy as np
import torch
import torch.nn as nn
from tqdm import tqdm
from torch.utils.data import Dataset, DataLoader
import logging
from torch.nn import init
import torch.nn.functional as F


parser = argparse.ArgumentParser()
parser.add_argument('--batch_size', type=int, default=1, help='batch size')
parser.add_argument('--epoch_num', type=int, default=1000, help='number of epochs')
parser.add_argument('--lr', type=float, default=5e-5, help='learning rate')
parser.add_argument('--weight_decay', type=float, default=0., help='decay weight of optimizer')
parser.add_argument('--output_path', type=str, default="./saved", help='checkpoint path')
parser.add_argument('--input_size', type=int, default=768)
parser.add_argument('--hidden_size', type=int, default=384)
parser.add_argument('--cuda_pos', type=str, default='1', help='which GPU to use')
parser.add_argument('--seed', type=int, default=42, help='max length of each case')
args = parser.parse_args()

np.random.seed(args.seed)
torch.manual_seed(args.seed)
torch.cuda.manual_seed_all(args.seed)
torch.backends.cudnn.deterministic = True

args.output_path = join(args.output_path, 'bsz{}_lr{}'.format(args.seed, args.batch_size, args.lr))
if not os.path.exists(args.output_path):
    os.makedirs(args.output_path)

log_name = "log_DGCNN"
logging.basicConfig(level=logging.INFO,  # 控制台打印的日志级别
                    filename=args.output_path + '/{}.log'.format(log_name),
                    filemode='a',  ##模式，有w和a，w就是写模式，每次都会重新写日志，覆盖之前的日志
                    # a是追加模式，默认如果不写的话，就是追加模式
                    format=
                    '%(asctime)s - %(pathname)s[line:%(lineno)d] - %(levelname)s: %(message)s'
                    # 日志格式
                    )
device = torch.device('cuda:' + args.cuda_pos) if torch.cuda.is_available() else torch.device('cpu')


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


class RankMLP(nn.Module):
    def __init__(self, input_size, filters):
        super(RankMLP, self).__init__()
        self.dense1 = nn.Linear(input_size, filters, bias=False)
        self.dense2 = nn.Linear(filters, 1,)
        self.dense3 = nn.Linear(input_size, filters, bias=False)
        self.dense4 = nn.Linear(filters, 1,)
        self.dense5 = nn.Linear(2, 1)

    def forward(self, inputs):
        inputs_sep = [inputs[..., :768], inputs[..., 768:1536]]

        key_out = self.dense1(nn.Dropout(0.1)(inputs_sep[0]))
        key_out = nn.ReLU()(key_out)
        key_out = self.dense2(nn.Dropout(0.1)(key_out))

        case_out = self.dense3(nn.Dropout(0.1)(inputs_sep[1]))
        case_out = nn.ReLU()(case_out)
        case_out = self.dense4(nn.Dropout(0.1)(case_out))

        output = self.dense5(torch.cat([key_out, case_out], -1))
        output = torch.sigmoid(output)
        return output


class PrepareDataset(Dataset):
    def __init__(self, data_x, data_y):
        super(PrepareDataset, self).__init__()
        self.data_x_tensor = torch.from_numpy(data_x)
        self.data_y_tensor = torch.from_numpy(data_y)

    def __len__(self):
        return len(self.data_x_tensor)

    def __getitem__(self, idx):
        return self.data_x_tensor[idx], self.data_y_tensor[idx]


def cosent(y_pred, y_true):
    y_pred = y_pred * 20
    y_pred = y_pred.unsqueeze(1) - y_pred.unsqueeze(0)  # 这里是算出所有位置 两两之间余弦的差值
    y_true = y_true.unsqueeze(1) < y_true.unsqueeze(0)  # 取出负例-正例的差值
    y_true = y_true.float()
    y_pred = y_pred - (1 - y_true) * 1e12
    y_pred = y_pred.view(-1)
    y_pred = torch.cat((torch.tensor([0]).float().to(device), y_pred), dim=0)  # 这里加0是因为e^0 = 1相当于在log中加了1
    return torch.logsumexp(y_pred, dim=0)


def ndcg(ranks, K):
    dcg_value = 0.
    idcg_value = 0.
    sranks = sorted(ranks, reverse=True)
    for i in range(0, K):
        logi = math.log(i+2, 2)
        dcg_value += ranks[i] / logi
        idcg_value += sranks[i] / logi
    if idcg_value == 0.0:
        idcg_value += 0.00000001
    return dcg_value/idcg_value


def cal_ndcg(all_preds, all_labels):
    ndcgs = []
    for qidx, pred_ids in all_preds.items():
        did2rel = all_labels[str(qidx)]
        ranks = [did2rel[str(idx)] if str(idx) in did2rel else 0 for idx in pred_ids]
        ndcgs.append(ndcg(ranks, 30))
    return sum(ndcgs) / len(ndcgs)


def evaluate(model, dataloader, all_labels=None):
    model.eval()
    with torch.no_grad():
        all_preds = {}
        for data in dataloader:
            x_batch, labels = data
            x_batch = x_batch.to(device)
            scores = model(x_batch)
            labels = labels.squeeze().cpu().tolist()
            scores = scores.squeeze().cpu().tolist()
            preds, dids, qidx = [], [], []
            for label, score in zip(labels, scores):
                preds.append(score)
                dids.append(int(label[1]))
                qidx.append(int(label[0]))
            sorted_r = sorted(list(zip(dids, preds)), key=lambda x: x[1], reverse=True)
            pred_ids = [x[0] for x in sorted_r]
            assert len(set(qidx)) == 1
            all_preds[qidx[0]] = pred_ids[:30]
    ndcg_30 = cal_ndcg(all_preds, all_labels)
    torch.cuda.empty_cache()
    return ndcg_30


def test_result(dataloader, model_path):
    model = torch.load(model_path)
    model.eval()
    params = list(model.named_parameters())
    a = params.__len__()
    model.to(device)
    with torch.no_grad():
        all_preds = {}
        for data in tqdm(dataloader):
            x_batch, labels = data
            x_batch = x_batch.to(device)
            scores, _ = model(x_batch, )
            labels = labels.squeeze().cpu().tolist()
            scores = scores.squeeze().cpu().tolist()
            preds, dids, qidx = [], [], []
            for label, score in zip(labels, scores):
                preds.append(score)
                dids.append(int(label[1]))
                qidx.append(int(label[0]))
            sorted_r = sorted(list(zip(dids, preds)), key=lambda x: x[1], reverse=True)
            pred_ids = [x[0] for x in sorted_r]
            assert len(set(qidx)) == 1
            all_preds[qidx[0]] = pred_ids[:30]
    return all_preds


def train(model, train_dataloader, valid_dataloader):
    all_labels = json.load(open(args.label_file, 'r', encoding='utf8'))
    model.to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    best = 0
    for epoch in tqdm(range(args.epoch_num)):
        epoch_loss = 0.0
        current_step = 0
        model.train()
        for batch_data in train_dataloader:
            x_batch, label_batch = batch_data
            x_batch = x_batch.to(device)
            label_batch = label_batch[..., 2].to(device)
            output_batch = model(x_batch)
            loss = cosent(output_batch.squeeze(), label_batch.squeeze())
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            loss_item = loss.cpu().detach().item()
            epoch_loss += loss_item
            current_step += 1

        epoch_loss = epoch_loss / current_step
        ndcg30 = evaluate(model, valid_dataloader, all_labels)
        print('Epoch[{}/{}], loss:{}, ndcg30: {}'.format(epoch + 1, args.epoch_num, epoch_loss, ndcg30))
        if best < ndcg30:
            best = ndcg30
            torch.save(model, args.output_path + '/' + f"rank_mlp.pt")
            print('higher_ndcg30: {}, Epoch[{}/{}], loss:{}, save model\n'.format(best, epoch + 1, args.epoch_num, epoch_loss))
            logging.info('higher_ndcg30: {}, Epoch[{}/{}], loss:{}, save model\n'.format(best, epoch + 1, args.epoch_num,
                                                                                  epoch_loss))


def ranking(mode='train',
            dev_data=None,
            test_data=None,
            train_data=None,
            dev_embeddings=None,
            test_embeddings=None,
            train_embeddings=None,
            rank_mlp_path=None
            ):
    if mode == 'test':
        test_x = np.load(test_embeddings)
        test_y = load_labels(test_data, test_x, 'test')
        test_dataloader = DataLoader(PrepareDataset(test_x, test_y), batch_size=1, shuffle=False)
        return test_result(test_dataloader, rank_mlp_path)
    elif mode == 'train':
        dev_x = np.load(dev_embeddings)
        train_x = np.load(train_embeddings)
        dev_y = load_labels(dev_data, dev_x)
        train_y = load_labels(train_data, train_x)
        train_dataloader = DataLoader(PrepareDataset(train_x, train_y), batch_size=args.batch_size, shuffle=True, drop_last=True)
        valid_dataloader = DataLoader(PrepareDataset(dev_x, dev_y, ), batch_size=1, shuffle=False)
        model = RankMLP(args.input_size, args.hidden_size)
        train(model, train_dataloader, valid_dataloader)
    else:
        raise ValueError('Error!')
