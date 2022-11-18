
import torch
import logging
import numpy as np
from tqdm import tqdm
from models import RankMLP
from utils import cal_ndcg, cosent, load_labels
from torch.utils.data import Dataset, DataLoader


class PrepareDataset(Dataset):
    def __init__(self, data_x, data_y):
        super(PrepareDataset, self).__init__()
        self.data_x_tensor = torch.from_numpy(data_x)
        self.data_y_tensor = torch.from_numpy(data_y)

    def __len__(self):
        return len(self.data_x_tensor)

    def __getitem__(self, idx):
        return self.data_x_tensor[idx], self.data_y_tensor[idx]


def evaluate(model, dataloader, device):
    model.eval()
    with torch.no_grad():
        all_preds, all_labels, sub_labels = {}, {}, {}
        for data in dataloader:
            x_batch, labels = data
            x_batch = x_batch.to(device)
            scores = model(x_batch)
            labels = labels.squeeze().cpu().tolist()
            scores = scores.squeeze().cpu().tolist()
            preds, dids, qidx, label_value = [], [], [], []
            for label, score in zip(labels, scores):
                preds.append(score)
                dids.append(int(label[1]))
                qidx.append(int(label[0]))
                if label[0] not in all_labels.keys():
                    sub_labels[label[1]] = label[2]
                    all_labels[label[0]] = sub_labels
                else:
                    all_labels[label[0]][label[1]] = label[2]
            sorted_r = sorted(list(zip(dids, preds)), key=lambda x: x[1], reverse=True)
            pred_ids = [x[0] for x in sorted_r]
            assert len(set(qidx)) == 1
            all_preds[qidx[0]] = pred_ids[:30]
    ndcg_30 = cal_ndcg(all_preds, all_labels)
    torch.cuda.empty_cache()
    return ndcg_30


def test_result(dataloader, model, device):
    model.eval()
    model.to(device)
    with torch.no_grad():
        all_preds = {}
        for data in tqdm(dataloader):
            x_batch, labels = data
            x_batch = x_batch.to(device)
            scores = model(x_batch, )
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


def train(model, train_dataloader, valid_dataloader, args):
    model.to(args.device)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.mlp_lr, weight_decay=args.weight_decay)
    best = 0
    for epoch in tqdm(range(args.mlp_epoch_num)):
        epoch_loss = 0.0
        current_step = 0
        model.train()
        for batch_data in train_dataloader:
            x_batch, label_batch = batch_data
            x_batch = x_batch.to(args.device)
            label_batch = label_batch[..., 2].to(args.device)
            output_batch = model(x_batch)
            loss = cosent(output_batch.squeeze(), label_batch.squeeze(), args.device)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            loss_item = loss.cpu().detach().item()
            epoch_loss += loss_item
            current_step += 1

        epoch_loss = epoch_loss / current_step
        ndcg30 = evaluate(model, valid_dataloader, args.device)
        print('Epoch[{}/{}], loss:{}, ndcg30: {}'.format(epoch + 1, args.mlp_epoch_num, epoch_loss, ndcg30))
        if best < ndcg30:
            best = ndcg30
            torch.save(model, args.mlp_output_path + '/' + f"rank_mlp.pt")
            print('higher_ndcg30: {}, Epoch[{}/{}], loss:{}, save model\n'.format(best, epoch + 1, args.mlp_epoch_num, epoch_loss))
            logging.info('higher_ndcg30: {}, Epoch[{}/{}], loss:{}, save model\n'.format(best, epoch + 1, args.mlp_epoch_num, epoch_loss))


def ranking(args=None,
            mode=None,
            model=None,
            dev_data=None,
            test_data=None,
            train_data=None,
            dev_embeddings=None,
            test_embeddings=None,
            train_embeddings=None
            ):
    if mode == 'test':
        test_x = np.load(test_embeddings)
        test_y = load_labels(test_data, test_x, 'test')
        test_dataloader = DataLoader(PrepareDataset(test_x, test_y), batch_size=1, shuffle=False)
        result = test_result(test_dataloader, model, args.device)
        return result
    elif mode == 'train':
        logging.basicConfig(filemode='a',
                            level=logging.INFO,
                            filename=args.mlp_output_path + '/{}.log'.format("RANK_MLP"),
                            format='%(asctime)s - %(pathname)s[line:%(lineno)d] - %(levelname)s: %(message)s')
        dev_x = np.load(dev_embeddings)
        train_x = np.load(train_embeddings)
        dev_y = load_labels(dev_data, dev_x)
        train_y = load_labels(train_data, train_x)
        train_dataloader = DataLoader(PrepareDataset(train_x, train_y), batch_size=args.mlp_batch_size, shuffle=True, drop_last=True)
        valid_dataloader = DataLoader(PrepareDataset(dev_x, dev_y, ), batch_size=args.mlp_batch_size, shuffle=False)
        model = RankMLP(args.input_size, args.hidden_size)
        train(model, train_dataloader, valid_dataloader, args)
    else:
        raise ValueError('Error!')


