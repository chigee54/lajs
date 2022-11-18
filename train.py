
import os
import json
import time
import torch
import argparse
import torch.nn as nn
from tqdm import tqdm
from os.path import join
from loguru import logger
from rank_mlp import ranking
from models import Lawformer
from data_preprocessing import load_data
from torch.utils.data import DataLoader, Dataset
from interact_extract import save_interact_embeddings
from utils import seed_everything, _move_to_device, cal_ndcg, pad_seq
from transformers import AutoTokenizer, AdamW, get_linear_schedule_with_warmup


class LongDataset(Dataset):
    def __init__(self, data_file, tokenizer, max_length):
        self.max_length = max_length
        self.tokenizer = tokenizer
        with open(data_file, 'r', encoding='utf8') as f:
            self.data = json.load(f)
            print(len(self.data))

    def __getitem__(self, index):
        data = self.data[index]
        query_cut = self.tokenizer.tokenize(data['crime']+'☢'+data['query'])[:509]
        candidate_cut = self.tokenizer.tokenize(data['ajName']+'☢'+data['candidate'])[:1020]
        tokens = ['[CLS]'] + query_cut + ['[SEP]'] + candidate_cut + ['[SEP]']
        token_ids = self.tokenizer.convert_tokens_to_ids(tokens)
        types = [0] * (len(query_cut) + 2) + [1] * (len(candidate_cut) + 1)
        input_ids, token_type_ids, attention_mask = pad_seq([token_ids], [types], self.max_length)
        feature = {'input_ids': torch.LongTensor(input_ids),
                   'token_type_ids': torch.LongTensor(token_type_ids),
                   'attention_mask': torch.LongTensor(attention_mask),
                   'label': data['labels']
                   }
        return feature

    def __len__(self):
        return len(self.data)


def scheduler_with_optimizer(model, train_loader, args):
    param_optimizer = list(model.named_parameters())
    no_decay = ['bias', 'LayerNorm.weight']
    optimizer_grouped_parameters = [
        {'params': [p for n, p in param_optimizer if not any(nd in n for nd in no_decay)], 'weight_decay': 0.01},
        {'params': [p for n, p in param_optimizer if any(nd in n for nd in no_decay)], 'weight_decay': 0.0}
    ]
    optimizer = AdamW(optimizer_grouped_parameters, lr=args.learning_rate, correct_bias=False)
    scheduler = get_linear_schedule_with_warmup(
        optimizer,
        num_warmup_steps=len(train_loader) * args.num_epochs * args.warm_up_proportion // args.gradient_accumulation_step,
        num_training_steps=len(train_loader) * args.num_epochs // args.gradient_accumulation_step)
    return optimizer, scheduler


def evaluate(model, dataloader):
    model.eval()
    all_labels, sub_labels = {}, {}
    with torch.no_grad():
        all_preds, info = {}, {}
        for data in dataloader:
            data = _move_to_device(data)
            score, label = model(**data)
            for n, i in enumerate(zip(label[0], label[1], label[2])):
                if i[0] not in info.keys():
                    sub_labels[i[1]] = i[2]
                    all_labels[i[0]] = sub_labels
                    info[i[0]] = [[i[1]], [score[n]]]
                else:
                    all_labels[i[0]][i[1]] = i[2]
                    info[i[0]][1].append(score[n])
                    info[i[0]][0].append(i[1])
        for qidx in info.keys():
            dids, preds = info[qidx]
            sorted_r = sorted(list(zip(dids, preds)), key=lambda x: x[1], reverse=True)
            pred_ids = [x[0] for x in sorted_r]
            all_preds[qidx] = pred_ids[:30]
    ndcg_30 = cal_ndcg(all_preds, all_labels)
    del info, all_preds
    torch.cuda.empty_cache()
    return ndcg_30


def train(model, tokenizer, train_file, dev_file, saved_model_path, args):
    train_data = LongDataset(train_file, tokenizer, args.max_length)
    dev_data = LongDataset(dev_file, tokenizer, args.max_length)
    dev_loader = DataLoader(dev_data, batch_size=args.eval_batch_size, shuffle=False)
    train_loader = DataLoader(train_data, batch_size=args.batch_size, shuffle=True)
    optimizer, scheduler = scheduler_with_optimizer(model, train_loader, args)
    loss_function = nn.MSELoss()
    best = 0
    for epoch in range(args.num_epochs):
        torch.cuda.empty_cache()
        model.train()
        model.zero_grad()
        for batch_idx, data in enumerate(tqdm(train_loader)):
            step = epoch * len(train_loader) + batch_idx
            data = _move_to_device(data)
            score, label = model(**data)
            label = label[:][2]
            loss = loss_function(score.view(-1), label.to(torch.float).to(args.device))
            optimizer.zero_grad()
            loss.backward()
            step += 1
            optimizer.step()
            model.zero_grad()
            if step % args.report_step == 0:
                torch.cuda.empty_cache()
                ndcg30 = evaluate(model, dev_loader)
                logger.info('Epoch[{}/{}], loss:{}, ndcg30: {}'.format(epoch + 1, args.num_epochs, loss.item(), ndcg30))
                model.train()
                if best < ndcg30:
                    best = ndcg30
                    torch.save(model.state_dict(), saved_model_path)
                    logger.info('higher_ndcg30: {}, step {}, epoch {}, save model\n'.format(best, step, epoch + 1))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-device', default='cuda', type=str)
    parser.add_argument('-seed', default=42, type=int)
    parser.add_argument('-input', default='../data/phase_2/train/', type=str, help='input path of the train dataset.')
    parser.add_argument('-dev_id_file', default='../data/phase_1/train/label_top30_dict.json', type=str, help='dev id')
    parser.add_argument('-model_path', default='../../pretrained_model/lawformer', type=str)
    parser.add_argument('-tokenizer_path', default="../../pretrained_model/roberta", type=str)
    parser.add_argument('-output_path', default='./saved', type=str)
    # Lawformer Finetune Argument
    parser.add_argument('-report_step', default=2000, type=int)
    parser.add_argument('-max_length', default=1533, type=int)
    parser.add_argument('-eval_batch_size', default=20, type=int)
    parser.add_argument('-batch_size', default=1, type=int)
    parser.add_argument('-num_epochs', default=3, type=int)
    parser.add_argument('-learning_rate', default=1e-5, type=float)
    parser.add_argument('-warm_up_proportion', default=0, type=float)
    parser.add_argument('-gradient_accumulation_step', default=1, type=int)
    # Interact Extractor Argument
    parser.add_argument('-extract_batch_size', default=1, type=int)
    # RANK MLP Argument
    parser.add_argument('-mlp_batch_size', type=int, default=1, help='batch size')
    parser.add_argument('-mlp_epoch_num', type=int, default=1000, help='number of epochs')
    parser.add_argument('-mlp_lr', type=float, default=5e-5, help='learning rate')
    parser.add_argument('-weight_decay', type=float, default=0., help='decay weight of optimizer')
    parser.add_argument('-input_size', type=int, default=768)
    parser.add_argument('-hidden_size', type=int, default=384)
    args = parser.parse_args()

    # config environment
    os.environ['CUDA_VISIBLE_DEVICES'] = '0'
    args.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    seed_everything(args.seed)
    model_output_path = join(args.output_path, 'bsz{}_lr{}'.format(args.batch_size, args.learning_rate))
    if not os.path.exists(model_output_path):
        os.makedirs(model_output_path)

    cur_time = time.strftime("%Y%m%d_%H_%M", time.localtime())
    logger.add(join(model_output_path, 'train-{}.log'.format(cur_time)))
    logger.info(args)

    new_train_file = os.path.join(os.path.dirname(__file__), 'saved/new_train_data.json')
    new_dev_file = os.path.join(os.path.dirname(__file__), 'saved/new_dev_data.json')
    input_path = args.input
    input_query_path = os.path.join(input_path, 'query.json')
    input_label_path = os.path.join(input_path, 'label_top30_dict.json')
    input_candidate_path = os.path.join(input_path, 'candidates')

    print('Data preprocessing...')
    if not os.path.exists(new_train_file):
        load_data(query_file=input_query_path,
                  candidate_dir=input_candidate_path,
                  saved_dev_file=new_dev_file,
                  saved_train_file=new_train_file,
                  dev_label_file=args.dev_id_file,
                  label_top30_file=input_label_path,
                  data_type='train')
    time.sleep(1)
    print('Data preprocessing finished...')

    tokenizer = AutoTokenizer.from_pretrained(args.tokenizer_path)
    tokenizer.add_tokens(['☢'])
    saved_lawformer_path = join(model_output_path, 'lawformer.pt')
    if not os.path.exists(saved_lawformer_path):
        model = Lawformer(args.model_path, tokenizer).to(args.device)
        print('Lawformer training...')
        start_time = time.time()
        train(model, tokenizer, new_train_file, new_dev_file, saved_lawformer_path, args)
        logger.info('run time: {:.2f}'.format((time.time() - start_time) / 60))
        print('Train Done!')

    new_train_embedding_path = os.path.join(os.path.dirname(__file__), 'saved/new_train_embeddings.npy')
    new_dev_embedding_path = os.path.join(os.path.dirname(__file__), 'saved/new_dev_embeddings.npy')
    print('Extracting interact embeddings...')
    if not os.path.exists(new_dev_embedding_path):
        save_interact_embeddings(args=args,
                                 tokenizer=tokenizer,
                                 saved_data=new_dev_file,
                                 saved_encoder_path=saved_lawformer_path,
                                 saved_embeddings=new_dev_embedding_path)
    if not os.path.exists(new_train_embedding_path):
        save_interact_embeddings(args=args,
                                 tokenizer=tokenizer,
                                 saved_data=new_train_file,
                                 saved_encoder_path=saved_lawformer_path,
                                 saved_embeddings=new_train_embedding_path)
    print('Interact embeddings have saved')

    args.mlp_output_path = join(args.output_path, 'bsz{}_lr{}'.format(args.mlp_batch_size, args.mlp_lr))
    if not os.path.exists(args.mlp_output_path):
        os.makedirs(args.mlp_output_path)
    print('RANK MLP training...')
    ranking(args=args,
            mode='train',
            dev_data=new_dev_file,
            train_data=new_train_file,
            dev_embeddings=new_dev_embedding_path,
            train_embeddings=new_train_embedding_path)
    print('Done!')


