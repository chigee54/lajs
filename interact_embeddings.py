# coding: utf-8
'''
长文本编码器
'''

from tqdm import tqdm
import json
import os, re, jieba
import numpy as np
import torch
import random
import argparse
from torch.utils.data import DataLoader, Dataset
import torch.nn as nn
from gensim.summarization import bm25
from transformers import AutoModel, AutoTokenizer


def save_interact_embeddings(saved_data=None, saved_embeddings=None, saved_encoder_path=None):
    parser = argparse.ArgumentParser()
    parser.add_argument('-device', default='cuda', type=str)
    parser.add_argument('-seed', default=42, type=int)
    parser.add_argument('-model_path', default='../pretrained_model/lawformer', type=str)
    parser.add_argument('-tokenizer_path', default="../pretrained_model/roberta", type=str)
    parser.add_argument('-max_length', default=1533, type=int)
    parser.add_argument('-batch_size', default=10, type=int)
    args = parser.parse_args()

    # config environment
    os.environ['CUDA_VISIBLE_DEVICES'] = '0'
    args.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    seed_everything(args.seed)

    tokenizer = AutoTokenizer.from_pretrained(args.tokenizer_path)
    tokenizer.add_tokens(['☢'])
    model = LongModel(args.model_path, tokenizer).cuda()
    model.load_state_dict(torch.load(saved_encoder_path))
    model.eval()
    longdataset = LongDataset(saved_data, tokenizer, args.max_length)
    dataset = DataLoader(longdataset, batch_size=args.batch_size, shuffle=False)

    all_embeddings, batch_embeddings, i = [], None, 0
    for data in tqdm(dataset, desc='数据向量化'):
        torch.cuda.empty_cache()
        i += 1
        for k, v in data.items():
            data[k] = v.view(2*args.batch_size, -1).cuda()
        embeddings = model(**data)
        if i == 1:
            batch_embeddings = embeddings
            continue
        batch_embeddings = torch.cat([batch_embeddings, embeddings], 0)
        if i == 100 // args.batch_size:
            all_embeddings.append(batch_embeddings.cpu().detach().numpy())
            i = 0

    print('The number of {} data: {}'.format(data_type, len(all_embeddings)))
    np.save(saved_embeddings, all_embeddings)
    print(u'输出路径：%s' % args.saved_embeddings)


def seed_everything(seed):
    '''
    设置整个开发环境的seed
    :param seed:
    :param device:
    :return:
    '''
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    # torch.backends.cudnn.benchmark = False
    # some cudnn methods can be random even after fixing the seed
    # unless you tell it to be deterministic
    torch.backends.cudnn.deterministic = True


class LongModel(nn.Module):
    def __init__(self, model_path, tokenizer):
        super(LongModel, self).__init__()
        self.model = AutoModel.from_pretrained(model_path)
        self.model.resize_token_embeddings(len(tokenizer))
        self.linear = nn.Linear(self.model.config.hidden_size, 1)

    def forward(self, input_ids, attention_mask, token_type_ids):
        with torch.no_grad():
            output = self.model(input_ids, attention_mask, token_type_ids)
            case_cls = output[1][1::2]
            key_cls = output[1][::2]
        return torch.cat([key_cls, case_cls], -1)


class LongDataset(Dataset):
    def __init__(self, saved_data_file, tokenizer, maxlength):
        self.maxlength = maxlength
        self.tokenizer = tokenizer
        with open(saved_data_file, 'r', encoding='utf8') as f:
            self.data = json.load(f)
            print(len(self.data))

    def __getitem__(self, index):
        data = self.data[index]
        q_crime_tokens, d_crime_tokens = self.tokenizer.tokenize(data['crime']), self.tokenizer.tokenize(data['ajName'])
        crime_tokens = ['[CLS]'] + q_crime_tokens + ['[SEP]'] + d_crime_tokens + ['[SEP]']
        crime_ids = self.tokenizer.convert_tokens_to_ids(crime_tokens)
        crime_types = [0] * (len(q_crime_tokens) + 2) + [1] * (len(d_crime_tokens) + 1)
        query_cut = self.tokenizer.tokenize(data['crime']+'☢'+data['query'])[:509]
        candidate_cut = self.tokenizer.tokenize(data['ajName']+'☢'+data['candidate'])[:1020]
        tokens = ['[CLS]'] + query_cut + ['[SEP]'] + candidate_cut + ['[SEP]']
        token_ids = self.tokenizer.convert_tokens_to_ids(tokens)
        types = [0] * (len(query_cut) + 2) + [1] * (len(candidate_cut) + 1)
        input_ids, token_type_ids, attention_mask = self.pad_seq([crime_ids, token_ids], [crime_types, types])
        feature = {'input_ids': torch.LongTensor(input_ids),
                   'token_type_ids': torch.LongTensor(token_type_ids),
                   'attention_mask': torch.LongTensor(attention_mask),
                   }
        return feature

    def pad_seq(self, ids_list, types_list):
        batch_len = self.maxlength
        new_ids_list, new_types_list, new_masks_list = [], [], []
        for ids, types in zip(ids_list, types_list):
            masks = [1] * len(ids) + [0] * (batch_len - len(ids))
            types += [0] * (batch_len - len(ids))
            ids += [0] * (batch_len - len(ids))
            new_ids_list.append(ids)
            new_types_list.append(types)
            new_masks_list.append(masks)
        return new_ids_list, new_types_list, new_masks_list

    def __len__(self):
        return len(self.data)


if __name__ == '__main__':
    save_interact_embeddings()
