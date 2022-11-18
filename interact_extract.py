# coding: utf-8
'''
交互特征提取器
'''

import json
import torch
import numpy as np
from tqdm import tqdm
from utils import pad_seq
from models import InteractExtractor
from torch.utils.data import DataLoader, Dataset


def save_interact_embeddings(args=None,
                             tokenizer=None,
                             saved_data=None,
                             saved_embeddings=None,
                             saved_encoder_path=None
                             ):
    model = InteractExtractor(args.model_path, tokenizer).cuda()
    model.load_state_dict(torch.load(saved_encoder_path))
    model.eval()
    extract_dataset = ExtractDataset(saved_data, tokenizer, args.max_length)
    dataset = DataLoader(extract_dataset, batch_size=args.extract_batch_size, shuffle=False)

    all_embeddings, batch_embeddings, i = [], None, 0
    for data in tqdm(dataset, desc='提取交互向量'):
        torch.cuda.empty_cache()
        i += 1
        for k, v in data.items():
            data[k] = v.view(2*args.extract_batch_size, -1).cuda()
        embeddings = model(**data)
        if i == 1:
            batch_embeddings = embeddings
            continue
        batch_embeddings = torch.cat([batch_embeddings, embeddings], 0)
        if i == 100 // args.extract_batch_size:
            all_embeddings.append(batch_embeddings.cpu().detach().numpy())
            i = 0

    print('The number of data: {}'.format(len(all_embeddings)))
    np.save(saved_embeddings, all_embeddings)
    print(u'输出路径：%s' % saved_embeddings)


class ExtractDataset(Dataset):
    def __init__(self, saved_data_file, tokenizer, max_length):
        self.max_length = max_length
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
        input_ids, token_type_ids, attention_mask = pad_seq([crime_ids, token_ids], [crime_types, types], self.max_length)
        feature = {'input_ids': torch.LongTensor(input_ids),
                   'token_type_ids': torch.LongTensor(token_type_ids),
                   'attention_mask': torch.LongTensor(attention_mask),
                   }
        return feature

    def __len__(self):
        return len(self.data)


if __name__ == '__main__':
    save_interact_embeddings()
