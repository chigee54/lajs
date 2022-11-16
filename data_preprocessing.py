import os, json
from tqdm import tqdm
import jieba
import numpy as np
from gensim.summarization import bm25


stw_path = os.path.join(os.path.dirname(__file__), 'stopword.txt')


def filter_jbqk(query, doc, pre_sent_size=5, group_size=700, windows_size=200, select=1):
    sents_list = [s for s in re.split(r'[。！；？]', doc) if len(s) > 0]
    pre_sents = sents_list[:pre_sent_size]
    group_sents = []
    sents_string = '。'.join(sents_list[pre_sent_size:])
    if len(sents_string) // group_size == 0:
        return doc
    for i in range(0, len(sents_string), windows_size):
        group_sents.append(sents_string[i:i + group_size])
    rel_sents, scores = search_for_related_sents(query, group_sents, select=select)
    for s in [s for s in re.split(r'[。！；，：？]', rel_sents[0]) if len(s) > 0]:
        if s not in pre_sents:
            pre_sents.append(s)
    filter_sents = '。'.join(pre_sents)
    return filter_sents


def search_for_related_sents(query, sents_list, select=1):
    corpus = []
    with open(stw_path, 'r', encoding='utf8') as g:
        words = g.readlines()
    stopwords = [i.strip() for i in words]
    stopwords.extend(['.','（','）','-','×'])
    for sent in sents_list:
        sent_tokens = [w for w in jieba.lcut(sent.strip()) if w not in stopwords]
        corpus.append(sent_tokens)
    bm25model = bm25.BM25(corpus)

    q_tokens = [w for w in jieba.lcut(query) if w not in stopwords]
    scores = bm25model.get_scores(q_tokens)
    rank_index = np.array(scores).argsort().tolist()[::-1]
    rank_index = rank_index[:select]
    return [sents_list[i] for i in rank_index], [scores[i] for i in rank_index]


def load_data(query_file=None,
              candidate_dir=None,
              saved_train_file=None,
              saved_dev_file=None,
              saved_test_file=None,
              label_top30_file=None,
              dev_label_file=None,
              data_type='train'):
    queries = []
    fq = open(query_file, 'r', encoding='utf8')
    for line in fq:
        queries.append(json.loads(line.strip()))
    all_label_top30 = json.load(open(label_top30_file, 'r', encoding='utf8')) if label_top30_file else None
    data, dev_data, test_data = [], [], []
    for query in tqdm(queries, desc=u'数据转换'):
        qidx, q, crime = str(query['ridx']), str(query['q']), '、'.join(query['crime'])
        # if qidx in self.filter_qids:
        #     continue
        doc_dir = os.path.join(candidate_dir, qidx)
        doc_files = os.listdir(doc_dir)
        if len(doc_files) != 100:
            print(qidx)
            exit(0)
        for doc_file in doc_files:
            doc_path = os.path.join(doc_dir, doc_file)
            didx = str(doc_file.split('.')[0])
            with open(doc_path, 'r', encoding='utf8') as fd:
                sample_d = json.load(fd)
            ajjbqk, ajName = sample_d['ajjbqk'], sample_d['ajName']
            ajjbqk = filter_jbqk(q, ajjbqk).strip()
            if label_top30_file:
                filter_qids = json.load(open(dev_label_file, 'r', encoding='utf8')).keys()
                label = all_label_top30[qidx][didx] if didx in all_label_top30[qidx] else 0
                all_label = [qidx, didx, label]
                if qidx in filter_qids:
                    dev_data.append({'crime': crime, 'query': q, 'ajName': ajName, 'candidate': ajjbqk, 'labels': all_label})
                else:
                    data.append({'crime': crime, 'query': q, 'ajName': ajName, 'candidate': ajjbqk, 'labels': all_label})
            else:
                all_label = [qidx, didx]
                test_data.append({'crime': crime, 'query': q, 'ajName': ajName, 'candidate': ajjbqk, 'labels': all_label})
    if data_type == 'train':
        print(len(data))
        with open(saved_train_file, 'w', encoding='utf8') as fs:
            json.dump(data, fs, ensure_ascii=False, indent=2)
        print(len(dev_data))
        with open(saved_dev_file, 'w', encoding='utf8') as fd:
            json.dump(dev_data, fd, ensure_ascii=False, indent=2)
        return dev_data
    else:
        print(len(test_data))
        with open(saved_test_file, 'w', encoding='utf8') as ft:
            json.dump(test_data, ft, ensure_ascii=False, indent=2)
        return test_data


