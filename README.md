
![](https://img.shields.io/badge/Python-3.8.13-blue.svg)
![](https://img.shields.io/badge/torch-1.7.1-brightgreen.svg)
![](https://img.shields.io/badge/transformers-4.2.1-brightgreen.svg)
![](https://img.shields.io/badge/gensim-3.8.0-brightgreen.svg)
![](https://img.shields.io/badge/jieba-0.42.1-brightgreen.svg)
![](https://img.shields.io/badge/numpy-1.22.3-brightgreen.svg)
![](https://img.shields.io/badge/loguru-0.6.0-brightgreen.svg)



# Simple MLP for Learning to Rank based on Lawformer
基于Lawformer的简易MLP排序学习。

## 环境配置
Python3.8.13

GPU: RTX3090*1

具体请参考`requirments.txt`


## 概述
目的：在显卡资源有限的情况下尽可能的发挥Lawformer模型的效果

分析：在Lawformer发布的论文中提到，对于类案检索长文本数据，query长度为509，candidates长度为3072，则模型的输入总长度为3584。
在本方法中限制模型的输入总长度为1533（实际上还未达到显存的极限），query长度为509，candidates长度为1020（是论文中的1/3）。

方法：
- **数据预处理**：既然限制了candidates的长度为1020，那么就得筛选最相关的那一部分作为模型的输入。
本方法仅考虑ajjbqk字段内容，前五句必选，后面的内容利用BM25进行筛选，筛选出的内容长度≤700，最后加起来总长度不定。具体请参考`data_preprocessing.py`

- **Lawformer-Finetune**：预处理后的数据即可用于微调Lawformer，训练时将crime+query作为查询案例的输入，将ajName+candidate作为候选案例的输入，最后查询案例+候选案例输入模型中进行交互，训练batch_size设置为1，采用MSE-Loss以PointWise的方式进行排序学习，评估时采用NDCG@30。具体请参考`train.py`

- **提取交互向量**：在Lawformer微调之后，取出所有查询与候选的交互向量。具体请参考`interact_extract.py`

- **训练Rank_MLP**：将所有交互向量分组，每个查询案例对应100个候选案例，搭建一个简易的MLP，对这100个交互向量重新以PairWise的方式进行排序学习，损失函数采用CoSENT（[https://spaces.ac.cn/archives/9341](https://spaces.ac.cn/archives/9341)）。具体请参考`rank_mlp.py`


## 预测
直接运行`run.sh`，其中data_path需改成封测阶段数据的路径。
```bash
#!/bin/bash

data_path=cail2022_类案检索_封测阶段/

python predict.py \
    --input $data_path \
    --output ./ \
    --encoder_path saved/bsz1_lr1e-05/lawformer_best.pt \
    --rank_mlp_path saved/bsz1_lr5e-05/rank_mlp_best.pt \
    --model_path thunlp/Lawformer \
    --tokenizer_path hfl/chinese-roberta-wwm-ext \
    --extract_batch_size 10 \
    --max_length 1533
```


## 训练
直接运行`train.sh`，其中train_data_path需改成第二阶段训练集的路径，dev_data_path需改成第一阶段训练集的路径。
```bash
#!/bin/bash

train_data_path=cail2022_类案检索_第二阶段/train/
dev_data_path=cail2022_类案检索_第一阶段/train/label_top30_dict.json

python train.py \
    --input $train_data_path \
    --dev_id_file $dev_data_path \
    --output_path ./saved \
    --model_path thunlp/Lawformer \
    --tokenizer_path hfl/chinese-roberta-wwm-ext \
    --max_length 1533 
```

## 结果
以下为第二阶段测试结果


|                   |           NDCG@30 ||
| :--------------   | --------- | ---------- |
|       模型        |     Dev   |    Test    |
| Lawformer-Finetune| 0.9386    |            |
| Lawformer-RankMLP | 0.9491    | 0.9355     |
