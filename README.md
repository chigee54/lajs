# Simple MLP for Ranking based on Lawformer under Limited Resources
在有限资源下基于lawformer的简易MLP排序。

## 环境配置
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

- **提取交互向量**：在Lawformer微调之后，取出所有查询与候选的交互向量。具体请参考`interact_embeddings.py`

- **训练Rank_MLP**：将所有交互向量分组，每个查询案例对应100个候选案例，搭建一个简易的MLP，对这100个交互向量重新以PairWise的方式进行排序学习，损失函数参考CoSENT（[https://spaces.ac.cn/archives/9341](https://spaces.ac.cn/archives/9341)）。具体请参考`rank_mlp.py`



## 训练

训练代码：
```bash
python train.py
```

## 预测
直接运行`run.sh`
```bash
python predict.py
```

