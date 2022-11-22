
import torch
import numpy as np
import torch.nn as nn
from torch.nn import init
from transformers import AutoModel


class Lawformer(nn.Module):
    def __init__(self, model_path, tokenizer):
        super(Lawformer, self).__init__()
        self.model = AutoModel.from_pretrained(model_path)
        self.model.resize_token_embeddings(len(tokenizer))
        self.linear = nn.Linear(self.model.config.hidden_size, 1)

    def forward(self, input_ids, attention_mask, token_type_ids, label=None):
        output = self.model(input_ids, attention_mask, token_type_ids)
        logits = self.linear(output[1])
        return logits, label


class InteractExtractor(nn.Module):
    def __init__(self, model_path, tokenizer):
        super(InteractExtractor, self).__init__()
        self.model = AutoModel.from_pretrained(model_path)
        self.model.resize_token_embeddings(len(tokenizer))
        self.linear = nn.Linear(self.model.config.hidden_size, 1)

    def forward(self, input_ids, attention_mask, token_type_ids):
        with torch.no_grad():
            output = self.model(input_ids, attention_mask, token_type_ids)
            pre_embedding = torch.mean(output[0][1::2, 1:510], dim=1)
            post_embedding = torch.mean(output[0][1::2, 1:-1], dim=1)
            case_cls = output[1][1::2]
            key_cls = output[1][::2]
        return torch.cat([key_cls, case_cls, pre_embedding, post_embedding], -1)


class SimplifiedScaledDotProductAttention(nn.Module):
    '''
    Scaled dot-product attention
    '''
    def __init__(self, d_model, h, dropout=.1):
        '''
        :param d_model: Output dimensionality of the model
        :param h: Number of heads
        '''
        super(SimplifiedScaledDotProductAttention, self).__init__()
        self.d_model = d_model
        self.d_k = d_model//h
        self.d_v = d_model//h
        self.h = h
        self.fc_o = nn.Linear(h * self.d_v, d_model)
        self.dropout=nn.Dropout(dropout)
        self.init_weights()

    def init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                init.kaiming_normal_(m.weight, mode='fan_out')
                if m.bias is not None:
                    init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                init.constant_(m.weight, 1)
                init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                init.normal_(m.weight, std=0.001)
                if m.bias is not None:
                    init.constant_(m.bias, 0)

    def forward(self, queries, keys, values, attention_mask=None, attention_weights=None):
        '''
        Computes
        :param queries: Queries (b_s, nq, d_model)
        :param keys: Keys (b_s, nk, d_model)
        :param values: Values (b_s, nk, d_model)
        :param attention_mask: Mask over attention values (b_s, h, nq, nk). True indicates masking.
        :param attention_weights: Multiplicative weights for attention values (b_s, h, nq, nk).
        :return:
        '''
        b_s, nq = queries.shape[:2]
        nk = keys.shape[1]

        q = queries.view(b_s, nq, self.h, self.d_k).permute(0, 2, 1, 3)  # (b_s, h, nq, d_k)
        k = keys.view(b_s, nk, self.h, self.d_k).permute(0, 2, 3, 1)  # (b_s, h, d_k, nk)
        v = values.view(b_s, nk, self.h, self.d_v).permute(0, 2, 1, 3)  # (b_s, h, nk, d_v)

        att = torch.matmul(q, k) / np.sqrt(self.d_k)  # (b_s, h, nq, nk)
        if attention_weights is not None:
            att = att * attention_weights
        if attention_mask is not None:
            att = att.masked_fill(attention_mask, -np.inf)
        att = torch.softmax(att, -1)
        att = self.dropout(att)

        out = torch.matmul(att, v).permute(0, 2, 1, 3).contiguous().view(b_s, nq, self.h * self.d_v)  # (b_s, nq, h*d_v)
        out = self.fc_o(out)  # (b_s, nq, d_model)
        return out


class RankAttention(nn.Module):
    def __init__(self, input_size, filters):
        super(RankAttention, self).__init__()
        self.ssa1 = SimplifiedScaledDotProductAttention(input_size, 1)
        self.ssa2 = SimplifiedScaledDotProductAttention(input_size, 1)
        self.ssa3 = SimplifiedScaledDotProductAttention(input_size, 2)
        self.dense1 = nn.Linear(input_size, filters,)
        self.dense2 = nn.Linear(input_size, filters,)
        self.dense3 = nn.Linear(input_size, filters,)
        self.dense4 = nn.Linear(filters, 1,)
        self.dense5 = nn.Linear(filters, 1,)
        self.dense6 = nn.Linear(filters, 1,)
        self.final_out = nn.Linear(3, 1)

    def forward(self, inputs):
        key_out, case_out, dual_out = inputs[..., :768], inputs[..., 768:1536], inputs[..., 1536:]
        dual_out = torch.mean(dual_out.view(1, 100, 2, 768), dim=2)
        key_out = self.ssa1(key_out, key_out, key_out)
        case_out = self.ssa2(case_out, case_out, case_out)
        dual_out = self.ssa3(dual_out, dual_out, dual_out)

        key_out = self.dense1(key_out)
        case_out = self.dense2(case_out)
        dual_out = self.dense3(dual_out)

        key_out = self.dense4(key_out)
        case_out = self.dense5(case_out)
        dual_out = self.dense6(dual_out)

        output = self.final_out(torch.cat([key_out, case_out, dual_out], -1))
        output = torch.sigmoid(output)
        return output


class RankMLP(nn.Module):
    def __init__(self, input_size, filters):
        super(RankMLP, self).__init__()
        self.dense1 = nn.Linear(input_size, filters, bias=False)
        self.dense2 = nn.Linear(input_size, filters, bias=False)
        self.dense3 = nn.Linear(filters, 1,)
        self.dense4 = nn.Linear(filters, 1,)
        self.dense5 = nn.Linear(2, 1)

    def forward(self, inputs):
        inputs_sep = [inputs[..., :768], inputs[..., 768:1536]]

        key_out = self.dense1(nn.Dropout(0.1)(inputs_sep[0]))
        key_out = nn.ReLU()(key_out)
        key_out = self.dense3(nn.Dropout(0.1)(key_out))

        case_out = self.dense2(nn.Dropout(0.1)(inputs_sep[1]))
        case_out = nn.ReLU()(case_out)
        case_out = self.dense4(nn.Dropout(0.1)(case_out))

        output = self.dense5(torch.cat([key_out, case_out], -1))
        output = torch.sigmoid(output)
        return output

