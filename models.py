import torch.nn as nn
from transformers import AutoModel
import torch


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
            case_cls = output[1][1::2]
            key_cls = output[1][::2]
        return torch.cat([key_cls, case_cls], -1)


class RankMLP(nn.Module):
    def __init__(self, input_size, filters):
        super(RankMLP, self).__init__()
        self.dense1 = nn.Linear(input_size, filters, bias=False)
        self.dense2 = nn.Linear(input_size, filters, bias=False)
        self.dense3 = nn.Linear(filters, 1,)
        self.dense4 = nn.Linear(filters, 1,)
        self.dense5 = nn.Linear(2, 1)

    def forward(self, inputs):
        inputs_sep = [inputs[..., :768], inputs[..., 768:]]

        key_out = self.dense1(nn.Dropout(0.1)(inputs_sep[0]))
        key_out = nn.ReLU()(key_out)
        key_out = self.dense3(nn.Dropout(0.1)(key_out))

        case_out = self.dense2(nn.Dropout(0.1)(inputs_sep[1]))
        case_out = nn.ReLU()(case_out)
        case_out = self.dense4(nn.Dropout(0.1)(case_out))

        output = self.dense5(torch.cat([key_out, case_out], -1))
        output = torch.sigmoid(output)
        return output
