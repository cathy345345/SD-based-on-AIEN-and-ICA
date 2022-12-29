import math
import torch
import torch.nn as nn
from torch.nn import CrossEntropyLoss
from transformers import BertModel
d_model = 768
from transformers import BertTokenizer
tokenizer = BertTokenizer.from_pretrained('bert-base-uncased', do_lower_case=True)



def gelu(x):
    """Implementation of the gelu activation function.
        For information: OpenAI GPT's gelu is slightly different (and gives slightly different results):
        0.5 * x * (1 + torch.tanh(math.sqrt(2 / math.pi) * (x + 0.044715 * torch.pow(x, 3))))
    """
    return x * 0.5 * (1.0 + torch.erf(x / math.sqrt(2.0)))


class BertLayerNorm(nn.Module):
    def __init__(self, hidden_size, eps=1e-12):
        """Construct a layernorm module in the TF style (epsilon inside the square root).
        """
        super(BertLayerNorm, self).__init__()
        self.weight = nn.Parameter(torch.ones(hidden_size))    #
        self.bias = nn.Parameter(torch.zeros(hidden_size))
        self.variance_epsilon = eps

    def forward(self, x):
        u = x.mean(-1, keepdim=True)
        s = (x - u).pow(2).mean(-1, keepdim=True)
        x = (x - u) / torch.sqrt(s + self.variance_epsilon)
        return self.weight * x + self.bias


class BertOutput(nn.Module):
    def __init__(self):
        super(BertOutput, self).__init__()
        self.dense = nn.Linear(3072, 768)
        self.LayerNorm = BertLayerNorm(768, eps=1e-12)
        self.dropout = nn.Dropout(0.1)

    def forward(self, hidden_states, input_tensor):
        hidden_states = self.dense(hidden_states)
        hidden_states = self.dropout(hidden_states)
        hidden_states = self.LayerNorm(hidden_states + input_tensor)
        return hidden_states


class BertIntermediate(nn.Module):
    def __init__(self):
        super(BertIntermediate, self).__init__()
        self.dense = nn.Linear(768, 3072)

    def forward(self, hidden_states):
        hidden_states = self.dense(hidden_states)
        hidden_states = gelu(hidden_states)
        return hidden_states


class BertSelfOutput(nn.Module):
    def __init__(self):
        super(BertSelfOutput, self).__init__()
        self.dense = nn.Linear(768, 768)
        self.LayerNorm = BertLayerNorm(768, eps=1e-12)
        self.dropout = nn.Dropout(0.1)

    def forward(self, hidden_states, input_tensor):
        hidden_states = self.dense(hidden_states)
        hidden_states = self.dropout(hidden_states)
        hidden_states = self.LayerNorm(hidden_states + input_tensor)
        return hidden_states


class KLBert(nn.Module):
    def __init__(self):
        super(KLBert, self).__init__()
        self.text_bert = BertModel.from_pretrained('bert-base-uncased', return_dict=False)
        self.know_bert = BertModel.from_pretrained('bert-base-uncased', return_dict=False)
        self.W_gate = nn.Linear(768 * 2, 1)
        self.intermediate = BertIntermediate()
        self.output = BertSelfOutput()
        self.classifier = nn.Linear(768, 2)
        self.secode_output = BertOutput()
        self.incongruence1 = Incongruence()
        self.incongruence2 = Incongruence()
        self.incongruence_cross = IncongruenceCross()

    def forward(self, text_ids, text_mask, know_ids, know_mask, labels=None):
        text_info, pooled_text_info = self.text_bert(input_ids=text_ids, attention_mask=text_mask)
        know_info, pooled_know_info = self.know_bert(input_ids=know_ids, attention_mask=know_mask)

        text_info_mask_mtr = get_attn_pad_mask(text_mask, text_mask)
        know_info_mask_mtr = get_attn_pad_mask(know_mask, know_mask)
        text_know_mask_mtr = get_attn_pad_mask(text_mask, know_mask)

        know_info_incong = self.incongruence2(know_info, know_info_mask_mtr,know_ids)
        text_info_incong = self.incongruence1(text_info, text_info_mask_mtr, text_ids)
        know_text_incong = self.incongruence_cross(text_info_incong, text_know_mask_mtr, know_info_incong, text_ids, know_ids)

        know_info_mean1 = torch.mean(know_info, dim=1).unsqueeze(1).expand(text_info.size(0),text_info.size(1),text_info.size(-1))
        text_info_mean1 = torch.mean(text_info, dim=1).unsqueeze(1).expand(text_info.size(0),text_info.size(1),text_info.size(-1))
        combine_info = torch.cat([text_info, know_info_mean1+text_info_mean1], dim=-1)



        # ablation study with "add" method
        '''res_add = torch.mean((text_info_incong + know_text_incong), dim=1, keepdim=True)
        res_add = self.add_lin2(self.add_lin1(res_add))
        intermediate_res = self.intermediate(res_add)
        # 32*1*768
        res = self.secode_output(intermediate_res, res_add)'''



        # ablation study with no "gate" method
        '''res_nogate = (text_info_incong[:, 0, :] + know_text_incong[:, 0, :]).unsqueeze(1)
        intermediate_res = self.intermediate(res_nogate)
        # 32*1*768
        res = self.secode_output(intermediate_res, res_nogate)'''



        alpha = self.W_gate(combine_info)
        alpha = torch.sigmoid(alpha)

        text_info_end= torch.matmul(alpha.transpose(1, 2), text_info_incong)
        know_text = torch.matmul((1 - alpha).transpose(1, 2), know_text_incong)

        res = self.output(know_text, text_info_end)

        intermediate_res = self.intermediate(res)
        res = self.secode_output(intermediate_res, res)
        logits = self.classifier(res)

        if labels is not None:

            loss_fct = CrossEntropyLoss()
            loss = loss_fct(logits.view(-1, 2), labels.view(-1))
            return loss
        else:
            return logits




class PoswiseFeedForwardNet(nn.Module):
    def __init__(self):
        super(PoswiseFeedForwardNet, self).__init__()
        self.lin1 = nn.Linear(d_model, 3072)
        self.lin2 = nn.Linear(3072, d_model)
        self.layer_norm = BertLayerNorm(768, eps=1e-12)
        self.dropout = nn.Dropout(0.1)
    def forward(self, inputs):
        residual = inputs
        output = gelu(self.lin1(inputs))
        output = self.lin2(output)
        output = self.dropout(output)
        return self.layer_norm(output + residual)



def get_attn_pad_mask(seq_q, seq_k):
    batch_size, len_q= seq_q.size()
    batch_size, len_k= seq_k.size()
    pad_attn_mask = seq_k.data.eq(0).unsqueeze(1)
    return pad_attn_mask.expand(batch_size, len_q, len_k)


class Incongruence(nn.Module):
    def __init__(self):
        super(Incongruence, self).__init__()
        self.incong_lin1 = nn.Linear(d_model, d_model)
        self.incong_lin2 = nn.Linear(d_model, d_model)
        self.incong_lin3 = nn.Linear(d_model, d_model)
        self.layer_norm = BertLayerNorm(768, eps=1e-12)
        self.pos_ffn = PoswiseFeedForwardNet()

    def forward(self, text_info, text_info_mask_mtr):
        text_info_incon1 = self.incong_lin1(text_info)
        text_info_incon2 = self.incong_lin2(text_info)
        text_info_incon3 = self.incong_lin3(text_info)
        text_info_cdist = torch.cdist(text_info_incon1, text_info_incon2)
        text_info_cdist1 = text_info_cdist.masked_fill(text_info_mask_mtr, -1e9)
        scores_dist = torch.softmax(text_info_cdist1, dim=-1)
        incon_info = torch.matmul(scores_dist, text_info_incon3)
        text_info_add = self.layer_norm(text_info + incon_info)
        text_info_add = self.pos_ffn(text_info_add)
        return text_info_add



class IncongruenceCross(nn.Module):
    def __init__(self):
        super(IncongruenceCross, self).__init__()
        self.incong_lin1 = nn.Linear(d_model, d_model)
        self.incong_lin2 = nn.Linear(d_model, d_model)
        self.incong_lin3 = nn.Linear(d_model, d_model)
        self.layer_norm = BertLayerNorm(768, eps=1e-12)
        self.pos_ffn = PoswiseFeedForwardNet()

    def forward(self, text_info, text_know_mask_mtr, know_info):
        text_info_incon_q = self.incong_lin1(text_info)
        know_info_incon_k = self.incong_lin2(know_info)
        know_info_incon_v = self.incong_lin3(know_info)
        text_info_cdist = torch.cdist(text_info_incon_q, know_info_incon_k)
        text_know_cdist1 = text_info_cdist.masked_fill(text_know_mask_mtr, -1e9)
        scores_dist = torch.softmax(text_know_cdist1, dim=-1)
        incon_info = torch.matmul(scores_dist, know_info_incon_v)
        text_info_add = self.layer_norm(text_info + incon_info)
        text_info_add = self.pos_ffn(text_info_add)
        return text_info_add

