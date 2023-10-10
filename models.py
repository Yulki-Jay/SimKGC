from abc import ABC
from copy import deepcopy

import torch
import torch.nn as nn
from peft import get_peft_model, prepare_model_for_int8_training,LoraConfig


from dataclasses import dataclass
from transformers import AutoModel, AutoConfig

from triplet_mask import construct_mask



def build_model(args) -> nn.Module:
    return CustomBertModel(args)


@dataclass
class ModelOutput:
    logits: torch.tensor
    labels: torch.tensor
    inv_t: torch.tensor
    hr_vector: torch.tensor
    tail_vector: torch.tensor


def get_lora_model(model_name): #这个目前还是有问题的，等一会再次进行修改
    model = AutoModel.from_pretrained(model_name)
    config = LoraConfig(
        r = 32,
        lora_alpha=16,
        lora_dropout=0.05,
        bias = "none",
        target_modules=["query","value"],
        task_type="CAUSAL_LM",
    )
    model = get_peft_model(model, config)
    model.print_trainable_parameters()
    return model
    
    


class CustomBertModel(nn.Module, ABC):
    def __init__(self, args):
        super().__init__()
        self.args = args
        self.config = AutoConfig.from_pretrained(args.pretrained_model)
        self.log_inv_t = torch.nn.Parameter(torch.tensor(1.0 / args.t).log(), requires_grad=args.finetune_t) # 1/t
        self.add_margin = args.additive_margin # γ = 0.02
        self.batch_size = args.batch_size # 256
        self.pre_batch = args.pre_batch # 0
        num_pre_batch_vectors = max(1, self.pre_batch) * self.batch_size # [256]
        random_vector = torch.randn(num_pre_batch_vectors, self.config.hidden_size) # [256 , 768]
        self.register_buffer("pre_batch_vectors",
                             nn.functional.normalize(random_vector, dim=1),
                             persistent=False) # 通过这行代码，你可以在模型中创建一个名为pre_batch_vectors的缓冲区，并且将其初始化为经过归一化处理的随机向量。这个缓冲区可以在模型的前向传播中使用，并且不会作为模型的可训练参数进行更新。
        self.offset = 0
        self.pre_batch_exs = [None for _ in range(num_pre_batch_vectors)] # [256]

        self.hr_bert = AutoModel.from_pretrained(args.pretrained_model) # 这个地方是我要进行修改的重点
        #self.hr_bert = get_lora_model(args.pretrained_model)
        self.tail_bert = deepcopy(self.hr_bert)

    def _encode(self, encoder, token_ids, mask, token_type_ids): # 这地方是进行编码的工作，他这个有点意思，正常来说应该在dataloader的时候就应该已经tokenize了
        outputs = encoder(input_ids=token_ids,
                          attention_mask=mask,
                          token_type_ids=token_type_ids,
                          return_dict=True)

        last_hidden_state = outputs.last_hidden_state # [bsz,seq_len,hidden_size] [256,768]
        cls_output = last_hidden_state[:, 0, :] # [bsz,hidden_size] [256,768]
        cls_output = _pool_output(self.args.pooling, cls_output, mask, last_hidden_state)
        return cls_output # 按照池化方式选择的cls_output 不一定就是选择最开始的cls token

    def forward(self, hr_token_ids, hr_mask, hr_token_type_ids,
                tail_token_ids, tail_mask, tail_token_type_ids,
                head_token_ids, head_mask, head_token_type_ids,
                only_ent_embedding=False, **kwargs) -> dict:
        if only_ent_embedding:
            return self.predict_ent_embedding(tail_token_ids=tail_token_ids,
                                              tail_mask=tail_mask,
                                              tail_token_type_ids=tail_token_type_ids)

        hr_vector = self._encode(self.hr_bert,
                                 token_ids=hr_token_ids,
                                 mask=hr_mask,
                                 token_type_ids=hr_token_type_ids)

        tail_vector = self._encode(self.tail_bert,
                                   token_ids=tail_token_ids,
                                   mask=tail_mask,
                                   token_type_ids=tail_token_type_ids)

        head_vector = self._encode(self.tail_bert, # 加入自身的embedding 到这个如果bsz=256 显存占用为19391M
                                   token_ids=head_token_ids,
                                   mask=head_mask,
                                   token_type_ids=head_token_type_ids)

        # DataParallel only support tensor/dict 这句话是什么意思
        return {'hr_vector': hr_vector,
                'tail_vector': tail_vector,
                'head_vector': head_vector}

    def compute_logits(self, output_dict: dict, batch_dict: dict) -> dict: # 这个是自己定义的计算logit的方法
        hr_vector, tail_vector = output_dict['hr_vector'], output_dict['tail_vector']
        batch_size = hr_vector.size(0) # 256
        labels = torch.arange(batch_size).to(hr_vector.device) # 0~bsz-1

        logits = hr_vector.mm(tail_vector.t()) # [bsz,bsz] 只有主对角线的元素为真
        if self.training:
            logits -= torch.zeros(logits.size()).fill_diagonal_(self.add_margin).to(logits.device) # 减去margin
        logits *= self.log_inv_t.exp()

        triplet_mask = batch_dict.get('triplet_mask', None) # [bsz,bsz]
        if triplet_mask is not None:
            logits.masked_fill_(~triplet_mask, -1e4) # 使用-1e4将logits张量中在triplet_mask为True的位置上的元素进行填充，将其值设为-1e4。

        if self.pre_batch > 0 and self.training:
            pre_batch_logits = self._compute_pre_batch_logits(hr_vector, tail_vector, batch_dict)
            logits = torch.cat([logits, pre_batch_logits], dim=-1)

        if self.args.use_self_negative and self.training:
            head_vector = output_dict['head_vector']
            self_neg_logits = torch.sum(hr_vector * head_vector, dim=1) * self.log_inv_t.exp()
            self_negative_mask = batch_dict['self_negative_mask']
            self_neg_logits.masked_fill_(~self_negative_mask, -1e4)
            logits = torch.cat([logits, self_neg_logits.unsqueeze(1)], dim=-1)

        return {'logits': logits,
                'labels': labels,
                'inv_t': self.log_inv_t.detach().exp(),
                'hr_vector': hr_vector.detach(),
                'tail_vector': tail_vector.detach()}

    def _compute_pre_batch_logits(self, hr_vector: torch.tensor,
                                  tail_vector: torch.tensor,
                                  batch_dict: dict) -> torch.tensor:
        assert tail_vector.size(0) == self.batch_size
        batch_exs = batch_dict['batch_data']
        # batch_size x num_neg
        pre_batch_logits = hr_vector.mm(self.pre_batch_vectors.clone().t())
        pre_batch_logits *= self.log_inv_t.exp() * self.args.pre_batch_weight
        if self.pre_batch_exs[-1] is not None:
            pre_triplet_mask = construct_mask(batch_exs, self.pre_batch_exs).to(hr_vector.device)
            pre_batch_logits.masked_fill_(~pre_triplet_mask, -1e4)

        self.pre_batch_vectors[self.offset:(self.offset + self.batch_size)] = tail_vector.data.clone()
        self.pre_batch_exs[self.offset:(self.offset + self.batch_size)] = batch_exs
        self.offset = (self.offset + self.batch_size) % len(self.pre_batch_exs)

        return pre_batch_logits

    @torch.no_grad()
    def predict_ent_embedding(self, tail_token_ids, tail_mask, tail_token_type_ids, **kwargs) -> dict:
        ent_vectors = self._encode(self.tail_bert,
                                   token_ids=tail_token_ids,
                                   mask=tail_mask,
                                   token_type_ids=tail_token_type_ids)
        return {'ent_vectors': ent_vectors.detach()}


def _pool_output(pooling: str,
                 cls_output: torch.tensor,# [bsz,hidden_size]
                 mask: torch.tensor,# [bsz,seq_len]
                 last_hidden_state: torch.tensor) -> torch.tensor: # [bsz,seq_len,hidden_size]
    if pooling == 'cls':
        output_vector = cls_output
    elif pooling == 'max': # mask的地方设置一个非常小的负数，然后再取最大值，这样就可以避免mask的地方取到最大值
        input_mask_expanded = mask.unsqueeze(-1).expand(last_hidden_state.size()).long()
        last_hidden_state[input_mask_expanded == 0] = -1e4
        output_vector = torch.max(last_hidden_state, 1)[0]
    elif pooling == 'mean': # 根据烟吗计算有效位置的embedding和，然后除以有效位置的个数
        input_mask_expanded = mask.unsqueeze(-1).expand(last_hidden_state.size()).float()
        sum_embeddings = torch.sum(last_hidden_state * input_mask_expanded, 1)
        sum_mask = torch.clamp(input_mask_expanded.sum(1), min=1e-4)
        output_vector = sum_embeddings / sum_mask
    else:
        assert False, 'Unknown pooling mode: {}'.format(pooling)

    output_vector = nn.functional.normalize(output_vector, dim=1) # 进行归一化
    return output_vector
