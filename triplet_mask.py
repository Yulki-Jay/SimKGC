import torch

from typing import List

from config import args
from dict_hub import get_train_triplet_dict, get_entity_dict, EntityDict, TripletDict

entity_dict: EntityDict = get_entity_dict() # 获得全部的实体 对于wn18rr来说，一共有40943个实体
train_triplet_dict: TripletDict = get_train_triplet_dict() if not args.is_test else None


# 这个代码其实还是没太看明白，最后再仔细看一下把
def construct_mask(row_exs: List, col_exs: List = None) -> torch.tensor: # row_exs:是一个batch的data，batch_exs是一个列表
    positive_on_diagonal = col_exs is None # 这是一个布尔值，如果col_exs是None，那么positive_on_diagonal就是True
    num_row = len(row_exs) # 这目前岂不是他俩一样
    col_exs = row_exs if col_exs is None else col_exs
    num_col = len(col_exs)

    # exact match
    row_entity_ids = torch.LongTensor([entity_dict.entity_to_idx(ex.tail_id) for ex in row_exs]) # 将每个实体的id映射为idx
    col_entity_ids = row_entity_ids if positive_on_diagonal else \
        torch.LongTensor([entity_dict.entity_to_idx(ex.tail_id) for ex in col_exs]) 
    # num_row x num_col
    triplet_mask = (row_entity_ids.unsqueeze(1) != col_entity_ids.unsqueeze(0)) # [num_row,num_col]
    if positive_on_diagonal:
        triplet_mask.fill_diagonal_(True) # 对角线元素设置成True

    # mask out other possible neighbors
    for i in range(num_row):
        head_id, relation = row_exs[i].head_id, row_exs[i].relation
        neighbor_ids = train_triplet_dict.get_neighbors(head_id, relation)
        # exact match is enough, no further check needed
        if len(neighbor_ids) <= 1:
            continue

        for j in range(num_col):
            if i == j and positive_on_diagonal:
                continue
            tail_id = col_exs[j].tail_id
            if tail_id in neighbor_ids:
                triplet_mask[i][j] = False

    return triplet_mask


def construct_self_negative_mask(exs: List) -> torch.tensor:  #在exs列表中构建一个掩码张量，head_id存在于neighbor_ids中的花，就是0，其余元素被设置为1。
    mask = torch.ones(len(exs))
    for idx, ex in enumerate(exs):
        head_id, relation = ex.head_id, ex.relation
        neighbor_ids = train_triplet_dict.get_neighbors(head_id, relation)
        if head_id in neighbor_ids:
            mask[idx] = 0
    return mask.bool()
