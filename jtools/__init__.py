#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time : Fri Jan 28 15:42:16 2022
# @Author : JRP - Ruipeng Jia

##################################################################
## Common
seed = 1024

##################################################################
## Home
from pathlib import Path
home = str(Path.home())

##################################################################
## Print
from pprint import pprint
def jprint(object):
    if isinstance(object, list) and isinstance(object[0], str):
        for txt in object:
            print(txt, '\n')

##################################################################
## Huggingface Path
hf_cache_datasets = home + '/.cache/huggingface/datasets/'

bert_base_cased = home + '/datasets/Language_Model/transformers/bert-base-cased/'
bert_base_uncased = home + '/datasets/Language_Model/transformers/bert-base-uncased/'
albert_base = home + '/datasets/Language_Model/transformers/albert-base/'
albert_large = home + '/datasets/Language_Model/transformers/albert-large/'
albert_xlarge = home + '/datasets/Language_Model/transformers/albert-xlarge/'
albert_xxlarge = home + '/datasets/Language_Model/transformers/albert-xxlarge/'
roberta_base = home + '/datasets/Language_Model/transformers/roberta-base/'
roberta_large = home + '/datasets/Language_Model/transformers/roberta-large/'
xlnet_base_cased = home + '/datasets/Language_Model/transformers/xlnet-base-cased/'
xlmr_base = home + '/datasets/Language_Model/transformers/xlmr-base/'

hf_dict = {
    "bert_base_cased": bert_base_cased,
    "BertCase_base_cased": bert_base_cased,
    "bert_base_uncased": bert_base_uncased,
    "albert_base": albert_base,
    "albert_large": albert_large,
    "albert_xlarge": albert_xlarge,
    "albert_xxlarge": albert_xxlarge,
    "roberta_base": roberta_base,
    "roberta_large": roberta_large,
    "xlnet_base_cased": xlnet_base_cased,
    "xlmr_base": xlmr_base
    }

##################################################################
## pad
def pad(nest_list, value=0, return_tensors=None):
    max_len = max([len(item) for item in nest_list])
    input_ids = [item + [value] * (max_len - len(item)) for item in nest_list]
    attention_mask = [[1] * len(item) + [0] * (max_len - len(item)) for item in nest_list]
    if return_tensors == 'pt':
        import torch
        input_ids = torch.tensor(input_ids)
        attention_mask = torch.tensor(attention_mask)
    return {'input_ids': input_ids, 'attention_mask': attention_mask}
