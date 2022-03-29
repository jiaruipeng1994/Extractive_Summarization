#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time : Mon Feb  7 18:20:58 2022
# @Author : JRP - Ruipeng Jia

from datasets import concatenate_datasets

def concatenate_datasetdicts(dataset_dicts, axis=1):
    dict_1, dict_2 = dataset_dicts
    for key in dict_1.keys():
        dict_1[key] = concatenate_datasets([dict_1[key], dict_2[key]], axis=axis)
    return dict_1
