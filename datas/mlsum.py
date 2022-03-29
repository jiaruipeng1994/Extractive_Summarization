#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time : Mon Mar 28 17:09:43 2022
# @Author : JRP - Ruipeng Jia

import os
from torch.utils.data import DataLoader
from datasets import load_dataset, load_from_disk

from datas.cnndm import CNNDMDatamodule
from jtools import home
from datasets import concatenate_datasets


class MLSUMDatamodule(CNNDMDatamodule):
    def __init__(self, pre_train: str='xlmr_base', eval_batch_size: int=32, num_proc: int=1, max_pos: int=512, lang: str='de'):
        super().__init__(pre_train=pre_train, eval_batch_size=eval_batch_size, num_proc=num_proc, max_pos=max_pos)
        self.lang = lang
        self.sent_dataset_dir = home + '/.cache/huggingface/datasets/mlsum/distill_sum/' + self.lang + '/sent_dataset/1.0/'
        self.subword_dataset_dir = home + '/.cache/huggingface/datasets/mlsum/distill_sum/' + self.lang + '/' + pre_train.split('_')[0] + '_' + str(max_pos) + '_subword_dataset/1.0/'

    def _get_info_from_example(self, example):
        article, highlights = example['text'], example['summary']
        return article, highlights

    def prepare_data(self):  # called only on 1 GPU, automatically
        if os.path.exists(self.sent_dataset_dir):
            sent_dataset = load_from_disk(self.sent_dataset_dir)
        else:
            original_dataset = load_dataset('mlsum', 'de', split='test', ignore_verifications=True)
            assert len(original_dataset) == 10701
            sent_dataset = original_dataset.map(self._distill_label, num_proc=self.num_proc, remove_columns=['text', 'summary', 'topic', 'url', 'title', 'date'])
            sent_dataset.save_to_disk(self.sent_dataset_dir)

        if os.path.exists(self.subword_dataset_dir):
            subword_dataset = load_from_disk(self.subword_dataset_dir)
        else:
            subword_dataset = sent_dataset.map(self._subword_tokenizer, num_proc=self.num_proc, remove_columns=['doc_sents', 'doc_sents_words', 'ref', 'oracle_ids', 'distill_labels'])
            subword_dataset.save_to_disk(self.subword_dataset_dir)

        dataset = concatenate_datasets([sent_dataset.remove_columns('rouge'), subword_dataset], axis=1)
        self.train, self.val, self.test = dataset, dataset, dataset


if __name__ == "__main__":
    mlsum = MLSUMDatamodule(eval_batch_size=2, num_proc=1)
    mlsum.prepare_data()
    print(len(mlsum.predict_dataloader()))
    batch = next(iter(mlsum.train_dataloader()))
    print(batch.keys())
    print(batch['input_ids'].shape, batch['input_ids'].shape, batch['distill_labels'].shape)
    print(batch['distill_labels'])
