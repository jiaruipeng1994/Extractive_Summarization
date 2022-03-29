#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time : Tue Jan 25 17:09:43 2022
# @Author : JRP - Ruipeng Jia

import os, sys
import torch
from torch.utils.data import DataLoader
from datasets import load_dataset, load_from_disk
from pytorch_lightning import LightningDataModule
from transformers import AutoTokenizer
from nltk import sent_tokenize, word_tokenize
from rouge import Rouge

from jtools import home, pad
from jtools import hf_dict
from jtools.sentence_label import greedy_selection
from jtools.text import sents_words_tokenize
from jtools.dataset import concatenate_datasetdicts


class CNNDMDatamodule(LightningDataModule):
    def __init__(self, pre_train: str='roberta_base', train_batch_size: int=8, eval_batch_size: int=0, temperature: int=5, iter_num: int=1, num_proc: int=1, max_pos: int=512, filter_score: float=0.5):
        super().__init__()
        self.pre_train = hf_dict[pre_train]
        self.train_batch_size = train_batch_size
        self.eval_batch_size = train_batch_size * 4 if not eval_batch_size else eval_batch_size
        self.temperature = temperature
        self.iter_num = iter_num
        self.num_proc = num_proc
        self.max_pos = max_pos
        self.filter_score = filter_score
        self.sent_dataset_dir = home + '/.cache/huggingface/datasets/cnn_dailymail/distill_sum/sent_dataset/1.0/'
        self.subword_dataset_dir = home + '/.cache/huggingface/datasets/cnn_dailymail/distill_sum/' + pre_train.split('_')[0] + '_' + str(max_pos) + '_subword_dataset/1.0/'
        self.tokenizer = AutoTokenizer.from_pretrained(self.pre_train, use_fast=True)
        self.rouge = Rouge()

    def _get_info_from_example(self, example):
        article, highlights = example['article'], example['highlights']
        article = article.replace('(CNN)', ' ')
        return article, highlights

    def _distill_label(self, example):
        article, highlights = self._get_info_from_example(example)
        doc_sents, doc_sents_words = sents_words_tokenize(article)
        _, abs_sents_words = sents_words_tokenize(highlights)
        ref = ' '.join(sum(abs_sents_words, [])).lower()
        oracle_ids, distill_labels = greedy_selection(doc_sents, ref, temperature=self.temperature, iter_num=self.iter_num)
        rouge = self.rouge.get_scores(' '.join(sum([doc_sents_words[idx] for idx in oracle_ids], [])).lower(), ref)[0]
        return {'doc_sents': doc_sents, 'doc_sents_words': doc_sents_words, 'ref': ref, 'oracle_ids': oracle_ids, 'distill_labels': distill_labels, 'rouge': rouge}

    def _subword_tokenizer(self, example):
        doc_sents = example['doc_sents']
        input_ids , sents_len = [], []
        for sent in doc_sents:
            res = self.tokenizer(sent)
            sent_len = len(res['input_ids'])
            if sum(sents_len) + sent_len > self.max_pos:
                break
            input_ids.append(res['input_ids'])
            sents_len.append(sent_len)
        input_ids = sum(input_ids, [])
        return {'input_ids': input_ids, 'sents_len': sents_len}

    def _batch_process(self, examples):
        data = {}
        for key in examples[0].keys():
            data[key] = [example[key] for example in examples]
        input_ids, distill_labels = data['input_ids'], data['distill_labels']
        distill_labels = [item[0] for item in distill_labels]

        res = pad(input_ids, return_tensors='pt')
        input_ids, attention_mask = res['input_ids'], res['attention_mask']
        token_type_ids = [sum([[1] * item if idx % 2 else [0] * item for idx, item in enumerate(sents_len)], []) for sents_len in data['sents_len']]
        token_type_ids = pad(token_type_ids, return_tensors='pt')['input_ids']

        res = pad(distill_labels, return_tensors='pt')
        distill_labels, label_attention_mask = res['input_ids'], res['attention_mask']
        max_len = max([len(item) for item in data['sents_len']])
        distill_labels, label_attention_mask = distill_labels[:, :max_len], label_attention_mask[:, :max_len]
        oracle_labels = (distill_labels == 1.0).float()

        return {'sents_len': data['sents_len'], 'ref': data['ref'], 'doc_sents_words': data['doc_sents_words'],
                'input_ids': input_ids, 'attention_mask': attention_mask, 'token_type_ids': token_type_ids,
                'distill_labels': distill_labels, 'oracle_labels': oracle_labels, 'label_attention_mask': label_attention_mask,
                'all_distill_labels': data['distill_labels']}

    def prepare_data(self):  # called only on 1 GPU, automatically
        if os.path.exists(self.sent_dataset_dir):
            sent_dataset = load_from_disk(self.sent_dataset_dir)
        else:
            sys.setrecursionlimit(8735 * 2080 + 10)
            original_dataset = load_dataset("cnn_dailymail", '3.0.0', ignore_verifications=True)
            assert len(original_dataset['test']) == 11490
            sent_dataset = original_dataset.map(self._distill_label, num_proc=self.num_proc, remove_columns=['article', 'highlights'])
            sent_dataset.save_to_disk(self.sent_dataset_dir)

        if os.path.exists(self.subword_dataset_dir):
            subword_dataset = load_from_disk(self.subword_dataset_dir)
        else:
            subword_dataset = sent_dataset.map(self._subword_tokenizer, num_proc=self.num_proc, remove_columns=['doc_sents', 'doc_sents_words', 'ref', 'oracle_ids', 'distill_labels', 'rouge'])
            subword_dataset.save_to_disk(self.subword_dataset_dir)
        assert sent_dataset['train'][0]['id'] == subword_dataset['train'][0]['id'] and sent_dataset['train'][-1]['id'] == subword_dataset['train'][-1]['id']

        dataset = concatenate_datasetdicts([sent_dataset.remove_columns("id"), subword_dataset], axis=1)
        self.train, self.val, self.test = dataset['train'], dataset['validation'], dataset['test']

    def setup(self, stage):
        pass

    def train_dataloader(self):
        return DataLoader(self.train, batch_size=self.train_batch_size, collate_fn=self._batch_process, shuffle=True)

    def val_dataloader(self):
        # return DataLoader(self.val, batch_size=self.eval_batch_size, collate_fn=self._batch_process)
        return self.test_dataloader()

    def test_dataloader(self):
        return DataLoader(self.test, batch_size=self.eval_batch_size, collate_fn=self._batch_process)

    def predict_dataloader(self):
        return self.test_dataloader()


if __name__ == "__main__":
    cnndm_dataset = CNNDMDatamodule(train_batch_size=2, num_proc=2)
    cnndm_dataset.prepare_data()
    print(len(cnndm_dataset.predict_dataloader()))
    batch = next(iter(cnndm_dataset.train_dataloader()))
    print(batch.keys())
    print(batch['input_ids'].shape, batch['input_ids'].shape, batch['distill_labels'].shape)
    print(batch['distill_labels'])
