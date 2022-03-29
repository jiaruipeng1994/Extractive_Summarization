#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time : Fri Jan 28 17:06:46 2022
# @Author : JRP - Ruipeng Jia

import numpy as np
from rouge_score.rouge_scorer import RougeScorer

from rouge_score.rouge_scorer import _score_ngrams
from rouge_score.rouge_scorer import _create_ngrams

##################################################################
## greedy_selection
def greedy_selection(doc_sents, abstract, summary_size=5, temperature=1, iter_num=1, stemmer=False, lang='en'):
    assert iter_num == temperature or iter_num == 1
    rouge = RougeScorer(['rouge1', 'rouge2', 'rougeL'], use_stemmer=stemmer)

    scores = np.array([0] * len(doc_sents))
    selected = []
    for _ in range(temperature):
        cur_selected = []
        cur_max_rouge = 0.0
        for s in range(summary_size):
            cur_id = -1
            for i in range(len(doc_sents)):
                if (i in selected):
                    continue

                c = cur_selected + [i]
                prediction = ' '.join([doc_sents[idx] for idx in c])
                results = rouge.score(abstract, prediction)
                rouge_score = results['rouge1'].fmeasure + results['rouge2'].fmeasure + results['rougeL'].fmeasure
                if rouge_score > cur_max_rouge:
                    cur_max_rouge = rouge_score
                    cur_id = i
            if (cur_id == -1):
                break
            cur_selected.append(cur_id)
            selected.append(cur_id)

        for item in selected:
            scores[item] += 1

    iter_scores = []
    for idx, _iter in enumerate(range(iter_num)):
        _scores = np.around((scores - idx) / (temperature - idx), decimals=2)
        _scores[_scores < 0] = 0
        iter_scores.append(list(_scores))

    return sorted(selected[:list(scores).count(temperature)]), iter_scores

##################################################################
## trigram-blocking
def trigram_blocking(candidate, references):
    tri_grams = _create_ngrams(candidate, 3)
    for sent_words in references:
        _tri_grams = _create_ngrams(sent_words, 3)
        for key in _tri_grams.keys():
            if key in tri_grams:
                return True
    return False
