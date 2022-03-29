#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time : Fri Mar 18 10:25:17 2022
# @Author : JRP - Ruipeng Jia

import torch
from torch import nn

from modules.distillation_summarization import DistilSum


class DifferSum(DistilSum):
    def __init__(self, args):
        super().__init__(args)
        self.num_iteration = 3
        self.context_sents_linears = nn.ModuleList([nn.Linear(self.bert.config.hidden_size, self.bert.config.hidden_size).apply(self.init_weights) for _ in range(self.num_iteration)])
        self.differential_amplifiers = nn.ModuleList([nn.Linear(self.bert.config.hidden_size, self.bert.config.hidden_size).apply(self.init_weights) for _ in range(self.num_iteration)])
        self.common_amplifier = nn.Linear(self.bert.config.hidden_size, self.bert.config.hidden_size).apply(self.init_weights)

    def forward(self, input_ids, attention_mask, token_type_ids, sents_len, *args, **kwargs):
        sents_hidden_state = super().forward(input_ids, attention_mask, token_type_ids, sents_len, *args, **kwargs)
        B, L, D = sents_hidden_state.size()

        for i in range(self.num_iteration):
            context_sents = torch.stack([torch.stack([torch.cat((doc[0:idx], doc[idx+1:sum(kwargs['label_attention_mask'][_idx])]), dim=0).mean(dim=0) for idx in range(L)]) for _idx, doc in enumerate(sents_hidden_state)])
            context_sents = self.context_sents_linears[i](context_sents)
            sents_hidden_state = self.differential_amplifiers[i](sents_hidden_state - context_sents) + sents_hidden_state

        return sents_hidden_state

    def _get_loss_from_logits(self, batch, logits):
        loss = self.criterion(logits, batch['oracle_labels']) + 0.5 * self.criterion(logits, batch['distill_labels'])
        B, L = loss.size()
        for _idx in range(B):
            for idx in range(L):
                if batch['oracle_labels'][_idx][idx]:
                    loss[_idx][idx] *= (sum(batch['label_attention_mask'][_idx]) - sum(batch['oracle_labels'][_idx]) + 3) / (sum(batch['oracle_labels'][_idx]) + 3)
        return loss


if __name__ == '__main__':
    pass
