#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time : Sat Mar 19 17:10:06 2022
# @Author : JRP - Ruipeng Jia

import torch
from torch import nn

from modules.distillation_summarization import DistilSum


class ThresSum(DistilSum):
    def __init__(self, args):
        super().__init__(args)
        self.num_iteration = 3
        self.redundancy_linears = nn.ModuleList([nn.Linear(self.bert.config.hidden_size, self.bert.config.hidden_size).apply(self.init_weights) for _ in range(self.num_iteration)])
        self.sents_linears = nn.ModuleList([nn.Linear(self.bert.config.hidden_size, self.bert.config.hidden_size).apply(self.init_weights) for _ in range(self.num_iteration)])

    def forward(self, input_ids, attention_mask, token_type_ids, sents_len, *args, **kwargs):
        sents_hidden_state = super().forward(input_ids, attention_mask, token_type_ids, sents_len, *args, **kwargs)
        label_attention_mask = kwargs['label_attention_mask']
        B, L, D = sents_hidden_state.size()

        iter_sent_logits = []
        for i in range(self.num_iteration):
            logits = self.sent_classifier(sents_hidden_state).squeeze(-1)
            sent_scores = self.sigmoid(logits)
            iter_sent_logits.append(logits)

            redundancy = []
            for _idx in range(L):  # L_sent
                update_mask = (label_attention_mask.float() * torch.ones(L, device=input_ids.device).scatter_(0, torch.tensor(_idx), 0.0)).unsqueeze(-1)
                _redundancy = self.redundancy_linears[i](torch.tanh((sents_hidden_state * update_mask * sent_scores.unsqueeze(-1)).mean(1))).unsqueeze(-1)
                _redundancy = torch.bmm(sents_hidden_state[:, _idx, :].unsqueeze(1), _redundancy).squeeze()
                redundancy.append(_redundancy)
            redundancy = torch.stack(redundancy, -1)
            redundancy = nn.functional.normalize(redundancy)
            redundancy = redundancy.unsqueeze(-1)

            sents_hidden_state = sents_hidden_state - redundancy
            sents_hidden_state = self.sents_linears[i](sents_hidden_state)

        logits = self.sent_classifier(sents_hidden_state).squeeze(-1)
        iter_sent_logits.append(logits)

        return iter_sent_logits

    def _get_logits_and_loss(self, batch):
        iter_sent_logits = self(**batch)
        loss = None
        for logits in iter_sent_logits:
            _loss = self._get_loss_from_logits(batch, logits)
            _loss = _loss.masked_select(batch['label_attention_mask'] == 1).mean()
            if not loss:
                loss = _loss
            else:
                loss += _loss
        return iter_sent_logits[-1], loss


if __name__ == '__main__':
    pass
