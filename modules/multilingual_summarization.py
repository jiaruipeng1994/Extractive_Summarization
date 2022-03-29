#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time : Mon Mar 28 10:25:17 2022
# @Author : JRP - Ruipeng Jia

import torch
from torch import nn
from torch.utils.data import DataLoader

from modules.distillation_summarization import DistilSum
from datas.mlsum import MLSUMDatamodule


class NLSSum(DistilSum):
    def __init__(self, args):
        super().__init__(args)
        self.mlsum = None
        self.alpha_classifier = nn.Linear(self.bert.config.hidden_size, 1).apply(self.init_weights)
        self.beta_classifier = nn.Linear(self.bert.config.hidden_size, 1).apply(self.init_weights)

    def forward(self, input_ids, attention_mask, token_type_ids, sents_len, *args, **kwargs):
        if not self.mlsum:
            self.mlsum = MLSUMDatamodule.from_argparse_args(self.args)
            self.mlsum.prepare_data()
            self.trainer.val_dataloaders = [self.mlsum.val_dataloader()]

        sents_hidden_state = super().forward(input_ids, attention_mask, token_type_ids, sents_len, *args, **kwargs)
        logits = self.alpha_classifier(sents_hidden_state)
        sents_weights = self.sigmoid(logits).squeeze(-1)  # (B, L_sent)
        return sents_logits, sents_weights

    def _get_loss_from_logits(self, batch, logits, sents_weights):
        alpha_labels = sents_weights * batch['oracle_labels']
        loss = self.criterion(logits, batch['oracle_labels']) + 0.5 * self.criterion(logits, batch['distill_labels']) + 0.5 * self.criterion(logits, alpha_labels)
        return loss

    def _get_logits_and_loss(self, batch):
        logits, sents_weights = self(**batch)
        loss = self._get_loss_from_logits(batch, logits, sents_weights)
        loss = loss.masked_select(batch['label_attention_mask'] == 1).mean()
        return logits, loss


if __name__ == '__main__':
    pass
