#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time : Wed Mar 23 01:16:14 2022
# @Author : JRP - Ruipeng Jia

import torch
from torch import nn
from torch_geometric.nn import GATConv

from modules.distillation_summarization import DistilSum
from jtools.sentence_label import trigram_blocking


class HAHSum(DistilSum):
    def __init__(self, args):
        super().__init__(args)
        self.num_iteration = 3
        self.redundancy_linears = nn.ModuleList([nn.Linear(self.bert.config.hidden_size, self.bert.config.hidden_size).apply(self.init_weights) for _ in range(self.num_iteration)])
        self.gats = nn.ModuleList([GATConv(self.bert.config.hidden_size, self.bert.config.hidden_size).apply(self.init_weights) for _ in range(self.num_iteration)])

    def _init_graph(self, batch):
        sim_graph_edge_index = []

        ## Sim Graph
        leng = len(batch['doc_sents_words'][0])
        for i in range(leng - 1):
            for j in range(i + 1, leng):
                if trigram_blocking(batch['doc_sents_words'][0][i], [batch['doc_sents_words'][0][j]]):
                    sim_graph_edge_index.append([i, j])
                    sim_graph_edge_index.append([j, i])
        for i in range(leng):
            sim_graph_edge_index.append([i, i])

        sim_graph = torch.tensor(sim_graph_edge_index, dtype=torch.long, device=batch['input_ids'].device).transpose(0, 1)
        res = {'sim_graph': sim_graph}
        batch.update(res)

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
            for _idx in range(L):
                update_mask = (label_attention_mask.float() * torch.ones(L, device=input_ids.device).scatter_(0, torch.tensor(_idx), 0.0)).unsqueeze(-1)
                _redundancy = self.redundancy_linears[i](torch.tanh((sents_hidden_state * update_mask * sent_scores.unsqueeze(-1)).mean(1))).unsqueeze(-1)
                _redundancy = torch.bmm(sents_hidden_state[:, _idx, :].unsqueeze(1), _redundancy).squeeze()
                redundancy.append(_redundancy)
            redundancy = torch.stack(redundancy, -1).unsqueeze(-1)

            sents_hidden_state = sents_hidden_state - redundancy
            sents_hidden_state = self.gats[i](sents_hidden_state[0], kwargs['sim_graph']).unsqueeze(0)

        logits = self.sent_classifier(sents_hidden_state).squeeze(-1)  # (B, L_sent)
        iter_sent_logits.append(logits)

        return iter_sent_logits

    def _get_logits_and_loss(self, batch):
        assert self.args.train_batch_size == 1 and self.args.eval_batch_size == 1
        self._init_graph(batch)
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
