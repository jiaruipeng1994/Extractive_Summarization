#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time : Wed Jan 26 00:46:14 2022
# @Author : JRP - Ruipeng Jia

from itertools import islice
from pytorch_lightning import LightningModule
from transformers import AutoModel, AdamW, get_linear_schedule_with_warmup
from transformers.models.bert.modeling_bert import BertAttention
import torch
from torch import nn
import numpy as np

from jtools import hf_dict
from jtools.sentence_label import trigram_blocking


class DistilSum(LightningModule):
    def __init__(self, args):
        super().__init__()
        self.save_hyperparameters()
        self.args = args
        self.bert = AutoModel.from_pretrained(hf_dict[args.pre_train])
        self.attention = BertAttention(self.bert.config).apply(self.init_weights)
        self.attention_word = torch.zeros(1, 1, self.bert.config.hidden_size)  # (1, 768) a high level representation of a fixed query "what is the informative word" over the words
        self.sent_linear = nn.Linear(self.bert.config.hidden_size, self.bert.config.hidden_size).apply(self.init_weights)
        self.sent_classifier = nn.Linear(self.bert.config.hidden_size, 1).apply(self.init_weights)

        self.layer_norm = nn.LayerNorm(self.bert.config.hidden_size, eps=1e-6)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(p=0.1)
        self.sigmoid = nn.Sigmoid()
        self.criterion = nn.BCEWithLogitsLoss(reduction='none')

        nn.init.xavier_uniform_(self.attention_word)

        if self.args.max_pos > self.bert.config.max_position_embeddings:
            pos_l, pos_d = self.bert.embeddings.position_embeddings.weight.data.size()
            my_pos_embeddings = nn.Embedding(args.max_pos, pos_d)
            my_pos_embeddings.weight.data[:pos_l] = self.bert.embeddings.position_embeddings.weight.data
            my_pos_embeddings.weight.data[pos_l:] = self.bert.embeddings.position_embeddings.weight.data[-1][None, :].repeat(self.args.max_pos - pos_l, 1)
            nn.init.xavier_uniform_(my_pos_embeddings.weight.data[pos_l:])
            self.bert.embeddings.position_embeddings = my_pos_embeddings

    def init_weights(self, m):
        if isinstance(m, nn.Linear):
            nn.init.xavier_uniform_(m.weight)
            m.bias.data.fill_(0.01)

    @staticmethod
    def add_model_specific_args(parent_parser):
        parser = parent_parser.add_argument_group("Module")
        parser.add_argument("--lr", type=float, default=5e-5)
        parser.add_argument("--warmup_steps", type=int, default=3000)
        return parent_parser

    def forward(self, input_ids, attention_mask, token_type_ids, sents_len, *args, **kwargs):
        if 'roberta' in self.args.pre_train:
            token_type_ids = torch.zeros(input_ids.size(), dtype=torch.long, device=input_ids.device)
        position_ids = torch.arange(input_ids.size(1), device=input_ids.device).expand((1, -1))
        words_hidden_state = self.bert(input_ids, attention_mask, token_type_ids, position_ids).last_hidden_state

        ## words -> sent
        B, L, D = words_hidden_state.size()
        sents_index, max_len = [], max([len(item) for item in sents_len])
        for _sents_len in sents_len:
            indexs = iter(range(L))
            ranges = [list(islice(indexs, idx)) for idx in _sents_len]
            sents_index.append([(item[0], item[-1] + 1) for item in ranges] + [(-2, -1)] * (max_len - len(_sents_len)))

        sents_words = sum([[words_hidden_state[idx][l:r] for l, r in sents_index[idx]] for idx in range(B)], [])
        sents_words = nn.utils.rnn.pad_sequence(sents_words, batch_first=True)
        sents_words_mask = torch.tensor(sum([[[1] * item + [0] * (sents_words.size(1) - item) for item in sent_len + [0] * (max_len - len(sent_len))] for sent_len in sents_len], []))
        sents_words_mask = self.bert.get_extended_attention_mask(sents_words_mask, sents_words.size()[:2], sents_words.device).to(sents_words.device)
        attention_words = self.attention_word.repeat(sents_words.size(0), 1, 1).to(sents_words.device)
        output = self.attention(hidden_states=attention_words, encoder_hidden_states=sents_words, encoder_attention_mask=sents_words_mask)
        sents_init = output[0].reshape(B, -1, D)

        sents_hidden_state = self.sent_linear(sents_init).squeeze(-1)
        return sents_hidden_state

    def _get_loss_from_logits(self, batch, logits):
        loss = self.criterion(logits, batch['oracle_labels']) + 0.5 * self.criterion(logits, batch['distill_labels'])
        return loss

    def _get_logits_and_loss(self, batch):
        sents_hidden_state = self(**batch)
        logits = self.sent_classifier(sents_hidden_state).squeeze(-1)
        loss = self._get_loss_from_logits(batch, logits)
        loss = loss.masked_select(batch['label_attention_mask'] == 1).mean()
        return logits, loss

    def _select_sents_from_logits(self, logits, label_attention_mask, doc_sents_words):
        logits = logits.cpu()  # (B, L)
        sent_scores = (logits - logits.min(-1, keepdim=True)[0]) / (logits.max(-1, keepdim=True)[0] - logits.min(-1, keepdim=True)[0] + 0.0001)
        sent_scores = sent_scores.data.numpy() + label_attention_mask.cpu().data.numpy()
        selected_ids = np.argsort(-sent_scores, axis=-1)
        pred = []
        for i in range(len((selected_ids))):
            _pred = []
            for j in selected_ids[i][:len(doc_sents_words[i])]:
                candidate = doc_sents_words[i][j]
                if not trigram_blocking(candidate, _pred):
                    _pred.append(candidate)
                if len(_pred) == 3:
                    break
            pred.append(' '.join(sum(_pred, [])).lower())
        return pred

    def training_step(self, batch, batch_idx):
        logits, loss = self._get_logits_and_loss(batch)
        self.log("train_loss", loss, batch_size=self.args.train_batch_size, on_step=True, prog_bar=True, logger=True)

        if self.trainer._lightning_optimizers:
            lr = self.trainer._lightning_optimizers[0].param_groups[0]['lr']
            param = self.sent_classifier.weight[0][0]
            self.log("lr", lr, on_step=True, prog_bar=True, logger=True)
            self.log("param", param, on_step=True, prog_bar=True, logger=True)
            self.log("logits", logits[0][0], on_step=True, prog_bar=True, logger=True)
        return loss

    def validation_step(self, batch, batch_idx):
        logits, loss = self._get_logits_and_loss(batch)
        self.log("val_loss", loss, on_epoch=True, prog_bar=True, logger=True)
        pred = self._select_sents_from_logits(logits, batch['label_attention_mask'], batch['doc_sents_words'])
        return pred

    def test_step(self, batch, batch_idx):
        logits, loss = self._get_logits_and_loss(batch)
        self.log("test_loss", loss)
        pred = self._select_sents_from_logits(logits, batch['label_attention_mask'], batch['doc_sents_words'])
        return pred

    def predict_step(self, batch, batch_idx):
        logits, loss = self._get_logits_and_loss(batch)
        pred = self._select_sents_from_logits(logits, batch['label_attention_mask'], batch['doc_sents_words'])
        return pred

    def configure_optimizers(self):
        num_training_steps = len(self.trainer.datamodule.train_dataloader()) * self.args.max_epochs // self.args.accumulate_grad_batches
        optimizer = AdamW([
            {'params': [p for n, p in self.named_parameters() if 'bert' in n]},
            {'params': [p for n, p in self.named_parameters() if 'bert' not in n], 'lr': self.args.lr * 10}], lr=self.args.lr)
        scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=self.args.warmup_steps, num_training_steps=num_training_steps)
        scheduler = {'scheduler': scheduler, 'interval': 'step', 'frequency': 1, 'reduce_on_plateau': False, 'monitor': 'val_loss'}
        return [optimizer], [scheduler]

if __name__ == '__main__':
    pass
