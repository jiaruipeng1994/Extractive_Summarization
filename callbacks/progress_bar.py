#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time : Fri Feb 11 23:19:53 2022
# @Author : JRP - Ruipeng Jia

from pytorch_lightning.callbacks import TQDMProgressBar


class LitProgressBar(TQDMProgressBar):

   def init_train_tqdm(self):
       bar = super().init_train_tqdm()
       bar.set_description('Training ...')
       return bar

   def get_metrics(self, trainer, model):
       items = super().get_metrics(trainer, model)
       items.pop("v_num", None)
       return items
