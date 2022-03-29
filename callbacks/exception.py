#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time : Mon Mar 21 21:15:39 2022
# @Author : JRP - Ruipeng Jia

import torch
from pytorch_lightning.callbacks import Callback


class MyException(Callback):

    def __init__(self):
        super().__init__()
        self.invalid_loss = False

    def on_before_backward(self, trainer, pl_module, loss):
        if torch.isnan(loss):
            print("Detected inf or nan values in gradients. Not updating model parameters")
            self.invalid_loss = True

    def on_after_backward(self, trainer, pl_module):
        if self.invalid_loss:
            for optimizer in trainer.accelerator.optimizers:
                optimizer.zero_grad()
            self.invalid_loss = False
