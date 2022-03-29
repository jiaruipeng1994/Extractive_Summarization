#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time : Wed Jan 26 01:05:02 2022
# @Author : JRP - Ruipeng Jia

from pytorch_lightning.callbacks import Callback

class EchoCallback(Callback):

    def on_train_start(self, trainer, pl_module):
        print("Training is started, ^v^!")

    def on_train_end(self, trainer, pl_module):
        print("Training is done, ^v^!")

    def on_validation_start(self, trainer, pl_module):
        print("Validation is started, ^v^!")

    def on_validation_end(self, trainer, pl_module):
        print("Validation is done, ^v^!")

    def on_test_start(self, trainer, pl_module):
        print("Test is started, ^v^!")

    def on_test_end(self, trainer, pl_module):
        print("Test is done, ^v^!")

    def on_predict_start(self, trainer, pl_module):
        print("Prediction is started, ^v^!")

    def on_predict_end(self, trainer, pl_module):
        print("Prediction is done, ^v^!")
