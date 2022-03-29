#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time : Wed Feb  9 20:05:29 2022
# @Author : JRP - Ruipeng Jia

import copy
from pytorch_lightning import Trainer
from pytorch_lightning.callbacks import Callback
from rouge import Rouge

from callbacks.echo_callback import EchoCallback
from jtools.text import printable


class RougeEvaluator(Callback):

    def __init__(self):
        super().__init__()
        self.val_predictions = []
        self.val_references = []
        self.test_predictions = []
        self.test_references = []
        self.rouge = Rouge()

    def _val_test(self, trainer, pl_module):
        callbacks = [EchoCallback(), self.__class__()]
        new_trainer = Trainer.from_argparse_args(pl_module.args, callbacks=callbacks)
        new_trainer.test(copy.copy(pl_module), datamodule=copy.copy(trainer.datamodule))

    def on_validation_batch_end(self, trainer, pl_module, outputs, batch, batch_idx, dataloader_idx):
        self.val_predictions += [''.join(filter(lambda x: x in printable, pred)) for pred in outputs]
        self.val_references += [''.join(filter(lambda x: x in printable, ref)) for ref in batch['ref']]

    def on_validation_epoch_end(self, trainer, pl_module):
        results = self.rouge.get_scores(self.val_predictions, self.val_references, avg=True)
        print(results)
        self.log("val_rouge", results['rouge-1']['f'], on_epoch=True, logger=True)
        self.val_predictions, self.val_references = [], []

        if pl_module.args.val_test:
            if trainer.num_sanity_val_steps > 0:
                trainer.num_sanity_val_steps = 0
            else:
                self._val_test(trainer, pl_module)  # will not work!!!

    def on_test_batch_end(self, trainer, pl_module, outputs, batch, batch_idx, dataloader_idx):
        self.val_predictions += [''.join(filter(lambda x: x in printable, pred)) for pred in outputs]
        self.val_references += [''.join(filter(lambda x: x in printable, ref)) for ref in batch['ref']]

    def on_test_epoch_end(self, trainer, pl_module):
        results = self.rouge.get_scores(self.test_predictions, self.test_references, avg=True)
        print(results)
        self.log("val_rouge", results['rouge-1']['f'], on_epoch=True, logger=True)
        self.test_predictions, self.test_references = [], []
