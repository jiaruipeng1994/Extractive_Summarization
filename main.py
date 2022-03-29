#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time : Tue Jan 25 16:50:29 2022
# @Author : JRP - Ruipeng Jia

import torch
from argparse import ArgumentParser
import pytorch_lightning as pl
from pytorch_lightning import Trainer
from pytorch_lightning.loggers import WandbLogger

from datas.cnndm import CNNDMDatamodule
from modules.distillation_summarization import DistilSum
from modules.iteration_summarization import ThresSum
from modules.differential_summarization import DifferSum
from modules.heterogeneous_summarization import HAHSum
from modules.multilingual_summarization import NLSSum
from callbacks import EchoCallback, RougeEvaluator, LitProgressBar, MyException


def main(args):
    pl.seed_everything(args.seed)
    wandb_logger = WandbLogger(project="jNLP")
    model = eval(args.version)(args)  # model = DistilSum(args)
    callbacks = [EchoCallback(), RougeEvaluator(), LitProgressBar(), MyException()]
    cnn_dm = CNNDMDatamodule.from_argparse_args(args)
    trainer = Trainer.from_argparse_args(args, enable_checkpointing=False, gradient_clip_val=0.5, callbacks=callbacks, logger=wandb_logger)
    trainer.fit(model, datamodule=cnn_dm)


if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("--version", type=str, default="DistilSum")
    parser.add_argument("--seed", type=int, default=1024)
    parser.add_argument("--val_test", type=bool, default=False)
    temp_args, _ = parser.parse_known_args()

    parser = Trainer.add_argparse_args(parser)
    parser = DistilSum.add_model_specific_args(parser)
    parser = CNNDMDatamodule.add_argparse_args(parser)

    if not torch.cuda.is_available():
        mocked_args = """ --max_epochs 3 --accumulate_grad_batches 4 --val_check_interval 2 --num_proc 1 --gpus 0 --train_batch_size 2 --max_pos 800 --pre_train roberta_base --version DistilSum """.split()
        args = parser.parse_args(mocked_args)
    else:
        mocked_args = """ --max_epochs 5 --accumulate_grad_batches 2 --precision 32 --val_check_interval 3000 --gpus 1 --pre_train roberta_base --lr 1e-5 --warmup_steps 30000 --max_pos 800 --version DifferSum """.split()
        # args = parser.parse_args(mocked_args)
        args = parser.parse_args()
    main(args)
