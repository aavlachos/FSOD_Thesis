#!/usr/bin/env python3
# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved

"""
Created on Thursday, April 14, 2022

This script is a simplified version of the training script in detectron2/tools.

@author: Guangxing Han
"""

import os

from detectron2.checkpoint import DetectionCheckpointer
from detectron2.config import get_cfg
from detectron2.engine import DefaultTrainer, default_argument_parser, default_setup, launch
from detectron2.data import build_batch_data_loader
from detectron2.evaluation import (
    DatasetEvaluator,
    inference_on_dataset,
    print_csv_format,
    verify_results,
)

from meta_faster_rcnn_withGCN.config import get_cfg
from meta_faster_rcnn_withGCN.data import DatasetMapperWithSupportCOCO, DatasetMapperWithSupportVOC
from meta_faster_rcnn_withGCN.data.build import build_detection_train_loader, build_detection_test_loader
from meta_faster_rcnn_withGCN.solver import build_optimizer
from meta_faster_rcnn_withGCN.evaluation import COCOEvaluator, PascalVOCDetectionEvaluator

import bisect
import copy
import itertools
import logging
import numpy as np
import operator
import pickle
import torch.utils.data
from collections import OrderedDict

import detectron2.utils.comm as comm
from detectron2.utils.logger import setup_logger

import time
from detectron2.utils.events import EventStorage, get_event_storage
from detectron2.utils.logger import _log_api_usage
from torch.nn.parallel import DataParallel, DistributedDataParallel
import torch
from typing import List, Mapping, Optional
import weakref
import logging
import concurrent.futures


class Trainer(DefaultTrainer):

    def __init__(self, cfg):

        self.grad_acc = cfg.GRAD_ACC
        super().__init__(cfg)
        print("Initializing Trainer with grad accumulation ",self.grad_acc)
        for name ,param in self.model.named_parameters():
            if 'gcn_model' not in name:
                param.requires_grad = False
                print("Parameter", name , "is frozen\n")



    def run_step(self):

        grad_acc=self.grad_acc
        self._trainer.iter = self.iter
        assert self._trainer.model.training, "[SimpleTrainer] model was changed to eval mode!"
        start = time.perf_counter()
        """
        If you want to do something with the data, you can wrap the dataloader.
        """
        data = next(self._trainer._data_loader_iter)
        data_time = time.perf_counter() - start

        if(self._trainer.iter==0):
            self._trainer.optimizer.zero_grad()

        """
        If you want to do something with the losses, you can wrap the model.
        """
        loss_dict = self._trainer.model(data)
        if isinstance(loss_dict, torch.Tensor):
            losses = loss_dict
            loss_dict = {"total_loss": loss_dict}
        else:
            losses = sum(loss_dict.values())

        losses = losses/grad_acc

        losses.backward()

        self._trainer.after_backward()

        if self._trainer.async_write_metrics:
            # write metrics asynchronically
            self._trainer.concurrent_executor.submit(
                self._trainer._write_metrics, loss_dict, data_time, iter=self._trainer.iter
            )
        else:
            self._trainer._write_metrics(loss_dict, data_time)

        """
        If you need gradient clipping/scaling or other processing, you can
        wrap the optimizer with your custom `step()` method. But it is
        suboptimal as explained in https://arxiv.org/abs/2006.15704 Sec 3.2.4
        """
        if ((self._trainer.iter + 1) % grad_acc == 0 or self._trainer.iter == 0):
            #print("Updating weights\n")
            self._trainer.optimizer.step()
            self._trainer.optimizer.zero_grad()
<<<<<<< HEAD
        if ((self._trainer.iter + 1) % 500 == 0):
            print(self.model.gcn_model.gcn_layer.graph_conv.weight)
=======
            #print(self.model.gcn_model.gcn_layer.graph_conv.weight)
>>>>>>> 9eb66849f127e6baf45464c0cfcde2e8e54d2496


    @classmethod
    def build_train_loader(cls, cfg):
        """
        Returns:
            iterable
        It calls :func:`detectron2.data.build_detection_train_loader` with a customized
        DatasetMapper, which adds categorical labels as a semantic mask.
        """
        print("Build train loader")
        if 'coco' in cfg.DATASETS.TRAIN[0]:
            mapper = DatasetMapperWithSupportCOCO(cfg)
        else:
            mapper = DatasetMapperWithSupportVOC(cfg)
        return build_detection_train_loader(cfg, mapper)

    @classmethod
    def build_test_loader(cls, cfg, dataset_name):
        """
        Returns:
            iterable
        It now calls :func:`detectron2.data.build_detection_test_loader`.
        Overwrite it if you'd like a different data loader.
        """
        return build_detection_test_loader(cfg, dataset_name)

    @classmethod
    def build_optimizer(cls, cfg, model):
        """
        Returns:
            torch.optim.Optimizer:
        It now calls :func:`detectron2.solver.build_optimizer`.
        Overwrite it if you'd like a different optimizer.
        """
        return build_optimizer(cfg, model)

    @classmethod
    def build_evaluator(cls, cfg, dataset_name, output_folder=None):
        if output_folder is None:
            output_folder = os.path.join(cfg.OUTPUT_DIR, "inference")
        if 'coco' in dataset_name:
            return COCOEvaluator(dataset_name, cfg, True, output_folder)
        else:
            return PascalVOCDetectionEvaluator(dataset_name)

    @classmethod
    def test(cls, cfg, model, evaluators=None):
        """
        Args:
            cfg (CfgNode):
            model (nn.Module):
            evaluators (list[DatasetEvaluator] or None): if None, will call
                :meth:`build_evaluator`. Otherwise, must have the same length as
                `cfg.DATASETS.TEST`.

        Returns:
            dict: a dict of result metrics
        """
        logger = logging.getLogger(__name__)
        if isinstance(evaluators, DatasetEvaluator):
            evaluators = [evaluators]
        if evaluators is not None:
            assert len(cfg.DATASETS.TEST) == len(evaluators), "{} != {}".format(
                len(cfg.DATASETS.TEST), len(evaluators)
            )

        results = OrderedDict()
        for idx, dataset_name in enumerate(cfg.DATASETS.TEST):
            data_loader = cls.build_test_loader(cfg, dataset_name)
            # When evaluators are passed in as arguments,
            # implicitly assume that evaluators can be created before data_loader.
            if evaluators is not None:
                evaluator = evaluators[idx]
            else:
                try:
                    evaluator = cls.build_evaluator(cfg, dataset_name)
                except NotImplementedError:
                    logger.warn(
                        "No evaluator found. Use `DefaultTrainer.test(evaluators=)`, "
                        "or implement its `build_evaluator` method."
                    )
                    results[dataset_name] = {}
                    continue

            test_seeds = cfg.DATASETS.SEEDS
            test_shots = cfg.DATASETS.TEST_SHOTS
            cur_test_shots_set = set(test_shots)
            if 'coco' in cfg.DATASETS.TRAIN[0]:
                evaluation_dataset = 'coco'
                coco_test_shots_set = set([1,2,3,5,10,30])
                test_shots_join = cur_test_shots_set.intersection(coco_test_shots_set)
                test_keepclasses = cfg.DATASETS.TEST_KEEPCLASSES
            else:
                evaluation_dataset = 'voc'
                voc_test_shots_set = set([1,2,3,5,10])
                test_shots_join = cur_test_shots_set.intersection(voc_test_shots_set)
                test_keepclasses = cfg.DATASETS.TEST_KEEPCLASSES

            if cfg.INPUT.FS.FEW_SHOT:
                test_shots = [cfg.INPUT.FS.SUPPORT_SHOT]
                test_shots_join = set(test_shots)

            print("================== test_shots_join=", test_shots_join)
            for shot in test_shots_join:
                print("evaluating {}.{} for {} shot".format(evaluation_dataset, test_keepclasses, shot))
                if isinstance(model, torch.nn.parallel.DistributedDataParallel):
                    model.module.init_support_features(evaluation_dataset, shot, test_keepclasses, test_seeds)
                else:
                    model.init_support_features(evaluation_dataset, shot, test_keepclasses, test_seeds)

                results_i = inference_on_dataset(model, data_loader, evaluator)
                results[dataset_name] = results_i
                if comm.is_main_process():
                    assert isinstance(
                        results_i, dict
                    ), "Evaluator must return a dict on the main process. Got {} instead.".format(
                        results_i
                    )
                    logger.info("Evaluation results for {} in csv format:".format(dataset_name))
                    print_csv_format(results_i)

        if len(results) == 1:
            results = list(results.values())[0]
        return results


def setup(args):
    """
    Create configs and perform basic setups.
    """
    cfg = get_cfg()
    cfg.merge_from_file(args.config_file)
    cfg.merge_from_list(args.opts)
    cfg.freeze()
    default_setup(cfg, args)

    rank = comm.get_rank()
    setup_logger(cfg.OUTPUT_DIR, distributed_rank=rank, name="meta_faster_rcnn")

    return cfg


def main(args):
    cfg = setup(args)
    # print("\n\n\n\n\n\nI'm in main(fsod_train_net.py)\n\n\n\n\n\n")

    if args.eval_only:
        model = Trainer.build_model(cfg)
        DetectionCheckpointer(model, save_dir=cfg.OUTPUT_DIR).resume_or_load(
            cfg.MODEL.WEIGHTS, resume=args.resume
        )
        res = Trainer.test(cfg, model)
        return res
    print("\nCreating Trainer...\n")
    trainer = Trainer(cfg)
    print("\nLoading previous weights...\n")
    trainer.resume_or_load(resume=args.resume)
    print("Start Training...")
    return trainer.train()


if __name__ == "__main__":
    args = default_argument_parser().parse_args()
    print("Command Line Args:", args)
    launch(
        main,
        args.num_gpus,
        num_machines=args.num_machines,
        machine_rank=args.machine_rank,
        dist_url=args.dist_url,
        args=(args,),
    )
