"""Exercise runtime methods without loading optional detector/Mamba dependencies."""
import ast
import copy
import logging
import os.path as osp
from pathlib import Path
import random
import subprocess
import sys
import tempfile
import unittest
from unittest.mock import patch

import numpy as np
import torch
from torch import nn, optim
import torchvision
import yaml


ROOT = Path(__file__).resolve().parents[1]
tree = ast.parse((ROOT / 'hyperssm.py').read_text())
runtime = next(n for n in tree.body if isinstance(n, ast.ClassDef) and n.name == 'HyperSSM')
methods = {'__init__', '_build_optimizer', 'train', '_evaluate_results',
           '_postprocess_results', 'postprocess'}
runtime.body = [n for n in runtime.body if isinstance(n, ast.FunctionDef) and n.name in methods]


class Progress(list):
    def __init__(self, iterable, **kwargs):
        super().__init__(iterable)

    def set_description(self, text):
        pass


namespace = dict(torch=torch, torchvision=torchvision, optim=optim, np=np, random=random,
                 osp=osp, subprocess=subprocess, sys=sys, __file__=str(ROOT / 'hyperssm.py'),
                 logger=logging.getLogger(__name__), tqdm=Progress)
exec(compile(ast.Module(body=[runtime], type_ignores=[]), '<runtime methods>', 'exec'), namespace)
Runtime = namespace['HyperSSM']
Runtime._build = lambda self: None


class Config(dict):
    __getattr__ = dict.__getitem__
    __setattr__ = dict.__setitem__


class LossModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.weight = nn.Parameter(torch.tensor(1.0))

    def forward(self, batch):
        return self.weight.square()


class RuntimeTests(unittest.TestCase):
    def agent(self, **overrides):
        config = Config(eval_mode=False, use_diffmot=False, use_detection_model=False,
                        info_dir='unused', lr=0.01, epochs=3, eval_every=1, dataset='test')
        config.update(overrides)
        agent = Runtime(config)
        agent.model = LossModel()
        agent.train_data_loader = [{}]
        return agent

    def test_seeds_and_unsupported_options(self):
        self.agent(seed=42)
        first = (random.random(), np.random.rand(), torch.rand(3))
        self.agent(seed=42)
        second = (random.random(), np.random.rand(), torch.rand(3))
        self.assertEqual(first[:2], second[:2])
        torch.testing.assert_close(first[2], second[2])
        for option in ['augment', 'use_diffmot']:
            with self.assertRaises(NotImplementedError):
                self.agent(**{option: True})

    def test_resume_matches_uninterrupted_training(self):
        for enabled in [False, True]:
            with tempfile.TemporaryDirectory() as directory:
                full = self.agent(use_scheduler=enabled)
                full.model_dir = directory
                full._build_optimizer()
                full.train()
                expected = copy.deepcopy(full.model.state_dict())
                expected_lr = full.optimizer.param_groups[0]['lr']
                checkpoint = torch.load(Path(directory) / 'test_epoch1.pt')
                resumed = self.agent(use_scheduler=enabled, resume=True)
                resumed.model_dir = directory
                resumed.checkpoint = checkpoint
                resumed.model.load_state_dict(checkpoint['ddpm'])
                resumed._build_optimizer()
                self.assertEqual(resumed.start_epoch, 2)
                resumed.train()
                torch.testing.assert_close(resumed.model.state_dict()['weight'], expected['weight'])
                self.assertEqual(resumed.optimizer.param_groups[0]['lr'], expected_lr)
                if enabled:
                    self.assertAlmostEqual(expected_lr, 0.01 * 0.98 ** 3)
                else:
                    self.assertEqual(expected_lr, 0.01)
                mismatch = self.agent(resume=True, use_scheduler=not enabled)
                mismatch.checkpoint = checkpoint
                with self.assertRaises(ValueError):
                    mismatch._build_optimizer()

    def test_empty_online_detections(self):
        agent = self.agent(use_detection_model=True)
        outputs = torch.zeros(1, 2, 6)
        result = agent._postprocess_results(outputs, 640, 480)
        self.assertEqual(result.shape, (0, 6))
        self.assertEqual(result[:, :5].numpy().shape, (0, 5))

    def test_evaluation_uses_config_and_can_be_disabled(self):
        agent = self.agent()
        with patch.object(subprocess, 'run') as run:
            agent._evaluate_results()
            run.assert_not_called()
            agent.config.run_evaluation = True
            with self.assertRaises(ValueError):
                agent._evaluate_results()
            with tempfile.TemporaryDirectory(prefix='hyperssm eval ') as directory:
                seqmap = Path(directory) / 'sequences.txt'
                seqmap.touch()
                agent.config.update(gt_folder=directory, seqmap_file=str(seqmap),
                                    split_to_eval='val', save_dir=directory)
                agent._evaluate_results()
            command = run.call_args.args[0]
            self.assertEqual(command[0], sys.executable)
            self.assertEqual(command[command.index('--GT_FOLDER') + 1], directory)
            self.assertEqual(command[command.index('--SPLIT_TO_EVAL') + 1], 'val')
            self.assertTrue(run.call_args.kwargs['check'])

    def test_preprocessing_keeps_target_that_leaves_after_window(self):
        for path in (ROOT / 'data_process').glob('*_data_process_interval.py'):
            tree = ast.parse(path.read_text())
            loop = next(n for n in ast.walk(tree) if isinstance(n, ast.For)
                        and isinstance(n.target, ast.Name) and n.target.id == 'i'
                        and any(isinstance(x, ast.Name) and x.id == 'common_elements' for x in ast.walk(n)))
            values = dict(interval=5, object_matrix=[[1]] * 7 + [[2]], common_object_matrix=[[], []])
            exec(compile(ast.Module(body=[loop], type_ignores=[]), str(path), 'exec'), values)
            self.assertEqual(values['common_object_matrix'], [{1}, set()])

    def test_configs(self):
        for path in (ROOT / 'configs').glob('*.yaml'):
            config = yaml.safe_load(path.read_text())
            self.assertFalse(config['augment'])
            self.assertEqual(config['batch_size'], 2)
            self.assertFalse(config.get('load_pretrain', False))
            if config.get('run_evaluation'):
                for key in ['gt_folder', 'seqmap_file', 'split_to_eval']:
                    self.assertTrue(config[key])


if __name__ == '__main__':
    unittest.main()
