import copy
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import torch
from torch import nn

from dataset.hyper_dataset import HyperDataset
from models.autoencoder import D2MP
from models.hyper_block import HyperComputeModule
from models.motion_decoder import Time_info_decoder


class TrajectoryEmbedding(nn.Module):
    """Small encoder to test motion plumbing without the optional Mamba kernels."""
    def __init__(self):
        super().__init__()
        self.proj = nn.Linear(8, 32)

    def forward(self, x, decoder_only=False, active_hyper=False):
        return self.proj(x)


class HyperMotionTests(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(7)
        torch.set_num_threads(1)

    def model(self, active_hyper=True, one2one=False):
        config = SimpleNamespace(use_diffmot=False, active_hyper=active_hyper,
                                 one2one_predict=one2one, decoder_only=True, interval=5)
        model = D2MP(config, encoder=TrajectoryEmbedding(), device='cpu')
        model.ssm_decoder = Time_info_decoder(d_model=32, n_layer=2, d_s=4)
        return model

    def batch(self):
        return {'condition': torch.rand(2, 5, 3, 8) * 0.3 + 0.1,
                'gt_bbox': torch.rand(2, 1, 3, 4) * 0.3 + 0.1,
                'valid_mask': torch.tensor([[True, True, False], [True, False, False]])}

    def test_hypergraph_interacts_only_with_same_frame_and_scene(self):
        graph = HyperComputeModule(8, 8, threshold=10)
        x = (torch.randn(2, 3, 5, 8) * 0.1).requires_grad_()
        y = graph(x)
        y[0, 0, 0, 0].backward()
        self.assertGreater(x.grad[0, 1:, 0].abs().sum().item(), 0)
        self.assertEqual(x.grad[1].abs().sum().item(), 0)
        self.assertEqual(x.grad[0, :, 1:].abs().sum().item(), 0)

    def test_hypergraph_padding_and_permutation(self):
        graph = HyperComputeModule(8, 8, threshold=10).train()
        x = torch.randn(1, 3, 5, 8)
        reference = graph(x)
        padded = torch.cat((x, torch.full((1, 2, 5, 8), float('nan'))), dim=1)
        mask = torch.tensor([[True, True, True, False, False]])
        result = graph(padded, mask)
        torch.testing.assert_close(result[:, :3], reference)
        self.assertEqual(result[:, 3:].abs().sum().item(), 0)
        order = [2, 0, 1]
        torch.testing.assert_close(graph(x[:, order]), reference[:, order])

    def test_training_backward_and_padding_invariance(self):
        for one2one in [False, True]:
            model = self.model(one2one=one2one).train()
            batch = self.batch()
            original = copy.deepcopy(batch)
            loss = model(batch)
            self.assertTrue(torch.isfinite(loss))
            loss.backward()
            for head in model.ssm_decoder.head_series:
                grad = head.hyperssm.hgconv.fc.weight.grad
                self.assertIsNotNone(grad)
                self.assertTrue(torch.isfinite(grad).all())
            for key in batch:
                torch.testing.assert_close(batch[key], original[key])
            padded = {key: value.clone() for key, value in batch.items()}
            for key in ['condition', 'gt_bbox']:
                shape = list(padded[key].shape)
                shape[2] = 2
                padded[key] = torch.cat((padded[key], torch.full(shape, float('nan'))), dim=2)
            padded['valid_mask'] = torch.cat((padded['valid_mask'], torch.zeros(2, 2, dtype=torch.bool)), dim=1)
            torch.testing.assert_close(model(padded), loss)

    def test_all_padding_has_finite_zero_loss(self):
        model = self.model().train()
        batch = self.batch()
        batch['valid_mask'].fill_(False)
        batch['condition'].fill_(float('nan'))
        batch['gt_bbox'].fill_(float('nan'))
        loss = model(batch)
        self.assertEqual(loss.item(), 0)
        loss.backward()
        self.assertTrue(all(p.grad is None or torch.isfinite(p.grad).all() for p in model.parameters()))

    def test_default_decoder_with_300_slots(self):
        decoder = Time_info_decoder().eval()
        positions = torch.rand(1, 2, 5, 4) * 0.2 + 0.1
        features = torch.randn(1, 2, 5, 256)
        padded_positions = torch.cat((positions, torch.zeros(1, 298, 5, 4)), dim=1)
        padded_features = torch.cat((features, torch.zeros(1, 298, 5, 256)), dim=1)
        mask = torch.arange(300)[None] < 2
        with torch.no_grad():
            expected = decoder(positions, features, active_hyper=True)
            actual = decoder(padded_positions, padded_features, active_hyper=True, valid_mask=mask)
        torch.testing.assert_close(actual[:, :2], expected)
        self.assertEqual(actual[:, 2:].abs().sum().item(), 0)

    def test_generation_matches_joint_decoder_and_object_order(self):
        for one2one in [False, True]:
            model = self.model(one2one=one2one).eval()
            condition = torch.rand(1, 3, 5, 8) * 0.2 + 0.1
            encoded = model.encoder(condition)
            positions = condition[..., :4]
            if one2one:
                positions, encoded = positions[..., -1:, :], encoded[..., -1:, :]
            expected = model.ssm_decoder(positions, encoded, active_hyper=True)[0, :, -1]
            histories = condition[0].numpy() * np.array([100, 200] * 4)
            actual = model.generate(list(histories), 1, True, img_w=100, img_h=200)
            np.testing.assert_allclose(actual, expected.detach().numpy(), rtol=1e-5, atol=1e-6)
            reordered = model.generate(list(histories[[2, 0, 1]]), 1, True, img_w=100, img_h=200)
            np.testing.assert_allclose(reordered, actual[[2, 0, 1]], rtol=1e-5, atol=1e-6)
            self.assertEqual(model.generate([], 1, True, img_w=100, img_h=200).shape, (0, 4))
            self.assertEqual(model.generate([histories[0, :1]], 1, True, img_w=100, img_h=200).shape, (1, 4))

    def test_legacy_single_object_path(self):
        for one2one in [False, True]:
            model = self.model(active_hyper=False, one2one=one2one)
            batch = {'condition': torch.rand(2, 5, 8) * 0.2 + 0.1,
                     'cur_bbox': torch.rand(2, 4) * 0.2 + 0.1}
            loss = model(batch)
            self.assertTrue(torch.isfinite(loss))
            loss.backward()
            model.eval()
            result = model.generate(list(batch['condition'].numpy()), 1, True, img_w=1, img_h=1)
            self.assertEqual(result.shape, (2, 4))

    def test_dataset_mask_and_truncation(self):
        with tempfile.TemporaryDirectory() as directory:
            seq = Path(directory) / 'seq'
            seq.mkdir()
            rows = [[frame, tid, 0.1 * tid, 0.2, 0.1, 0.1, 1]
                    for frame in range(7) for tid in [1, 2, 3]]
            np.savetxt(seq / '000001.txt', rows)
            dataset = HyperDataset(directory, SimpleNamespace(interval=5))
            sample = dataset[0]
            self.assertEqual(sample['condition'].shape, (5, 300, 8))
            self.assertEqual(sample['valid_mask'].sum(), 3)
            self.assertEqual(sample['valid_mask'].dtype, np.bool_)
            dataset.pad = 2
            self.assertEqual(dataset[0]['valid_mask'].tolist(), [True, True])


if __name__ == '__main__':
    unittest.main()
