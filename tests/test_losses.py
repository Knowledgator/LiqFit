import unittest

import torch
from kornia.losses import focal_loss
from liqfit.losses import focal_loss_with_mask


class TestCorrectness(unittest.TestCase):
    def test_focal_loss_with_ignore_index(self):
        x = torch.tensor(
            [[[1, 2, 3, 4], [5, 6, 7, 8], [9, 10, 11, 12]]],
            dtype=torch.float32,
        )
        y = torch.tensor([[1, 2, 3]], dtype=torch.int64)
        y[:, -1] = -100
        loss = round(
            focal_loss_with_mask(
                x.reshape(-1, x.shape[-1]), y.reshape(-1)
            ).item(),
            4,
        )
        output = 0.1795
        self.assertEqual(loss, output)

    def test_modified_loss_with_kornia_impl(self):
        x = torch.tensor(
            [[[1, 2, 3, 4], [5, 6, 7, 8], [9, 10, 11, 12]]],
            dtype=torch.float32,
        )
        y = torch.tensor([[1, 2, 3]], dtype=torch.int64)
        modified_loss = round(
            focal_loss_with_mask(
                x.reshape(-1, x.shape[-1]), y.reshape(-1), alpha=0.5
            ).item(),
            4,
        )
        kornia_loss = round(
            focal_loss(
                x.reshape(-1, x.shape[-1]),
                y.reshape(-1),
                alpha=0.5,
                reduction="mean",
            ).item(),
            4,
        )
        self.assertEqual(modified_loss, kornia_loss)
