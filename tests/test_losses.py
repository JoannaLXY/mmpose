import pytest
import torch
from mmpose.models.losses import JointsMSELoss, JointsOHKMMSELoss


def test_mse_loss():
    loss = JointsMSELoss()
    fake_pred = torch.zeros((1, 3, 64, 64))
    fake_label = torch.zeros((1, 3, 64, 64))
    assert torch.allclose(loss(fake_pred, fake_label, None), torch.tensor(0.))

    fake_pred = torch.ones((1, 3, 64, 64))
    fake_label = torch.zeros((1, 3, 64, 64))
    assert torch.allclose(loss(fake_pred, fake_label, None), torch.tensor(1.))

    fake_pred = torch.zeros((1, 2, 64, 64))
    fake_pred[0, 0] += 1
    fake_label = torch.zeros((1, 2, 64, 64))
    assert torch.allclose(loss(fake_pred, fake_label, None), torch.tensor(0.5))

    with pytest.raises(ValueError):
        loss = JointsOHKMMSELoss()
        fake_pred = torch.zeros((1, 3, 64, 64))
        fake_label = torch.zeros((1, 3, 64, 64))
        assert torch.allclose(
            loss(fake_pred, fake_label, None), torch.tensor(0.))

    with pytest.raises(AssertionError):
        loss = JointsOHKMMSELoss(topk=-1)
        fake_pred = torch.zeros((1, 3, 64, 64))
        fake_label = torch.zeros((1, 3, 64, 64))
        assert torch.allclose(
            loss(fake_pred, fake_label, None), torch.tensor(0.))

    loss = JointsOHKMMSELoss(topk=2)
    fake_pred = torch.ones((1, 3, 64, 64))
    fake_label = torch.zeros((1, 3, 64, 64))
    assert torch.allclose(loss(fake_pred, fake_label, None), torch.tensor(1.))

    fake_pred = torch.zeros((1, 3, 64, 64))
    fake_pred[0, 0] += 1
    fake_label = torch.zeros((1, 3, 64, 64))
    assert torch.allclose(loss(fake_pred, fake_label, None), torch.tensor(0.5))
