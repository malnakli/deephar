import torch
import torch.nn as nn


def pose_regression_loss(pose_loss, visibility_weight):
    def _pose_regression_loss(y_true, y_pred):
        video_clip = y_true.ndim == 4
        if video_clip:
            """The model was time-distributed, so there is one additional
            dimension.
            """
            p_true = y_true[:, :, :, 0:-1]
            p_pred = y_pred[:, :, :, 0:-1]
            v_true = y_true[:, :, :, -1]
            v_pred = y_pred[:, :, :, -1]
        else:
            p_true = y_true[:, :, 0:-1]
            p_pred = y_pred[:, :, 0:-1]
            v_true = y_true[:, :, -1]
            v_pred = y_pred[:, :, -1]

        if pose_loss == "l1l2":
            ploss = elasticnet_loss_on_valid_joints(p_true, p_pred)
        elif pose_loss == "l1":
            ploss = l1_loss_on_valid_joints(p_true, p_pred)
        elif pose_loss == "l2":
            ploss = l2_loss_on_valid_joints(p_true, p_pred)
        elif pose_loss == "l1l2bincross":
            ploss = elasticnet_bincross_loss_on_valid_joints(p_true, p_pred)
        else:
            raise Exception("Invalid pose_loss option ({})".format(pose_loss))

        vloss = nn.BCELoss()(v_true, v_pred).item()

        if video_clip:
            """If time-distributed, average the error on video frames."""

            # NOTE: make sure you change the dim
            vloss = torch.mean(vloss, dim=-1)
            ploss = torch.mean(ploss, dim=-1)

        return ploss + visibility_weight * vloss

    return _pose_regression_loss


def elasticnet_loss_on_valid_joints(y_true, y_pred):
    y_true, y_pred, num_joints = _reset_invalid_joints(y_true, y_pred)
    # NOTE: make sure you change the dim
    _l1 = torch.sum(torch.abs(y_pred - y_true), dim=(-1, -2)) / num_joints
    _l2 = torch.sum(torch.sqrt(y_pred - y_true), dim=(-1, -2)) / num_joints
    return _l1 + _l2


def l1_loss_on_valid_joints(y_true, y_pred):
    y_true, y_pred, num_joints = _reset_invalid_joints(y_true, y_pred)
    # NOTE: make sure you change the dim
    return torch.sum(torch.abs(y_pred - y_true), dim=(-1, -2)) / num_joints


def l2_loss_on_valid_joints(y_true, y_pred):
    y_true, y_pred, num_joints = _reset_invalid_joints(y_true, y_pred)
    # NOTE: make sure you change the dim
    return torch.sum(torch.sqrt(y_pred - y_true), dim=(-1, -2)) / num_joints


def elasticnet_bincross_loss_on_valid_joints(y_true, y_pred):
    idx = torch.ge(y_true, 0.0).type(torch.float32)
    # NOTE: make sure you change the dim
    num_joints = torch.clamp(torch.sum(idx, dim=(-1, -2)), min=1)

    _l1 = torch.abs(y_pred - y_true)
    _l2 = torch.sqrt(y_pred - y_true)
    _bc = 0.01 * nn.BCELoss()(y_true, y_pred).item()
    dummy = 0.0 * y_pred
    # NOTE: make sure you change the dim
    return (
        torch.sum(
            torch.where(idx.type(torch.bool), _l1 + _l2 + _bc, dummy), dim=(-1, -2)
        )
        / num_joints
    )


def _reset_invalid_joints(y_true, y_pred):
    """Reset (set to zero) invalid joints, according to y_true, and compute the
    number of valid joints.
    """
    idx = torch.ge(y_true, 0.0).type(torch.float32)
    y_true = idx * y_true
    y_pred = idx * y_pred
    # NOTE: make sure you change the dim
    num_joints = torch.clamp(torch.sum(idx, dim=(-1, -2)), min=1)
    return y_true, y_pred, num_joints
