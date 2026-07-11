import torch

from models.SAMMOO.utils import compute_present_group_losses

__all__ = ["compute_present_group_losses", "fw_lmo_vertex", "fw_online_update"]


def fw_lmo_vertex(num_groups, present_group_ids, present_losses):
    """
    Frank-Wolfe Linear Maximization Oracle for the worst-group scalarization
    (generalizes Eq. 3's binary "1 if L1 > L2 else 0" to num_groups groups):
    the one-hot vertex on the simplex maximizing s^T losses, restricted to the
    groups actually present this step.
    """
    stacked = torch.stack([loss.detach() for loss in present_losses])
    winner_local_idx = int(torch.argmax(stacked).item())
    winner_group = present_group_ids[winner_local_idx]

    vertex = torch.zeros(num_groups, device=stacked.device, dtype=stacked.dtype)
    vertex[winner_group] = 1.0
    return vertex


def fw_online_update(beta, vertex, gamma):
    """One online Frank-Wolfe step (Eq. 4): beta <- (1-gamma) * beta + gamma * vertex."""
    return (1.0 - gamma) * beta + gamma * vertex
