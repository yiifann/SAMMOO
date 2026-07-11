import torch

from models.SAMMOO.utils import compute_present_group_losses

__all__ = ["compute_present_group_losses", "gram_matrix", "mgda_frank_wolfe"]


def gram_matrix(flat_grads):
    """
    flat_grads: list of 1D tensors of equal length (per-group gradients, flattened).
    Returns the (T, T) Gram matrix M with M[i, j] = <flat_grads[i], flat_grads[j]>.
    """
    stacked = torch.stack(flat_grads)
    return stacked @ stacked.T


def mgda_frank_wolfe(M, max_iter=20, stop_tol=1e-4):
    """
    Solve min_{alpha in simplex} alpha^T M alpha via the Frank-Wolfe algorithm of
    Desideri (2012), following Algorithm 2 of Sener & Koltun (2018), "Multi-Task
    Learning as Multi-Objective Optimization". M is the Gram matrix of per-group
    gradient vectors, M[i, j] = <grad_i, grad_j>.

    Returns alpha, a (T,) tensor on the simplex minimizing ||sum_t alpha_t grad_t||^2.
    """
    T = M.shape[0]
    if T == 1:
        return torch.ones(1, device=M.device, dtype=M.dtype)

    alpha = torch.full((T,), 1.0 / T, device=M.device, dtype=M.dtype)

    for _ in range(max_iter):
        Ma = M @ alpha
        t_hat = int(torch.argmin(Ma).item())

        a_dot_a = float((alpha @ Ma).item())
        a_dot_b = float(Ma[t_hat].item())
        b_dot_b = float(M[t_hat, t_hat].item())

        # Exact line search for gamma in [0, 1] minimizing
        # ||(1-gamma) alpha + gamma e_t_hat||^2_M (Algorithm 1, Sener & Koltun 2018).
        if a_dot_b >= b_dot_b:
            gamma = 1.0
        elif a_dot_b >= a_dot_a:
            gamma = 0.0
        else:
            denom = b_dot_b - 2 * a_dot_b + a_dot_a
            gamma = (a_dot_a - a_dot_b) / denom if denom > 1e-12 else 0.0
            gamma = min(max(gamma, 0.0), 1.0)

        vertex = torch.zeros_like(alpha)
        vertex[t_hat] = 1.0
        alpha = (1 - gamma) * alpha + gamma * vertex

        if gamma < stop_tol:
            break

    return alpha
