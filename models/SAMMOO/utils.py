import torch


def compute_group_loss_weights(
    group_losses,
    mode="frank_wolfe",
    temperature=1.0,
    fw_max_iter=1,
    fw_max_gamma=0.5,
    tol=1e-6,
):
    """
    Compute detached group weights from scalar group losses.

    The returned weights sum to one. Gradients do not flow through
    the weight-selection procedure, but do flow through group_losses
    when the weighted loss is constructed.
    """
    if not group_losses:
        raise ValueError("group_losses cannot be empty.")

    losses = torch.stack(
        [loss.detach().float() for loss in group_losses]
    )

    n_groups = losses.numel()

    # If only one group is present in the batch, use its loss directly.
    if n_groups == 1:
        return torch.ones_like(losses)

    # Avoid arbitrary preference when losses are effectively equal.
    if torch.max(losses) - torch.min(losses) < tol:
        return torch.ones_like(losses) / n_groups

    if mode == "softmax":
        if temperature <= 0:
            raise ValueError("temperature must be positive.")

        weights = torch.softmax(losses / temperature, dim=0)

    elif mode == "normalized":
        weights = losses / losses.sum().clamp_min(1e-12)

    elif mode == "worst":
        weights = torch.zeros_like(losses)
        weights[torch.argmax(losses)] = 1.0

    elif mode == "frank_wolfe":
        if fw_max_iter < 1:
            raise ValueError("fw_max_iter must be at least 1.")

        weights = torch.ones_like(losses) / n_groups

        # Minimize -alpha^T losses, so the FW oracle selects
        # the group with the largest loss.
        grad_alpha = -losses

        for iteration in range(fw_max_iter):
            vertex = torch.zeros_like(weights)
            vertex[torch.argmin(grad_alpha)] = 1.0

            gamma = 2.0 / (iteration + 2.0)

            if fw_max_gamma is not None:
                if not 0.0 <= fw_max_gamma <= 1.0:
                    raise ValueError("fw_max_gamma must be in [0, 1].")
                gamma = min(gamma, fw_max_gamma)

            new_weights = (
                (1.0 - gamma) * weights
                + gamma * vertex
            )

            if torch.norm(new_weights - weights) < tol:
                weights = new_weights
                break

            weights = new_weights

    else:
        raise ValueError(
            "mode must be one of: "
            "'softmax', 'normalized', 'worst', 'frank_wolfe'."
        )

    return weights


def compute_present_group_losses(
    per_sample_loss,
    sensitive_attr,
    sens_classes,
):
    """
    Compute the mean loss and sample count for every sensitive group
    present in the current batch.

    Returns:
        group_ids:
            Present sensitive-group IDs.

        group_losses:
            Mean loss of each present group.

        group_counts:
            Number of samples from each present group.
    """
    per_sample_loss = per_sample_loss.reshape(-1)
    sensitive_attr = sensitive_attr.reshape(-1).long()

    if per_sample_loss.numel() != sensitive_attr.numel():
        raise ValueError(
            "Loss and sensitive attribute must contain the same "
            "number of samples."
        )

    group_ids = []
    group_losses = []
    group_counts = []

    for group_id in range(sens_classes):
        mask = sensitive_attr == group_id

        if mask.any():
            group_ids.append(group_id)
            group_losses.append(per_sample_loss[mask].mean())
            group_counts.append(mask.sum())

    if not group_losses:
        raise RuntimeError(
            "The current batch does not contain a valid sensitive group."
        )

    return group_ids, group_losses, group_counts