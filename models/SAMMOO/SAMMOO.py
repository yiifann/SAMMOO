import numpy as np
import torch
import torch.nn as nn

from importlib import import_module
from torch.optim.lr_scheduler import CosineAnnealingLR

from utils.evaluation import calculate_auc
from models.basenet import BaseNet

from models.SAM.utils import (
    SAM_optimizer,
    disable_running_stats,
    enable_running_stats,
)

from models.SAMMOO.utils import (
    compute_group_loss_weights,
    compute_present_group_losses,
)


class SAMMOO(BaseNet):
    def __init__(self, opt, wandb):
        super(SAMMOO, self).__init__(opt, wandb)

        self.alpha_mode = opt["alpha_mode"]
        self.temperature = opt["temperature"]
        self.fw_max_iter = opt["fw_max_iter"]
        self.fw_max_gamma = opt["fw_max_gamma"]
        self.recompute_alpha_at_adv = opt["recompute_alpha_at_adv"]

        self.set_network(opt)
        self.set_optimizer(opt)

    def set_network(self, opt):
        """Define the classification network."""
        if self.is_3d:
            mod = import_module("models.basemodels_3d")
            model_class = getattr(mod, self.backbone)
            self.network = model_class(
                n_classes=self.output_dim,
                pretrained=self.pretrained,
            ).to(self.device)

        elif self.is_tabular:
            mod = import_module("models.basemodels_mlp")
            model_class = getattr(mod, self.backbone)
            self.network = model_class(
                n_classes=self.output_dim,
                in_features=self.in_features,
                hidden_features=1024,
            ).to(self.device)

        else:
            mod = import_module("models.basemodels")
            model_class = getattr(mod, self.backbone)
            self.network = model_class(
                n_classes=self.output_dim,
                pretrained=self.pretrained,
            ).to(self.device)

    def set_optimizer(self, opt):
        optimizer_setting = opt["optimizer_setting"]

        self.base_optimizer = torch.optim.Adam

        self.optimizer = SAM_optimizer(
            params=self.network.parameters(),
            base_optimizer=self.base_optimizer,
            rho=opt["rho"],
            adaptive=opt["adaptive"],
            lr=optimizer_setting["lr"],
            weight_decay=optimizer_setting["weight_decay"],
        )

        # Preserve the original MEDFAIR SAM scheduler behavior.
        self.scheduler = CosineAnnealingLR(
            self.optimizer.base_optimizer,
            T_max=opt["T_max"],
        )

    def _criterion(self, output, target):
        criterion = nn.BCEWithLogitsLoss(reduction="none")
        return criterion(output, target)

    def state_dict(self):
        return {
            "model": self.network.state_dict(),
            "optimizer": self.optimizer.state_dict(),
            "epoch": self.epoch,
        }

    def _weighted_group_loss(
        self,
        outputs,
        targets,
        sensitive_attr,
        fixed_weights=None,
    ):
        """
        Compute group losses and their weighted combination.
        """
        per_sample_loss = self._criterion(
            outputs,
            targets,
        ).reshape(-1)

        group_ids, group_losses, group_counts = compute_present_group_losses(
            per_sample_loss=per_sample_loss,
            sensitive_attr=sensitive_attr,
            sens_classes=self.sens_classes,
        )

        if fixed_weights is None:
            if self.alpha_mode == "sample_mean":
                # Ordinary sample-level mean loss.
                #
                # Example:
                # 24 male + 8 female samples
                # alpha_male = 24 / 32 = 0.75
                # alpha_female = 8 / 32 = 0.25
                counts = torch.stack([
                    count.to(
                        device=per_sample_loss.device,
                        dtype=per_sample_loss.dtype,
                    )
                    for count in group_counts
                ])

                weights = counts / counts.sum()

            else:
                weights = compute_group_loss_weights(
                    group_losses=group_losses,
                    mode=self.alpha_mode,
                    temperature=self.temperature,
                    fw_max_iter=self.fw_max_iter,
                    fw_max_gamma=self.fw_max_gamma,
                )
        else:
            weights = fixed_weights

        if len(weights) != len(group_losses):
            raise RuntimeError(
                "The number of weights does not match the "
                "number of groups present in the batch."
            )

        combined_loss = torch.stack([
            weight * group_loss
            for weight, group_loss in zip(weights, group_losses)
        ]).sum()

        return combined_loss, group_ids, group_losses, weights

    def _train(self, loader):
        """Train SAM + loss-based group weighting for one epoch."""
        self.network.train()

        total_loss = 0.0
        auc_sum = 0.0
        auc_batches = 0
        no_iter = 0

        group_loss_sums = {
            group_id: 0.0
            for group_id in range(self.sens_classes)
        }

        alpha_sums = {
            group_id: 0.0
            for group_id in range(self.sens_classes)
        }

        group_batch_counts = {
            group_id: 0
            for group_id in range(self.sens_classes)
        }

        missing_group_batches = 0

        for i, (
            images,
            targets,
            sensitive_attr,
            index,
        ) in enumerate(loader):

            images = images.to(self.device)
            targets = targets.to(self.device)
            sensitive_attr = sensitive_attr.to(self.device)

            # =====================================================
            # SAM step 1: loss and gradient at original weights
            # =====================================================
            enable_running_stats(self.network)

            outputs, _ = self.network(images)

            (
                combined_loss,
                group_ids,
                group_losses,
                weights,
            ) = self._weighted_group_loss(
                outputs=outputs,
                targets=targets,
                sensitive_attr=sensitive_attr,
            )

            if len(group_ids) < self.sens_classes:
                missing_group_batches += 1

            combined_loss.backward()

            self.optimizer.first_step(zero_grad=True)
            self.scheduler.step()

            # =====================================================
            # SAM step 2: loss and gradient at perturbed weights
            # =====================================================
            disable_running_stats(self.network)

            outputs_adv, _ = self.network(images)

            if self.recompute_alpha_at_adv:
                (
                    combined_loss_adv,
                    group_ids_adv,
                    group_losses_adv,
                    weights_adv,
                ) = self._weighted_group_loss(
                    outputs=outputs_adv,
                    targets=targets,
                    sensitive_attr=sensitive_attr,
                )
            else:
                (
                    combined_loss_adv,
                    group_ids_adv,
                    group_losses_adv,
                    weights_adv,
                ) = self._weighted_group_loss(
                    outputs=outputs_adv,
                    targets=targets,
                    sensitive_attr=sensitive_attr,
                    fixed_weights=weights,
                )

            if group_ids_adv != group_ids:
                raise RuntimeError(
                    "Sensitive groups changed between the two SAM steps."
                )

            combined_loss_adv.backward()

            self.optimizer.second_step(zero_grad=True)
            self.scheduler.step()

            # =====================================================
            # Logging
            # =====================================================
            batch_auc = calculate_auc(
                torch.sigmoid(outputs_adv).detach().cpu().numpy(),
                targets.detach().cpu().numpy(),
            )

            if not np.isnan(batch_auc):
                auc_sum += batch_auc
                auc_batches += 1

            total_loss += combined_loss.item()
            no_iter += 1

            for local_idx, group_id in enumerate(group_ids):
                group_loss_sums[group_id] += (
                    group_losses[local_idx].item()
                )

                alpha_sums[group_id] += weights[local_idx].item()
                group_batch_counts[group_id] += 1

            if self.log_freq and i % self.log_freq == 0:
                log_dict = {
                    "Training loss": total_loss / no_iter,
                }

                for group_id in range(self.sens_classes):
                    count = group_batch_counts[group_id]

                    if count > 0:
                        log_dict[
                            f"Group {group_id} loss"
                        ] = group_loss_sums[group_id] / count

                        log_dict[
                            f"Group {group_id} alpha"
                        ] = alpha_sums[group_id] / count

                self.log_wandb(log_dict)

        average_loss = total_loss / max(no_iter, 1)

        if auc_batches > 0:
            average_auc = 100.0 * auc_sum / auc_batches
        else:
            average_auc = float("nan")

        print(
            f"Training epoch {self.epoch}: "
            f"AUC:{average_auc}"
        )

        print(
            f"Training epoch {self.epoch}: "
            f"weighted loss:{average_loss}"
        )

        # For Sex in MEDFAIR:
        # group 0 = Male, group 1 = Female
        for group_id in range(self.sens_classes):
            count = group_batch_counts[group_id]

            if count == 0:
                continue

            average_group_loss = (
                group_loss_sums[group_id] / count
            )

            average_alpha = (
                alpha_sums[group_id] / count
            )

            group_name = {
                0: "Male",
                1: "Female",
            }.get(group_id, f"Group {group_id}")

            print(
                f"{group_name}: "
                f"loss={average_group_loss:.6f}, "
                f"alpha={average_alpha:.6f}"
            )

        print(
            "Batches missing at least one sensitive group:",
            missing_group_batches,
            "/",
            no_iter,
        )

        self.epoch += 1
