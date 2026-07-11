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

from models.SAMFW.utils import (
    compute_present_group_losses,
    fw_lmo_vertex,
    fw_online_update,
)


class SAMFW(BaseNet):
    """
    Online (persistent) Frank-Wolfe group-fairness weighting composed with SAM,
    implementing the "Group Fairness via Sharpness-Aware Minimization and
    Frank-Wolfe Optimization" design (Eq. 1-5): a single fairness-weight vector
    beta evolves across the entire training run via one Frank-Wolfe step per
    batch on the *perturbed*-point group losses, rather than being reset every
    batch (SAMMOO) or solved from gradients (MGDASAM). Generalized here from
    the paper's two-group beta in [0,1] to a sens_classes-dimensional simplex.
    See docs/SAMFW.md for the full formulation and a documented deviation
    (beta_reset_every) from the literal algorithm.
    """

    def __init__(self, opt, wandb):
        super(SAMFW, self).__init__(opt, wandb)

        self.beta_reset_every = opt["beta_reset_every"]
        self.beta = torch.full(
            (self.sens_classes,), 1.0 / self.sens_classes, device=self.device
        )
        self.fw_step = 0

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
            "beta": self.beta,
            "fw_step": self.fw_step,
        }

    def _weighted_combined_loss(self, outputs, targets, sensitive_attr, beta):
        """
        Eq. 1's inner objective for the present groups, weighted by beta
        (renormalized over present groups; a no-op whenever every group is
        present, which is the paper's own binary two-groups-always-present
        setting).
        """
        per_sample_loss = self._criterion(outputs, targets).reshape(-1)

        group_ids, group_losses, group_counts = compute_present_group_losses(
            per_sample_loss=per_sample_loss,
            sensitive_attr=sensitive_attr,
            sens_classes=self.sens_classes,
        )

        present_beta = torch.stack([beta[g] for g in group_ids])
        present_beta = present_beta / present_beta.sum()

        combined_loss = torch.stack([
            w * group_loss for w, group_loss in zip(present_beta, group_losses)
        ]).sum()

        return combined_loss, group_ids, group_losses

    def _train(self, loader):
        """Train online-Frank-Wolfe-weighted SAM for one epoch."""
        self.network.train()

        if self.beta_reset_every and self.beta_reset_every > 0:
            if self.epoch % self.beta_reset_every == 0:
                self.fw_step = 0

        total_loss = 0.0
        auc_sum = 0.0
        auc_batches = 0
        no_iter = 0
        skipped_fw_batches = 0

        for i, (images, targets, sensitive_attr, index) in enumerate(loader):
            images = images.to(self.device)
            targets = targets.to(self.device)
            sensitive_attr = sensitive_attr.to(self.device)

            # =====================================================
            # Step 1 (Eq. 2): SAM perturbation using the current beta_t
            # =====================================================
            enable_running_stats(self.network)
            outputs, _ = self.network(images)

            combined_loss, group_ids, group_losses = self._weighted_combined_loss(
                outputs, targets, sensitive_attr, self.beta
            )

            combined_loss.backward()
            self.optimizer.first_step(zero_grad=True)

            # =====================================================
            # Step 2 (Eq. 3-4): online Frank-Wolfe update for beta, using
            # perturbed-point group losses
            # =====================================================
            disable_running_stats(self.network)
            outputs_adv, _ = self.network(images)

            per_sample_loss_adv = self._criterion(outputs_adv, targets).reshape(-1)
            group_ids_adv, group_losses_adv, group_counts_adv = compute_present_group_losses(
                per_sample_loss=per_sample_loss_adv,
                sensitive_attr=sensitive_attr,
                sens_classes=self.sens_classes,
            )

            if len(group_ids_adv) >= 2:
                vertex = fw_lmo_vertex(self.sens_classes, group_ids_adv, group_losses_adv)
                gamma_t = 2.0 / (self.fw_step + 2.0)
                self.beta = fw_online_update(self.beta, vertex, gamma_t)
                self.fw_step += 1
            else:
                skipped_fw_batches += 1

            # =====================================================
            # Step 3 (Eq. 5): weight update using the just-updated beta_{t+1}
            # =====================================================
            present_beta_adv = torch.stack([self.beta[g] for g in group_ids_adv])
            present_beta_adv = present_beta_adv / present_beta_adv.sum()

            combined_loss_adv = torch.stack([
                w * group_loss
                for w, group_loss in zip(present_beta_adv, group_losses_adv)
            ]).sum()

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

            if self.log_freq and i % self.log_freq == 0:
                log_dict = {"Training loss": total_loss / no_iter}
                for g in range(self.sens_classes):
                    log_dict[f"beta_{g}"] = self.beta[g].item()
                self.log_wandb(log_dict)

        average_loss = total_loss / max(no_iter, 1)

        if auc_batches > 0:
            average_auc = 100.0 * auc_sum / auc_batches
        else:
            average_auc = float("nan")

        print(f"Training epoch {self.epoch}: AUC:{average_auc}")
        print(f"Training epoch {self.epoch}: weighted loss:{average_loss}")

        # For Sex in MEDFAIR: group 0 = Male, group 1 = Female
        group_name_map = {0: "Male", 1: "Female"}
        for g in range(self.sens_classes):
            group_name = group_name_map.get(g, f"Group {g}")
            print(f"{group_name}: beta={self.beta[g].item():.6f}")

        reset_desc = self.beta_reset_every if self.beta_reset_every else "never"
        print(f"fw_step: {self.fw_step} (resets every {reset_desc} epoch(s))")
        print(
            "Batches skipped for the beta update (fewer than 2 groups present):",
            skipped_fw_batches,
            "/",
            no_iter,
        )

        self.epoch += 1
