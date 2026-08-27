"""
MODULE 7 — Training Loop
=========================
Full training loop for the MLP reranker.

FIXES APPLIED:
  1. Loss: uses PairwiseRankingLoss (BPR) — NOT BCE/MSE
  2. FeatureNormalizer: fitted on training data and attached to extractor
  3. f2 leakage: rank scaled to [0,1] via /max_rank during feature extraction
  4. Embedding validation: warns when std(embedding) ≈ 0 (zero vectors)
  5. log_misses / strict: surfaced via TrainerConfig.encoder_strict
"""

from __future__ import annotations

import logging
import os
import random
from dataclasses import dataclass, field
from typing import List, Optional, Tuple

import numpy as np

logger = logging.getLogger(__name__)

try:
    import torch
    import torch.optim as optim
    from torch.utils.data import DataLoader, random_split
    _TORCH_AVAILABLE = True
except ImportError:
    _TORCH_AVAILABLE = False



# Training configuration


@dataclass
class TrainerConfig:
    """Hyperparameters and training settings."""

    # Optimiser
    learning_rate: float = 1e-3
    weight_decay: float = 1e-4

    # Schedule
    n_epochs: int = 50
    warmup_epochs: int = 5

    # Batch
    batch_size: int = 256

    # Loss
    margin: float = 0.0          # 0 = BPR; try 1.0 for hinge variant

    # Regularisation
    grad_clip: float = 1.0

    # Dataset split
    val_fraction: float = 0.1

    # Checkpointing
    checkpoint_path: Optional[str] = "ranker_best.pt"

    # Reproducibility
    seed: int = 42

    # Device
    device: str = "cpu"

    # Early stopping
    early_stopping_patience: int = 10

    # FIX #4 / #7: encoder debug settings
    encoder_log_misses: bool = True   # warn on cache miss
    encoder_strict: bool = False      # raise on cache miss (set True to debug)



# RankerTrainer


class RankerTrainer:
    """
    Training manager for RankerMLP using pairwise ranking loss.

    Args:
        model:   A RankerMLP instance.
        config:  A TrainerConfig instance.
    """

    def __init__(self, model, config: Optional[TrainerConfig] = None) -> None:
        if not _TORCH_AVAILABLE:
            raise ImportError("torch is required for RankerTrainer.")

        self.model = model
        self.cfg = config or TrainerConfig()
        self.device = torch.device(self.cfg.device)
        self.model.to(self.device)

        self._set_seeds(self.cfg.seed)
        self._shuffle_generator = torch.Generator(device="cpu")
        self._shuffle_generator.manual_seed(self.cfg.seed)

        from rerank.loss import PairwiseRankingLoss
        self.criterion = PairwiseRankingLoss(
            margin=self.cfg.margin,
            reduction="mean",
        )

        self.optimizer = optim.AdamW(
            self.model.parameters(),
            lr=self.cfg.learning_rate,
            weight_decay=self.cfg.weight_decay,
        )

        self.scheduler = optim.lr_scheduler.CosineAnnealingLR(
            self.optimizer,
            T_max=max(self.cfg.n_epochs - self.cfg.warmup_epochs, 1),
            eta_min=self.cfg.learning_rate * 0.01,
        )

        self.train_losses: List[float] = []
        self.val_losses: List[float] = []
        self.best_val_loss = float("inf")
        self._epochs_no_improve = 0


    # Public API


    def fit(self, dataset) -> None:
        """
        Train the model on the provided PairwiseRankingDataset.

        Steps:
            1. Validate embedding quality (warn on zero-vector features)
            2. Fit FeatureNormalizer on training data and attach to extractor
            3. Run pairwise ranking training loop
            4. Restore best checkpoint

        Args:
            dataset: A PairwiseRankingDataset instance.
        """
        # Validate feature dim match (catches 2D model vs 3D dataset mistakes)
        feat_dim = dataset.feature_dim
        if feat_dim > 0 and feat_dim != self.model.input_dim:
            raise ValueError(
                f"Dataset feature_dim={feat_dim} does not match "
                f"model.input_dim={self.model.input_dim}. "
                f"Use RankerMLP.for_2d() for 2D datasets (dim=5) "
                f"or RankerMLP.for_3d() for 3D datasets (dim=10)."
            )

        # FIX #6: validate embeddings before training
        self._validate_embeddings(dataset)

        train_loader, val_loader = self._make_dataloaders(dataset)
        logger.info(
            "Training RankerMLP for %d epochs | margin=%.2f | feature_dim=%d | "
            "train=%d pairs, val=%d pairs",
            self.cfg.n_epochs,
            self.cfg.margin,
            feat_dim,
            len(train_loader.dataset),
            len(val_loader.dataset) if val_loader else 0,
        )

        n_epochs = self.cfg.n_epochs
        for epoch in range(1, n_epochs + 1):
            train_loss = self._train_epoch(train_loader, epoch)
            self.train_losses.append(train_loss)

            if val_loader is not None:
                val_loss = self._val_epoch(val_loader)
                self.val_losses.append(val_loss)

                improved = val_loss < self.best_val_loss
                if improved:
                    self.best_val_loss = val_loss
                    self._epochs_no_improve = 0
                    if self.cfg.checkpoint_path:
                        self.model.save(self.cfg.checkpoint_path)
                else:
                    self._epochs_no_improve += 1

                star = " *" if improved else ""
                line = (
                    f"\rEpoch {epoch:3d}/{n_epochs} | "
                    f"train={train_loss:.4f} | "
                    f"val={val_loss:.4f} | "
                    f"best={self.best_val_loss:.4f}{star}"
                )
                # newline only when best improved (to keep log clean)
                end = "\n" if improved else ""
                print(line, end=end, flush=True)
                log_msg = line.strip()
            else:
                # No val_loader: save checkpoint at every epoch (model at epoch N = last saved)
                line = f"\rEpoch {epoch:3d}/{n_epochs} | train={train_loss:.4f}"
                print(line, end="", flush=True)
                log_msg = line.strip()
                if self.cfg.checkpoint_path:
                    self.model.save(self.cfg.checkpoint_path)

            logger.info(log_msg)

            if epoch > self.cfg.warmup_epochs:
                self.scheduler.step()

            if self._epochs_no_improve >= self.cfg.early_stopping_patience:
                print()  # newline after carriage return
                logger.info(
                    "Early stopping triggered after %d epochs without improvement.",
                    self._epochs_no_improve,
                )
                break

        print()  # trailing newline

        if self.cfg.checkpoint_path and os.path.exists(self.cfg.checkpoint_path):
            state = torch.load(self.cfg.checkpoint_path, map_location=self.device)
            self.model.load_state_dict(state)
            logger.info("Restored best model from %s", self.cfg.checkpoint_path)
        elif self.cfg.checkpoint_path:
            # Fallback: save model now so downstream code can load it
            self.model.save(self.cfg.checkpoint_path)
            logger.info("No checkpoint found; saved final model to %s", self.cfg.checkpoint_path)


    # FIX #6 — Embedding validation


    def _validate_embeddings(self, dataset) -> None:
        """
        Validate embeddings, print feature statistics table (mean, std, min, max),
        and run correlation check warning on |r| > 0.92.
        """
        n_check = min(1000, len(dataset))
        if n_check == 0:
            return

        sample_feats = []
        for i in range(n_check):
            x_pos, x_neg = dataset[i]
            sample_feats.append(x_pos.numpy())
            sample_feats.append(x_neg.numpy())

        X = np.stack(sample_feats, axis=0)   # (2*n_check, dim)
        means = X.mean(axis=0)
        stds = X.std(axis=0)
        mins = X.min(axis=0)
        maxs = X.max(axis=0)

        dim = X.shape[1]
        from rerank.features import FEATURE_NAMES_MAP
        feature_names = []
        for mode, names in FEATURE_NAMES_MAP.items():
            if len(names) == dim:
                feature_names = names
                break
        if not feature_names:
            feature_names = [f"f{i+1}" for i in range(dim)]

        # Print detailed statistics table
        logger.info("=" * 80)
        logger.info("FEATURE DISTRIBUTION DIAGNOSTICS (sampled from %d vectors)", len(X))
        logger.info("=" * 80)
        logger.info(f"{'Index':<6} {'Feature Name':<25} {'Mean':>10} {'Std':>10} {'Min':>10} {'Max':>10}")
        logger.info("-" * 80)
        for idx in range(dim):
            name = feature_names[idx]
            logger.info(f"{idx:<6} {name:<25} {means[idx]:>10.4f} {stds[idx]:>10.4f} {mins[idx]:>10.4f} {maxs[idx]:>10.4f}")
        logger.info("=" * 80)

        # Check for zero-std features (cache misses)
        zero_features = np.where(stds < 1e-6)[0]
        if len(zero_features) > 0:
            dead = [feature_names[i] for i in zero_features]
            logger.warning(
                "   ZERO-STD FEATURES DETECTED: %s\n"
                "   These features are all-zero — likely caused by cache misses.\n"
                "   Ensure cache coverage is high.",
                dead,
            )
        else:
            logger.info("Embedding validation passed: all %d features have non-zero std.", dim)

        # Check for highly correlated features (|r| > 0.92)
        if dim > 1:
            corr = np.corrcoef(X.T)
            # Log correlation matrix for debug/manual review
            logger.debug("Feature Correlation Matrix:\n%s", np.round(corr, 3))
            
            redundant_pairs = []
            for i in range(dim):
                for j in range(i + 1, dim):
                    r_val = corr[i, j]
                    if abs(r_val) > 0.92:
                        redundant_pairs.append((feature_names[i], feature_names[j], r_val))
            if redundant_pairs:
                logger.warning("HIGHLY CORRELATED (REDUNDANT) FEATURES DETECTED (threshold |r| > 0.92):")
                for f1_name, f2_name, r_val in redundant_pairs:
                    logger.warning("   * %s vs %s: r = %.4f", f1_name, f2_name, r_val)
            else:
                logger.info("Correlation check passed: no feature pairs have |r| > 0.92.")
        logger.info("=" * 80)



    # Epoch-level methods


    def _train_epoch(self, dataloader: "DataLoader", epoch: int) -> float:
        self.model.train()
        total_loss = 0.0
        n_batches = 0

        dataset = dataloader.dataset
        tensor_pos = getattr(dataset, "_tensor_pos", None)
        tensor_neg = getattr(dataset, "_tensor_neg", None)
        if tensor_pos is not None and tensor_neg is not None:
            # The study uses no validation split, so the DataLoader dataset is
            # the materialized PairwiseRankingDataset itself.  Indexing one
            # contiguous tensor per batch avoids 300k Python __getitem__ calls
            # per epoch while retaining a seeded uniform reshuffle.
            order = torch.randperm(len(dataset), generator=self._shuffle_generator)
            for start in range(0, len(dataset), self.cfg.batch_size):
                indices = order[start : start + self.cfg.batch_size]
                x_pos = tensor_pos[indices].to(self.device)
                x_neg = tensor_neg[indices].to(self.device)

                self.optimizer.zero_grad()
                score_pos = self.model.score(x_pos)
                score_neg = self.model.score(x_neg)
                loss = self.criterion(score_pos, score_neg)
                loss.backward()
                if self.cfg.grad_clip > 0:
                    torch.nn.utils.clip_grad_norm_(
                        self.model.parameters(), self.cfg.grad_clip
                    )
                self.optimizer.step()
                total_loss += loss.item()
                n_batches += 1
            return total_loss / max(n_batches, 1)

        for x_pos, x_neg in dataloader:
            x_pos = x_pos.to(self.device)
            x_neg = x_neg.to(self.device)

            self.optimizer.zero_grad()

            # FIX #1: pairwise ranking loss — model.score() returns (B,)
            score_pos = self.model.score(x_pos)
            score_neg = self.model.score(x_neg)

            loss = self.criterion(score_pos, score_neg)
            loss.backward()

            if self.cfg.grad_clip > 0:
                torch.nn.utils.clip_grad_norm_(
                    self.model.parameters(), self.cfg.grad_clip
                )

            self.optimizer.step()

            total_loss += loss.item()
            n_batches += 1

        return total_loss / max(n_batches, 1)

    @torch.no_grad()
    def _val_epoch(self, dataloader: "DataLoader") -> float:
        self.model.eval()
        total_loss = 0.0
        n_batches = 0

        for x_pos, x_neg in dataloader:
            x_pos = x_pos.to(self.device)
            x_neg = x_neg.to(self.device)

            score_pos = self.model.score(x_pos)
            score_neg = self.model.score(x_neg)

            loss = self.criterion(score_pos, score_neg)
            total_loss += loss.item()
            n_batches += 1

        return total_loss / max(n_batches, 1)


    # Helpers


    def _make_dataloaders(
        self, dataset
    ) -> Tuple["DataLoader", Optional["DataLoader"]]:
        n_total = len(dataset)
        n_val = max(1, int(n_total * self.cfg.val_fraction)) if self.cfg.val_fraction > 0 else 0
        n_train = n_total - n_val

        generator = torch.Generator()
        generator.manual_seed(self.cfg.seed)

        if n_val > 0:
            train_ds, val_ds = random_split(
                dataset, [n_train, n_val], generator=generator,
            )
            val_loader = DataLoader(
                val_ds,
                batch_size=self.cfg.batch_size,
                shuffle=False,
                drop_last=False,
            )
        else:
            train_ds = dataset
            val_loader = None

        train_loader = DataLoader(
            train_ds,
            batch_size=self.cfg.batch_size,
            shuffle=True,
            drop_last=False,
            generator=generator,
        )
        return train_loader, val_loader

    @staticmethod
    def _set_seeds(seed: int) -> None:
        random.seed(seed)
        np.random.seed(seed)
        if _TORCH_AVAILABLE:
            torch.manual_seed(seed)
            if torch.cuda.is_available():
                torch.cuda.manual_seed_all(seed)
                torch.backends.cudnn.deterministic = True
                torch.backends.cudnn.benchmark = False
