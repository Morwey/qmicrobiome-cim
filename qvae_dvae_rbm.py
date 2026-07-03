# -*- coding: utf-8 -*-
# QVAE 阶段① 编码器：DVAE + RBM（离散变分自编码器 + 受限玻尔兹曼机）
# 原始实现由项目内 QVAE 负责人提供；此处为去除硬编码凭据的版本，
# 许可凭据一律经 kw.license.init 由环境注入，不写入源码。
# 负相采样：optimizer_type="sa"(kw.classical.SimulatedAnnealingOptimizer) 或 "cim"(kw.cim.CIMOptimizer)。

import anndata
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Uniform
from torch.utils.data import Dataset, DataLoader
import numpy as np
import scipy.sparse as sp
from sklearn.model_selection import train_test_split
from tqdm import tqdm
import kaiwu as kw
import time
import os
from collections import defaultdict
import logging
import copy

logger = logging.getLogger("kaiwu.common._logger")
logger.setLevel(logging.ERROR)
logger.propagate = False


def cuda_sync(device):
    if torch.cuda.is_available() and str(device).startswith("cuda"):
        torch.cuda.synchronize(device)


class SparseDataset(Dataset):
    """Sparse-aware dataset that avoids converting the full matrix to dense."""
    def __init__(self, X, batch_indices_stacked):
        """
        X: scipy sparse matrix or numpy array
        batch_indices_stacked: Tensor of shape (n_cells, n_keys)
        """
        if sp.issparse(X):
            self.X = X.tocsr()
            self.is_sparse = True
        else:
            self.X = X
            self.is_sparse = False
        self.batch_indices_stacked = batch_indices_stacked

    def __len__(self):
        return self.X.shape[0]

    def __getitem__(self, idx):
        if self.is_sparse:
            x = torch.tensor(
                np.asarray(self.X[idx].todense()).squeeze(0), dtype=torch.float32
            )
        else:
            x = torch.tensor(self.X[idx], dtype=torch.float32)
        return x, self.batch_indices_stacked[idx]


class scDataset(Dataset):
    def __init__(self, anndata_info, batch_indices_list=None):
        """
        batch_indices_list: list of LongTensors, one per batch key
        """
        self.rna_tensor = torch.tensor(anndata_info, dtype=torch.float32)

        if batch_indices_list is not None:
            self.batch_indices_list = batch_indices_list
            for i, bi in enumerate(self.batch_indices_list):
                if len(bi) != self.rna_tensor.shape[0]:
                    raise ValueError(
                        f"Length of batch_indices[{i}] must match number of samples"
                    )
        else:
            self.batch_indices_list = [
                torch.zeros(self.rna_tensor.shape[0], dtype=torch.long)
            ]

    def __len__(self):
        return self.rna_tensor.shape[0]

    def __getitem__(self, idx):
        batch_indices = torch.stack([bi[idx] for bi in self.batch_indices_list])
        return self.rna_tensor[idx, :], batch_indices


class Encoder(nn.Module):
    def __init__(self, input_dim, hidden_dim, latent_dim, normalization_method="batch"):
        super(Encoder, self).__init__()
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        if normalization_method == "batch":
            self.norm = nn.BatchNorm1d(hidden_dim)
        elif normalization_method == "layer":
            self.norm = nn.LayerNorm(hidden_dim)
        else:
            raise ValueError("normalization_method must be 'batch' or 'layer'")
        self.dropout = nn.Dropout(0.1)
        self.fc2 = nn.Linear(hidden_dim, latent_dim)

    def forward(self, x):
        h = F.relu(self.norm(self.fc1(x)))
        q_logits = self.fc2(h)
        return q_logits


class Decoder(nn.Module):
    def __init__(self, latent_dim, hidden_dim, output_dim, normalization_method="batch"):
        super(Decoder, self).__init__()
        self.fc1 = nn.Linear(latent_dim, hidden_dim)
        if normalization_method == "batch":
            self.norm = nn.BatchNorm1d(hidden_dim)
        elif normalization_method == "layer":
            self.norm = nn.LayerNorm(hidden_dim)
        else:
            raise ValueError("normalization_method must be 'batch' or 'layer'")
        self.dropout = nn.Dropout(0.1)
        self.fc2 = nn.Linear(hidden_dim, output_dim)

    def forward(self, zeta):
        h = F.relu(self.norm(self.fc1(zeta)))
        x_recon = self.fc2(h)
        return x_recon


class RBM(nn.Module):
    def __init__(self, latent_dim,
                 sample_method="ising_noise",
                 optimizer_type="cim",
                 user_id=None,
                 sdk_code=None,
                 project_no=None):
        super(RBM, self).__init__()
        self.h = nn.Parameter(torch.zeros(latent_dim))
        self.W = nn.Parameter(torch.randn(latent_dim, latent_dim) * 0.001)
        self.latent_dim = latent_dim
        self.sample_method = sample_method
        self.optimizer_type = optimizer_type
        self.ising_matrix = self._create_ising_matrix(self.latent_dim)
        self.timing = defaultdict(float)
        self.user_id = user_id
        self.sdk_code = sdk_code
        self.project_no = project_no

        if self.optimizer_type == "sa":
            self.worker = kw.classical.SimulatedAnnealingOptimizer(
                initial_temperature=1000,
                alpha=0.5,
                cutoff_temperature=0.001,
                iterations_per_t=10,
                size_limit=10,
                rand_seed=512
            )
        elif self.optimizer_type == "cim":
            kw.common.CheckpointManager.save_dir = './tmp'
            self.worker = kw.cim.CIMOptimizer(
                task_name="666",
                wait=True,
                project_no=self.project_no
            )
        else:
            raise ValueError("optimizer_type must be either 'sa' or 'cim'")

    def reset_timing(self):
        self.timing = defaultdict(float)

    def get_para(self):
        return self.W, self.h

    def energy(self, z):
        z = z.float()
        h_term = torch.sum(z * self.h, dim=-1)
        w_term = torch.sum((z @ self.W) * z, dim=-1)
        return h_term + w_term

    def _create_ising_matrix(self, number_of_hidden_units):
        W = self.W.detach().float().cpu().numpy()
        h = self.h.detach().float().cpu().numpy()

        n = number_of_hidden_units
        Wn = W[:n, :n]
        hn = h[:n]

        upper_right = 0.25 * Wn
        lower_left = 0.25 * Wn.T

        adjacency_matrix = np.block([
            [np.zeros((n, n), dtype=np.float32), upper_right.astype(np.float32)],
            [lower_left.astype(np.float32), np.zeros((n, n), dtype=np.float32)]
        ])

        bias_first = 0.5 * hn + 0.25 * Wn.sum(axis=1)
        bias_second = 0.5 * hn + 0.25 * Wn.sum(axis=0)
        bias_terms = np.concatenate([bias_first, bias_second]).astype(np.float32)

        ising_matrix = np.block([
            [adjacency_matrix, bias_terms[:, None]],
            [bias_terms[None, :], np.zeros((1, 1), dtype=np.float32)]
        ])

        return -ising_matrix

    def adjust_precision(self, ising_matrix, method='adjust'):
        if method == 'scale':
            return np.round(ising_matrix * 100, 2)
        elif method == 'adjust':
            return kw.ising.adjust_ising_matrix_precision(ising_matrix, bit_width=14)
        elif method == 'truncate':
            return np.round(ising_matrix, 2)
        else:
            print("no adjust!")
            return ising_matrix

    def ising_sampling_noise(self, number_of_samples, number_of_hidden_units):
        pass  # (debug print removed)

        total_units = number_of_hidden_units + number_of_hidden_units
        adjacency_matrix = torch.zeros((total_units, total_units), device=self.h.device)
        for i in range(number_of_hidden_units):
            for j in range(number_of_hidden_units):
                value = 0.25 * self.W[i, j]
                adjacency_matrix[i, number_of_hidden_units + j] = value
                adjacency_matrix[number_of_hidden_units + j, i] = value

        bias_terms = torch.zeros(total_units, device=self.h.device)
        for i in range(number_of_hidden_units):
            bias_terms[i] = 0.5 * self.h[i] + 0.25 * torch.sum(self.W[i, :])
        for j in range(number_of_hidden_units):
            bias_terms[number_of_hidden_units + j] = 0.5 * self.h[j] + 0.25 * torch.sum(self.W[:, j])

        self_strength = 0.9
        neighbor_strength = 0.1
        noise_strength = 0.12

        chain_state = torch.zeros(total_units, device=self.h.device)
        samples = []
        for _ in range(number_of_samples):
            chain_state = self.torch_map_clip(chain_state, adjacency_matrix, bias_terms, self_strength,
                                              neighbor_strength, noise_strength)
            samples.append(chain_state.clone())
        result = torch.stack(samples, dim=0)
        latent_variable = 0.5 * (torch.sign(result) + 1)
        return latent_variable[:, :number_of_hidden_units]

    def torch_map_clip(self, chain_state, adjacency_matrix, bias_terms, self_strength, neighbor_strength,
                       noise_strength):
        noise = torch.randn(chain_state.shape, device=chain_state.device) * noise_strength
        out = self_strength * chain_state + neighbor_strength * torch.matmul(adjacency_matrix,
                                                                             chain_state) + 0.40 * neighbor_strength * bias_terms + noise
        return torch.clamp(out, min=-0.4, max=0.4)

    def ising_sampling_sa(self, number_of_samples, number_of_hidden_units, step, behavior):
        pass  # (debug print removed)

        if hasattr(self.worker, "size_limit"):
            self.worker.size_limit = number_of_samples

        solver_name = self.optimizer_type.upper()

        t_all = time.perf_counter()

        t0 = time.perf_counter()
        ising_matrix = self._create_ising_matrix(number_of_hidden_units)
        build_t = time.perf_counter() - t0
        self.timing[f'{behavior}_build_matrix'] += build_t

        t0 = time.perf_counter()
        ising_matrix = self.adjust_precision(ising_matrix, method="adjust")
        adjust_t = time.perf_counter() - t0
        self.timing[f'{behavior}_adjust_precision'] += adjust_t
        self.ising_matrix = ising_matrix

        if hasattr(self.worker, "task_name"):
            self.worker.task_name = f"step-{step}_{behavior}_{int(time.time())}"

        t0 = time.perf_counter()
        output = None
        while output is None:
            try:
                output = list(self.worker.solve(ising_matrix))
            except ValueError as e:
                msg = str(e)
                if (
                    "Failed to retrieve task" in msg and
                    "the JSON object must be str, bytes or bytearray, not NoneType" in msg
                ):
                    time.sleep(1)
                else:
                    raise
            except Exception as e:
                print('------error-------', e)
                time.sleep(5)
        solve_time = time.perf_counter() - t0
        self.timing[f'{behavior}_solve'] += solve_time
        self.timing[f'{behavior}_calls'] += 1

        t0 = time.perf_counter()
        result = []
        for sample in output:
            sample = np.asarray(sample, dtype=np.float32).reshape(-1)
            if len(sample) > number_of_hidden_units:
                sample = sample[:-1] * sample[-1]
            sample[sample == -1] = 0
            result.append(sample[:number_of_hidden_units])

        result = np.asarray(result, dtype=np.float32)
        post_t = time.perf_counter() - t0
        self.timing[f'{behavior}_postprocess'] += post_t

        total_t = time.perf_counter() - t_all
        self.timing[f'{behavior}_total'] += total_t

        print(
            f"[{solver_name}][{behavior}] step={step} | "
            f"n_samples={number_of_samples} | matrix_shape={ising_matrix.shape} | "
            f"build={build_t:.4f}s | adjust={adjust_t:.4f}s | "
            f"solve={solve_time:.4f}s | post={post_t:.4f}s | total={total_t:.4f}s",
            flush=True
        )

        return torch.tensor(result, device=self.h.device, dtype=torch.float32)

    def compute_gradients(self, positive_latent_variable, step, number_of_negative_samples=16):
        pass  # (debug print removed)

        positive_latent_variable = positive_latent_variable.float()
        positive_hidden_gradient = positive_latent_variable.mean(dim=0)
        positive_weight_gradient = torch.einsum('bi,bj->ij', positive_latent_variable,
                                                positive_latent_variable) / positive_latent_variable.size(0)

        if self.sample_method == "ising_noise":
            negative_latent_variable = self.ising_sampling_noise(number_of_negative_samples, self.latent_dim)
        elif self.sample_method == "ising_sa":
            negative_latent_variable = self.ising_sampling_sa(
                number_of_negative_samples,
                self.latent_dim,
                step=step,
                behavior="sa_rbm_grad"
            )
        else:
            raise ValueError(f"Invalid sample method: {self.sample_method}")

        negative_hidden_gradient = negative_latent_variable.mean(dim=0)
        negative_weight_gradient = torch.einsum('bi,bj->ij', negative_latent_variable,
                                                negative_latent_variable) / negative_latent_variable.size(0)

        hidden_gradient = positive_hidden_gradient - negative_hidden_gradient
        weight_gradient = positive_weight_gradient - negative_weight_gradient
        weight_gradient = (weight_gradient + weight_gradient.T) / 2
        return {'hidden_biases': hidden_gradient, 'weights': weight_gradient}


class DVAE_RBM(nn.Module):
    def __init__(self,
                 hidden_dim=512,
                 latent_dim=256,
                 beta=0.5,
                 beta_kl=0.00001,
                 normalization_method="batch",
                 sample_method='ising_sa',
                 optimizer_type='cim',
                 user_id=None,
                 sdk_code=None,
                 project_no=None,
                 device=torch.device('cpu')):
        super(DVAE_RBM, self).__init__()
        self.hidden_dim = hidden_dim
        self.latent_dim = latent_dim
        self.beta = beta
        self.beta_kl = beta_kl
        self._base_beta_kl = beta_kl  # Store original for warmup
        self.device = device
        self.normalization_method = normalization_method
        self.sample_method = sample_method
        self.optimizer_type = optimizer_type
        self.user_id = user_id
        self.sdk_code = sdk_code
        self.project_no = project_no

        self.input_dim = None
        self.n_batches_list = None
        self.total_batch_dim = None
        self.encoder = None
        self.decoder = None
        self.rbm = None

        self.adata = None
        self.batch_keys = None
        self.batch_indices_list = None

        self.to(device)

    def set_adata(self, adata: anndata.AnnData, batch_key='batch'):
        """
        batch_key: str or list of str
        """
        self.adata = adata.copy()

        if isinstance(batch_key, str):
            self.batch_keys = [batch_key]
        elif isinstance(batch_key, (list, tuple)):
            self.batch_keys = list(batch_key)
        else:
            raise ValueError("batch_key must be a str or list of str")

        self.input_dim = adata.X.shape[1]

        self.n_batches_list = []
        self.batch_indices_list = []

        for key in self.batch_keys:
            if key not in adata.obs:
                raise ValueError(f"Batch key '{key}' not found in adata.obs")
            batch_categories = adata.obs[key].astype('category')
            indices = torch.tensor(batch_categories.cat.codes.values, dtype=torch.long)
            n_categories = len(batch_categories.cat.categories)

            self.batch_indices_list.append(indices)
            self.n_batches_list.append(n_categories)

        self.total_batch_dim = sum(self.n_batches_list)

        self.encoder = Encoder(
            self.input_dim,
            self.hidden_dim,
            self.latent_dim,
            normalization_method=self.normalization_method
        ).to(self.device)

        self.decoder = Decoder(
            self.latent_dim + self.total_batch_dim,
            self.hidden_dim,
            self.input_dim,
            normalization_method=self.normalization_method
        ).to(self.device)

        self.rbm = RBM(
            self.latent_dim,
            sample_method=self.sample_method,
            optimizer_type=self.optimizer_type,
            user_id=self.user_id,
            sdk_code=self.sdk_code,
            project_no=self.project_no
        ).to(self.device)

        pass  # (debug print removed)

    def reparameterize(self, q_logits, rho):
        q = torch.sigmoid(q_logits)
        zeta = torch.zeros_like(rho)
        mask = rho > (1 - q)
        beta_tensor = torch.tensor(self.beta, dtype=torch.float32, device=rho.device)
        exp_beta_minus_1 = torch.exp(beta_tensor) - 1
        zeta[mask] = (1 / beta_tensor) * torch.log(
            (torch.clamp(rho[mask] - (1 - q[mask]), min=0) / q[mask]) * exp_beta_minus_1 + 1
        )
        z = (zeta > 0).float()
        return zeta, z, q

    def _get_batch_one_hot_multi(self, batch_indices_stacked):
        """
        batch_indices_stacked: Tensor of shape (batch_size, n_keys)
        Returns: concatenated one-hot of shape (batch_size, total_batch_dim)
        """
        one_hots = []
        for i, n_cat in enumerate(self.n_batches_list):
            idx = batch_indices_stacked[:, i]
            oh = F.one_hot(idx, num_classes=n_cat).float()
            one_hots.append(oh)
        return torch.cat(one_hots, dim=-1).to(self.device)

    def kl_divergence(self, z, q, behavior, step):
        pass  # (debug print removed)

        q = torch.clamp(q, min=1e-7, max=1 - 1e-7)
        log_q = z * torch.log(q) + (1 - z) * torch.log(1 - q)
        entropy = -log_q.sum(dim=-1)
        energy_pos = self.rbm.energy(z)

        if self.sample_method == "ising_noise":
            z_negative = self.rbm.ising_sampling_noise(16, self.latent_dim)
        elif self.sample_method == "ising_sa":
            z_negative = self.rbm.ising_sampling_sa(z.size(0), self.latent_dim, step=step, behavior="sa_kl")
        else:
            raise ValueError(f"Invalid sample method: {self.sample_method}")

        energy_neg = self.rbm.energy(z_negative)
        logZ = energy_neg.mean()
        kl = (energy_pos - entropy + logZ).mean()
        return kl

    def forward(self, x, batch_indices_stacked, step):
        """
        batch_indices_stacked: Tensor of shape (batch_size, n_keys)
        """
        batch_one_hot = self._get_batch_one_hot_multi(batch_indices_stacked)

        q_logits = self.encoder(x)
        rho = Uniform(0, 1).sample(q_logits.shape).to(x.device)
        zeta, z, q = self.reparameterize(q_logits, rho)

        decoder_input = torch.cat([zeta, batch_one_hot], dim=-1)
        x_recon = self.decoder(decoder_input)
        recon_loss = F.mse_loss(x_recon, x, reduction='sum') / x.size(0)

        kl_loss = self.kl_divergence(z, q, behavior='_', step=step)
        elbo = -recon_loss - self.beta_kl * kl_loss
        return elbo, recon_loss, kl_loss, z, zeta

    def get_representation(self, step, adata=None, batch_size=5000):
        if adata is None and self.adata is None:
            raise ValueError("No AnnData object provided or set")
        adata = adata if adata is not None else self.adata

        # Build batch indices for potentially new adata
        batch_indices_list = []
        for key in self.batch_keys:
            if key not in adata.obs:
                raise ValueError(f"Batch key '{key}' not found in adata.obs")
            batch_categories = adata.obs[key].astype('category')
            indices = torch.tensor(batch_categories.cat.codes.values, dtype=torch.long)
            batch_indices_list.append(indices)

        batch_indices_stacked = torch.stack(batch_indices_list, dim=1)

        dataset = SparseDataset(adata.X, batch_indices_stacked)
        dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=0)

        self.eval()
        latent_reps = []
        with torch.no_grad():
            for x, batch_idx in dataloader:
                x = x.to(self.device)
                zeta = self.encoder(x)
                latent_reps.append(zeta.cpu().numpy())

        reps = np.concatenate(latent_reps, axis=0)
        pass  # (debug print removed)
        return reps

    def fit(self,
            adata,
            val_percentage=0.1,
            batch_size=128,
            epochs=100,
            lr=1e-4,
            rbm_lr=1e-3,
            early_stopping=True,
            early_stopping_patience=10,
            n_epochs_kl_warmup=None,
            verbose=0,
            ckpt_dir="./models_gastric_cancer",
            resume_ckpt_path=None):

        if adata is None and self.adata is None:
            raise ValueError("No AnnData object provided or set")
        adata = adata if adata is not None else self.adata

        # Build batch indices for all keys
        batch_indices_list = []
        for key in self.batch_keys:
            batch_categories = adata.obs[key].astype('category')
            indices = torch.tensor(batch_categories.cat.codes.values, dtype=torch.long)
            batch_indices_list.append(indices)

        batch_indices_stacked = torch.stack(batch_indices_list, dim=1)

        if early_stopping:
            train_indices, val_indices = train_test_split(
                np.arange(adata.shape[0]), test_size=val_percentage, random_state=0
            )

            train_X = adata.X[train_indices]
            val_X = adata.X[val_indices]
            train_batch_stacked = batch_indices_stacked[train_indices]
            val_batch_stacked = batch_indices_stacked[val_indices]

            val_dataset = SparseDataset(val_X, val_batch_stacked)
            val_dataloader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=0)
        else:
            train_X = adata.X
            train_batch_stacked = batch_indices_stacked

        train_dataset = SparseDataset(train_X, train_batch_stacked)
        train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=0)
        pass  # (debug print removed)

        optimizer = torch.optim.Adam(
            list(self.encoder.parameters()) + list(self.decoder.parameters()), lr=lr
        )
        rbm_optimizer = torch.optim.Adam(self.rbm.parameters(), lr=rbm_lr)

        best_val_elbo = float('-inf')
        patience_counter = 0
        best_state_dict = None
        start_epoch = 1
        step = 0

        if resume_ckpt_path is not None and os.path.exists(resume_ckpt_path):
            ckpt = torch.load(resume_ckpt_path, map_location=self.device)

            self.load_state_dict(ckpt["model_state_dict"])
            optimizer.load_state_dict(ckpt["optimizer_state_dict"])
            rbm_optimizer.load_state_dict(ckpt["rbm_optimizer_state_dict"])

            start_epoch = ckpt["epoch"] + 1
            step = ckpt.get("step", 0)
            best_val_elbo = ckpt.get("best_val_elbo", float('-inf'))
            patience_counter = ckpt.get("patience_counter", 0)
            best_state_dict = ckpt.get("best_state_dict", None)

            print(f"Resume training from {resume_ckpt_path}, start_epoch={start_epoch}")

        epoch_pbar = tqdm(range(start_epoch, epochs + 1), desc="Training Progress", total=epochs - start_epoch + 1)

        kl_warmup_epochs = n_epochs_kl_warmup if n_epochs_kl_warmup is not None else 0

        intermediate_results = {
            'all_train_elbo': [],
            'all_train_recon_loss': [],
            'all_train_kl': [],
            'all_val_elbo': [],
            'all_val_recon_loss': [],
            'all_val_kl': [],
            'time': [],
            'epoch_timing': []
        } if verbose == 1 else None

        for epoch in epoch_pbar:
            # --- KL warmup ---
            if kl_warmup_epochs > 0 and epoch <= kl_warmup_epochs:
                self.beta_kl = self._base_beta_kl * (epoch / kl_warmup_epochs)
            else:
                self.beta_kl = self._base_beta_kl

            self.rbm.reset_timing()
            epoch_timing = defaultdict(float)
            epoch_start_time = time.perf_counter()
            self.train()
            total_elbo, total_recon, total_kl = 0, 0, 0
            for x, batch_idx in train_dataloader:
                batch_start = time.perf_counter()

                t0 = time.perf_counter()
                x = x.to(self.device)
                batch_idx = batch_idx.to(self.device)
                cuda_sync(self.device)
                epoch_timing['data_to_device'] += time.perf_counter() - t0

                optimizer.zero_grad()
                rbm_optimizer.zero_grad()

                cuda_sync(self.device)
                t0 = time.perf_counter()
                elbo, recon_loss, kl_loss, z, zeta = self(x, batch_idx, step=step)
                cuda_sync(self.device)
                epoch_timing['forward_total'] += time.perf_counter() - t0

                cuda_sync(self.device)
                t0 = time.perf_counter()
                loss = -elbo
                loss.backward()
                cuda_sync(self.device)
                epoch_timing['backward'] += time.perf_counter() - t0

                t0 = time.perf_counter()
                rbm_grads = self.rbm.compute_gradients(z.detach(), step=step)
                epoch_timing['rbm_grad_total'] += time.perf_counter() - t0

                with torch.no_grad():
                    self.rbm.h.grad = rbm_grads['hidden_biases']
                    self.rbm.W.grad = rbm_grads['weights']

                cuda_sync(self.device)
                t0 = time.perf_counter()
                optimizer.step()
                rbm_optimizer.step()
                cuda_sync(self.device)
                epoch_timing['optimizer_step'] += time.perf_counter() - t0

                step += 1

                total_elbo += elbo.item()
                total_recon += recon_loss.item()
                total_kl += kl_loss.item()

                batch_total = time.perf_counter() - batch_start
                epoch_timing['batch_total'] += batch_total

                if verbose == 1:
                    intermediate_results['time'].append(batch_total)

            avg_elbo = total_elbo / len(train_dataloader)
            avg_recon = total_recon / len(train_dataloader)
            avg_kl = total_kl / len(train_dataloader)
            epoch_pbar.set_postfix({
                'KL_weight': f'{self.beta_kl:.6f}',
                'ELBO': f'{avg_elbo:.4f}',
                'Recon': f'{avg_recon:.4f}',
                'KL': f'{avg_kl:.4f}'
            })

            epoch_timing['epoch_total'] = time.perf_counter() - epoch_start_time

            for k, v in self.rbm.timing.items():
                epoch_timing[k] += v

            if verbose == 1:
                intermediate_results['epoch_timing'].append(dict(epoch_timing))

            print(
                f"[Epoch {epoch}] "
                f"beta_kl={self.beta_kl:.6f} | "
                f"epoch_total={epoch_timing['epoch_total']:.2f}s | "
                f"forward={epoch_timing['forward_total']:.2f}s | "
                f"backward={epoch_timing['backward']:.2f}s | "
                f"rbm_grad={epoch_timing['rbm_grad_total']:.2f}s | "
                f"sa_kl={epoch_timing.get('sa_kl_solve', 0.0):.2f}s | "
                f"sa_rbm_grad={epoch_timing.get('sa_rbm_grad_solve', 0.0):.2f}s"
            )

            if verbose == 1:
                intermediate_results['all_train_elbo'].append(avg_elbo)
                intermediate_results['all_train_recon_loss'].append(avg_recon)
                intermediate_results['all_train_kl'].append(avg_kl)

            if early_stopping:
                self.eval()
                val_total_elbo, val_total_recon, val_total_kl = 0, 0, 0
                for x, batch_idx in val_dataloader:
                    x = x.to(self.device)
                    batch_idx = batch_idx.to(self.device)
                    with torch.no_grad():
                        elbo, recon_loss, kl_loss, z, zeta = self(x, batch_idx, step=step)
                    val_total_elbo += elbo.item()
                    val_total_recon += recon_loss.item()
                    val_total_kl += kl_loss.item()

                avg_val_elbo = val_total_elbo / len(val_dataloader)
                avg_val_recon = val_total_recon / len(val_dataloader)
                avg_val_kl = val_total_kl / len(val_dataloader)

                if verbose == 1:
                    intermediate_results['all_val_elbo'].append(avg_val_elbo)
                    intermediate_results['all_val_recon_loss'].append(avg_val_recon)
                    intermediate_results['all_val_kl'].append(avg_val_kl)

                if avg_val_elbo > best_val_elbo:
                    best_val_elbo = avg_val_elbo
                    patience_counter = 0
                    best_state_dict = copy.deepcopy(self.state_dict())
                else:
                    patience_counter += 1

                os.makedirs(ckpt_dir, exist_ok=True)
                ckpt_path = os.path.join(ckpt_dir, f"model_epoch{epoch}.pth")

                torch.save({
                    "epoch": epoch,
                    "step": step,
                    "model_state_dict": self.state_dict(),
                    "optimizer_state_dict": optimizer.state_dict(),
                    "rbm_optimizer_state_dict": rbm_optimizer.state_dict(),
                    "scheduler_state_dict": None,
                    "best_val_elbo": best_val_elbo,
                    "patience_counter": patience_counter,
                    "best_state_dict": best_state_dict,
                }, ckpt_path)

                if early_stopping and patience_counter >= early_stopping_patience:
                    tqdm.write(f"Early stopping triggered after {epoch} epochs")
                    if best_state_dict is not None:
                        self.load_state_dict(best_state_dict)
                    epoch_pbar.close()
                    break

        epoch_pbar.close()

        # Restore original beta_kl
        self.beta_kl = self._base_beta_kl

        if verbose == 1:
            return intermediate_results
        else:
            return None
