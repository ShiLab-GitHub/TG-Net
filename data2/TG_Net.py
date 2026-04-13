#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
TG-Net: A Token-Guided Multi-Scale Neural Network for Cell Type Annotation in scRNA-seq.
Includes extended hyperparameter search, cosine annealing, gradient clipping,
and better defaults for large datasets (e.g., Zheng68K).

Tunable coefficients:
    --residual-guidance-weight (α) : residual guidance weight in encoder
    --latent-modulation-weight (γ) : latent modulation weight before decoder
"""

import os
import sys
import argparse
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    accuracy_score, confusion_matrix,
    f1_score, precision_score, recall_score,
    balanced_accuracy_score
)
from sklearn.manifold import TSNE
from collections import Counter, defaultdict
import warnings
import copy
from tqdm import tqdm
import time
import gc
import random
import scanpy as sc
import matplotlib.pyplot as plt
import seaborn as sns
from upsetplot import UpSet
from matplotlib_venn import venn2, venn3
import torch_geometric
from torch_geometric.data import Data as PyGData

warnings.filterwarnings('ignore')

# ====================== Configuration ======================
BASE_DATA_DIR = "dataset/pre_data/scRNAseq_datasets/"
BASE_5FOLD_DIR = "dataset/5fold_data/"
BASE_WCSN_DIR = "dataset/5fold_data/"

# Extended hyperparameter search space (optimized for large-scale datasets)
PARAM_RANGES = {
    'latent_dim': [128, 256, 384, 512],
    'token_dim': [8, 12, 16],
    'recon_weight': [0.02, 0.05, 0.08, 0.1],
    'layer_cls_weight': [0.6, 0.8, 1.0],
    'hidden_dims': ['1024,512', '2048,1024', '2048,1024,512'],
    'encoder_hidden_dims': ['2048,1024', '1024,512', '512']
}


def sample_random_params():
    return {
        'latent_dim': random.choice(PARAM_RANGES['latent_dim']),
        'token_dim': random.choice(PARAM_RANGES['token_dim']),
        'recon_weight': random.choice(PARAM_RANGES['recon_weight']),
        'layer_cls_weight': random.choice(PARAM_RANGES['layer_cls_weight']),
        'hidden_dims': random.choice(PARAM_RANGES['hidden_dims']),
        'encoder_hidden_dims': random.choice(PARAM_RANGES['encoder_hidden_dims'])
    }


# ====================== TG-Net Model Components ======================
class LinearEncoderLayer(nn.Module):
    """Encoder layer: Linear -> BatchNorm -> ReLU -> Dropout"""
    def __init__(self, in_dim, out_dim, dropout_rate=0.3):
        super(LinearEncoderLayer, self).__init__()
        self.linear = nn.Linear(in_dim, out_dim)
        self.bn = nn.BatchNorm1d(out_dim)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(dropout_rate)

    def forward(self, x):
        x = self.linear(x)
        x = self.bn(x)
        x = self.relu(x)
        x = self.dropout(x)
        return x


class LinearDecoderLayer(nn.Module):
    """Decoder layer: Linear -> BatchNorm -> ReLU -> Dropout"""
    def __init__(self, in_dim, out_dim, dropout_rate=0.3):
        super(LinearDecoderLayer, self).__init__()
        self.linear = nn.Linear(in_dim, out_dim)
        self.bn = nn.BatchNorm1d(out_dim)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(dropout_rate)

    def forward(self, x):
        x = self.linear(x)
        x = self.bn(x)
        x = self.relu(x)
        x = self.dropout(x)
        return x


class ClassTokenLayer(nn.Module):
    """Learnable class tokens serving as dynamic cell type prototypes"""
    def __init__(self, num_classes, token_dim, dropout_rate=0.3):
        super(ClassTokenLayer, self).__init__()
        self.num_classes = num_classes
        self.token_dim = token_dim
        init_tokens = torch.randn(num_classes, token_dim) * 0.1
        self.class_tokens = nn.Parameter(init_tokens)
        self.token_updater = nn.Sequential(
            nn.Linear(token_dim, token_dim),
            nn.LayerNorm(token_dim),
            nn.ReLU(),
            nn.Dropout(dropout_rate),
            nn.Linear(token_dim, token_dim)
        )

    def forward(self):
        base_tokens = self.class_tokens
        updated_tokens = self.token_updater(base_tokens)
        updated_tokens = base_tokens + 0.1 * updated_tokens
        return updated_tokens


class LayerWiseClassifier(nn.Module):
    """Layer-wise classifier using normalized cosine similarity with class tokens"""
    def __init__(self, feature_dim, num_classes, token_dim, dropout_rate=0.3):
        super(LayerWiseClassifier, self).__init__()
        self.feature_dim = feature_dim
        self.num_classes = num_classes
        self.token_dim = token_dim
        self.feature_to_token = nn.Sequential(
            nn.Linear(feature_dim, token_dim),
            nn.LayerNorm(token_dim),
            nn.ReLU(),
            nn.Dropout(dropout_rate)
        )
        self.classifier = nn.Sequential(
            nn.Linear(num_classes, 64),
            nn.ReLU(),
            nn.Dropout(dropout_rate),
            nn.Linear(64, num_classes)
        )

    def forward(self, features, class_tokens):
        feature_tokens = self.feature_to_token(features)
        feature_norm = F.normalize(feature_tokens, p=2, dim=1)
        token_norm = F.normalize(class_tokens, p=2, dim=1)
        similarity = torch.matmul(feature_norm, token_norm.T)
        logits = self.classifier(similarity)
        return logits, similarity


class ResidualGuidanceLayer(nn.Module):
    """Maps classification logits back to feature space for residual guidance (α)"""
    def __init__(self, in_dim, out_dim, dropout_rate=0.3):
        super(ResidualGuidanceLayer, self).__init__()
        self.transform = nn.Sequential(
            nn.Linear(in_dim, out_dim),
            nn.LayerNorm(out_dim),
            nn.ReLU(),
            nn.Dropout(dropout_rate)
        )

    def forward(self, guidance):
        return self.transform(guidance)


class TGNet(nn.Module):
    """
    TG-Net: Token-Guided Multi-Scale Neural Network for cell type annotation.

    Components:
        - Hierarchical encoder-decoder with reconstruction regularization.
        - Learnable class tokens (dynamic prototypes).
        - Multi-scale supervision via layer-wise classifiers.
        - Residual guidance (α) and latent modulation (γ).
    """
    def __init__(self, input_dim, num_classes, config_args):
        super(TGNet, self).__init__()
        self.input_dim = input_dim
        self.num_classes = num_classes
        self.config = config_args

        # Tunable coefficients α and γ from the paper
        self.alpha = getattr(config_args, 'residual_guidance_weight', 0.2)   # α
        self.gamma = getattr(config_args, 'latent_modulation_weight', 0.3)   # γ

        # Encoder dimensions: [input] -> ... -> [latent]
        encoder_dims = [input_dim] + config_args.encoder_hidden_dims + [config_args.latent_dim]
        # Decoder dimensions: [latent] -> ... -> [input]
        decoder_dims = [config_args.latent_dim] + config_args.hidden_dims + [input_dim]

        # Encoder layers
        self.encoder_layers = nn.ModuleList()
        for i in range(len(encoder_dims) - 1):
            self.encoder_layers.append(
                LinearEncoderLayer(encoder_dims[i], encoder_dims[i + 1], config_args.dropout)
            )

        # Decoder layers
        self.decoder_layers = nn.ModuleList()
        for i in range(len(decoder_dims) - 1):
            self.decoder_layers.append(
                LinearDecoderLayer(decoder_dims[i], decoder_dims[i + 1], config_args.dropout)
            )

        # Class token layer
        self.class_token_layer = ClassTokenLayer(num_classes, config_args.token_dim, config_args.dropout)

        # Layer-wise classifiers (multi-scale supervision)
        self.layer_classifiers = nn.ModuleList()
        for dim in encoder_dims:
            self.layer_classifiers.append(
                LayerWiseClassifier(dim, num_classes, config_args.token_dim, config_args.dropout)
            )

        # Residual guidance layers for encoder
        self.residual_layers = nn.ModuleList()
        for i in range(len(encoder_dims) - 1):
            self.residual_layers.append(
                ResidualGuidanceLayer(num_classes, encoder_dims[i], config_args.dropout)
            )

        # Final classifier
        self.final_classifier = nn.Sequential(
            nn.Linear(config_args.latent_dim, 128),
            nn.LayerNorm(128),
            nn.ReLU(),
            nn.Dropout(config_args.dropout),
            nn.Linear(128, num_classes)
        )

        # Class guidance projection for latent modulation
        self.class_guidance = nn.Sequential(
            nn.Linear(config_args.token_dim, config_args.latent_dim),
            nn.LayerNorm(config_args.latent_dim),
            nn.ReLU(),
            nn.Dropout(config_args.dropout)
        )

        self._init_weights()

    def _init_weights(self):
        for name, module in self.named_modules():
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)
            elif isinstance(module, nn.LayerNorm):
                nn.init.ones_(module.weight)
                nn.init.zeros_(module.bias)
            elif isinstance(module, ClassTokenLayer):
                nn.init.normal_(module.class_tokens, mean=0.0, std=0.1)

    def forward(self, x):
        batch_size = x.shape[0]
        class_tokens = self.class_token_layer()  # [C, token_dim]

        layer_logits = []
        current_features = x

        # Input layer classification
        logits, _ = self.layer_classifiers[0](x, class_tokens)
        layer_logits.append(logits)

        # Pass through encoder layers
        for i, encoder_layer in enumerate(self.encoder_layers):
            # Residual guidance: add α * transformed previous logits
            if i < len(self.residual_layers):
                guidance = self.residual_layers[i](layer_logits[-1])
                current_features = current_features + self.alpha * guidance
            current_features = encoder_layer(current_features)
            logits, _ = self.layer_classifiers[i + 1](current_features, class_tokens)
            layer_logits.append(logits)

        latent_features = current_features
        final_logits = self.final_classifier(latent_features)

        # Latent modulation for decoding: add γ * projected class token
        _, predicted = torch.max(final_logits, dim=1)
        selected_tokens = class_tokens[predicted]  # [batch, token_dim]
        guidance_info = self.class_guidance(selected_tokens)  # [batch, latent_dim]
        x_decoded = latent_features + self.gamma * guidance_info

        for decoder_layer in self.decoder_layers:
            x_decoded = decoder_layer(x_decoded)

        reconstructed = x_decoded

        return {
            'reconstructed': reconstructed,
            'final_logits': final_logits,
            'layer_logits': layer_logits,
            'latent_features': latent_features,
        }


# ====================== Data Loading Functions ======================
def load_npz_data(npz_path):
    data = np.load(npz_path, allow_pickle=True)
    expr = data['count']
    gene_symbols = data['gene_symbol']
    barcodes = data['barcode']
    str_labels = data['str_labels']
    label_ints = data['label']
    return expr, gene_symbols, barcodes, str_labels, label_ints


def load_wcsn_graphs(dataset_name, fold, cell_indices, mode='train'):
    wcsn_dir = os.path.join(BASE_WCSN_DIR, dataset_name, f'WCSN_a0.01_hvgs2000', f'{mode}_f{fold}', 'processed')
    graphs = []
    for idx in cell_indices:
        pt_file = os.path.join(wcsn_dir, f'cell_{idx}.pt')
        if os.path.exists(pt_file):
            data = torch.load(pt_file)
            graphs.append(data)
    return graphs


# ====================== Dataset Class ======================
class NumpyDataset(Dataset):
    def __init__(self, X, y):
        self.X = torch.tensor(X, dtype=torch.float32)
        self.y = torch.tensor(y, dtype=torch.long)

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]


# ====================== Preprocessing ======================
def preprocess_train(train_X, train_y, args):
    adata_train = sc.AnnData(train_X)
    adata_train.obs['label'] = train_y
    sc.pp.filter_cells(adata_train, min_genes=args.min_genes)
    sc.pp.filter_genes(adata_train, min_cells=args.min_cells)
    sc.pp.normalize_total(adata_train, target_sum=1e4)
    sc.pp.log1p(adata_train)
    sc.pp.highly_variable_genes(adata_train, n_top_genes=args.n_top_genes)
    hvgs = adata_train.var.highly_variable
    adata_train = adata_train[:, hvgs]
    return adata_train.X, adata_train.obs['label'].values, hvgs


def apply_preprocess(adata, hvgs, args):
    adata = adata[:, hvgs]
    sc.pp.filter_cells(adata, min_genes=args.min_genes)
    sc.pp.normalize_total(adata, target_sum=1e4)
    sc.pp.log1p(adata)
    return adata.X, adata.obs['label'].values


# ====================== Training and Evaluation ======================
def train_one_epoch(model, loader, criterion, optimizer, device, args, scaler=None, accum_steps=1):
    model.train()
    total_loss = 0.0
    correct = 0
    total = 0
    optimizer.zero_grad()
    pbar = tqdm(loader, desc="Training", leave=False)
    for i, (data, target) in enumerate(pbar):
        data, target = data.to(device), target.to(device)
        if scaler is not None:
            with torch.cuda.amp.autocast():
                output = model(data)
                final_cls_loss = criterion(output['final_logits'], target)
                layer_cls_loss = 0
                for layer_logits in output['layer_logits']:
                    layer_cls_loss += criterion(layer_logits, target)
                layer_cls_loss /= len(output['layer_logits'])
                recon_criterion = nn.MSELoss()
                recon_loss = recon_criterion(output['reconstructed'], data)
                loss = (args.cls_weight * final_cls_loss +
                        args.recon_weight * recon_loss +
                        args.layer_cls_weight * layer_cls_loss)
                loss = loss / accum_steps
            scaler.scale(loss).backward()
            if (i + 1) % accum_steps == 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=args.grad_clip)
                scaler.step(optimizer)
                scaler.update()
                optimizer.zero_grad()
        else:
            output = model(data)
            final_cls_loss = criterion(output['final_logits'], target)
            layer_cls_loss = 0
            for layer_logits in output['layer_logits']:
                layer_cls_loss += criterion(layer_logits, target)
            layer_cls_loss /= len(output['layer_logits'])
            recon_criterion = nn.MSELoss()
            recon_loss = recon_criterion(output['reconstructed'], data)
            loss = (args.cls_weight * final_cls_loss +
                    args.recon_weight * recon_loss +
                    args.layer_cls_weight * layer_cls_loss)
            loss = loss / accum_steps
            loss.backward()
            if (i + 1) % accum_steps == 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=args.grad_clip)
                optimizer.step()
                optimizer.zero_grad()

        total_loss += loss.item() * accum_steps * data.size(0)
        _, predicted = torch.max(output['final_logits'], 1)
        correct += (predicted == target).sum().item()
        total += target.size(0)
        pbar.set_postfix(loss=loss.item() * accum_steps, acc=100. * correct / total)

    if (i + 1) % accum_steps != 0:
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=args.grad_clip)
        if scaler is not None:
            scaler.step(optimizer)
            scaler.update()
        else:
            optimizer.step()
        optimizer.zero_grad()

    return total_loss / total, 100. * correct / total


def evaluate(model, loader, criterion, device, args):
    model.eval()
    total_loss = 0.0
    final_cls_loss_total = 0.0
    correct = 0
    total = 0
    all_targets = []
    all_predicted = []
    with torch.no_grad():
        for data, target in loader:
            data, target = data.to(device), target.to(device)
            output = model(data)
            final_cls_loss = criterion(output['final_logits'], target)
            layer_cls_loss = 0
            for layer_logits in output['layer_logits']:
                layer_cls_loss += criterion(layer_logits, target)
            layer_cls_loss /= len(output['layer_logits'])
            recon_criterion = nn.MSELoss()
            recon_loss = recon_criterion(output['reconstructed'], data)
            loss = (args.cls_weight * final_cls_loss +
                    args.recon_weight * recon_loss +
                    args.layer_cls_weight * layer_cls_loss)

            total_loss += loss.item() * data.size(0)
            final_cls_loss_total += final_cls_loss.item() * data.size(0)
            _, predicted = torch.max(output['final_logits'], 1)
            correct += (predicted == target).sum().item()
            total += target.size(0)
            all_targets.extend(target.cpu().numpy())
            all_predicted.extend(predicted.cpu().numpy())

    avg_loss = total_loss / total
    avg_final_cls_loss = final_cls_loss_total / total
    acc = 100. * correct / total

    metrics = compute_metrics(all_targets, all_predicted, len(np.unique(all_targets)))
    metrics['loss'] = avg_loss
    metrics['final_cls_loss'] = avg_final_cls_loss
    metrics['accuracy_percent'] = acc
    metrics['accuracy_decimal'] = acc / 100.0
    return metrics


def compute_metrics(y_true, y_pred, num_classes):
    metrics = {}
    metrics['accuracy'] = accuracy_score(y_true, y_pred)
    metrics['balanced_accuracy'] = balanced_accuracy_score(y_true, y_pred)
    metrics['f1_macro'] = f1_score(y_true, y_pred, average='macro', zero_division=0)
    metrics['f1_micro'] = f1_score(y_true, y_pred, average='micro', zero_division=0)
    metrics['precision_macro'] = precision_score(y_true, y_pred, average='macro', zero_division=0)
    metrics['precision_micro'] = precision_score(y_true, y_pred, average='micro', zero_division=0)
    metrics['recall_macro'] = recall_score(y_true, y_pred, average='macro', zero_division=0)
    metrics['recall_micro'] = recall_score(y_true, y_pred, average='micro', zero_division=0)

    cm = confusion_matrix(y_true, y_pred, labels=range(num_classes))
    class_accuracies = {}
    for i in range(num_classes):
        if i in y_true:
            class_accuracies[f'class_{i}_accuracy'] = cm[i, i] / cm[i, :].sum() if cm[i, :].sum() > 0 else 0.0
        else:
            class_accuracies[f'class_{i}_accuracy'] = np.nan
    metrics['class_accuracies'] = class_accuracies
    metrics['confusion_matrix'] = cm
    return metrics


# ====================== WCSN Analysis Functions (unchanged) ======================
def analyze_hub_genes(graphs, gene_names, cell_type_labels):
    from collections import defaultdict
    cell_type_degrees = defaultdict(list)
    for g, label in zip(graphs, cell_type_labels):
        if g is None:
            continue
        edge_index = g.edge_index
        deg = torch.bincount(edge_index[0], minlength=len(gene_names)).cpu().numpy()
        cell_type_degrees[label].append(deg)
    top_genes_per_type = {}
    for label, degs in cell_type_degrees.items():
        avg_deg = np.mean(degs, axis=0)
        top_idx = np.argsort(avg_deg)[-100:][::-1]
        top_genes = [gene_names[i] for i in top_idx]
        top_genes_per_type[label] = top_genes
    return top_genes_per_type


def analyze_high_weight_edges(graphs, gene_names, cell_type_labels, top_k=100):
    from collections import defaultdict, Counter
    cell_type_edge_counts = defaultdict(Counter)
    for g, label in zip(graphs, cell_type_labels):
        if g is None or not hasattr(g, 'edge_attr'):
            continue
        edge_index = g.edge_index.cpu().numpy()
        edge_weights = g.edge_attr.cpu().numpy().flatten()
        if len(edge_weights) > top_k:
            top_idx = np.argsort(edge_weights)[-top_k:]
        else:
            top_idx = np.arange(len(edge_weights))
        for idx in top_idx:
            u, v = edge_index[0][idx], edge_index[1][idx]
            if u > v:
                u, v = v, u
            edge_key = f"{gene_names[u]}-{gene_names[v]}"
            cell_type_edge_counts[label][edge_key] += 1
    top_edges_per_type = {}
    for label, counter in cell_type_edge_counts.items():
        top_edges = [k for k, v in counter.most_common(top_k)]
        top_edges_per_type[label] = top_edges
    return top_edges_per_type


def plot_venn_upset(items_per_type, save_dir, prefix):
    types = list(items_per_type.keys())
    if len(types) == 2:
        set1 = set(items_per_type[types[0]])
        set2 = set(items_per_type[types[1]])
        plt.figure(figsize=(5, 5))
        venn2([set1, set2], set_labels=types)
        plt.title(f"{prefix} (2 types)")
        plt.savefig(os.path.join(save_dir, f"{prefix}_venn2.png"), dpi=300)
        plt.close()
    elif len(types) == 3:
        set1 = set(items_per_type[types[0]])
        set2 = set(items_per_type[types[1]])
        set3 = set(items_per_type[types[2]])
        plt.figure(figsize=(6, 6))
        venn3([set1, set2, set3], set_labels=types)
        plt.title(f"{prefix} (3 types)")
        plt.savefig(os.path.join(save_dir, f"{prefix}_venn3.png"), dpi=300)
        plt.close()

    if len(types) > 1:
        all_items = set()
        for items in items_per_type.values():
            all_items.update(items)
        data = {}
        for item in all_items:
            row = {t: (item in items_per_type[t]) for t in types}
            data[item] = row
        df = pd.DataFrame(data).T
        upset = UpSet(df, subset_size='count', show_counts=True)
        upset.plot()
        plt.title(f"{prefix} UpSet")
        plt.savefig(os.path.join(save_dir, f"{prefix}_upset.png"), dpi=300)
        plt.close()


def plot_tsne_with_features(latent_features, cell_types, type_names, top_genes_per_type, graphs, gene_names, save_dir):
    if latent_features.shape[0] > 10000:
        idx = np.random.choice(latent_features.shape[0], 10000, replace=False)
        latent_features = latent_features[idx]
        cell_types = cell_types[idx]
        graphs = [graphs[i] for i in idx]
    tsne = TSNE(n_components=2, random_state=42, perplexity=30)
    emb = tsne.fit_transform(latent_features)

    plt.figure(figsize=(10, 8))
    for i, label in enumerate(np.unique(cell_types)):
        mask = cell_types == label
        plt.scatter(emb[mask, 0], emb[mask, 1], s=1, label=type_names[label])
    plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.title("t-SNE of latent features (true labels)")
    plt.savefig(os.path.join(save_dir, "tsne_true_labels.png"), dpi=300, bbox_inches='tight')
    plt.close()

    for cell_type, top_genes in top_genes_per_type.items():
        if not top_genes:
            continue
        gene = top_genes[0]
        if gene not in gene_names:
            continue
        gene_idx = np.where(gene_names == gene)[0][0]
        degrees = []
        for g in graphs:
            if g is None:
                degrees.append(0)
            else:
                edge_index = g.edge_index.cpu().numpy()
                deg = np.bincount(edge_index[0], minlength=len(gene_names))[gene_idx]
                degrees.append(deg)
        plt.figure(figsize=(8, 6))
        sc = plt.scatter(emb[:, 0], emb[:, 1], c=degrees, cmap='viridis', s=2)
        plt.colorbar(sc, label=f"Degree of {gene}")
        plt.title(f"t-SNE with degree of {gene} (cell type: {type_names[cell_type]})")
        plt.savefig(os.path.join(save_dir, f"tsne_degree_{gene}.png"), dpi=300)
        plt.close()


# ====================== Single Fold Training ======================
def train_fold(dataset_name, fold, X_train, y_train, X_val, y_val, X_test, y_test, args, save_dir):
    print(f"\n{'=' * 60}\nFold {fold} for {dataset_name}\n{'=' * 60}")

    X_train_processed, y_train_processed, hvgs = preprocess_train(X_train, y_train, args)

    adata_val = sc.AnnData(X_val)
    adata_val.obs['label'] = y_val
    X_val_processed, y_val_processed = apply_preprocess(adata_val, hvgs, args)

    adata_test = sc.AnnData(X_test)
    adata_test.obs['label'] = y_test
    X_test_processed, y_test_processed = apply_preprocess(adata_test, hvgs, args)

    train_dataset = NumpyDataset(X_train_processed, y_train_processed)
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, drop_last=True)
    val_dataset = NumpyDataset(X_val_processed, y_val_processed)
    val_loader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False)
    test_dataset = NumpyDataset(X_test_processed, y_test_processed)
    test_loader = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False)

    input_dim = X_train_processed.shape[1]
    num_classes = len(np.unique(np.concatenate([y_train_processed, y_val_processed, y_test_processed])))
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}, input_dim={input_dim}, num_classes={num_classes}")

    hidden_dims = [int(d) for d in args.hidden_dims.split(',')]
    encoder_hidden_dims = [int(d) for d in args.encoder_hidden_dims.split(',')]
    model_args = type('Args', (), {
        'latent_dim': args.latent_dim,
        'token_dim': args.token_dim,
        'hidden_dims': hidden_dims,
        'encoder_hidden_dims': encoder_hidden_dims,
        'dropout': args.dropout,
        'residual_guidance_weight': args.residual_guidance_weight,
        'latent_modulation_weight': args.latent_modulation_weight
    })()
    model = TGNet(input_dim, num_classes, model_args).to(device)

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.num_epochs, eta_min=args.lr * 0.01)

    scaler = torch.cuda.amp.GradScaler() if args.use_amp and torch.cuda.is_available() else None

    best_val_acc = 0.0
    best_model_state = None
    patience_counter = 0

    for epoch in range(1, args.num_epochs + 1):
        train_loss, train_acc = train_one_epoch(model, train_loader, criterion, optimizer, device, args,
                                                scaler=scaler, accum_steps=args.accum_steps)
        val_metrics = evaluate(model, val_loader, criterion, device, args)
        val_acc = val_metrics['accuracy_percent']
        scheduler.step()

        print(
            f"Epoch {epoch:3d} | Train Loss: {train_loss:.4f} | Train Acc: {train_acc:.2f}% | Val Acc: {val_acc:.2f}% | LR: {optimizer.param_groups[0]['lr']:.6f}")

        if val_acc > best_val_acc:
            best_val_acc = val_acc
            best_model_state = copy.deepcopy(model.state_dict())
            patience_counter = 0
        else:
            patience_counter += 1
            if patience_counter >= args.patience:
                print(f"Early stopping at epoch {epoch}")
                break

    if best_model_state is not None:
        model.load_state_dict(best_model_state)

    test_metrics = evaluate(model, test_loader, criterion, device, args)
    print(f"Test Acc: {test_metrics['accuracy_percent']:.2f}%")
    print(f"Test F1 macro: {test_metrics['f1_macro']:.4f}")
    print(f"Test Loss: {test_metrics['loss']:.4f}")
    print(f"Test Final Cls Loss: {test_metrics['final_cls_loss']:.4f}")

    os.makedirs(save_dir, exist_ok=True)
    torch.save(model.state_dict(), os.path.join(save_dir, f'model_fold{fold}.pth'))
    pd.DataFrame([test_metrics]).to_csv(os.path.join(save_dir, f'test_metrics_fold{fold}.csv'), index=False)

    if args.analyze:
        seq_dict_path = os.path.join(BASE_5FOLD_DIR, dataset_name, 'seq_dict.npz')
        if os.path.exists(seq_dict_path):
            seq_dict = np.load(seq_dict_path, allow_pickle=True)
            test_indices = seq_dict[f'test_index_{fold}'].astype(int)
            graphs = load_wcsn_graphs(dataset_name, fold, test_indices, mode='test')
            if graphs:
                y_test_raw = y_test_processed
                _, gene_names, _, _, _ = load_npz_data(os.path.join(BASE_DATA_DIR, f'{dataset_name}.npz'))
                top_genes = analyze_hub_genes(graphs, gene_names, y_test_raw)
                plot_venn_upset(top_genes, save_dir, f"fold{fold}_hub")
                top_edges = analyze_high_weight_edges(graphs, gene_names, y_test_raw, top_k=100)
                plot_venn_upset(top_edges, save_dir, f"fold{fold}_edges")
                model.eval()
                all_latents = []
                with torch.no_grad():
                    for data, _ in test_loader:
                        data = data.to(device)
                        out = model(data)
                        all_latents.append(out['latent_features'].cpu().numpy())
                latent_features = np.concatenate(all_latents, axis=0)
                plot_tsne_with_features(latent_features, y_test_raw, np.unique(y_test_raw), top_genes, graphs,
                                        gene_names, save_dir)
        else:
            print("seq_dict.npz not found, skipping WCSN analysis.")

    return test_metrics


# ====================== Run a Full Dataset ======================
def run_dataset(dataset_name, args, output_dir, use_5fold=True, force=False):
    import time
    start_time = time.time()

    if use_5fold:
        summary_path = os.path.join(output_dir, dataset_name, '5fold_summary.csv')
        if not force and os.path.exists(summary_path):
            print(f"Results already exist for {dataset_name} (5-fold). Loading from {summary_path}")
            summary_df = pd.read_csv(summary_path)
            summary = summary_df.iloc[0].to_dict()
            if 'accuracy' not in summary:
                summary['accuracy'] = f"{summary['accuracy_percent_mean']:.2f} ± {summary['accuracy_percent_std']:.2f}"
            if 'f1_macro' not in summary:
                summary['f1_macro'] = f"{summary['f1_macro_mean']:.4f} ± {summary['f1_macro_std']:.4f}"
            if 'balanced_accuracy' not in summary:
                summary[
                    'balanced_accuracy'] = f"{summary['balanced_accuracy_mean']:.4f} ± {summary['balanced_accuracy_std']:.4f}"
            if 'loss' not in summary:
                summary['loss'] = f"{summary['loss_mean']:.4f} ± {summary['loss_std']:.4f}"
            if 'final_cls_loss' not in summary:
                summary[
                    'final_cls_loss'] = f"{summary['final_cls_loss_mean']:.4f} ± {summary['final_cls_loss_std']:.4f}"
            elapsed = 0.0
            return summary, elapsed
    else:
        single_path = os.path.join(output_dir, dataset_name, 'single_split', 'test_metrics_fold0.csv')
        if not force and os.path.exists(single_path):
            print(f"Results already exist for {dataset_name} (single split). Loading from {single_path}")
            metrics_df = pd.read_csv(single_path)
            metrics = metrics_df.iloc[0].to_dict()
            summary = {
                'accuracy_percent_mean': metrics.get('accuracy_percent', np.nan),
                'accuracy_percent_std': 0.0,
                'f1_macro_mean': metrics.get('f1_macro', np.nan),
                'f1_macro_std': 0.0,
                'balanced_accuracy_mean': metrics.get('balanced_accuracy', np.nan),
                'balanced_accuracy_std': 0.0,
                'loss_mean': metrics.get('loss', np.nan),
                'loss_std': 0.0,
                'final_cls_loss_mean': metrics.get('final_cls_loss', np.nan),
                'final_cls_loss_std': 0.0,
                'accuracy': f"{metrics.get('accuracy_percent', np.nan):.2f} ± 0.00",
                'f1_macro': f"{metrics.get('f1_macro', np.nan):.4f} ± 0.0000",
                'balanced_accuracy': f"{metrics.get('balanced_accuracy', np.nan):.4f} ± 0.0000",
                'loss': f"{metrics.get('loss', np.nan):.4f} ± 0.0000",
                'final_cls_loss': f"{metrics.get('final_cls_loss', np.nan):.4f} ± 0.0000",
            }
            elapsed = 0.0
            return summary, elapsed

    npz_path = os.path.join(BASE_DATA_DIR, f'{dataset_name}.npz')
    if not os.path.exists(npz_path):
        print(f"Error: {npz_path} not found.")
        return None, None

    expr, _, _, _, label_ints = load_npz_data(npz_path)

    if use_5fold:
        seq_dict_path = os.path.join(BASE_5FOLD_DIR, dataset_name, 'seq_dict.npz')
        if not os.path.exists(seq_dict_path):
            print(f"Warning: {seq_dict_path} not found, falling back to random split.")
            use_5fold = False
        else:
            seq_dict = np.load(seq_dict_path, allow_pickle=True)
            folds = []
            for fold in range(1, 6):
                train_idx = seq_dict[f'train_index_{fold}'].astype(int)
                test_idx = seq_dict[f'test_index_{fold}'].astype(int)
                X_train_all = expr[train_idx]
                y_train_all = np.array([label_ints[i] for i in train_idx])
                val_ratio = args.val_size / (1 - args.test_size)
                X_train, X_val, y_train, y_val = train_test_split(
                    X_train_all, y_train_all, test_size=val_ratio, random_state=args.random_state,
                    stratify=y_train_all if len(np.unique(y_train_all)) > 1 else None
                )
                X_test = expr[test_idx]
                y_test = np.array([label_ints[i] for i in test_idx])
                folds.append((X_train, y_train, X_val, y_val, X_test, y_test))

            fold_results = []
            for fold, (X_train, y_train, X_val, y_val, X_test, y_test) in enumerate(folds, 1):
                print(f"\n--- Fold {fold} ---")
                save_dir = os.path.join(output_dir, dataset_name, f'fold_{fold}')
                res = train_fold(dataset_name, fold, X_train, y_train, X_val, y_val, X_test, y_test, args, save_dir)
                fold_results.append(res)

            summary = {}
            for metric in ['accuracy_percent', 'f1_macro', 'f1_micro', 'balanced_accuracy', 'loss', 'final_cls_loss']:
                values = [r[metric] for r in fold_results]
                summary[f'{metric}_mean'] = np.mean(values)
                summary[f'{metric}_std'] = np.std(values, ddof=1)

            summary['accuracy'] = f"{summary['accuracy_percent_mean']:.2f} ± {summary['accuracy_percent_std']:.2f}"
            summary['f1_macro'] = f"{summary['f1_macro_mean']:.4f} ± {summary['f1_macro_std']:.4f}"
            summary[
                'balanced_accuracy'] = f"{summary['balanced_accuracy_mean']:.4f} ± {summary['balanced_accuracy_std']:.4f}"
            summary['loss'] = f"{summary['loss_mean']:.4f} ± {summary['loss_std']:.4f}"
            summary['final_cls_loss'] = f"{summary['final_cls_loss_mean']:.4f} ± {summary['final_cls_loss_std']:.4f}"

            summary_df = pd.DataFrame([summary])
            summary_df.to_csv(os.path.join(output_dir, dataset_name, '5fold_summary.csv'), index=False)

            print("\n5-fold cross-validation completed.")
            print("Summary (mean ± std):")
            print(f"  Accuracy:        {summary['accuracy']}%")
            print(f"  F1 macro:        {summary['f1_macro']}")
            print(f"  Balanced Acc:    {summary['balanced_accuracy']}")
            print(f"  Test Loss:       {summary['loss']}")
            print(f"  Final Cls Loss:  {summary['final_cls_loss']}")

            elapsed = time.time() - start_time
            return summary, elapsed
    else:
        X_train_val, X_test, y_train_val, y_test = train_test_split(
            expr, label_ints, test_size=args.test_size, random_state=args.random_state,
            stratify=label_ints if len(np.unique(label_ints)) > 1 else None)
        val_ratio = args.val_size / (1 - args.test_size)
        X_train, X_val, y_train, y_val = train_test_split(
            X_train_val, y_train_val, test_size=val_ratio, random_state=args.random_state,
            stratify=y_train_val if len(np.unique(y_train_val)) > 1 else None)
        save_dir = os.path.join(output_dir, dataset_name, 'single_split')
        res = train_fold(dataset_name, 0, X_train, y_train, X_val, y_val, X_test, y_test, args, save_dir)
        summary = {f'{k}_mean': v for k, v in res.items() if
                   k in ['accuracy_percent', 'f1_macro', 'f1_micro', 'balanced_accuracy', 'loss', 'final_cls_loss']}
        for k in list(summary.keys()):
            summary[k.replace('_mean', '_std')] = 0.0
        summary['accuracy'] = f"{summary['accuracy_percent_mean']:.2f} ± 0.00"
        summary['f1_macro'] = f"{summary['f1_macro_mean']:.4f} ± 0.0000"
        summary['balanced_accuracy'] = f"{summary['balanced_accuracy_mean']:.4f} ± 0.0000"
        summary['loss'] = f"{summary['loss_mean']:.4f} ± 0.0000"
        summary['final_cls_loss'] = f"{summary['final_cls_loss_mean']:.4f} ± 0.0000"
        elapsed = time.time() - start_time
        return summary, elapsed


# ====================== Hyperparameter Search ======================
def search_dataset(dataset_name, base_args, output_dir, max_trials=10, patience=3, force=False):
    import time
    start_time = time.time()

    print(f"\n{'=' * 70}\nHyperparameter search for {dataset_name}\n{'=' * 70}")

    search_dir = os.path.join(output_dir, dataset_name, 'search')
    best_info_path = os.path.join(output_dir, dataset_name, 'best_info.csv')
    history_path = os.path.join(search_dir, 'search_history.csv')

    if not force and os.path.exists(best_info_path):
        print(f"Best result already exists for {dataset_name}. Loading from {best_info_path}")
        best_info = pd.read_csv(best_info_path).iloc[0].to_dict()
        if 'best_params' in best_info and isinstance(best_info['best_params'], str):
            import ast
            best_info['best_params'] = ast.literal_eval(best_info['best_params'])
        elapsed = 0.0
        return best_info

    completed_trials = {}
    best_val = -1
    best_params = None
    best_summary = None
    if not force and os.path.exists(history_path):
        history_df = pd.read_csv(history_path)
        for idx, row in history_df.iterrows():
            trial = int(row['trial'])
            params_str = row['params']
            import ast
            params = ast.literal_eval(params_str)
            trial_summary_path = os.path.join(search_dir, f'trial_{trial}', '5fold_summary.csv')
            if os.path.exists(trial_summary_path):
                summary_df = pd.read_csv(trial_summary_path)
                summary = summary_df.iloc[0].to_dict()
                completed_trials[trial] = (params, summary)
                current_val = summary['accuracy_percent_mean']
                if current_val > best_val:
                    best_val = current_val
                    best_params = params
                    best_summary = summary
        print(
            f"Resuming search: {len(completed_trials)} trials already completed. Best accuracy so far: {best_val:.4f}")

    next_trial = max(completed_trials.keys()) + 1 if completed_trials else 1
    no_improve = 0

    for trial in range(next_trial, max_trials + 1):
        print(f"\nTrial {trial}/{max_trials}")
        sampled = sample_random_params()
        trial_args = copy.deepcopy(base_args)
        for k, v in sampled.items():
            setattr(trial_args, k, v)

        summary, _ = run_dataset(dataset_name, trial_args,
                                 os.path.join(output_dir, dataset_name, 'search', f'trial_{trial}'), use_5fold=True,
                                 force=force)
        if summary is None:
            continue

        trial_result = {
            'trial': trial,
            'params': str(sampled),
            'summary': str(summary)
        }
        os.makedirs(search_dir, exist_ok=True)
        if os.path.exists(history_path):
            history_df = pd.read_csv(history_path)
            new_row = pd.DataFrame([trial_result])
            history_df = pd.concat([history_df, new_row], ignore_index=True)
        else:
            history_df = pd.DataFrame([trial_result])
        history_df.to_csv(history_path, index=False)

        current_val = summary['accuracy_percent_mean']
        if current_val > best_val:
            best_val = current_val
            best_params = sampled
            best_summary = summary
            no_improve = 0
            print(f"New best: accuracy = {best_val:.4f}")
        else:
            no_improve += 1
            print(f"No improvement, patience {no_improve}/{patience}")

        if no_improve >= patience:
            print(f"Early stopping after {trial} trials.")
            break

    if best_summary is None:
        print("No valid trial completed.")
        return None

    print(f"\nBest parameters found: {best_params}")
    print(f"Best accuracy: {best_val:.4f}")

    final_summary = best_summary
    final_time = 0.0

    best_info = {
        'dataset': dataset_name,
        'accuracy_mean': final_summary['accuracy_percent_mean'],
        'accuracy_std': final_summary['accuracy_percent_std'],
        'accuracy': f"{final_summary['accuracy_percent_mean']:.2f} ± {final_summary['accuracy_percent_std']:.2f}",
        'f1_macro_mean': final_summary['f1_macro_mean'],
        'f1_macro_std': final_summary['f1_macro_std'],
        'f1_macro': f"{final_summary['f1_macro_mean']:.4f} ± {final_summary['f1_macro_std']:.4f}",
        'balanced_accuracy_mean': final_summary['balanced_accuracy_mean'],
        'balanced_accuracy_std': final_summary['balanced_accuracy_std'],
        'balanced_accuracy': f"{final_summary['balanced_accuracy_mean']:.4f} ± {final_summary['balanced_accuracy_std']:.4f}",
        'test_loss_mean': final_summary['loss_mean'],
        'test_loss_std': final_summary['loss_std'],
        'loss': f"{final_summary['loss_mean']:.4f} ± {final_summary['loss_std']:.4f}",
        'test_final_cls_loss_mean': final_summary['final_cls_loss_mean'],
        'test_final_cls_loss_std': final_summary['final_cls_loss_std'],
        'final_cls_loss': f"{final_summary['final_cls_loss_mean']:.4f} ± {final_summary['final_cls_loss_std']:.4f}",
        'total_time_seconds': final_time,
        'best_params': str(best_params)
    }
    pd.DataFrame([best_info]).to_csv(os.path.join(output_dir, dataset_name, 'best_info.csv'), index=False)

    return best_info


# ====================== Main ======================
if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='TG-Net: Token-Guided Multi-Scale Neural Network for scRNA-seq annotation.')
    parser.add_argument('--dataset', type=str, default=None, help='Dataset name. If not given, process all.')
    parser.add_argument('--no-5fold', action='store_true', help='Disable 5-fold cross-validation.')
    parser.add_argument('--output-dir', type=str, default='results', help='Output directory.')
    parser.add_argument('--analyze', action='store_true',
                        help='Perform WCSN structure analysis (requires graph files).')
    parser.add_argument('--search', action='store_true', help='Enable hyperparameter search.')
    parser.add_argument('--max-trials', type=int, default=15, help='Maximum trials for hyperparameter search.')
    parser.add_argument('--search-patience', type=int, default=5, help='Patience for early stopping in search.')
    parser.add_argument('--force', action='store_true', help='Force re-run even if results exist.')

    # Training hyperparameters (optimized defaults)
    parser.add_argument('--n-top-genes', type=int, default=2000)
    parser.add_argument('--batch-size', type=int, default=128)
    parser.add_argument('--accum-steps', type=int, default=1)
    parser.add_argument('--use-amp', action='store_true', help='Use mixed precision training')
    parser.add_argument('--dropout', type=float, default=0.3)
    parser.add_argument('--lr', type=float, default=5e-4)
    parser.add_argument('--weight-decay', type=float, default=5e-4)
    parser.add_argument('--grad-clip', type=float, default=1.0, help='Gradient clipping max norm')
    parser.add_argument('--num-epochs', type=int, default=150)
    parser.add_argument('--min-genes', type=int, default=0)
    parser.add_argument('--min-cells', type=int, default=0)
    parser.add_argument('--test-size', type=float, default=0.2)
    parser.add_argument('--val-size', type=float, default=0.1)
    parser.add_argument('--random-state', type=int, default=42)
    parser.add_argument('--patience', type=int, default=10, help='Early stopping patience on validation accuracy')
    parser.add_argument('--cls-weight', type=float, default=1.2)
    parser.add_argument('--recon-weight', type=float, default=0.05)
    parser.add_argument('--layer-cls-weight', type=float, default=0.8)

    # Model architecture (optimized defaults)
    parser.add_argument('--latent-dim', type=int, default=256)
    parser.add_argument('--token-dim', type=int, default=12)
    parser.add_argument('--hidden-dims', type=str, default="2048,1024")
    parser.add_argument('--encoder-hidden-dims', type=str, default="2048,1024")

    # Tunable coefficients α and γ
    parser.add_argument('--residual-guidance-weight', type=float, default=0.2,
                        help='Residual guidance weight (α) for encoder feature enhancement')
    parser.add_argument('--latent-modulation-weight', type=float, default=0.3,
                        help='Latent modulation weight (γ) for class guidance before decoding')

    args = parser.parse_args()

    if args.dataset:
        datasets = [args.dataset]
    else:
        datasets = [f.replace('.npz', '') for f in os.listdir(BASE_DATA_DIR) if f.endswith('.npz')]
        print(f"Found datasets: {datasets}")

    use_5fold = not args.no_5fold
    all_results = []

    for ds in datasets:
        print(f"\n{'=' * 70}\nProcessing {ds}\n{'=' * 70}")
        if args.search:
            best_info = search_dataset(ds, args, args.output_dir, max_trials=args.max_trials,
                                       patience=args.search_patience, force=args.force)
            if best_info is not None:
                all_results.append(best_info)
        else:
            summary, elapsed = run_dataset(ds, args, args.output_dir, use_5fold=use_5fold, force=args.force)
            if summary is not None:
                all_results.append({
                    'dataset': ds,
                    'accuracy_mean': summary.get('accuracy_percent_mean', np.nan),
                    'accuracy_std': summary.get('accuracy_percent_std', np.nan),
                    'accuracy': summary.get('accuracy',
                                            f"{summary.get('accuracy_percent_mean', np.nan):.2f} ± {summary.get('accuracy_percent_std', np.nan):.2f}"),
                    'f1_macro_mean': summary.get('f1_macro_mean', np.nan),
                    'f1_macro_std': summary.get('f1_macro_std', np.nan),
                    'f1_macro': summary.get('f1_macro',
                                            f"{summary.get('f1_macro_mean', np.nan):.4f} ± {summary.get('f1_macro_std', np.nan):.4f}"),
                    'balanced_accuracy_mean': summary.get('balanced_accuracy_mean', np.nan),
                    'balanced_accuracy_std': summary.get('balanced_accuracy_std', np.nan),
                    'balanced_accuracy': summary.get('balanced_accuracy',
                                                     f"{summary.get('balanced_accuracy_mean', np.nan):.4f} ± {summary.get('balanced_accuracy_std', np.nan):.4f}"),
                    'test_loss_mean': summary.get('loss_mean', np.nan),
                    'test_loss_std': summary.get('loss_std', np.nan),
                    'loss': summary.get('loss',
                                        f"{summary.get('loss_mean', np.nan):.4f} ± {summary.get('loss_std', np.nan):.4f}"),
                    'test_final_cls_loss_mean': summary.get('final_cls_loss_mean', np.nan),
                    'test_final_cls_loss_std': summary.get('final_cls_loss_std', np.nan),
                    'final_cls_loss': summary.get('final_cls_loss',
                                                  f"{summary.get('final_cls_loss_mean', np.nan):.4f} ± {summary.get('final_cls_loss_std', np.nan):.4f}"),
                    'total_time_seconds': elapsed,
                    'best_params': {k: getattr(args, k) for k in
                                    ['latent_dim', 'token_dim', 'recon_weight', 'layer_cls_weight', 'hidden_dims',
                                     'encoder_hidden_dims']}
                })

    if all_results:
        summary_df = pd.DataFrame(all_results)
        summary_df['total_time_minutes'] = summary_df['total_time_seconds'] / 60
        summary_df.to_csv(os.path.join(args.output_dir, 'all_results_summary.csv'), index=False)

        print("\n" + "=" * 80)
        print("GLOBAL RESULTS SUMMARY (mean ± std)")
        print("=" * 80)
        for _, row in summary_df.iterrows():
            ds = row['dataset']
            acc = row.get('accuracy', f"{row.get('accuracy_mean', np.nan):.2f} ± {row.get('accuracy_std', np.nan):.2f}")
            f1 = row.get('f1_macro', f"{row.get('f1_macro_mean', np.nan):.4f} ± {row.get('f1_macro_std', np.nan):.4f}")
            time_min = row.get('total_time_minutes', np.nan)
            print(f"{ds:20} | Acc: {acc:>15} | F1(macro): {f1:>15} | Time: {time_min:.2f} min")
        print(f"\nGlobal summary saved to {os.path.join(args.output_dir, 'all_results_summary.csv')}")
