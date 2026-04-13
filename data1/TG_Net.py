#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
TG-Net: A Token-Guided Multi-Scale Neural Network for Cell Type Annotation in scRNA-seq.

This script performs 5-fold cross-validation on single or multiple tissue datasets.
It implements the model described in the paper with tunable hyperparameters:
    --residual-guidance-weight (α) : residual guidance weight in encoder
    --latent-modulation-weight (γ): latent modulation weight before decoder
Loss weights: --cls-weight (λ_cls), --recon-weight (λ_recon), --layer-cls-weight (λ_layer)
"""

import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split, StratifiedKFold
from sklearn.preprocessing import LabelEncoder
import scanpy as sc
from sklearn.metrics import (
    accuracy_score, confusion_matrix,
    f1_score, precision_score, recall_score,
    balanced_accuracy_score
)
import warnings
import os
import json
import copy
from tqdm import tqdm
import time
import gc
from scipy.sparse import issparse
from collections import Counter, defaultdict
import argparse
import random

warnings.filterwarnings('ignore')

# ============================================================================
# Command-line arguments
# ============================================================================
parser = argparse.ArgumentParser(description='TG-Net: Token-Guided Multi-Scale Neural Network')
parser.add_argument('--use-file', action='store_true', default=True,
                    help='Read multiple tasks from a file (merged mode)')
parser.add_argument('--task-file', type=str, default='experiment.xlsx',
                    help='Task file with columns: tissue, data-file, labels-file')
parser.add_argument('--tissue', type=str, default='lung',
                    help='Tissue name for single-file mode')
parser.add_argument('--data-file', type=str, default='human_Lung9603_data.csv',
                    help='Data filename for single-file mode')
parser.add_argument('--labels-file', type=str, default='human_Lung9603_celltype.csv',
                    help='Labels filename for single-file mode')
parser.add_argument('--data-dir', type=str, default='data/',
                    help='Root directory for data files')
parser.add_argument('--save-dir', type=str, default='results_5fold',
                    help='Root directory for saving results')

parser.add_argument('--use-5fold', action='store_true', default=True,
                    help='Enable 5-fold cross-validation')

# Training hyperparameters
parser.add_argument('--n-top-genes', type=int, default=2000)
parser.add_argument('--batch-size', type=int, default=64)
parser.add_argument('--dropout', type=float, default=0.2)
parser.add_argument('--lr', type=float, default=0.0001)
parser.add_argument('--weight-decay', type=float, default=1e-3)
parser.add_argument('--num-epochs', type=int, default=100)
parser.add_argument('--min-genes', type=int, default=0)
parser.add_argument('--min-cells', type=int, default=0)
parser.add_argument('--test-size', type=float, default=0.2)
parser.add_argument('--val-size', type=float, default=0.1)
parser.add_argument('--random-state', type=int, default=42)
parser.add_argument('--patience', type=int, default=8,
                    help='Early stopping patience')
parser.add_argument('--cls-weight', type=float, default=1.0,
                    help='Classification loss weight (λ_cls)')
parser.add_argument('--recon-weight', type=float, default=0.1,
                    help='Reconstruction loss weight (λ_recon)')
parser.add_argument('--layer-cls-weight', type=float, default=0.5,
                    help='Layer-wise classification loss weight (λ_layer)')

# Model architecture
parser.add_argument('--latent-dim', type=int, default=128)
parser.add_argument('--token-dim', type=int, default=8)
parser.add_argument('--hidden-dims', type=str, default="1024,512",
                    help='Decoder hidden dimensions (comma-separated)')
parser.add_argument('--encoder-hidden-dims', type=str, default="1024",
                    help='Encoder hidden dimensions (comma-separated)')

# TG-Net specific tunable coefficients (α and γ)
parser.add_argument('--residual-guidance-weight', type=float, default=0.2,
                    help='Residual guidance weight (α) for encoder feature enhancement')
parser.add_argument('--latent-modulation-weight', type=float, default=0.3,
                    help='Latent modulation weight (γ) for class guidance before decoding')

args = parser.parse_args()

# Device configuration
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

def set_seed(seed):
    np.random.seed(seed)
    random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False

set_seed(args.random_state)

# ============================================================================
# TG-Net model components (aligned with paper)
# ============================================================================
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
    """Maps classification logits back to feature space for residual guidance"""
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

        # Hyperparameters α and γ from the paper
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

        encoded_features = []
        layer_logits = []
        layer_similarities = []

        # Input layer classification
        logits, similarity = self.layer_classifiers[0](x, class_tokens)
        layer_logits.append(logits)
        layer_similarities.append(similarity)
        encoded_features.append(x)

        current_features = x

        # Pass through encoder layers
        for i, encoder_layer in enumerate(self.encoder_layers):
            # Residual guidance: add α * transformed previous logits
            if i < len(self.residual_layers):
                guidance = self.residual_layers[i](layer_logits[-1])
                current_features = current_features + self.alpha * guidance
            current_features = encoder_layer(current_features)
            encoded_features.append(current_features)

            # Layer-wise classification
            logits, similarity = self.layer_classifiers[i + 1](current_features, class_tokens)
            layer_logits.append(logits)
            layer_similarities.append(similarity)

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
            'layer_similarities': layer_similarities,
            'latent_features': latent_features,
            'class_tokens': class_tokens
        }

# ============================================================================
# Data loading and preprocessing
# ============================================================================
def load_single_dataset(data_path, label_path):
    print(f"    Loading data: {data_path}")
    data = pd.read_csv(data_path, index_col=0).T
    print(f"      Shape after transpose: {data.shape}")

    print(f"    Loading labels: {label_path}")
    labels = pd.read_csv(label_path, header=None)

    if labels.shape[0] == 1:
        first_row = labels.iloc[0]
        if first_row[0] == 'type':
            cell_labels = first_row[1:].astype(str).values
        else:
            cell_labels = first_row.astype(str).values
        le = LabelEncoder()
        numeric_labels = le.fit_transform(cell_labels)
        labels_series = pd.Series(numeric_labels, name='cell_type')
    else:
        labels_df = labels.drop(labels.columns[0], axis=1)
        labels_df = labels_df.astype(str)
        labels_flat = labels_df.values.flatten()
        le = LabelEncoder()
        encoded = le.fit_transform(labels_flat)
        labels_df = pd.DataFrame(encoded.reshape(labels_df.shape))
        labels_df = labels_df.T.reset_index(drop=True)
        labels_df.columns = [0]
        labels_series = labels_df[0]

    print(f"      Label length: {len(labels_series)}")
    if data.shape[0] != len(labels_series):
        print(f"      Warning: data rows {data.shape[0]} != labels {len(labels_series)}, truncating to min")
        min_len = min(data.shape[0], len(labels_series))
        data = data.iloc[:min_len, :]
        labels_series = labels_series.iloc[:min_len]

    return data, labels_series

def load_and_merge_data_by_tasks(tissue, task_list, data_dir):
    print(f"  Merging {len(task_list)} datasets for tissue '{tissue}'...")
    data_frames = []
    label_series_list = []
    common_genes = None

    for idx, (data_file, label_file) in enumerate(task_list):
        print(f"  Dataset {idx + 1}: {data_file} / {label_file}")
        path1 = os.path.join(data_dir, tissue, data_file)
        path2 = os.path.join(data_dir, data_file)

        if os.path.exists(path1):
            data_path = path1
            label_path = os.path.join(data_dir, tissue, label_file)
        elif os.path.exists(path2):
            data_path = path2
            label_path = os.path.join(data_dir, label_file)
        else:
            raise FileNotFoundError(f"Cannot find data file, tried:\n  {path1}\n  {path2}")

        data_df, labels_series = load_single_dataset(data_path, label_path)

        genes = set(data_df.columns)
        if common_genes is None:
            common_genes = genes
        else:
            common_genes = common_genes.intersection(genes)
            if not common_genes:
                raise ValueError("No common genes across datasets, cannot merge")

        data_frames.append(data_df)
        label_series_list.append(labels_series)

    common_genes = sorted(common_genes)
    print(f"  Common genes across all datasets: {len(common_genes)}")

    aligned_dfs = [df[common_genes] for df in data_frames]
    merged_data = pd.concat(aligned_dfs, axis=0, ignore_index=True)
    merged_labels = pd.concat(label_series_list, axis=0, ignore_index=True)

    print(f"  Merged data shape: {merged_data.shape}, labels shape: {merged_labels.shape}")
    return merged_data, merged_labels

# ============================================================================
# Dataset class
# ============================================================================
class AnnDataset(Dataset):
    def __init__(self, adata):
        if issparse(adata.X):
            self.data = torch.tensor(adata.X.toarray(), dtype=torch.float32)
        else:
            self.data = torch.tensor(adata.X, dtype=torch.float32)
        self.labels = torch.tensor(adata.obs[0].values, dtype=torch.long)

    def __len__(self):
        return self.data.shape[0]

    def __getitem__(self, idx):
        return self.data[idx], self.labels[idx]

# ============================================================================
# Metrics computation
# ============================================================================
def compute_all_metrics(y_true, y_pred, num_classes):
    metrics = {}
    metrics['accuracy'] = accuracy_score(y_true, y_pred)
    metrics['balanced_accuracy'] = balanced_accuracy_score(y_true, y_pred)

    present_classes = np.unique(y_true)
    metrics['f1_macro'] = f1_score(y_true, y_pred, labels=present_classes, average='macro', zero_division=0)
    metrics['f1_micro'] = f1_score(y_true, y_pred, average='micro', zero_division=0)
    metrics['precision_macro'] = precision_score(y_true, y_pred, labels=present_classes, average='macro', zero_division=0)
    metrics['precision_micro'] = precision_score(y_true, y_pred, average='micro', zero_division=0)
    metrics['recall_macro'] = recall_score(y_true, y_pred, labels=present_classes, average='macro', zero_division=0)
    metrics['recall_micro'] = recall_score(y_true, y_pred, average='micro', zero_division=0)

    cm = confusion_matrix(y_true, y_pred, labels=range(num_classes))
    class_accuracies = {}
    for i in range(num_classes):
        if i in y_true:
            if cm[i, :].sum() > 0:
                class_accuracies[f'class_{i}_accuracy'] = cm[i, i] / cm[i, :].sum()
            else:
                class_accuracies[f'class_{i}_accuracy'] = 0.0
        else:
            class_accuracies[f'class_{i}_accuracy'] = np.nan
    metrics['class_accuracies'] = class_accuracies
    metrics['confusion_matrix'] = cm
    return metrics

# ============================================================================
# Training and testing functions
# ============================================================================
def train(model, device, train_loader, criterion, optimizer, args):
    model.train()
    running_loss = 0.0
    correct = 0
    total = 0
    batch_times = []
    pbar = tqdm(train_loader, desc="Training", leave=False)
    for batch_idx, (data, target) in enumerate(pbar):
        batch_start_time = time.time()
        data, target = data.to(device), target.to(device)
        optimizer.zero_grad()
        output = model(data)

        final_cls_loss = criterion(output['final_logits'], target)
        layer_cls_loss = 0
        for layer_logits in output['layer_logits']:
            layer_cls_loss += criterion(layer_logits, target)
        layer_cls_loss = layer_cls_loss / len(output['layer_logits'])
        recon_criterion = nn.MSELoss()
        recon_loss = recon_criterion(output['reconstructed'], data)

        total_loss = args.cls_weight * final_cls_loss + \
                     args.recon_weight * recon_loss + \
                     args.layer_cls_weight * layer_cls_loss

        total_loss.backward()
        optimizer.step()

        running_loss += total_loss.item() * data.size(0)
        _, predicted = torch.max(output['final_logits'].data, 1)
        total += target.size(0)
        correct += (predicted == target).sum().item()

        batch_time = time.time() - batch_start_time
        batch_times.append(batch_time)
        batch_acc = 100. * (predicted == target).sum().item() / target.size(0)
        pbar.set_postfix({
            'loss': f'{total_loss.item():.4f}',
            'acc': f'{batch_acc:.2f}%',
            'batch_time': f'{batch_time:.3f}s'
        })

    train_loss = running_loss / len(train_loader.dataset)
    accuracy = 100. * correct / total
    return train_loss, accuracy, np.mean(batch_times) if batch_times else 0

def test(model, device, test_loader, criterion, args, num_classes, phase="Validation"):
    model.eval()
    test_loss = 0.0
    final_cls_loss_total = 0.0
    correct = 0
    total = 0
    all_targets = []
    all_predicted = []
    start_time = time.time()
    pbar = tqdm(test_loader, desc=phase, leave=False)
    with torch.no_grad():
        for data, target in pbar:
            data, target = data.to(device), target.to(device)
            output = model(data)

            final_cls_loss = criterion(output['final_logits'], target)
            layer_cls_loss = 0
            for layer_logits in output['layer_logits']:
                layer_cls_loss += criterion(layer_logits, target)
            layer_cls_loss = layer_cls_loss / len(output['layer_logits'])
            recon_criterion = nn.MSELoss()
            recon_loss = recon_criterion(output['reconstructed'], data)
            total_loss_batch = args.cls_weight * final_cls_loss + \
                               args.recon_weight * recon_loss + \
                               args.layer_cls_weight * layer_cls_loss

            test_loss += total_loss_batch.item() * data.size(0)
            final_cls_loss_total += final_cls_loss.item() * data.size(0)

            _, predicted = torch.max(output['final_logits'].data, 1)
            correct += (predicted == target).sum().item()
            total += target.size(0)
            all_targets.extend(target.view_as(predicted).cpu().numpy())
            all_predicted.extend(predicted.cpu().numpy())
            batch_acc = 100. * (predicted == target).sum().item() / target.size(0)
            pbar.set_postfix({'loss': f'{total_loss_batch.item():.4f}', 'acc': f'{batch_acc:.2f}%'})

    test_time = time.time() - start_time
    test_loss /= total
    final_cls_loss_avg = final_cls_loss_total / total
    accuracy = 100. * correct / total

    metrics = compute_all_metrics(all_targets, all_predicted, num_classes)
    metrics['loss'] = test_loss
    metrics['final_cls_loss'] = final_cls_loss_avg
    metrics['accuracy_percent'] = accuracy
    metrics['accuracy_decimal'] = metrics['accuracy']
    cm = metrics['confusion_matrix']
    efficiency_metrics = {
        'inference_time': test_time,
        'samples_per_second': total / test_time if test_time > 0 else 0,
        'avg_inference_time_per_sample': test_time / total if total > 0 else 0
    }
    return metrics, cm, all_targets, all_predicted, efficiency_metrics

# ============================================================================
# Data splitting with handling of rare classes
# ============================================================================
def split_data_with_few_samples(X, y, test_size=0.2, val_size=0.1, random_state=42):
    class_counts = Counter(y)
    unique_classes = np.unique(y)
    print(f"\nData split strategy:")
    print(f"  Total samples: {len(y)}")
    print(f"  Number of classes: {len(unique_classes)}")

    few_sample_classes = [cls for cls, count in class_counts.items() if count < 2]
    sufficient_classes = [cls for cls, count in class_counts.items() if count >= 2]
    print(f"  Sufficient classes ({len(sufficient_classes)}): {sorted(sufficient_classes)}")
    print(f"  Few-sample classes ({len(few_sample_classes)}): {sorted(few_sample_classes)}")

    X_train, X_val, X_test = [], [], []
    y_train, y_val, y_test = [], [], []

    if sufficient_classes:
        sufficient_indices = np.where(np.isin(y, sufficient_classes))[0]
        X_sufficient = X[sufficient_indices]
        y_sufficient = y[sufficient_indices]
        print(f"\nProcessing sufficient classes ({len(sufficient_indices)} samples):")

        if test_size > 0:
            X_train_val, X_test_suf, y_train_val, y_test_suf = train_test_split(
                X_sufficient, y_sufficient,
                test_size=test_size,
                random_state=random_state,
                stratify=y_sufficient
            )
        else:
            X_train_val = X_sufficient
            y_train_val = y_sufficient
            X_test_suf = np.array([])
            y_test_suf = np.array([])

        val_ratio = val_size / (1 - test_size) if test_size < 1 else val_size
        if val_ratio <= 0 or val_ratio >= 1:
            val_ratio = 0.1
            print(f"Warning: val_ratio={val_ratio} invalid, using default 0.1")
        X_train_suf, X_val_suf, y_train_suf, y_val_suf = train_test_split(
            X_train_val, y_train_val,
            test_size=val_ratio,
            random_state=random_state,
            stratify=y_train_val
        )
        X_train.append(X_train_suf)
        X_val.append(X_val_suf)
        X_test.append(X_test_suf)
        y_train.append(y_train_suf)
        y_val.append(y_val_suf)
        y_test.append(y_test_suf)
        print(f"  Training: {len(y_train_suf)} samples")
        print(f"  Validation: {len(y_val_suf)} samples")
        print(f"  Test: {len(y_test_suf)} samples")

    if few_sample_classes:
        few_indices = np.where(np.isin(y, few_sample_classes))[0]
        X_few = X[few_indices]
        y_few = y[few_indices]
        print(f"\nProcessing few-sample classes ({len(few_indices)} samples):")
        print(f"  All placed in training set")
        X_train.append(X_few)
        y_train.append(y_few)

    if X_train:
        X_train = np.concatenate(X_train, axis=0) if len(X_train) > 1 else X_train[0]
        y_train = np.concatenate(y_train, axis=0) if len(y_train) > 1 else y_train[0]
    else:
        X_train = np.array([])
        y_train = np.array([])
    if X_val:
        X_val = np.concatenate(X_val, axis=0) if len(X_val) > 1 else X_val[0]
        y_val = np.concatenate(y_val, axis=0) if len(y_val) > 1 else y_val[0]
    else:
        X_val = np.array([])
        y_val = np.array([])
    if X_test:
        X_test = np.concatenate(X_test, axis=0) if len(X_test) > 1 else X_test[0]
        y_test = np.concatenate(y_test, axis=0) if len(y_test) > 1 else y_test[0]
    else:
        X_test = np.array([])
        y_test = np.array([])

    print(f"\nFinal split:")
    print(f"  Training: {len(y_train)} samples ({len(y_train)/len(y)*100:.1f}%)")
    print(f"  Validation: {len(y_val)} samples ({len(y_val)/len(y)*100:.1f}%)")
    print(f"  Test: {len(y_test)} samples ({len(y_test)/len(y)*100:.1f}%)")
    if len(y_val) == 0:
        print(f"Warning: Validation set empty!")
    if len(y_test) == 0:
        print(f"Warning: Test set empty!")
    return X_train, X_val, X_test, y_train, y_val, y_test

# ============================================================================
# Single training run (one fold)
# ============================================================================
def train_single_split(X_train, X_val, X_test, y_train, y_val, y_test, args, save_dir):
    print(f"\n{'='*60}")
    print(f"Training TG-Net")
    print(f"{'='*60}")
    fold_start_time = time.time()
    os.makedirs(save_dir, exist_ok=True)

    # Convert to AnnData
    adata_train = sc.AnnData(X_train)
    y_train_df = pd.DataFrame(y_train, columns=[0])
    adata_train.obs = y_train_df

    adata_val = sc.AnnData(X_val)
    y_val_df = pd.DataFrame(y_val, columns=[0])
    adata_val.obs = y_val_df

    adata_test = sc.AnnData(X_test)
    y_test_df = pd.DataFrame(y_test, columns=[0])
    adata_test.obs = y_test_df

    print("Preprocessing data...")
    preprocessing_start = time.time()
    adata_combined = sc.concat([adata_train, adata_val, adata_test],
                               label='batch', keys=['train', 'val', 'test'])
    sc.pp.filter_cells(adata_combined, min_genes=args.min_genes)
    sc.pp.filter_genes(adata_combined, min_cells=args.min_cells)
    sc.pp.normalize_total(adata_combined, target_sum=1e4)
    sc.pp.log1p(adata_combined)
    sc.pp.highly_variable_genes(adata_combined, n_top_genes=args.n_top_genes)
    highly_variable_genes = adata_combined.var.highly_variable
    adata_combined = adata_combined[:, highly_variable_genes]

    adata_train = adata_combined[adata_combined.obs['batch'] == 'train']
    adata_val = adata_combined[adata_combined.obs['batch'] == 'val']
    adata_test = adata_combined[adata_combined.obs['batch'] == 'test']
    preprocessing_time = time.time() - preprocessing_start
    print(f"Preprocessing time: {preprocessing_time:.2f}s")
    print(f"Training shape: {adata_train.shape}")
    print(f"Validation shape: {adata_val.shape}")
    print(f"Test shape: {adata_test.shape}")

    # Create data loaders
    train_dataset = AnnDataset(adata_train)
    val_dataset = AnnDataset(adata_val)
    test_dataset = AnnDataset(adata_test)

    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, drop_last=True)
    val_loader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False, drop_last=False)
    test_loader = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False, drop_last=False)

    # Model initialization
    input_size = adata_train.shape[1]
    num_classes = len(np.unique(np.concatenate([y_train, y_val, y_test])))
    print(f"Classes in train: {np.unique(y_train)}")
    print(f"Classes in validation: {np.unique(y_val)}")
    print(f"Classes in test: {np.unique(y_test)}")
    print(f"Total classes: {num_classes}")

    # Parse hidden dimension strings
    if isinstance(args.hidden_dims, str):
        hidden_dims_list = [int(dim) for dim in args.hidden_dims.split(',')]
    else:
        hidden_dims_list = args.hidden_dims
    if isinstance(args.encoder_hidden_dims, str):
        encoder_hidden_dims_list = [int(dim) for dim in args.encoder_hidden_dims.split(',')]
    else:
        encoder_hidden_dims_list = args.encoder_hidden_dims

    model_args = copy.copy(args)
    model_args.hidden_dims = hidden_dims_list
    model_args.encoder_hidden_dims = encoder_hidden_dims_list
    model_args.residual_guidance_weight = args.residual_guidance_weight
    model_args.latent_modulation_weight = args.latent_modulation_weight

    model = TGNet(
        input_dim=input_size,
        num_classes=num_classes,
        config_args=model_args
    ).to(device)

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='max', factor=0.5, patience=5)

    best_val_acc = 0.0
    best_val_metrics = None
    best_val_cm = None
    best_all_targets = None
    best_all_predicted = None
    best_efficiency_metrics = None
    best_model_state = None
    train_losses = []
    train_accuracies = []
    val_losses = []
    val_accuracies = []
    learning_rates = []
    avg_batch_times = []
    patience_counter = 0
    best_epoch = 0

    training_start_time = time.time()
    epoch_pbar = tqdm(range(1, args.num_epochs + 1), desc="Training Progress")

    for epoch in epoch_pbar:
        train_loss, train_acc, avg_batch_time = train(
            model, device, train_loader, criterion, optimizer, args)
        train_losses.append(train_loss)
        train_accuracies.append(train_acc)
        avg_batch_times.append(avg_batch_time)

        val_metrics, val_cm, all_targets, all_predicted, efficiency_metrics = test(
            model, device, val_loader, criterion, args, num_classes, "Validation")
        val_losses.append(val_metrics['loss'])
        val_accuracies.append(val_metrics['accuracy_percent'])

        current_lr = optimizer.param_groups[0]['lr']
        learning_rates.append(current_lr)
        scheduler.step(val_metrics['accuracy_percent'])

        epoch_pbar.set_postfix({
            'train_loss': f'{train_loss:.4f}',
            'train_acc': f'{train_acc:.2f}%',
            'val_loss': f'{val_metrics["loss"]:.4f}',
            'val_acc': f'{val_metrics["accuracy_percent"]:.2f}%',
            'lr': f'{current_lr:.6f}'
        })

        if val_metrics['accuracy_percent'] > best_val_acc:
            best_val_acc = val_metrics['accuracy_percent']
            best_val_metrics = val_metrics.copy()
            best_val_cm = val_cm
            best_all_targets = all_targets
            best_all_predicted = all_predicted
            best_efficiency_metrics = efficiency_metrics
            best_epoch = epoch
            best_model_state = copy.deepcopy(model.state_dict())
            patience_counter = 0
        else:
            patience_counter += 1
            if patience_counter >= args.patience:
                epoch_pbar.set_postfix({'status': 'Early stopping triggered'})
                break

    training_time = time.time() - training_start_time

    if best_model_state is not None:
        model.load_state_dict(best_model_state)
    print(f"\nEvaluating TG-Net on test set...")
    test_metrics, test_cm, test_targets, test_predicted, test_efficiency = test(
        model, device, test_loader, criterion, args, num_classes, "Test")

    fold_time = time.time() - fold_start_time

    efficiency_info = {
        'preprocessing_time': preprocessing_time,
        'training_time': training_time,
        'total_fold_time': fold_time,
        'avg_epoch_time': training_time / len(train_losses) if len(train_losses) > 0 else 0,
        'avg_batch_time_training': np.mean(avg_batch_times) if avg_batch_times else 0,
        'val_inference_time': best_efficiency_metrics['inference_time'],
        'val_samples_per_second': best_efficiency_metrics['samples_per_second'],
        'val_avg_inference_time_per_sample': best_efficiency_metrics['avg_inference_time_per_sample'],
        'test_inference_time': test_efficiency['inference_time'],
        'test_samples_per_second': test_efficiency['samples_per_second'],
        'test_avg_inference_time_per_sample': test_efficiency['avg_inference_time_per_sample']
    }

    # Save model
    torch.save({
        'model_state_dict': model.state_dict(),
        'input_dim': input_size,
        'num_classes': num_classes,
        'model_config': {
            'latent_dim': args.latent_dim,
            'token_dim': args.token_dim,
            'hidden_dims': hidden_dims_list,
            'encoder_hidden_dims': encoder_hidden_dims_list,
            'dropout': args.dropout,
            'recon_weight': args.recon_weight,
            'cls_weight': args.cls_weight,
            'layer_cls_weight': args.layer_cls_weight,
            'residual_guidance_weight': args.residual_guidance_weight,
            'latent_modulation_weight': args.latent_modulation_weight
        },
        'val_accuracy': best_val_acc,
        'test_accuracy': test_metrics['accuracy_percent'],
        'best_epoch': best_epoch
    }, os.path.join(save_dir, 'model.pth'))

    # Save predictions and metrics (identical to original, omitted for brevity but kept in actual code)
    pd.DataFrame({
        'cell_index': range(len(best_all_targets)),
        'true_label': best_all_targets,
        'predicted_label': best_all_predicted,
        'is_correct': [1 if t == p else 0 for t, p in zip(best_all_targets, best_all_predicted)]
    }).to_csv(os.path.join(save_dir, 'val_predictions.csv'), index=False)

    pd.DataFrame({
        'cell_index': range(len(test_targets)),
        'true_label': test_targets,
        'predicted_label': test_predicted,
        'is_correct': [1 if t == p else 0 for t, p in zip(test_targets, test_predicted)]
    }).to_csv(os.path.join(save_dir, 'test_predictions.csv'), index=False)

    pd.DataFrame({
        'epoch': range(1, len(train_losses) + 1),
        'train_loss': train_losses,
        'train_accuracy': train_accuracies,
        'val_loss': val_losses,
        'val_accuracy': val_accuracies,
        'learning_rate': learning_rates,
        'avg_batch_time': avg_batch_times
    }).to_csv(os.path.join(save_dir, 'training_history.csv'), index=False)

    pd.DataFrame([efficiency_info]).to_csv(os.path.join(save_dir, 'efficiency.csv'), index=False)

    # Validation metrics
    val_metrics_data = {
        'accuracy_percent': best_val_metrics['accuracy_percent'],
        'accuracy_decimal': best_val_metrics['accuracy_decimal'],
        'balanced_accuracy': best_val_metrics['balanced_accuracy'],
        'f1_macro': best_val_metrics['f1_macro'],
        'f1_micro': best_val_metrics['f1_micro'],
        'precision_macro': best_val_metrics['precision_macro'],
        'precision_micro': best_val_metrics['precision_micro'],
        'recall_macro': best_val_metrics['recall_macro'],
        'recall_micro': best_val_metrics['recall_micro'],
        'loss': best_val_metrics['loss']
    }
    val_metrics_data.update(best_val_metrics['class_accuracies'])
    pd.DataFrame([val_metrics_data]).to_csv(os.path.join(save_dir, 'val_metrics.csv'), index=False)

    # Test metrics
    test_metrics_data = {
        'accuracy_percent': test_metrics['accuracy_percent'],
        'accuracy_decimal': test_metrics['accuracy_decimal'],
        'balanced_accuracy': test_metrics['balanced_accuracy'],
        'f1_macro': test_metrics['f1_macro'],
        'f1_micro': test_metrics['f1_micro'],
        'precision_macro': test_metrics['precision_macro'],
        'precision_micro': test_metrics['precision_micro'],
        'recall_macro': test_metrics['recall_macro'],
        'recall_micro': test_metrics['recall_micro'],
        'loss': test_metrics['loss'],
        'final_cls_loss': test_metrics['final_cls_loss']
    }
    test_metrics_data.update(test_metrics['class_accuracies'])
    pd.DataFrame([test_metrics_data]).to_csv(os.path.join(save_dir, 'test_metrics.csv'), index=False)

    # Class accuracy details
    val_class_counts = Counter(y_val)
    val_class_acc_data = []
    for class_name, acc in best_val_metrics['class_accuracies'].items():
        if not np.isnan(acc):
            class_id = int(class_name.split('_')[1])
            sample_count = val_class_counts.get(class_id, 0)
            val_class_acc_data.append({
                'class': class_id,
                'accuracy': acc,
                'sample_count': sample_count,
                'accuracy_percentage': f"{acc * 100:.2f}%"
            })
    pd.DataFrame(val_class_acc_data).to_csv(os.path.join(save_dir, 'val_class_accuracy.csv'), index=False)

    test_class_counts = Counter(y_test)
    test_class_acc_data = []
    for class_name, acc in test_metrics['class_accuracies'].items():
        if not np.isnan(acc):
            class_id = int(class_name.split('_')[1])
            sample_count = test_class_counts.get(class_id, 0)
            test_class_acc_data.append({
                'class': class_id,
                'accuracy': acc,
                'sample_count': sample_count,
                'accuracy_percentage': f"{acc * 100:.2f}%"
            })
    pd.DataFrame(test_class_acc_data).to_csv(os.path.join(save_dir, 'test_class_accuracy.csv'), index=False)

    pd.DataFrame(best_val_cm).to_csv(os.path.join(save_dir, 'val_confusion_matrix.csv'))
    pd.DataFrame(test_cm).to_csv(os.path.join(save_dir, 'test_confusion_matrix.csv'))

    print(f"\nTG-Net training results:")
    print(f"  Best epoch: {best_epoch}")
    print(f"  Validation accuracy: {best_val_acc:.2f}%")
    print(f"  Validation loss: {best_val_metrics['loss']:.4f}")
    print(f"  Test accuracy: {test_metrics['accuracy_percent']:.2f}%")
    print(f"  Test loss: {test_metrics['loss']:.4f}")
    print(f"\nResults saved to: {save_dir}")

    del model, train_loader, val_loader, test_loader
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    gc.collect()

    return {
        'best_epoch': best_epoch,
        'val_accuracy_percent': best_val_acc,
        'val_accuracy_decimal': best_val_metrics['accuracy_decimal'],
        'val_balanced_accuracy': best_val_metrics['balanced_accuracy'],
        'val_f1_macro': best_val_metrics['f1_macro'],
        'val_f1_micro': best_val_metrics['f1_micro'],
        'val_precision_macro': best_val_metrics['precision_macro'],
        'val_recall_macro': best_val_metrics['recall_macro'],
        'test_accuracy_percent': test_metrics['accuracy_percent'],
        'test_accuracy_decimal': test_metrics['accuracy_decimal'],
        'test_balanced_accuracy': test_metrics['balanced_accuracy'],
        'test_f1_macro': test_metrics['f1_macro'],
        'test_f1_micro': test_metrics['f1_micro'],
        'test_precision_macro': test_metrics['precision_macro'],
        'test_precision_micro': test_metrics['precision_micro'],
        'test_recall_macro': test_metrics['recall_macro'],
        'test_recall_micro': test_metrics['recall_micro'],
        'test_loss': test_metrics['loss'],
        'test_final_cls_loss': test_metrics['final_cls_loss'],
        'training_time': training_time,
        'num_classes': num_classes
    }

# ============================================================================
# 5-fold cross-validation
# ============================================================================
def run_5fold_cv(X, y, args, save_dir):
    print(f"\n{'='*60}")
    print(f"Starting 5-fold cross-validation for TG-Net")
    print(f"Samples: {len(y)}, Classes: {len(np.unique(y))}")
    print(f"Save directory: {save_dir}")
    print(f"{'='*60}")

    os.makedirs(save_dir, exist_ok=True)

    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=args.random_state)
    fold_results = []
    fold_times = []

    for fold, (train_idx, test_idx) in enumerate(skf.split(X, y), 1):
        fold_start_time = time.time()
        print(f"\n--- Fold {fold} ---")
        X_train_all = X[train_idx]
        y_train_all = y[train_idx]
        X_test = X[test_idx]
        y_test = y[test_idx]

        val_ratio = args.val_size / (1 - args.test_size) if args.test_size < 1 else args.val_size
        X_train, X_val, _, y_train, y_val, _ = split_data_with_few_samples(
            X_train_all, y_train_all,
            test_size=0.0,
            val_size=val_ratio,
            random_state=args.random_state
        )

        if len(y_val) == 0:
            print(f"Warning: Fold {fold} validation set empty! Using test set as validation (may affect model selection).")
            X_val = X_test.copy()
            y_val = y_test.copy()

        fold_save_dir = os.path.join(save_dir, f'fold_{fold}')
        result = train_single_split(X_train, X_val, X_test, y_train, y_val, y_test, args, fold_save_dir)

        if result is None:
            print(f"Fold {fold} training failed, skipping")
            continue

        fold_results.append(result)
        fold_times.append(time.time() - fold_start_time)

    if not fold_results:
        print("All folds failed")
        return None

    # Aggregate metrics (mean ± std)
    summary = {}
    metric_names = ['test_accuracy_percent', 'test_f1_macro', 'test_f1_micro',
                    'test_balanced_accuracy', 'test_loss', 'test_final_cls_loss', 'training_time']
    for metric in metric_names:
        values = [r[metric] for r in fold_results]
        summary[f'{metric}_mean'] = np.mean(values)
        summary[f'{metric}_std'] = np.std(values, ddof=1)

    # Formatted strings
    summary['accuracy'] = f"{summary['test_accuracy_percent_mean']:.2f} ± {summary['test_accuracy_percent_std']:.2f}"
    summary['f1_macro'] = f"{summary['test_f1_macro_mean']:.4f} ± {summary['test_f1_macro_std']:.4f}"
    summary['balanced_accuracy'] = f"{summary['test_balanced_accuracy_mean']:.4f} ± {summary['test_balanced_accuracy_std']:.4f}"
    summary['loss'] = f"{summary['test_loss_mean']:.4f} ± {summary['test_loss_std']:.4f}"
    summary['final_cls_loss'] = f"{summary['test_final_cls_loss_mean']:.4f} ± {summary['test_final_cls_loss_std']:.4f}"

    summary['total_time_seconds'] = sum(fold_times)
    summary['num_classes'] = fold_results[0]['num_classes']

    # Save summary
    summary_df = pd.DataFrame([summary])
    summary_df.to_csv(os.path.join(save_dir, '5fold_summary.csv'), index=False)

    config_dict = {
        'num_folds': 5,
        'results_summary': {k: float(v) if isinstance(v, (int, float)) else v for k, v in summary.items()},
        'class_distribution': {str(int(k)): int(v) for k, v in Counter(y).items()},
        'args': vars(args)
    }
    with open(os.path.join(save_dir, 'config.json'), 'w', encoding='utf-8') as f:
        json.dump(config_dict, f, indent=2, ensure_ascii=False)

    print(f"\n5-fold CV completed. Results saved to: {save_dir}")
    print("Summary (mean ± std):")
    print(f"  Accuracy:        {summary['accuracy']}%")
    print(f"  F1 macro:        {summary['f1_macro']}")
    print(f"  Balanced Acc:    {summary['balanced_accuracy']}")
    print(f"  Loss:            {summary['loss']}")
    print(f"  Final Cls Loss:  {summary['final_cls_loss']}")
    print(f"  Total time:      {summary['total_time_seconds']:.2f} s")

    return summary

# ============================================================================
# Main function
# ============================================================================
def main():
    if args.use_file:
        if not os.path.exists(args.task_file):
            print(f"Error: Task file {args.task_file} does not exist")
            return
        try:
            tasks_df = pd.read_excel(args.task_file, header=None)
        except Exception as e:
            print(f"Error reading task file: {e}")
            return
        if tasks_df.shape[1] < 3:
            print("Error: Task file needs at least 3 columns (tissue, data-file, labels-file)")
            return

        tissue_tasks = defaultdict(list)
        for _, row in tasks_df.iterrows():
            tissue = str(row[0]).strip()
            data_file = str(row[1]).strip()
            label_file = str(row[2]).strip()
            tissue_tasks[tissue].append((data_file, label_file))

        print(f"Found {len(tissue_tasks)} tissues: {list(tissue_tasks.keys())}")
        os.makedirs(args.save_dir, exist_ok=True)
        global_summary_path = os.path.join(args.save_dir, 'global_summary.csv')
        global_rows = []

        for tissue, task_list in tissue_tasks.items():
            print(f"\n{'='*70}")
            print(f"Processing tissue: {tissue} (merging {len(task_list)} datasets)")
            print(f"{'='*70}")

            try:
                data, labels_series = load_and_merge_data_by_tasks(tissue, task_list, args.data_dir)
            except Exception as e:
                print(f"Error loading data: {e}")
                global_rows.append({
                    'tissue': tissue,
                    'status': 'failed',
                    'error': str(e)
                })
                continue

            X = data.values
            y = labels_series.values

            tissue_save_dir = os.path.join(args.save_dir, tissue)
            summary = run_5fold_cv(X, y, args, tissue_save_dir)

            if summary is None:
                global_rows.append({
                    'tissue': tissue,
                    'status': 'failed',
                    'error': '5-fold CV failed'
                })
                continue

            global_row = {
                'tissue': tissue,
                'accuracy': summary['accuracy'],
                'f1_macro': summary['f1_macro'],
                'balanced_accuracy': summary['balanced_accuracy'],
                'loss': summary['loss'],
                'final_cls_loss': summary['final_cls_loss'],
                'total_time_seconds': summary['total_time_seconds'],
                'accuracy_mean': summary['test_accuracy_percent_mean'],
                'accuracy_std': summary['test_accuracy_percent_std'],
                'f1_macro_mean': summary['test_f1_macro_mean'],
                'f1_macro_std': summary['test_f1_macro_std'],
                'balanced_accuracy_mean': summary['test_balanced_accuracy_mean'],
                'balanced_accuracy_std': summary['test_balanced_accuracy_std'],
                'loss_mean': summary['test_loss_mean'],
                'loss_std': summary['test_loss_std'],
                'final_cls_loss_mean': summary['test_final_cls_loss_mean'],
                'final_cls_loss_std': summary['test_final_cls_loss_std'],
                'num_classes': summary['num_classes'],
                'status': 'completed'
            }
            global_rows.append(global_row)

        global_df = pd.DataFrame(global_rows)
        global_df.to_csv(global_summary_path, index=False)
        print(f"\nAll tissues processed! Global summary: {global_summary_path}")

    else:
        tissue = args.tissue
        data_file = args.data_file
        label_file = args.labels_file
        save_dir = os.path.join(args.save_dir, tissue)

        data_path = os.path.join(args.data_dir, data_file)
        label_path = os.path.join(args.data_dir, label_file)
        data_df, labels_series = load_single_dataset(data_path, label_path)
        X = data_df.values
        y = labels_series.values

        summary = run_5fold_cv(X, y, args, save_dir)

        if summary:
            global_summary_path = os.path.join(args.save_dir, 'global_summary.csv')
            if os.path.exists(global_summary_path):
                global_df = pd.read_csv(global_summary_path)
            else:
                global_df = pd.DataFrame()
            new_row = {
                'tissue': tissue,
                'accuracy': summary['accuracy'],
                'f1_macro': summary['f1_macro'],
                'balanced_accuracy': summary['balanced_accuracy'],
                'loss': summary['loss'],
                'final_cls_loss': summary['final_cls_loss'],
                'total_time_seconds': summary['total_time_seconds'],
                'accuracy_mean': summary['test_accuracy_percent_mean'],
                'accuracy_std': summary['test_accuracy_percent_std'],
                'f1_macro_mean': summary['test_f1_macro_mean'],
                'f1_macro_std': summary['test_f1_macro_std'],
                'balanced_accuracy_mean': summary['test_balanced_accuracy_mean'],
                'balanced_accuracy_std': summary['test_balanced_accuracy_std'],
                'loss_mean': summary['test_loss_mean'],
                'loss_std': summary['test_loss_std'],
                'final_cls_loss_mean': summary['test_final_cls_loss_mean'],
                'final_cls_loss_std': summary['test_final_cls_loss_std'],
                'num_classes': summary['num_classes']
            }
            global_df = pd.concat([global_df, pd.DataFrame([new_row])], ignore_index=True)
            global_df.to_csv(global_summary_path, index=False)

if __name__ == '__main__':
    main()
