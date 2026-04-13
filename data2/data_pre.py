#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Unified preprocessing script for all scRNA-seq datasets.
Reads raw data, filters cells and genes, and saves as .npz files.
"""

import os
import sys
import argparse
import numpy as np
import pandas as pd
from scipy.io import mmread
from pathlib import Path

# ------------------------------ Configuration ------------------------------
# Base paths (adjust these if your directory structure differs)
BASE_DATA_DIR = "../../data/scRNAseq_Benchmark_datasets/"
BASE_OUTPUT_DIR = "../../dataset/pre_data/scRNAseq_datasets/"

# Dataset-specific information
DATASETS = {
    "AMB": {
        "expr_file": "AMB/Filtered_mouse_allen_brain_data.csv",
        "label_file": "AMB/Labels.csv",
        "label_col": "Class",               # column name for cell types
        "expr_type": "csv",
        "special_filter": None,
    },
    "Baron_Human": {
        "expr_file": "Baron Human/Filtered_Baron_HumanPancreas_data.csv",
        "label_file": "Baron Human/Labels.csv",
        "label_col": "x",
        "expr_type": "csv",
        "special_filter": None,
    },
    "Baron_Mouse": {
        "expr_file": "Baron Mouse/Filtered_MousePancreas_data.csv",
        "label_file": "Baron Mouse/Labels.csv",
        "label_col": "x",
        "expr_type": "csv",
        "special_filter": None,
    },
    "Kang_ctrl": {
        "expr_file": "Kang_PBMCs/GSM2560248_2.1.mtx",
        "label_file": "Kang_PBMCs/GSE96583_batch2.total.tsne.df.tsv",
        "gene_file": "Kang_PBMCs/GSE96583_batch2.genes.tsv",
        "barcode_file": "Kang_PBMCs/GSM2560248_barcodes.tsv",
        "label_col": "cell",                # column for cell types after processing
        "expr_type": "mtx",
        "special_filter": "kang",           # special processing for Kang
    },
    "Muraro": {
        "expr_file": "Muraro/Filtered_Muraro_HumanPancreas_data.csv",
        "label_file": "Muraro/Labels.csv",
        "label_col": "x",
        "expr_type": "csv",
        "special_filter": None,
    },
    "Segerstolpe": {
        "expr_file": "Segerstolpe/Filtered_Segerstolpe_HumanPancreas_data.csv",
        "label_file": "Segerstolpe/Labels.csv",
        "label_col": "x",
        "expr_type": "csv",
        "special_filter": "segerstolpe",   # exclude 'co-expression' type
    },
    "TM": {
        "expr_file": "TM/Filtered_TM_data.csv",
        "label_file": "TM/Labels.csv",
        "label_col": "x",
        "expr_type": "csv",
        "special_filter": None,
    },
    # "Zhang_T": {
    #     "expr_file": "Zhang_T/GSE108989_CRC.TCell.S11138.count.txt",
    #     "label_file": "Zhang_T/GSE108989-tbl-1.txt",
    #     "gene_col": "symbol",               # column with gene symbols
    #     "label_col": None,                  # handled specially
    #     "expr_type": "zhang",
    #     "special_filter": "zhang",
    # },
    "Zheng68K": {
        "expr_file": "Zheng 68K/Filtered_68K_PBMC_data.csv",
        "label_file": "Zheng 68K/Labels.csv",
        "label_col": "x",
        "expr_type": "csv",
        "special_filter": None,
    }
}

# ------------------------------ Helper Functions ------------------------------
def filter_cell_types(label_series, min_cells=10, exclude=None):
    """
    Filter cell types with at least min_cells, optionally excluding a specific type.
    Returns boolean mask for rows to keep.
    """
    counts = label_series.value_counts()
    if exclude:
        counts = counts[counts.index != exclude]
    keep_types = counts[counts >= min_cells].index
    return label_series.isin(keep_types)

def filter_genes(expr_df, min_cells=10):
    """
    Filter genes expressed in at least min_cells cells.
    Returns filtered DataFrame.
    """
    gene_exp_counts = (expr_df > 0).sum(axis=0)
    keep_genes = gene_exp_counts[gene_exp_counts >= min_cells].index
    return expr_df[keep_genes]

def save_npz(expr_df, labels, str_labels, barcodes, gene_symbols, output_file):
    """Save processed data as .npz."""
    data_dict = {
        'gene_symbol': gene_symbols,
        'count': expr_df.values.astype(np.float32),
        'str_labels': str_labels,
        'label': labels,
        'barcode': barcodes,
    }
    np.savez(output_file, **data_dict)
    print(f"Saved to {output_file}")

# ------------------------------ Dataset-specific Readers ------------------------------
def read_csv_dataset(expr_file, label_file, label_col):
    """Read expression (CSV) and labels (CSV), align indices."""
    expr_df = pd.read_csv(expr_file, index_col=0)
    label_df = pd.read_csv(label_file, header=0)
    # Align labels with expression rows
    label_df = label_df.set_index(expr_df.index)
    labels_series = label_df[label_col]
    return expr_df, labels_series

def read_kang_dataset(expr_file, label_file, gene_file, barcode_file):
    """Special reader for Kang (MTX format)."""
    # Read genes
    gene_df = pd.read_csv(gene_file, sep='\t', header=None)
    genes = gene_df.iloc[:, 0].values
    # Read barcodes
    barcodes = pd.read_csv(barcode_file, sep='\t', header=None)[0].values
    # Read MTX
    sparse_mat = mmread(expr_file).toarray().astype(int)
    # Transpose so rows are cells, columns are genes
    expr_df = pd.DataFrame(sparse_mat.T, index=barcodes, columns=genes)
    # Read labels from the main TSV
    all_labels = pd.read_csv(label_file, sep='\t', index_col=0)
    # Keep only control group and singlet cells
    ctrl_df = all_labels[all_labels['stim'] == 'ctrl']
    singlet_indices = ctrl_df[ctrl_df['multiplets'] == 'singlet'].index
    expr_df = expr_df.loc[singlet_indices]
    labels_series = ctrl_df.loc[singlet_indices, 'cell']
    # Remove cells with missing labels
    valid = labels_series.notna()
    expr_df = expr_df[valid]
    labels_series = labels_series[valid]
    return expr_df, labels_series

def read_zhang_dataset(expr_file, label_file):
    """Special reader for Zhang_T dataset."""
    # Expression
    expr_df = pd.read_csv(expr_file, sep='\t')
    gene_symbol = expr_df['symbol']
    expr_df = expr_df.set_index('geneID')
    expr_df.drop(columns='symbol', inplace=True)
    expr_df = expr_df.T  # rows = cells, columns = genes
    # Labels
    label_df = pd.read_csv(label_file, delimiter='\t', header=None, index_col=0)
    # Keep only CD4/CD8 cells
    valid_cell_types = label_df.iloc[:, 1].str.startswith(('CD4', 'CD8'))
    label_df = label_df[valid_cell_types]
    expr_df = expr_df.loc[label_df.index]
    labels_series = label_df.iloc[:, 1]
    # Remove cells with missing labels
    valid = labels_series.notna()
    expr_df = expr_df[valid]
    labels_series = labels_series[valid]
    return expr_df, labels_series

# ------------------------------ Main Processing ------------------------------
def process_dataset(dataset_name):
    """Process a single dataset."""
    print(f"\n=== Processing {dataset_name} ===")
    cfg = DATASETS[dataset_name]

    # Construct full paths
    data_dir = Path(BASE_DATA_DIR)
    expr_path = data_dir / cfg['expr_file']
    label_path = data_dir / cfg['label_file'] if 'label_file' in cfg else None

    # Read expression and labels based on type
    if cfg['expr_type'] == 'csv':
        expr_df, labels_series = read_csv_dataset(expr_path, label_path, cfg['label_col'])
    elif cfg['expr_type'] == 'mtx':
        # Kang dataset
        gene_path = data_dir / cfg['gene_file']
        barcode_path = data_dir / cfg['barcode_file']
        expr_df, labels_series = read_kang_dataset(expr_path, label_path, gene_path, barcode_path)
    elif cfg['expr_type'] == 'zhang':
        expr_df, labels_series = read_zhang_dataset(expr_path, label_path)
    else:
        raise ValueError(f"Unknown expression type for {dataset_name}")

    print(f"Initial shape: {expr_df.shape}, labels: {len(labels_series)}")

    # Apply special filtering for Segerstolpe (exclude 'co-expression')
    exclude = None
    if cfg.get('special_filter') == 'segerstolpe':
        exclude = 'co-expression'

    # Filter cell types
    keep_cells = filter_cell_types(labels_series, min_cells=10, exclude=exclude)
    expr_df = expr_df[keep_cells]
    labels_series = labels_series[keep_cells]
    print(f"After cell filtering: {expr_df.shape}, labels: {len(labels_series)}")

    # Filter genes expressed in at least 10 cells
    expr_df = filter_genes(expr_df, min_cells=10)
    print(f"After gene filtering: {expr_df.shape}")

    # Convert to float32
    expr_df = expr_df.astype('float32')

    # Prepare labels
    unique_labels = labels_series.unique().tolist()
    label_indices = [unique_labels.index(x) for x in labels_series]

    # Save
    output_file = Path(BASE_OUTPUT_DIR) / f"{dataset_name}.npz"
    os.makedirs(BASE_OUTPUT_DIR, exist_ok=True)
    save_npz(expr_df, label_indices, unique_labels,
             expr_df.index.values, expr_df.columns.values, output_file)

def main():
    parser = argparse.ArgumentParser(description="Preprocess scRNA-seq datasets.")
    parser.add_argument("--dataset", type=str, default=None,
                        help="Dataset name to process (if not specified, process all)")
    args = parser.parse_args()

    if args.dataset:
        if args.dataset not in DATASETS:
            print(f"Error: Unknown dataset '{args.dataset}'. Available: {list(DATASETS.keys())}")
            sys.exit(1)
        process_dataset(args.dataset)
    else:
        for dname in DATASETS:
            process_dataset(dname)

if __name__ == "__main__":
    main()
