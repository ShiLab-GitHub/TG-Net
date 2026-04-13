```markdown
# TG-Net

Official code for **TG-Net: A Token-Guided Multi-Scale Neural Network for Cell Type Annotation in Single-Cell RNA Sequencing**.

## Datasets

### Dataset 1 (9 human tissues)
Download from the scFormer paper:  
[https://figshare.com/s/7651fa9725807dfb3dbc](https://figshare.com/s/7651fa9725807dfb3dbc)

### Dataset 2 (8 datasets from WCSGNet)
Use the following download links:

1. **Baron Human** – [Filtered Baron HumanPancreas data.csv](https://zenodo.org/record/3357167/files/Filtered%20Baron%20HumanPancreas%20data.csv)
2. **Baron Mouse** – [FilteredMousePancreas data.csv](https://zenodo.org/record/3357167/files/FilteredMousePancreas%20data.csv)
3. **Muraro** – [Filtered_Muraro_HumanPancreas_data.csv](https://zenodo.org/record/3357167/files/Filtered_Muraro_HumanPancreas_data.csv)
4. **Segerstolpe** – [Filtered_Segerstolpe_HumanPancreas_data.csv](https://zenodo.org/record/3357167/files/Filtered_Segerstolpe_HumanPancreas_data.csv)
5. **AMB** – [Filtered_mouse_allen_brain_data.csv](https://zenodo.org/record/3357167/files/Filtered_mouse_allen_brain_data.csv)
6. **TM** – [Filtered_TM_data.csv](https://zenodo.org/record/3357167/files/Filtered_TM_data.csv)
7. **Zheng68K** – [Filtered68KPBMC data.csv](https://zenodo.org/record/3357167/files/Filtered68KPBMC%20data.csv)
8. **Kang** – [GEO GSE96583](https://www.ncbi.nlm.nih.gov/geo/query/acc.cgi?acc=GSE96583) (download `GSE96583_batch2.genes.tsv.gz`, `GSM2560248_2.1.mtx`, `GSM2560248barcodes.tsv`)

## Requirements

```bash
pip install -r requirements.txt
```

## Usage

### Dataset 1 (no preprocessing)
```bash
python TG-Net.py
```

### Dataset 2 (preprocess first)
```bash
python data_pre.py    # run once
python TG-Net.py
```

> Each dataset folder already contains the corresponding code. Simply place the downloaded files in the correct folder and run `python TG-Net.py`.

## Citation

If you use our work, please cite:

- **TG-Net paper** (to be updated)
- **Dataset 1**: Qin et al., *scFormer: A Transformer-Based Cell Type Annotation Method for scRNA-seq Data Using Smooth Gene Embedding and Global Features*. Journal of Chemical Information and Modeling, 2024. DOI: [10.1021/acs.jcim.4c00616](https://doi.org/10.1021/acs.jcim.4c00616)
- **Dataset 2**: Wang et al., *WCSGNet: Weighted Cell-Specific Graph Neural Network for Cell Type Annotation*, 2025. (The paper that compiled these eight datasets)
```
