# CORAL
Implementation of the paper "Coordinative Learning with Ordinal and Relational Priors for Volumetric Medical Image Segmentation".

**Note:** This codebase is based on [https://github.com/dewenzeng/positional_cl](https://github.com/dewenzeng/positional_cl).

## Requirements

Install the required dependencies:

```bash
pip install -r requirements.txt
```
### Data Preparation

**Data preparation is based on [https://github.com/dewenzeng/positional_cl](https://github.com/dewenzeng/positional_cl).**

Use the preprocessing scripts in the `dataset/` folder to convert the original data into `.npy` format for training and testing. For example:

```bash
# Convert ACDC dataset
python dataset/generate_acdc.py -i raw_image_dir -out_labeled save_dir_for_labeled_data -out_unlabeled save_dir_for_unlabeled_data
```

## Usage

### 1. Contrastive Pre-training

To run contrastive pre-training, use `train_contrast.py`. **Use the `--contrastive_method` argument to specify different methods:**

- **`coral`**: Use the CORAL method (our proposed method)
- **`pcl`**: Use the Positional Contrastive Learning method
- **`gcl`**: Use the Global Contrastive Learning method
- **`simclr`**: Use the SimCLR baseline

#### CORAL on ACDC dataset

```bash
python train_contrast.py --batch_size 32 --epochs 300 \
  --data_dir /path/to/acdc_dataset --lr 0.1 --do_contrast --dataset acdc --patch_size 352 352 \
  --experiment_name coral_acdc_ --slice_threshold 0.35 --temp 0.1 --initial_filter_size 48 \
  --classes 512 --contrastive_method coral
```

#### PCL on ACDC dataset

```bash
python train_contrast.py --batch_size 32 --epochs 300 \
  --data_dir /path/to/acdc_dataset --lr 0.1 --do_contrast --dataset acdc --patch_size 352 352 \
  --experiment_name pcl_acdc_ --slice_threshold 0.35 --temp 0.1 --initial_filter_size 48 \
  --classes 512 --contrastive_method pcl
```

### 2. Supervised Fine-tuning

After pre-training, use `train_supervised.py` for supervised fine-tuning with limited labeled data.

## Citation

If you find this work useful, please cite the our paper:

```bibtex
@article{wang2025coordinative,
  title={Coordinative Learning with Ordinal and Relational Priors for Volumetric Medical Image Segmentation},
  author={Wang, Haoyi},
  journal={arXiv preprint arXiv:2511.11276},
  year={2025}
}
```