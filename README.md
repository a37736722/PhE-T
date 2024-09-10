# PhE-T: A Transformer-based Approach for Phenotype Representation and Multi-task Disease Risk Prediction

This repository contains the implementation of PhE-T, a novel framework for multi-task disease risk prediction using transformer architectures.

## Table of Contents

1. [Setup](#setup)
2. [Data Preparation](#data-preparation)
3. [Training](#training)
4. [Prediction](#prediction)
5. [Evaluation](#evaluation)

## Setup

### Prerequisites

- Git
- Python 3.12
- Access to UK Biobank data

### Installation

1. Clone the repository:

```bash
git clone --recurse-submodules -j8 https://github.com/a37736722/PhE-T.git
```

2. Navigate to the project directory:

```bash
cd PhE-T
```

3. Install dependencies:

```bash
pip install -r requirements.txt
```

## Data Preparation

### Environment Setup

Set the required environment variables:

```bash
export UKB_FOLDER=<path_to_ukb_folder>
export PROJECT_ID=<your_project_id>
```

### Raw Data Creation

Follow these steps to create and preprocess the raw data:

```bash
cd UKB-Tools
python commands/get_newest_baskets.py $UKB_FOLDER $PROJECT_ID ../data/ukb_fields.txt ../data/field_to_basket.json
python commands/create_data.py $UKB_FOLDER ../data/field_to_basket.json ../data/raw.csv
cd ..
```

2. Preprocess the raw data:

```bash
python preprocess.py data/raw.csv configs/preprocess_cfg.py data/preprocessed.csv
```

3. Generate train/val/test splits:

```bash
python scripts/split.py --data_path data/preprocessed.csv --val_size 10000 --test_size 10000 --save_dir data/
```

4. Prepare spirometry data by running the notebook: [notebooks/prepare_spiro.ipynb](notebooks/prepare_spiro.ipynb)

## Training

Train the models using the following commands:

1. PhE-T model:

```bash
python train.py --model='phet' --nb_epochs=10 --nb_gpus=1 --nb_nodes=1 --nb_workers=20 --pin_memory --config='configs/train_phet_cfg.py' --run_name='v0'
```

2. Asthma ResNet model:

```bash
python train.py --model='as-resnet' --nb_epochs=10 --nb_gpus=1 --nb_nodes=1 --nb_workers=20 --pin_memory --config='configs/train_as-resnet_cfg.py' --run_name='v0'
```

3. Asthma PhE-T model:

```bash
python train.py --model='as-phet' --nb_epochs=10 --nb_gpus=1 --nb_nodes=1 --nb_workers=20 --pin_memory --config='configs/train_as-phet_cfg.py' --run_name='v0'
```

## Prediction

Generate predictions using the trained models:

1. PhE-T predictions:

```bash
python predict.py --model='phet' --ckpt_path=ckpts/PhE-T/v0/best-epoch=3-step=3842.ckpt --out_dir=scores/phet --nb_workers=20 --config='configs/train_phet_cfg.py'
```

2. Asthma PhE-T predictions:

```bash
python predict.py --model='as-phet' --ckpt_path=ckpts/AsthmaPhE-T/v0/best-epoch=6-step=5030.ckpt --out_dir=scores/as-phet --nb_workers=20 --config='configs/train_as-phet_cfg.py'
```

## Evaluation

Generate results for the predictions:

1. PhE-T results:

```bash
python scripts/generate_results.py scores/phet results/phet
```

2. Asthma PhE-T results:

```bash
python scripts/generate_results.py scores/as-phet results/as-phet
```

## Additional Information

For more detailed information about the PhE-T framework, its performance, and comparisons with other models, please refer to the full paper.
