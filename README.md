# Harnessing Test-time Adaptation for NLU Tasks Involving Dialects of English

> “Adapting at test-time to bridge dialect gaps in NLP.”  

[![MIT License](https://img.shields.io/badge/License-MIT-blue.svg)](LICENSE.txt)  
[![Python Version](https://img.shields.io/badge/Python-3.12.7-blue.svg)](https://www.python.org/downloads/)

## Table of Contents

- [Overview](#overview)
- [Setup](#setup)
- [Experiments](#experiments)
- [License](#license)
- [Acknowledgements](#acknowledgements)

## Overview

This project leverages **Test-time Adaptation (TTA)**—specifically the SHOT technique—to enhance natural language understanding across various English dialects. Traditionally, models are trained on Standard American English (SAE) and then face performance drops when applied to dialects like Indian, Nigerian, or Singaporean English. Here, we address this gap by adapting models on-the-fly during inference without needing additional labeled data.

## Setup

Follow these steps to get your environment ready:

```bash
# Create and activate the conda environment with Python 3.12.7
conda create --name tta python=3.12.7
conda activate tta

# Install all necessary dependencies
% pip install -r requirements.txt
pip install torch==2.4.1
pip install numpy===1.26.4 pandas==2.2.2 torchaudio torchvision transformers==4.44.2 evaluate==0.4.3 datasets==2.19.1 accelerate==1.0.1 scipy==1.13.1
```

- **Clone & Install `multi_value`:**  
  Clone the [`multi_value`](https://github.com/SALT-NLP/multi-value) repository into your working directory and install it as per the instructions provided there.

- **Prepare the Dialectal Datasets:**  
  Use `multi_value` to generate dialectal GLUE datasets and save them under `DATASET_PATH/multivalue`.

- **Configure Dataset Paths:**  
  Update the `DATASET_PATH` constant in both `train.py` and `test_shot.py` to point to your datasets.

## Experiments

### Training

Run training with SHOT on the CoLA dataset by executing:

```bash
CUDA_VISIBLE_DEVICES=1,2,3,4,5,6 accelerate launch train.py --seed 42 --max_epoch 30 --dset cola
```

### Test-time Adaptation

Evaluate test-time adaptation using a pre-trained model (`roberta-base`):

```bash
CUDA_VISIBLE_DEVICES=1,2,3,4,5,6 accelerate launch --main_process_port 29600 test_shot.py --dset cola --model_name roberta-base --seed 42 --max_epoch 30 --int_filename cola_SAE_train_0 --validation_dataset cola_Indian_validation --val_size 0
```

These commands utilize multiple GPUs for accelerated training and testing.

## Acknowledgement
The production of this code was partially possible through the use of code assistants, generative AI tools.

## License

Distributed under the MIT License. See [LICENSE.txt](LICENSE.txt) for more details.

<p align="right"><a href="#top">Back to top</a></p>
