<!-- Original code

SHOT: https://github.com/tim-learn/SHOT
SHOT++: https://github.com/tim-learn/SHOT-plus

BMD: https://github.com/ispc-lab/BMD 
NRC: https://github.com/Albert0147/NRC_SFDA 
LD: https://github.com/fumyou13/LDBE 
ATP: https://github.com/yxiwang/ATP 
ASL: https://github.com/cnyanhao/ASL 
KUDA: https://github.com/tsun/KUDA
G-SFDA: https://github.com/Albert0147/G-SFDA 
APA: https://github.com/tsun/APA 

SSNLL: https://arxiv.org/pdf/2102.11614 https://ieeexplore.ieee.org/document/9981099/ https://github.com/mil-tokyo/MCD_DA/ 

Possibly add more methods if I have time -->

## Setup

```bash
conda create --name tta python=3.12
conda activate tta

pip install -r tta_env.yml
```

- Clone `multi_value` into current folder and install

## Experiment examples

```bash
# Pretrained SAE RTE Nigerian
python new_train.py --scratch --max_epoch 30 --validation_dataset datasets/rte_Nigerian_validation

# Pretrained Nigerian RTE Nigerian
python new_train.py --scratch --max_epoch 30 --training_dataset datasets/rte_Nigerian_train --validation_dataset datasets/rte_Nigerian_validation

# From scratch SAE RTE Nigerian
python new_train.py --max_epoch 30 --validation_dataset datasets/rte_Nigerian_validation
# From scratch Nigerian RTE Nigerian
python new_train.py --max_epoch 30 --training_dataset datasets/rte_Nigerian_train --validation_dataset datasets/rte_Nigerian_validation
```