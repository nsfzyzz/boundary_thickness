# Boundary thickness
Boundary thickness and robustness in learning models
## Introduction
This repository includes all necessary programs to reproduce the results of [the following paper](https://arxiv.org/abs/2007.05086). The code has been tested on Python 3.7.4 with PyTorch 1.5.0 and torchvision 0.6.0. We appreciate it if you would please cite the following paper if you found the repository useful for your work:
* [Y. Yang, R. Khanna, Y. Yu, A. Gholami, K. Keutzer, Joseph E. Gonzalez, K. Ramchandran, Michael W. Mahoney, “Boundary thickness and robustness in learning models.”](https://arxiv.org/abs/2007.05086)

## Usage
Please first clone the this library to your local system:

```
git clone https://github.com/nsfzyzz/boundary_thickness.git
```

After cloning, please prepare the experiments by downloading datasets and pretrained checkpoints:

```
cd boundary_thickness/
./download_models.sh
./prepare_cifar10C.sh
```

### Measure boundary thickness
To measure boundary thickness, please use the following command. If you don't give the checkpoint path, you will evaluate the downloaded example.

```
python measure_thickness.py --resume [Your_checkpoint_path]
```

### Evaluate adversarial robustness:
To measure the adversarial robustness of a model, please use one of the following commands. If you don't give the checkpoint path, you will evaluate the downloaded example.

```
python test_pgd.py --resume [Your_checkpoint_path]
python test_bb.py --resume [Your_checkpoint_path]
```

### Evaluate ood robustness 
To measure the out-of-distribution generalization of a model on CIFAR10-C or CIFAR100-C, please use the following command. If you don't give the checkpoint path, you will evaluate the downloaded example.

```
python test_ood.py --resume [Your_checkpoint_path]
```

### Train noisy mixup
To train a model using noisy mixup, please use the following code.

```
python noisy_mixup.py --noise-type Noisy
```

### Train ERM model
To train a model using empirical risk minimization, please use the following code:

```
python train_models.py --saving-folder [Your_folder] --arch [Your_model_choice]
```

### 3D visualization
Please use the following code to visualize 3D decision boundary. If you don't give the checkpoint path, you will evaluate the downloaded example.

```
python visualize_3D.py --resume [Your_checkpoint_path]
```

### Chessboard toy example
The following ipynb checkpoint contains the experiment in the Appendix C of our paper:

```
jupyter notebook does-chess-board-have-boundary-tilting.ipynb
```

### Non-robust feature
The following ipynb checkpoint contains the experiment in Section 4.2 on measuring the non-robust feature score:

```
jupyter notbeook measure_non_robust_feature_score.ipynb
```

License
----

MIT