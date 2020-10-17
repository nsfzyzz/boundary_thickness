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
To measure boundary thickness, please use the following code:

```
python measure_thickness.py
```

### Evaluate adversarial robustness:
To measure the adversarial robustness of a model, please use one of the following python codes:

```
python test_pgd.py
python test_bb.py
```

### Evaluate ood robustness 
To measure the out-of-distribution generalization of a model on CIFAR10-C or CIFAR100-C, please use the following code:

```
python test_ood.py
```

### Train noisy mixup
To train a model using mixup or noisy mixup, please use the following code:

```
python noisy_mixup.py
```

### Train ERM model
To train a model using empirical risk minimization, please use the following code:

```
python train_models.py --saving-folder [Your_folder] --arch [Your_model_choice]
```

### 3D visualization
Please use the following code to see an example of visualizing 3D decision boundary:

```
python visualize_3D.py
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