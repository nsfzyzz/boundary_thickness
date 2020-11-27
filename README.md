# Boundary thickness
Boundary thickness and robustness in learning models
## Introduction
This repository includes all necessary programs to reproduce the results of our NeurIPS paper [Boundary thickness and robustness in learning models](https://proceedings.neurips.cc/paper/2020/file/44e76e99b5e194377e955b13fb12f630-Paper.pdf). The code has been tested on Python 3.7.4 with PyTorch 1.5.0 and torchvision 0.6.0. 

## Usage
Please use the following commands to download the repo and the example checkpoints.

```
mkdir Boundary_thickness_repo
cd Boundary_thickness_repo
mkdir data
git clone https://github.com/nsfzyzz/boundary_thickness.git
cd boundary_thickness/
./download/download_models.sh
```

### Measure boundary thickness
To measure boundary thickness, please use the following command. If you don't give the arguments and directly use `python measure_thickness.py`, you will evaluate the downloaded example.

```
python measure_thickness.py --ckpt [Your_checkpoint_path] --arch [Your_model_choice]
```

### Reproduce the boundary thickness experiment on different models
Please first download the pretrained models. 

```
./download/download_pretrained_models.sh
```

Then, use the following command to evaluate the boundary thickness of all the downloaded models. For example, if you want to use GPUs 0 and 1, you can specify `--available-gpus 0 1`

```
python Reproduce_thickness_measurement_experiment.py --available-gpus [Your_GPUs]
```

Finally, use the following command to generate Figure 2 of our paper. You should be able to generate `visualization/Fig2.png`

```
python visualize_thickness_measurement_experiment.py
```

### Train noisy mixup and other models
To train a model using noisy mixup, please use the following code.

```
python noisy_mixup.py --noise-type Noisy
```

To train a model using empirical risk minimization, please use the following code:

```
python train_models.py --saving-folder [Your_folder] --arch [Your_model_choice]
```

### Some examples of evaluating robustness
Please use one of the following commands to evaluate the adversarial robustness of the downloaded checkpoint.

```
python test_pgd.py
python test_bb.py
```

We need CIFAR10-C and CIFAR100-C to evaluate ood robustness.

```
./download/prepare_cifar10C.sh
```

Please use the following command to evaluate the ood robustness of the downloaded checkpoint.

```
python test_ood.py
```

### Some examples of 3D visualization
Please use the following code to visualize 3D decision boundary of a downloaded checkpoint.

```
python visualize_3D.py
```

The following ipynb checkpoint contains the experiment in the Appendix C of our paper, which uses a chessboard toy example to visualize boundary tilting.

```
jupyter notebook does-chess-board-have-boundary-tilting.ipynb
```

## Citation
We appreciate it if you would please cite the following paper if you found the repository useful for your work:

```
@article{yang2020boundary,
  title={Boundary thickness and robustness in learning models},
  author={Yang, Yaoqing and Khanna, Rajiv and Yu, Yaodong and Gholami, Amir and Keutzer, Kurt and Gonzalez, Joseph E and Ramchandran, Kannan and Mahoney, Michael W},
  journal={Advances in Neural Information Processing Systems},
  volume={33},
  year={2020}
}
```

License
----

MIT