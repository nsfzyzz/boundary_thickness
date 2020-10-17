# boundary_robustness

## Prepare 

cd boundary_thickness/
./download_models.sh

## Measure boundary thickness

python measure_thickness.py

## Evaluate adversarial robustness:

python test_pgd.py
python test_bb.py

## Evaluate ood robustness (on CIFAR10-C or CIFAR100-C):

./prepare_cifar10C.sh
python test_ood.py

## Train noisy mixup

python noisy_mixup.py

## Train a model using empirical risk minimization

python train_models.py --saving-folder [Your_folder] --arch [Your_model_choice]

## See an example of visualizing 3D decision boundary

python visualize_3D.py

## See the chessboard example

jupyter notebook does-chess-board-have-boundary-tilting.ipynb

## See how we measure the non-robust feature score

jupyter notbeook measure_non_robust_feature_score.ipynb