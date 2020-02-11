# boundary_robustness

To test the functions used in the paper, execute the code in the folder Experiment_code

Evaluate the model robustness:

python eval_models_blackbox.py
python eval_models_ood.py
python eval_models_whitebox.py

Measure boundary thickness

python measure_boundary_thickness

Train a model

python train_models.py --saving-folder [Your_folder] --arch [Your_model_choice]

Train the two mixup models and compare

python train-mixup-random-noise.py
python train-mixup.py

See an example of visualizing 3D decision boundary

python visualize_3D.py

