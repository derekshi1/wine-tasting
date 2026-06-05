# Augmented MLP Experiment

Simple neural-network baseline for wine quality regression using scikit-learn's
`MLPRegressor`.

Run from the repo root:

```bash
python3 synthetic_data_gen/mlp_augmented/train_mlp_augmented.py
```

By default, the script:

- Uses the existing 60/20/20 train/validation/test split.
- Drops `Id`, fills missing values from the training means, and scales features.
- Creates a 10x training set with Mixup (`9x` synthetic rows plus original rows).
- Trains a baseline MLP and an augmented-data MLP.
- Reports RMSE, MAE, and R2 on the original validation and held-out test sets.

You can adjust the amount of synthetic data:

```bash
python3 synthetic_data_gen/mlp_augmented/train_mlp_augmented.py --augmentation-factor 15
```

`--augmentation-factor 15` means add `15x` synthetic rows, for `16x` total
training data.
