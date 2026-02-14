# Pre-trained Models

Place trained model checkpoints here. These files are not tracked by git.

## Expected files after training

- `best_model.pt` - checkpoint with best validation accuracy
- `final_model.pt` - checkpoint with full evaluation metadata
- `training_curves.png` - loss and accuracy plots
- `confusion_matrix.png` - per-class confusion matrix

## Training a model

```bash
python scripts/train.py --num-train 16000 --num-val 3200 --epochs 100
```

## Checkpoint format

Each `.pt` file contains:
- `model_state_dict` - model weights
- `epoch` - training epoch
- `val_acc` - validation accuracy
- `args` - training arguments
- `leads` - lead configuration used
- `class_names` - list of 16 class names
