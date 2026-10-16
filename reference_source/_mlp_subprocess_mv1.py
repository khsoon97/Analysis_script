#!/usr/bin/env python3
"""
_mlp_subprocess_mv1.py
======================
[mv1] Helper script that runs PyTorch GPU MLP training in an isolated subprocess.
Called from the main notebook through spawn_mlp().

Usage:
    python3 _mlp_subprocess_mv1.py <in_file.npz> <out_file.npy> <n_classes> [model_out.pt] [random_seed]

Input (npz):
    - X_train: (N_train, D) training feature matrix
    - y_train: (N_train,) integer labels
    - X_test:  (N_test, D) feature matrix for prediction

Output (npy):
    - pred_probs: (N_test, n_classes) softmax probability matrix

Optional output (pt):
    - CPU state_dict and network architecture metadata for external data annotation
"""

import os
import random
import sys

os.environ.setdefault('CUBLAS_WORKSPACE_CONFIG', ':4096:8')

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader


# ==============================================================================
# Define the MLP network architecture
# ==============================================================================
class SimpleMLP(nn.Module):
    """
    MLP (Multi-Layer Perceptron) classifier with two hidden layers.

    [Network architecture]
    Input(D) → Linear(256) → BatchNorm → ReLU → Dropout(0.2)
            → Linear(128) → BatchNorm → ReLU → Dropout(0.2)
             → Linear(n_classes) → Output

    ──────────────── Hyperparameter descriptions ────────────────
    (★ indicates adjustable parameters)

    ★ hidden_dims = [256, 128]
        Number of hidden-layer neurons: 256 in the first layer and 128 in the second.
        Adjustable according to data complexity (e.g., [512, 256] or [128, 64]).

    ★ dropout_rate = 0.5
        Dropout rate. The PyTorch Dropout default randomly disables 50% of neurons during training to prevent overfitting.
        Larger values increase regularization (recommended range: 0.0–0.5).

    (Fixed) BatchNorm1d
        Normalize each hidden-layer output to improve training stability. No adjustment needed.

    (Fixed) Activation function: ReLU
        Negative values → 0; positive values → unchanged. A commonly used activation.

    (Fixed) Output layer: Linear(n_classes)
        Output one logit per class. During training, softmax is included in CrossEntropyLoss.
    """
    def __init__(self, input_dim, n_classes, hidden_dims=None, dropout_rate=0.5):
        super().__init__()
        if hidden_dims is None:
            hidden_dims = [256, 128]  # ★ Hidden-layer sizes (adjustable)

        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dims[0]),   # Input → first hidden layer
            nn.BatchNorm1d(hidden_dims[0]),          # Batch normalization
            nn.ReLU(),                               # Activation
            nn.Dropout(dropout_rate),                # ★ Dropout (adjustable)
            nn.Linear(hidden_dims[0], hidden_dims[1]),  # First → second hidden layer
            nn.BatchNorm1d(hidden_dims[1]),
            nn.ReLU(),
            nn.Dropout(dropout_rate),
            nn.Linear(hidden_dims[1], n_classes)     # Second hidden layer → output
        )

    def forward(self, x):
        return self.net(x)


# ==============================================================================
# Main training and prediction logic
# ==============================================================================
def main():
    if len(sys.argv) < 4:
        sys.exit(
            "Usage: python3 _mlp_subprocess_mv1.py "
            "<in_file> <out_file> <n_classes> [model_out.pt] [random_seed]"
        )

    in_file = sys.argv[1]
    out_file = sys.argv[2]
    n_classes = int(sys.argv[3])
    model_out = sys.argv[4] if len(sys.argv) >= 5 else None
    if model_out in (None, '', '__NONE__'):
        model_out = None
    random_seed = int(sys.argv[5]) if len(sys.argv) >= 6 else 42

    random.seed(random_seed)
    np.random.seed(random_seed)
    torch.manual_seed(random_seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(random_seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    torch.use_deterministic_algorithms(True, warn_only=True)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # ── 1. Load data ─────────────────────────────────────────────────────────
    try:
        data = np.load(in_file)
        X_train = data['X_train']
        y_train = data['y_train']
        X_test  = data['X_test']
    except Exception as e:
        print(f"[MLP] Failed to load input data: {e}", file=sys.stderr)
        sys.exit(1)

    input_dim = X_train.shape[1]

    # ── 2. PyTorch Dataset & DataLoader ───────────────────────────────────────
    X_tr_t = torch.tensor(X_train, dtype=torch.float32)
    y_tr_t = torch.tensor(y_train, dtype=torch.long)

    # Preserve the 7:3 structure of the outer Step02 GroupShuffleSplit:
    # do not perform a separate validation split or early stopping inside the MLP.
    train_dataset = TensorDataset(X_tr_t, y_tr_t)

    # ★ batch_size = 4096: adjustable according to GPU memory (2048–8192)
    BATCH_SIZE = min(4096, len(train_dataset))
    if len(train_dataset) > 1 and len(train_dataset) % BATCH_SIZE == 1:
        BATCH_SIZE -= 1
    if BATCH_SIZE < 2:
        raise ValueError(f'BatchNorm MLP training requires at least 2 cells: n={len(train_dataset)}')
    loader_generator = torch.Generator()
    loader_generator.manual_seed(random_seed)
    train_loader = DataLoader(
        train_dataset,
        batch_size=BATCH_SIZE,
        shuffle=True,
        drop_last=False,
        generator=loader_generator,
    )

    # ── 3. Configure the model and training environment ──────────────────────
    """
    ──────────────── Training parameter descriptions ────────────────
    (Default) Adam optimizer
        Use PyTorch Adam defaults (lr=1e-3, betas=(0.9, 0.999), eps=1e-8, weight_decay=0).

    ★ epochs = 30 (fixed number of epochs)
        Number of passes through the full training data; fixed-length training without internal early stopping.

    (Fixed) Loss function: CrossEntropyLoss
        Standard multiclass classification loss, combining softmax and NLLLoss.

    (Fixed) Optimizer: Adam
        Adaptive learning rates can enable faster convergence than SGD.
    """
    model = SimpleMLP(input_dim, n_classes).to(device)
    optimizer = torch.optim.Adam(model.parameters())
    criterion = nn.CrossEntropyLoss()

    MAX_EPOCHS = 30  # ★ Fixed number of training epochs (adjustable)

    # ── 4. Training loop (no internal early stopping) ────────────────────────
    model.train()
    for ep in range(MAX_EPOCHS):
        train_loss = 0.0
        for bx, by in train_loader:
            bx, by = bx.to(device), by.to(device)
            optimizer.zero_grad()
            logits = model(bx)
            loss = criterion(logits, by)
            loss.backward()
            optimizer.step()
            train_loss += loss.item() * len(bx)
        train_loss /= len(train_dataset)

    # ── 5. Prediction (return softmax probabilities) ─────────────────────────
    model.eval()
    X_te_t = torch.tensor(X_test, dtype=torch.float32)
    test_dataset = TensorDataset(X_te_t)
    test_loader = DataLoader(test_dataset, batch_size=8192, shuffle=False)

    all_probs = []
    with torch.no_grad():
        for (bx,) in test_loader:
            bx = bx.to(device)
            logits = model(bx)
            probs = torch.softmax(logits, dim=1).cpu().numpy()
            all_probs.append(probs)

    pred_probs_full = np.concatenate(all_probs, axis=0)

    # ── 6. Save results ─────────────────────────────────────────────────────
    try:
        np.save(out_file, pred_probs_full)
    except Exception as e:
        print(f"[MLP] Failed to save results: {e}", file=sys.stderr)
        sys.exit(1)

    # Save the model after probability calculation finishes, preserving training/prediction formulas.
    if model_out is not None:
        try:
            os.makedirs(os.path.dirname(os.path.abspath(model_out)), exist_ok=True)
            torch.save(
                {
                    'format_version': '1.0',
                    'pipeline_version': 'V9',
                    'model_type': 'SimpleMLP',
                    'state_dict': {
                        key: value.detach().cpu()
                        for key, value in model.state_dict().items()
                    },
                    'input_dim': int(input_dim),
                    'n_classes': int(n_classes),
                    'hidden_dims': [256, 128],
                    'dropout_rate': 0.5,
                    'max_epochs': int(MAX_EPOCHS),
                    'train_batch_size': int(BATCH_SIZE),
                    'random_seed': int(random_seed),
                    'torch_version': torch.__version__,
                },
                model_out,
            )
        except Exception as e:
            # Warn if model-package saving fails, without invalidating existing probability outputs.
            print(f"[MLP] Warning while saving model package: {e}", file=sys.stderr)

    # ── 7. Release GPU VRAM ─────────────────────────────────────────────────
    del data, X_train, y_train, X_test
    del model, optimizer, criterion, train_loader, test_loader
    del X_tr_t, y_tr_t, X_te_t, all_probs, pred_probs_full

    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    sys.exit(0)


if __name__ == "__main__":
    main()
