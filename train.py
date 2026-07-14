import copy
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.metrics import roc_auc_score
from torch.utils.data import DataLoader

from ue import embedding_l2_mean


def _to_device(x, xd, y, device):
    return (x.to(device),
            xd.to(device) if xd.shape[1] else None,
            y.to(device))


def evaluate(model: nn.Module, loader: DataLoader, device: torch.device) -> float:
    model.eval()
    preds, targets = [], []
    with torch.no_grad():
        for x, xd, y in loader:
            x, xd, y = _to_device(x, xd, y, device)
            preds.append(torch.sigmoid(model(x, xd)).cpu().numpy())
            targets.append(y.cpu().numpy())
    return roc_auc_score(np.concatenate(targets), np.concatenate(preds))


def train_model(
    model:      nn.Module,
    tr_loader:  DataLoader,
    va_loader:  DataLoader,
    device:     torch.device,
    lr:         float = 1e-3,
    max_epochs: int   = 30,
    patience:   int   = 5,
    weight_decay: float = 1e-5,
    te_loader:  DataLoader = None,
    writer            = None,
) -> tuple:
    model = model.to(device)
    optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
    criterion = nn.BCEWithLogitsLoss()
    has_reg   = hasattr(model, "reg_loss")

    best_auc, best_state, no_improve = -1.0, None, 0
    history = []

    for epoch in range(1, max_epochs + 1):
        model.train()
        total_loss, n_batches = 0.0, 0
        for x, xd, y in tr_loader:
            x, xd, y = _to_device(x, xd, y, device)
            optimizer.zero_grad()
            loss = criterion(model(x, xd), y)
            if has_reg:
                loss = loss + model.reg_loss()
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
            n_batches  += 1

        train_loss = total_loss / max(n_batches, 1)
        val_auc    = evaluate(model, va_loader, device)
        emb_l2     = embedding_l2_mean(model.emb) if hasattr(model, "emb") else 0.0

        row = {
            "epoch":      epoch,
            "train_loss": round(train_loss, 5),
            "val_auc":    round(val_auc, 5),
            "emb_l2_mean": round(emb_l2, 5),
        }
        test_str = ""
        if te_loader is not None:
            row["test_auc"] = round(evaluate(model, te_loader, device), 5)
            test_str = f"  test_auc {row['test_auc']:.4f}"
        history.append(row)
        if writer is not None:
            writer.add_scalar("loss/train",  train_loss, epoch)
            writer.add_scalar("auc/val",     val_auc,    epoch)
            writer.add_scalar("emb/l2_mean", emb_l2,     epoch)
            if te_loader is not None:
                writer.add_scalar("auc/test_epoch", row["test_auc"], epoch)
        print(f"    epoch {epoch:>2}: loss {train_loss:.4f}  val_auc {val_auc:.4f}"
              f"{test_str}  emb_l2 {emb_l2:.3f}", flush=True)

        if val_auc > best_auc:
            best_auc   = val_auc
            best_state = copy.deepcopy(model.state_dict())
            no_improve = 0
        else:
            no_improve += 1

        if patience > 0 and no_improve >= patience:
            break

    if best_state is not None:
        model.load_state_dict(best_state)

    return model, history
