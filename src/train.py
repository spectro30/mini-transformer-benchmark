import torch
import torch.nn as nn
from torch.optim import Adam
import time

from data import get_dataloaders
from model import MiniTransformer


def train(model, train_loader, val_loader, epochs=10, lr=0.001, device="cpu"):
    optimizer = Adam(model.parameters(), lr=lr)
    criterion = nn.CrossEntropyLoss()

    model.to(device)

    history = {"train_loss": [], "val_acc": []}

    for epoch in range(epochs):
        model.train()
        total_loss = 0

        for batch in train_loader:
            tokens = batch["tokens"].to(device)
            mask = batch["mask"].to(device)
            labels = batch["label"].to(device)

            optimizer.zero_grad()
            output = model(tokens, mask)
            loss = criterion(output, labels)
            loss.backward()
            optimizer.step()

            total_loss += loss.item()

        avg_loss = total_loss / len(train_loader)
        val_acc = evaluate(model, val_loader, device)

        history["train_loss"].append(avg_loss)
        history["val_acc"].append(val_acc)

        print(f"epoch {epoch+1}/{epochs}  loss: {avg_loss:.4f}  val_acc: {val_acc:.4f}")

    return history


def evaluate(model, loader, device="cpu"):
    model.eval()
    correct = 0
    total = 0

    with torch.no_grad():
        for batch in loader:
            tokens = batch["tokens"].to(device)
            mask = batch["mask"].to(device)
            labels = batch["label"].to(device)

            output = model(tokens, mask)
            preds = output.argmax(dim=1)
            correct += (preds == labels).sum().item()
            total += labels.size(0)

    return correct / total


if __name__ == "__main__":
    device = "mps" if torch.backends.mps.is_available() else "cpu"
    print(f"using device: {device}")

    train_loader, val_loader, test_loader = get_dataloaders(
        train_path="data/train.csv",
        val_path="data/validation.csv",
        test_path="data/test.csv",
        batch_size=32
    )

    model = MiniTransformer(
        vocab_size=5,
        embed_dim=64,
        num_heads=4,
        ff_dim=128,
        num_layers=1,
        dropout=0.1,
        use_pos_enc=True
    )

    print(f"parameters: {sum(p.numel() for p in model.parameters())}")

    start = time.time()
    history = train(model, train_loader, val_loader, epochs=10, lr=0.001, device=device)
    elapsed = time.time() - start

    test_acc = evaluate(model, test_loader, device)

    print(f"\ntest accuracy: {test_acc:.4f}")
    print(f"training time: {elapsed/60:.1f} min")
