import torch
import time
import json
import sys
import os

sys.path.insert(0, os.path.dirname(__file__))

from data import get_dataloaders
from model import MiniTransformer
from train import train, evaluate
from utils import set_seed, plot_training_curves

set_seed(1971)

device = "mps" if torch.backends.mps.is_available() else "cpu"
print(f"using device: {device}\n")

train_loader, val_loader, test_loader = get_dataloaders(
    train_path="data/train.csv",
    val_path="data/validation.csv",
    test_path="data/test.csv",
    batch_size=32
)

configs = [
    {"name": "A", "num_heads": 1, "num_layers": 1, "use_pos_enc": True},
    {"name": "B", "num_heads": 4, "num_layers": 1, "use_pos_enc": True},
    {"name": "C", "num_heads": 4, "num_layers": 1, "use_pos_enc": False},
    {"name": "D", "num_heads": 4, "num_layers": 2, "use_pos_enc": True},
]

results = []

for cfg in configs:
    print(f"── Model {cfg['name']} | heads={cfg['num_heads']} layers={cfg['num_layers']} pos_enc={cfg['use_pos_enc']}")

    model = MiniTransformer(
        vocab_size=5,
        embed_dim=64,
        num_heads=cfg["num_heads"],
        ff_dim=128,
        num_layers=cfg["num_layers"],
        dropout=0.1,
        use_pos_enc=cfg["use_pos_enc"]
    )

    num_params = sum(p.numel() for p in model.parameters())

    start = time.time()
    history = train(model, train_loader, val_loader, epochs=10, lr=0.001, device=device)
    elapsed = time.time() - start

    val_acc  = evaluate(model, val_loader,  device)
    test_acc = evaluate(model, test_loader, device)

    results.append({
        "model":      cfg["name"],
        "pos_enc":    "Yes" if cfg["use_pos_enc"] else "No",
        "heads":      cfg["num_heads"],
        "layers":     cfg["num_layers"],
        "val_acc":    round(val_acc,  4),
        "test_acc":   round(test_acc, 4),
        "train_time": f"{elapsed/60:.1f} min",
        "params":     num_params,
        "history":    history
    })

    print(f"   val_acc={val_acc:.4f}  test_acc={test_acc:.4f}  time={elapsed/60:.1f}min  params={num_params}\n")


print("\n── Benchmark Results ─────────────────────────────────────────────")
print(f"{'Model':<8}{'Pos Enc':<10}{'Heads':<8}{'Layers':<8}{'Val Acc':<10}{'Test Acc':<10}{'Time':<10}{'Params'}")
print("-" * 70)
for r in results:
    print(f"{r['model']:<8}{r['pos_enc']:<10}{r['heads']:<8}{r['layers']:<8}{r['val_acc']:<10}{r['test_acc']:<10}{r['train_time']:<10}{r['params']}")

# plot first, then save json (removing history after)
plot_training_curves(results, save_path="training_curves.png")

for r in results:
    r.pop("history")

with open("benchmark_results.json", "w") as f:
    json.dump(results, f, indent=2)

print("saved to benchmark_results.json")