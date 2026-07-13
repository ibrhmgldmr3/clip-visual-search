"""Faz 4: CIFAR-10 uzerinde ResNet50 (CNN) ve ViT-Small (Transformer) karsilastirmasi.

Ikisi de ImageNet-pretrained'den fine-tune edilir (ayni veri/epoch/optimizer),
accuracy/egitim suresi/parametre sayisi/inference hizi tablosu uretilir. Amaç
"ViT ne zaman CNN'den daha iyi, ne zaman degil" sorusuna küçük veri rejiminde
kendi deneyiyle bir gözlem eklemek (literatürde küçük veride CNN'in daha
stabil oldugu bilinir - bkz. README Sonuçlar).

Egitim suresini makul tutmak icin CIFAR-10'un bir alt kumesi kullanilir
(varsayilan 8000 train / 2000 test) - amac SOTA accuracy degil, iki mimarinin
ayni kosullarda karsilastirilmasi.

Veri kaynagi: HF Hub'daki `uoft-cs/cifar10` (parquet, auth'suz) - torchvision'in
resmi CIFAR-10 indirmesi (cs.toronto.edu) bu ortamda ~40kB/s'e dusup 170MB
icin 1 saatten fazla surecegi icin (bkz. proje calisirken gozlemlenen sorun)
HF mirror'ina gecildi.
"""
import argparse
import json
import time
from pathlib import Path

import timm
import torch
import torch.nn as nn
from datasets import load_dataset
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms

MODELS_DIR = Path("models")
OUT_PATH = Path("outputs/predictions/vit_vs_cnn.json")
CIFAR10_CLASSES = ["airplane", "automobile", "bird", "cat", "deer", "dog", "frog", "horse", "ship", "truck"]

MODEL_CONFIGS = {
    "resnet50": "resnet50",
    "vit_small": "vit_small_patch16_224",
}


class HFImageDataset(Dataset):
    def __init__(self, hf_dataset, transform):
        self.hf_dataset = hf_dataset
        self.transform = transform

    def __len__(self):
        return len(self.hf_dataset)

    def __getitem__(self, idx):
        example = self.hf_dataset[idx]
        image = example["img"].convert("RGB")
        return self.transform(image), example["label"]


def build_dataloaders(n_train, n_test, batch_size, img_size, seed):
    normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    transform = transforms.Compose([
        transforms.Resize((img_size, img_size)),
        transforms.ToTensor(),
        normalize,
    ])

    train_full = load_dataset("uoft-cs/cifar10", split="train").shuffle(seed=seed).select(range(n_train))
    test_full = load_dataset("uoft-cs/cifar10", split="test").shuffle(seed=seed).select(range(n_test))

    train_loader = DataLoader(HFImageDataset(train_full, transform), batch_size=batch_size, shuffle=True, num_workers=0)
    test_loader = DataLoader(HFImageDataset(test_full, transform), batch_size=batch_size, shuffle=False, num_workers=0)
    return train_loader, test_loader, CIFAR10_CLASSES


def train_and_eval(model_name, timm_name, train_loader, test_loader, epochs, lr, device):
    model = timm.create_model(timm_name, pretrained=True, num_classes=10).to(device)
    n_params = sum(p.numel() for p in model.parameters())

    optimizer = torch.optim.AdamW(model.parameters(), lr=lr)
    loss_fn = nn.CrossEntropyLoss()

    t0 = time.time()
    for epoch in range(epochs):
        model.train()
        total_loss = 0.0
        for images, labels in train_loader:
            images, labels = images.to(device), labels.to(device)
            optimizer.zero_grad()
            logits = model(images)
            loss = loss_fn(logits, labels)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
        print(f"  [{model_name}] epoch {epoch + 1}/{epochs} loss={total_loss / len(train_loader):.4f}")
    train_time = time.time() - t0

    model.eval()
    n_correct, n_total = 0, 0
    inference_times = []
    with torch.no_grad():
        for images, labels in test_loader:
            images, labels = images.to(device), labels.to(device)
            t0 = time.perf_counter()
            logits = model(images)
            if device == "cuda":
                torch.cuda.synchronize()
            inference_times.append((time.perf_counter() - t0) * 1000 / images.size(0))
            preds = logits.argmax(dim=1)
            n_correct += (preds == labels).sum().item()
            n_total += labels.size(0)

    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    torch.save(model.state_dict(), MODELS_DIR / f"{model_name}_cifar10.pt")

    return {
        "model": model_name,
        "timm_name": timm_name,
        "accuracy": n_correct / n_total,
        "n_params": n_params,
        "train_time_sec": train_time,
        "avg_inference_ms": sum(inference_times) / len(inference_times),
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--n-train", type=int, default=8000)
    parser.add_argument("--n-test", type=int, default=2000)
    parser.add_argument("--epochs", type=int, default=5)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--lr", type=float, default=3e-5)
    parser.add_argument("--img-size", type=int, default=224)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    torch.manual_seed(args.seed)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Cihaz: {device}")

    train_loader, test_loader, classes = build_dataloaders(
        args.n_train, args.n_test, args.batch_size, args.img_size, args.seed
    )
    print(f"Train: {len(train_loader.dataset)}, Test: {len(test_loader.dataset)}, Siniflar: {classes}")

    results = []
    for model_name, timm_name in MODEL_CONFIGS.items():
        print(f"\n=== {model_name} ({timm_name}) egitiliyor ===")
        result = train_and_eval(model_name, timm_name, train_loader, test_loader, args.epochs, args.lr, device)
        results.append(result)
        print(f"  Sonuc: accuracy={result['accuracy']:.4f}, params={result['n_params']:,}, "
              f"train_time={result['train_time_sec']:.1f}s, inference={result['avg_inference_ms']:.2f}ms")

    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    with open(OUT_PATH, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2)

    header = "| Model | Accuracy | Parametre | Eğitim Süresi (s) | Inference (ms) |"
    sep = "|---|---|---|---|---|"
    rows = [
        f"| {r['model']} | {r['accuracy']:.4f} | {r['n_params']:,} | {r['train_time_sec']:.1f} | {r['avg_inference_ms']:.2f} |"
        for r in results
    ]
    table = "\n".join([header, sep, *rows])
    print("\n" + table)

    md_path = OUT_PATH.with_suffix(".md")
    md_path.write_text(f"# CNN vs ViT Karsilastirma (CIFAR-10)\n\n{table}\n", encoding="utf-8")
    print(f"\nKaydedildi: {OUT_PATH}, {md_path}")


if __name__ == "__main__":
    main()
