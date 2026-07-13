"""Faz 1: CLIP ile tum goruntulerin (ve prompt'larin) embedding'ini cikarir.

Hem goruntu hem prompt embedding'lerini tek gecistee hesapliyoruz:
goruntu embedding'leri arama indexinde kullanilacak (index.py), prompt
embedding'leri ise retrieval degerlendirmesinde (evaluate_retrieval.py) sorgu
olarak kullanilacak - ikisini ayri ayri hesaplamak CLIP modelini iki kez
calistirmak anlamina gelirdi.
"""
import argparse
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from PIL import Image
from transformers import CLIPModel, CLIPProcessor

MODEL_NAME = "openai/clip-vit-base-patch32"
IMAGES_DIR = Path("data/processed/images")
PROMPTS_CSV = Path("data/processed/prompts.csv")
OUT_DIR = Path("outputs/embeddings")


def l2_normalize(x):
    return x / np.linalg.norm(x, axis=1, keepdims=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--model", default=MODEL_NAME)
    args = parser.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Cihaz: {device}, model: {args.model}")

    model = CLIPModel.from_pretrained(args.model).to(device).eval()
    processor = CLIPProcessor.from_pretrained(args.model)

    df = pd.read_csv(PROMPTS_CSV)
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    image_embeds = []
    text_embeds = []
    with torch.no_grad():
        for start in range(0, len(df), args.batch_size):
            batch = df.iloc[start:start + args.batch_size]
            images = [Image.open(IMAGES_DIR / fn).convert("RGB") for fn in batch["filename"]]
            prompts = batch["prompt"].tolist()

            img_inputs = processor(images=images, return_tensors="pt").to(device)
            img_feat = model.get_image_features(**img_inputs).pooler_output
            img_feat = img_feat.cpu().numpy()

            txt_inputs = processor(text=prompts, return_tensors="pt", padding=True, truncation=True).to(device)
            txt_feat = model.get_text_features(**txt_inputs).pooler_output
            txt_feat = txt_feat.cpu().numpy()

            image_embeds.append(img_feat)
            text_embeds.append(txt_feat)

            if (start // args.batch_size + 1) % 10 == 0:
                print(f"  {start + len(batch)}/{len(df)}")

    image_embeds = l2_normalize(np.concatenate(image_embeds, axis=0)).astype("float32")
    text_embeds = l2_normalize(np.concatenate(text_embeds, axis=0)).astype("float32")

    np.save(OUT_DIR / "image_embeddings.npy", image_embeds)
    np.save(OUT_DIR / "text_embeddings.npy", text_embeds)
    df[["image_id", "filename"]].to_csv(OUT_DIR / "embedding_index.csv", index=False)

    print(f"\nGoruntu embedding'leri: {image_embeds.shape} -> {OUT_DIR / 'image_embeddings.npy'}")
    print(f"Prompt embedding'leri: {text_embeds.shape} -> {OUT_DIR / 'text_embeddings.npy'}")


if __name__ == "__main__":
    main()
