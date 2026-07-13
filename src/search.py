"""Faz 2: Metin ya da gorsel sorgusuyla FAISS index'te en yakin k sonucu bulur.

Kullanim:
    python src/search.py --query "red car in the city" --k 8
    python src/search.py --image-query data/processed/images/00001.jpg --k 8
"""
import argparse
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from PIL import Image
from transformers import CLIPModel, CLIPProcessor

from index import load_index

MODEL_NAME = "openai/clip-vit-base-patch32"
# Yollar proje koku (src/'nin ust dizini) baz alinarak cozulur - boylece script
# hangi calisma dizininden (repo koku, notebooks/, demo/) cagirilirsa cagirilsin
# ayni dosyalari bulur.
PROJECT_ROOT = Path(__file__).resolve().parent.parent
INDEX_CSV = PROJECT_ROOT / "outputs/embeddings/embedding_index.csv"
DEFAULT_INDEX_PATH = PROJECT_ROOT / "outputs/faiss_index/image_index.faiss"


class ClipSearchEngine:
    def __init__(self, index_path=None, model_name=MODEL_NAME):
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.model = CLIPModel.from_pretrained(model_name).to(self.device).eval()
        self.processor = CLIPProcessor.from_pretrained(model_name)
        self.index = load_index(str(index_path) if index_path else str(DEFAULT_INDEX_PATH))
        self.meta = pd.read_csv(INDEX_CSV)

    def _normalize(self, vec):
        return vec / np.linalg.norm(vec, axis=1, keepdims=True)

    def embed_text(self, text):
        with torch.no_grad():
            inputs = self.processor(text=[text], return_tensors="pt", padding=True).to(self.device)
            feat = self.model.get_text_features(**inputs).pooler_output.cpu().numpy()
        return self._normalize(feat).astype("float32")

    def embed_image(self, image):
        if isinstance(image, (str, Path)):
            image = Image.open(image).convert("RGB")
        with torch.no_grad():
            inputs = self.processor(images=[image], return_tensors="pt").to(self.device)
            feat = self.model.get_image_features(**inputs).pooler_output.cpu().numpy()
        return self._normalize(feat).astype("float32")

    def search(self, query_embedding, k=10):
        scores, indices = self.index.search(query_embedding, k)
        results = []
        for score, idx in zip(scores[0], indices[0]):
            row = self.meta.iloc[idx]
            results.append({"filename": row["filename"], "image_id": int(row["image_id"]), "score": float(score)})
        return results

    def search_by_text(self, text, k=10):
        return self.search(self.embed_text(text), k=k)

    def search_by_image(self, image, k=10):
        return self.search(self.embed_image(image), k=k)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--query", help="Metin sorgusu")
    parser.add_argument("--image-query", help="Gorsel sorgusu (dosya yolu)")
    parser.add_argument("--k", type=int, default=8)
    args = parser.parse_args()

    if not args.query and not args.image_query:
        raise SystemExit("--query veya --image-query belirtilmeli")

    engine = ClipSearchEngine()
    if args.query:
        results = engine.search_by_text(args.query, k=args.k)
        print(f"Sorgu: \"{args.query}\"")
    else:
        results = engine.search_by_image(args.image_query, k=args.k)
        print(f"Gorsel sorgu: {args.image_query}")

    for r in results:
        print(f"  {r['score']:.4f}  {r['filename']}")


if __name__ == "__main__":
    main()
