"""Faz 2: Goruntu embedding'lerinden bir FAISS index olusturur ve kaydeder.

Embedding'ler zaten L2-normalize edildigi icin (embed.py) `IndexFlatIP` (inner
product) kullanmak kosinus benzerligine esdegerdir - kucuk veri seti (10k)
icin yaklasik/approximate index'e gerek yok, tam (exact) arama yeterince hizli.
"""
import argparse
from pathlib import Path

import faiss
import numpy as np

EMBEDDINGS_PATH = Path("outputs/embeddings/image_embeddings.npy")
INDEX_PATH = Path("outputs/faiss_index/image_index.faiss")


def build_index(embeddings):
    dim = embeddings.shape[1]
    index = faiss.IndexFlatIP(dim)
    index.add(embeddings)
    return index


def load_index(path=INDEX_PATH):
    return faiss.read_index(str(path))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--embeddings", default=str(EMBEDDINGS_PATH))
    parser.add_argument("--out", default=str(INDEX_PATH))
    args = parser.parse_args()

    embeddings = np.load(args.embeddings)
    index = build_index(embeddings)

    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    faiss.write_index(index, args.out)
    print(f"FAISS index olusturuldu: {index.ntotal} vektor, boyut {embeddings.shape[1]} -> {args.out}")


if __name__ == "__main__":
    main()
