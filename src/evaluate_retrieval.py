"""Faz 2: Metin->goruntu retrieval degerlendirmesi (Recall@1/5/10).

Her prompt kendi goruntusunun "doğru cevabı" oldugu icin (unsplash_10k'daki
image-prompt eslesmesi), her prompt'u sorgu olarak FAISS index'te arayip kendi
goruntusunu top-k icinde bulma oranini olcuyoruz - sadece "demo guzel
gorunuyor" degil, somut bir dogruluk metrigi.
"""
import argparse
import json
from pathlib import Path

import numpy as np

from index import load_index

TEXT_EMB_PATH = Path("outputs/embeddings/text_embeddings.npy")
INDEX_PATH = Path("outputs/faiss_index/image_index.faiss")
OUT_PATH = Path("outputs/predictions/retrieval_eval.json")
K_VALUES = (1, 5, 10)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--text-embeddings", default=str(TEXT_EMB_PATH))
    parser.add_argument("--index", default=str(INDEX_PATH))
    parser.add_argument("--out", default=str(OUT_PATH))
    args = parser.parse_args()

    text_embeddings = np.load(args.text_embeddings)
    index = load_index(args.index)

    max_k = max(K_VALUES)
    _scores, top_indices = index.search(text_embeddings, max_k)

    ground_truth = np.arange(len(text_embeddings))
    hits_at = {k: 0 for k in K_VALUES}
    for i, retrieved in enumerate(top_indices):
        for k in K_VALUES:
            if ground_truth[i] in retrieved[:k]:
                hits_at[k] += 1

    n = len(text_embeddings)
    recall = {f"recall@{k}": hits_at[k] / n for k in K_VALUES}
    result = {"n_queries": n, **recall}

    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    with open(args.out, "w", encoding="utf-8") as f:
        json.dump(result, f, indent=2)

    print(json.dumps(result, indent=2))
    print(f"\nKaydedildi: {args.out}")


if __name__ == "__main__":
    main()
