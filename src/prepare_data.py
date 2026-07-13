"""Faz 1: Unsplash gorsel koleksiyonunu indirir, diske cikarir, EDA istatistikleri uretir.

Kaynak: Hugging Face Hub'daki `wtcherr/unsplash_10k` (parquet formatinda, auth
gerektirmez) - 10.000 gercek Unsplash fotografi + `prompt` (metin aciklamasi)
alani. `jamescalam/unsplash-25k-photos` (script tabanli, artik `datasets`
kutuphanesi tarafindan desteklenmiyor) yerine bilincli olarak bu kaynak
secildi (bkz. turkish-anpr projesindeki benzer sorun).

`prompt` alani sayesinde sadece kalitatif demo degil, kantitatif retrieval
degerlendirmesi de mumkun (bkz. evaluate_retrieval.py): her prompt kendi
goruntusunu top-k'ta bulmali.
"""
import argparse
import csv
import json
from pathlib import Path

from datasets import load_dataset

REPO_ID = "wtcherr/unsplash_10k"
IMAGES_DIR = Path("data/processed/images")
PROMPTS_CSV = Path("data/processed/prompts.csv")
STATS_PATH = Path("data/processed/eda_stats.json")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--n", type=int, default=10000, help="Kac gorsel kullanilacak (varsayilan: tumu)")
    args = parser.parse_args()

    IMAGES_DIR.mkdir(parents=True, exist_ok=True)
    PROMPTS_CSV.parent.mkdir(parents=True, exist_ok=True)

    print(f"{REPO_ID} indiriliyor/aciliyor...")
    ds = load_dataset(REPO_ID, split=f"train[:{args.n}]")
    print(f"{len(ds)} kayit yuklendi.")

    sizes = []
    prompt_lengths = []
    rows = []
    for i, example in enumerate(ds):
        image = example["image"].convert("RGB")
        prompt = example["prompt"]
        filename = f"{i:05d}.jpg"

        image.save(IMAGES_DIR / filename, "JPEG", quality=90)
        sizes.append(image.size)
        prompt_lengths.append(len(prompt.split()))
        rows.append({"image_id": i, "filename": filename, "prompt": prompt})

        if (i + 1) % 1000 == 0:
            print(f"  {i + 1}/{len(ds)}")

    with open(PROMPTS_CSV, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=["image_id", "filename", "prompt"])
        writer.writeheader()
        writer.writerows(rows)

    widths = [s[0] for s in sizes]
    heights = [s[1] for s in sizes]
    stats = {
        "n_images": len(rows),
        "image_width": {"mean": sum(widths) / len(widths), "min": min(widths), "max": max(widths)},
        "image_height": {"mean": sum(heights) / len(heights), "min": min(heights), "max": max(heights)},
        "prompt_word_count": {
            "mean": sum(prompt_lengths) / len(prompt_lengths),
            "min": min(prompt_lengths),
            "max": max(prompt_lengths),
        },
    }
    with open(STATS_PATH, "w", encoding="utf-8") as f:
        json.dump(stats, f, indent=2, ensure_ascii=False)

    print(f"\nGoruntuler yazildi: {IMAGES_DIR} ({len(rows)} dosya)")
    print(f"Prompts: {PROMPTS_CSV}")
    print(f"EDA istatistikleri: {STATS_PATH}")
    print(json.dumps(stats, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
