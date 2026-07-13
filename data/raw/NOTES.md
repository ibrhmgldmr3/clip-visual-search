# Veri Setlerini İndirme

Bu klasördeki veriler manuel indirme gerektirmez — script'ler Hugging Face
Hub'dan otomatik indirir (auth/API key gerekmez).

## 1. Arama korpusu — `wtcherr/unsplash_10k`

```bash
python src/prepare_data.py
```

Kaynak: [`wtcherr/unsplash_10k`](https://huggingface.co/datasets/wtcherr/unsplash_10k)
(parquet formatı, 10.000 Unsplash fotoğrafı + `prompt` metin açıklaması).
Görseller `data/processed/images/` altına, açıklamalar `data/processed/prompts.csv`'ye
çıkarılır. `jamescalam/unsplash-25k-photos` (script tabanlı, artık `datasets`
tarafından desteklenmiyor) yerine bilinçli olarak bu kaynak seçildi.

## 2. CNN vs ViT — `uoft-cs/cifar10`

```bash
python src/vit_vs_cnn.py
```

`src/vit_vs_cnn.py` CIFAR-10'u [`uoft-cs/cifar10`](https://huggingface.co/datasets/uoft-cs/cifar10)
HF mirror'ından otomatik indirir (torchvision'ın cs.toronto.edu indirmesi bu
ortamda çok yavaş kaldığı için). Görseller `~/.cache/huggingface` altına iner.
