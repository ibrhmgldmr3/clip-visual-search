# CLIP ile Metin-Görsel Arama Motoru

## Problem Tanımı

CLIP embedding'leri kullanarak doğal dil sorgusuyla ("red car in the city" gibi) görsel arayan bir sistem. Görseller ve metin sorguları aynı vektör uzayına gömülüp FAISS ile hızlı en-yakın-komşu araması yapılıyor. Ek olarak CNN (ResNet50) ve ViT (Vision Transformer) mimarileri aynı sınıflandırma görevinde karşılaştırılıyor, ve modellerin kararları Grad-CAM ile görselleştiriliyor (açıklanabilir yapay zeka).

## Veri Seti

- **Arama korpusu:** [`wtcherr/unsplash_10k`](https://huggingface.co/datasets/wtcherr/unsplash_10k) (Hugging Face Hub, parquet formatı, auth gerektirmez) — 10.000 gerçek Unsplash fotoğrafı + her görsel için bir `prompt` (metin açıklaması). Script-tabanlı `datasets` yükleyicileri artık desteklenmediği için (bkz. diğer projelerde yaşanan sorun) bilinçli olarak parquet-tabanlı bir kaynak seçildi.
  - **`prompt` alanının değeri:** Sadece kalitatif demo değil, kantitatif retrieval değerlendirmesi de mümkün — her prompt kendi görselinin "doğru cevabı" olduğu için Recall@k ölçülebiliyor (bkz. Sonuçlar).
  - **EDA** (`data/processed/eda_stats.json`): görseller 512-2021px genişlikte, prompt'lar ortalama 5.86 kelime.
- **CNN vs ViT karşılaştırması:** [`uoft-cs/cifar10`](https://huggingface.co/datasets/uoft-cs/cifar10) (HF Hub, parquet). torchvision'ın resmi CIFAR-10 indirmesi (cs.toronto.edu) bu ortamda ~40 kB/s'e düşüp kullanışsız hale geldiği için HF mirror'ına geçildi. Süreyi makul tutmak için bir alt küme kullanıldı (8000 train / 2000 test).

## Teknoloji Yığını

| Amaç | Araç |
|---|---|
| Görsel-metin embedding | Hugging Face `transformers` (`openai/clip-vit-base-patch32`) |
| Vektör benzerlik arama | `faiss-cpu` (`IndexFlatIP` — normalize embedding + inner product = kosinüs benzerliği) |
| CNN vs ViT | `timm` (ResNet50, ViT-Small — ikisi de ImageNet-pretrained'den fine-tune) |
| Açıklanabilirlik | `grad-cam` (CLIP ViT için `reshape_transform`'lı Grad-CAM) |
| Demo | Streamlit |

## Kurulum

```bash
conda activate yolov8_env   # torch+CUDA+timm+streamlit zaten kurulu bir ortam
pip install -r requirements.txt
```
GPU: NVIDIA RTX 4060 Laptop (8GB).

## Çalıştırma Sırası

```bash
python src/prepare_data.py                  # unsplash_10k indir, gorselleri + prompts.csv cikar
python src/embed.py                         # CLIP ile gorsel + prompt embedding'leri
python src/index.py                         # FAISS index olustur
python src/search.py --query "sunset over the ocean" --k 8   # CLI arama
python src/evaluate_retrieval.py            # Recall@1/5/10
python src/vit_vs_cnn.py                    # CIFAR-10'da ResNet50 vs ViT-Small
streamlit run demo/app.py                   # interaktif demo (metin + gorsel arama)
```

Notebook'lar (`notebooks/`): `pipeline_demo.ipynb` tüm arama akışını, `explainability.ipynb` Grad-CAM açıklamalarını tek yerde gösterir (önce yukarıdaki script'ler çalıştırılmış olmalı).

## Klasör Yapısı

```
clip-visual-search/
├── data/
│   ├── raw/NOTES.md               # veri kaynaklari + indirme talimati
│   └── processed/
│       ├── images/                # unsplash_10k'dan cikarilan 10.000 gorsel
│       ├── prompts.csv            # image_id, filename, prompt
│       └── eda_stats.json
├── src/
│   ├── prepare_data.py            # HF'ten indir, gorselleri + prompts.csv cikar
│   ├── embed.py                   # CLIP gorsel + prompt embedding'leri
│   ├── index.py                   # FAISS index olustur/yukle
│   ├── search.py                  # ClipSearchEngine: metin/gorsel -> en yakin k
│   ├── evaluate_retrieval.py      # Recall@1/5/10
│   ├── vit_vs_cnn.py              # CIFAR-10 ResNet50 vs ViT-Small
│   └── gradcam_utils.py           # CLIP ViT + CNN/ViT icin Grad-CAM yardimcilari
├── notebooks/
│   ├── pipeline_demo.ipynb        # uctan uca: veri -> arama -> sonuc galerisi
│   └── explainability.ipynb       # CLIP Grad-CAM + CNN vs ViT dikkat karsilastirmasi
├── demo/app.py                    # Streamlit: metin arama + gorsel-ile-gorsel arama
├── outputs/
│   ├── embeddings/                # image/text embeddings.npy + embedding_index.csv
│   ├── faiss_index/               # image_index.faiss
│   └── predictions/               # retrieval_eval.json, vit_vs_cnn.{json,md}
├── models/                        # resnet50_cifar10.pt, vit_small_cifar10.pt
├── requirements.txt
└── README.md
```

## Roadmap

- [x] **Faz 1 — Veri + Embedding**: unsplash_10k indirme, CLIP ile 10.000 görsel + prompt embedding'i.
- [x] **Faz 2 — Arama Motoru + Değerlendirme**: FAISS index, metin/görsel sorgu, Recall@k.
- [x] **Faz 3 — Demo**: Streamlit ile metin araması + görsel-ile-görsel (image-to-image) arama.
- [x] **Faz 4 — CNN vs ViT**: CIFAR-10 üzerinde ResNet50 ve ViT-Small karşılaştırması.
- [x] **Faz 5 — Açıklanabilirlik**: CLIP Grad-CAM ("model metinle görseli neden eşleştirdi") + CNN vs ViT dikkat karşılaştırması notebook'u.
- [x] **Faz 5b — Uçtan uca pipeline notebook'u**: tüm akışı tek notebook'ta çalıştırılabilir gösterim.

## Değerlendirme Metrikleri

Metin→görsel retrieval için Recall@1/5/10 (her prompt kendi görselini top-k'ta bulma oranı). CNN vs ViT için accuracy, parametre sayısı, eğitim süresi, inference hızı.

## Sonuçlar

**Metin→görsel retrieval** (10.000 sorgu, `outputs/predictions/retrieval_eval.json`):

| Recall@1 | Recall@5 | Recall@10 |
|---|---|---|
| 0.121 | 0.275 | 0.364 |

Bu rakamlar ilk bakışta düşük görünüyor, ama korpusun doğası gereği beklenen bir durum: veri setinde çok sayıda görsel/anlamsal olarak birbirine çok benzer fotoğraf var (onlarca kaktüs, onlarca "karda köpek" fotoğrafı). Bir prompt sorgulandığında ilk sıradaki sonuçlar genelde konu olarak tamamen doğru ama tam olarak o prompt'un eşlendiği *spesifik* görsel değil — yani metrik, aslında başarılı olan aramaları "yanlış" sayıyor. Elle kontrol edildiğinde (bkz. `notebooks/pipeline_demo.ipynb`) sonuçların anlamsal kalitesi Recall@1'in ima ettiğinden çok daha yüksek. Bu, near-duplicate içerikli veri setlerinde retrieval değerlendirmesinin bilinen bir sınırlaması.

**CNN vs ViT** (CIFAR-10, 8000 train / 2000 test, 5 epoch, ImageNet-pretrained'den fine-tune, `outputs/predictions/vit_vs_cnn.md`):

| Model | Accuracy | Parametre | Eğitim Süresi (s) | Inference (ms) |
|---|---|---|---|---|
| ResNet50 | 0.866 | 23,528,522 | 381.6 | 2.48 |
| ViT-Small | 0.974 | 21,669,514 | 465.5 | 2.69 |

Bu deneyde ViT-Small, ResNet50'den belirgin şekilde daha yüksek doğruluk verdi (%97.4 vs %86.6), üstelik daha az parametreyle. Literatürde "küçük veride CNN daha stabil" gözlemi genelde *sıfırdan eğitim* için geçerlidir; burada her iki model de ImageNet-pretrained olduğu için ViT'in güçlü ön-eğitilmiş temsilleri küçük veride bile iyi transfer oldu — planlama aşamasında beklediğimin tersi bir sonuç, ama transfer öğrenme senaryosuyla tutarlı ve dürüstçe raporlanıyor. Inference hızları neredeyse eşit.

**Açıklanabilirlik** (`notebooks/explainability.ipynb`):
- **CLIP Grad-CAM**: farklı metin sorgularında ısı haritası görselin farklı bölgelerine kayıyor ("red car" → araba gövdesi, "mountain landscape" → dağ zirveleri) — modelin gerçekten metne duyarlı, bölgesel bir eşleştirme yaptığının kanıtı, sadece global bir alaka skoru değil.
- **ResNet50 vs ViT-Small Grad-CAM**: aynı `grad-cam` kütüphanesiyle, ViT için sadece patch token'larını 2D ızgaraya geri şekillendiren bir `reshape_transform` farkıyla uygulandı — CNN ve Transformer mimarileri arasında aynı XAI tekniğinin nasıl taşınabildiğinin somut bir örneği.

## Sınırlamalar

- **Retrieval metriği near-duplicate'lerde yanıltıcı:** Recall@1=0.12, sistemin anlamsal kalitesini olduğundan kötü gösteriyor (yukarıda açıklandı). Daha adil bir değerlendirme için insan-etiketli alaka skorları veya küme-tabanlı bir metrik gerekir.
- **CLIP zero-shot, fine-tune edilmedi:** Türkçe sorgu desteği veya domain'e özel (ör. tıbbi, uydu) görseller için CLIP'in fine-tune edilmesi gerekebilir.
- **CNN vs ViT küçük ölçekli:** 5 epoch, CIFAR-10'un alt kümesi — kesin bir mimari üstünlüğü iddiası değil, aynı koşulda hızlı bir karşılaştırma. Sonuç seed/epoch/alt-küme seçimine duyarlı olabilir.
- **FAISS `IndexFlatIP` (exact):** 10k görsel için yeterince hızlı; milyonlarca görselde IVF/HNSW gibi approximate index'lere geçmek gerekir.
- **Grad-CAM tek katman:** CLIP ViT'in yalnızca son transformer bloğundan alındı; çok katmanlı/attention-rollout tabanlı bir analiz yapılmadı.

## Notlar

Veri seti seçimleri (arama için unsplash_10k, CNN/ViT karşılaştırması için CIFAR-10'un HF mirror'ı) ve açıklanabilirlik notebook'u, geliştirme sırasındaki gerçek ortam/veri kısıtlarına göre belirlenmiştir (bkz. yukarıdaki Sınırlamalar ve Sonuçlar).
