"""Faz 3: Streamlit demo - metin sorgusuyla goruntu ara, bonus: goruntu yukle -> benzerlerini bul.

Kullanim:
    streamlit run demo/app.py
"""
import sys
from pathlib import Path

import streamlit as st
from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))
from search import ClipSearchEngine  # noqa: E402

IMAGES_DIR = Path(__file__).resolve().parent.parent / "data" / "processed" / "images"
INDEX_PATH = Path(__file__).resolve().parent.parent / "outputs" / "faiss_index" / "image_index.faiss"


@st.cache_resource
def get_engine():
    return ClipSearchEngine(index_path=str(INDEX_PATH))


def show_results(results):
    cols = st.columns(4)
    for i, r in enumerate(results):
        with cols[i % 4]:
            st.image(str(IMAGES_DIR / r["filename"]), use_container_width=True)
            st.caption(f"{r['score']:.3f} · {r['filename']}")


def main():
    st.set_page_config(page_title="CLIP Görsel Arama", layout="wide")
    st.title("CLIP ile Metin-Görsel Arama Motoru")
    st.caption("openai/clip-vit-base-patch32 + FAISS · 10.000 Unsplash fotoğrafı")

    if not INDEX_PATH.exists():
        st.error(f"FAISS index bulunamadı: {INDEX_PATH}. Önce `python src/embed.py` ve `python src/index.py` çalıştırın.")
        return

    engine = get_engine()

    tab_text, tab_image = st.tabs(["Metin ile Ara", "Görsel ile Ara (bonus)"])

    with tab_text:
        query = st.text_input("Ne arıyorsun?", placeholder="ör. red car in the city, mountain sunset, cat sleeping")
        k = st.slider("Kaç sonuç?", 4, 20, 8, key="k_text")
        if query:
            with st.spinner("Aranıyor..."):
                results = engine.search_by_text(query, k=k)
            show_results(results)

    with tab_image:
        uploaded = st.file_uploader("Bir görsel yükle, benzerlerini bulalım", type=["jpg", "jpeg", "png"])
        k2 = st.slider("Kaç sonuç?", 4, 20, 8, key="k_image")
        if uploaded is not None:
            image = Image.open(uploaded).convert("RGB")
            st.image(image, caption="Sorgu görseli", width=250)
            with st.spinner("Aranıyor..."):
                results = engine.search_by_image(image, k=k2)
            show_results(results)


if __name__ == "__main__":
    main()
