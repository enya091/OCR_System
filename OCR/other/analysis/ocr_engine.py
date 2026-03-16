# analysis/ocr_engine.py

import streamlit as st
import easyocr

@st.cache_resource
def load_ocr_engine():
    """
    載入 EasyOCR 引擎，並透過 Streamlit 快取機制避免重複啟動
    """
    return easyocr.Reader(['ch_tra', 'en'], gpu=False)

def extract_text_from_image(img_array) -> str:
    """
    執行影像文字萃取
    """
    reader = load_ocr_engine()
    result = reader.readtext(
        img_array, 
        detail=0, 
        paragraph=True, 
        adjust_contrast=True
    )
    return "\n".join(result)