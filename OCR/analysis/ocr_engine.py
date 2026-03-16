from __future__ import annotations

import easyocr
import numpy as np
import streamlit as st


@st.cache_resource
def load_ocr_engine() -> easyocr.Reader:
    """
    Load EasyOCR once and reuse it across reruns.
    """
    return easyocr.Reader(["ch_tra", "en"], gpu=False)


def extract_text_from_image(img_array: np.ndarray) -> str:
    """
    Run OCR on an RGB image array.
    """
    reader = load_ocr_engine()
    result = reader.readtext(
        img_array,
        detail=0,
        paragraph=True,
        adjust_contrast=True,
    )
    return "\n".join(result).strip()
