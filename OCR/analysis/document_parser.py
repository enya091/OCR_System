from __future__ import annotations

from collections.abc import Callable

import cv2
import fitz
import numpy as np
from PIL import Image

from analysis.ocr_engine import extract_text_from_image

SUPPORTED_EXTENSIONS = {"pdf", "png", "jpg", "jpeg"}


def _pixmap_to_rgb_array(pixmap: fitz.Pixmap) -> np.ndarray:
    img_data = np.frombuffer(pixmap.samples, dtype=np.uint8).reshape(
        pixmap.h,
        pixmap.w,
        pixmap.n,
    )

    if pixmap.n == 4:
        return cv2.cvtColor(img_data, cv2.COLOR_RGBA2RGB)
    if pixmap.n == 1:
        return cv2.cvtColor(img_data, cv2.COLOR_GRAY2RGB)
    if pixmap.n >= 3:
        return img_data[:, :, :3]

    raise ValueError("Unsupported pixmap channel count.")


def parse_uploaded_file(
    uploaded_file,
    progress_callback: Callable[[float], None] | None = None,
) -> tuple[str, list[Image.Image]]:
    """
    Parse an uploaded PDF/image and return OCR text + preview images.
    """
    extension = uploaded_file.name.rsplit(".", 1)[-1].lower()
    if extension not in SUPPORTED_EXTENSIONS:
        raise ValueError(f"Unsupported file extension: {extension}")

    content_blocks: list[str] = []
    previews: list[Image.Image] = []

    if extension == "pdf":
        pdf_bytes = uploaded_file.getvalue()
        with fitz.open(stream=pdf_bytes, filetype="pdf") as pdf_doc:
            total_pages = max(len(pdf_doc), 1)
            for i, page in enumerate(pdf_doc):
                page_text = page.get_text().strip()

                if len(page_text) < 50:
                    pix = page.get_pixmap(matrix=fitz.Matrix(3, 3))
                    img_rgb = _pixmap_to_rgb_array(pix)
                    ocr_text = extract_text_from_image(img_rgb)
                    parsed_text = ocr_text if ocr_text else "(未擷取到可辨識文字)"
                    source_label = "[由 EasyOCR 引擎萃取]"
                else:
                    parsed_text = page_text
                    source_label = "[原生文字層提取]"

                content_blocks.append(f"--- 第 {i + 1} 頁 ---\n{source_label}\n{parsed_text}")

                if progress_callback:
                    progress_callback((i + 1) / total_pages)

    else:
        pil_img = Image.open(uploaded_file).convert("RGB")
        previews.append(pil_img)
        ocr_text = extract_text_from_image(np.array(pil_img))
        parsed_text = ocr_text if ocr_text else "(未擷取到可辨識文字)"
        content_blocks.append(f"--- 照片檔案內容 ---\n[由 EasyOCR 引擎萃取]\n{parsed_text}")
        if progress_callback:
            progress_callback(1.0)

    return "\n\n".join(content_blocks), previews
