# ocr_module/ocr.py

import pytesseract
from pdf2image import convert_from_bytes
from PIL import Image
from io import BytesIO
from typing import List, Union

class OCRProcessor:
    def __init__(self, lang: str = "fra"):
        self.lang = lang

    def extract_text_from_pdf_bytes(self, pdf_bytes: bytes) -> str:
        images = convert_from_bytes(pdf_bytes)
        return self._extract_from_images(images)

    def extract_text_from_file(self, path: str) -> str:
        ext = path.lower().split('.')[-1]
        if ext in ("jpg","jpeg","png"):
            return self.extract_text_from_image_file(path)
        elif ext == "pdf":
            with open(path,"rb") as f: pdf_bytes = f.read()
            return self.extract_text_from_pdf_bytes(pdf_bytes)
        else:
            raise ValueError(f"Format non pris en charge : {ext}")

    def extract_text_from_image_bytes(self, img_bytes: bytes) -> str:
        img = Image.open(BytesIO(img_bytes))
        return pytesseract.image_to_string(img, lang=self.lang)

    def extract_text_from_image_file(self, img_path: str) -> str:
        img = Image.open(img_path)
        return pytesseract.image_to_string(img, lang=self.lang)

    def _extract_from_images(self, images: List[Image.Image]) -> str:
        texts = [pytesseract.image_to_string(img, lang=self.lang) for img in images]
        return "\n".join(texts)
