from paddleocr import PaddleOCR
from PIL import Image
import fitz  # PyMuPDF
from io import BytesIO
from typing import List

class OCRProcessor:
    def __init__(self, lang: str = "fr"):
        self.lang = lang
        self.ocr = PaddleOCR(use_angle_cls=True, lang=lang)

    def extract_text_from_pdf_bytes(self, pdf_bytes: bytes) -> str:
        doc = fitz.open(stream=pdf_bytes, filetype="pdf")
        texts = []
        for page in doc:
            pix = page.get_pixmap()
            img = Image.frombytes("RGB", [pix.width, pix.height], pix.samples)
            result = self.ocr.ocr(img, cls=True)
            texts.append("\n".join([line[1][0] for line in result[0]]))
        return "\n".join(texts)

    def extract_text_from_file(self, path: str) -> str:
        ext = path.lower().split('.')[-1]
        if ext in ("jpg", "jpeg", "png"):
            return self.extract_text_from_image_file(path)
        elif ext == "pdf":
            with open(path, "rb") as f:
                pdf_bytes = f.read()
            return self.extract_text_from_pdf_bytes(pdf_bytes)
        else:
            raise ValueError(f"Format non pris en charge : {ext}")

    def extract_text_from_image_bytes(self, img_bytes: bytes) -> str:
        img = Image.open(BytesIO(img_bytes))
        result = self.ocr.ocr(img, cls=True)
        return "\n".join([line[1][0] for line in result[0]])

    def extract_text_from_image_file(self, img_path: str) -> str:
        img = Image.open(img_path)
        result = self.ocr.ocr(img, cls=True)
        return "\n".join([line[1][0] for line in result[0]])
