from paddleocr import PaddleOCR
from PIL import Image
import fitz  # PyMuPDF
from io import BytesIO
from typing import List

class OCRProcessor:
    def __init__(self, lang: str = "french"):
        """
        :param lang: language model name accepted by PaddleOCR, e.g., 'french', 'english', etc.
        """
        self.lang = lang
        # PaddleOCR expects full names: 'french', 'english', etc.
        self.ocr = PaddleOCR(use_angle_cls=True, lang=self.lang)

    def extract_text_from_pdf_bytes(self, pdf_bytes: bytes) -> str:
        doc = fitz.open(stream=pdf_bytes, filetype="pdf")
        texts: List[str] = []
        for page in doc:
            pix = page.get_pixmap()
            img = Image.frombytes("RGB", [pix.width, pix.height], pix.samples)
            result = self.ocr.ocr(img, cls=True)
            # result is List[List[box, (text, confidence)]], take the text
            page_text = "\n".join(entry[1][0] for entry in result[0])
            texts.append(page_text)
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
        return "\n".join(entry[1][0] for entry in result[0])

    def extract_text_from_image_file(self, img_path: str) -> str:
        img = Image.open(img_path)
        result = self.ocr.ocr(img, cls=True)
        return "\n".join(entry[1][0] for entry in result[0])
