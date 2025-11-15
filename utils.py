import pytesseract
import base64
import io
import torch
from PIL import Image
import fitz
from transformers import BlipProcessor, BlipForConditionalGeneration

import hashlib
import os

OCR_CACHE_DIR = "ocr_cache"
os.makedirs(OCR_CACHE_DIR, exist_ok=True)


def get_pdf_hash(pdf_path: str) -> str:
    with open(pdf_path, "rb") as f:
        pdf_bytes = f.read()
    return hashlib.md5(pdf_bytes).hexdigest()


def extract_text_with_fitz_ocr(pdf_path: str) -> str:
    pdf_hash = get_pdf_hash(pdf_path)
    cache_path = os.path.join(OCR_CACHE_DIR, f"{pdf_hash}.txt")

    if os.path.exists(cache_path):
        with open(cache_path, "r", encoding="utf-8") as f:
            return f.read()

    doc = fitz.open(pdf_path)
    full_text = ""
    for page in doc:
        pix = page.get_pixmap(dpi=300)
        img = Image.frombytes("RGB", [pix.width, pix.height], pix.samples)
        text = pytesseract.image_to_string(img)
        full_text += text + "\n"

    with open(cache_path, "w", encoding="utf-8") as f:
        f.write(full_text)

    return full_text


def is_answer_confident(answer: str) -> tuple[bool, float]:
    fallback_phrases = ["i'm not sure", "i do not know",
                        "cannot answer", "no relevant context"]
    lowered = answer.lower()

    if any(p in lowered for p in fallback_phrases):
        return False, 0.0

    token_count = len(answer.split())
    score = min(1.0, token_count / 100)  # Cap at 1.0
    return True, round(score, 2)


device = "cuda" if torch.cuda.is_available() else "cpu"
processor = BlipProcessor.from_pretrained(
    "Salesforce/blip-image-captioning-base")
model = BlipForConditionalGeneration.from_pretrained(
    "Salesforce/blip-image-captioning-base").to(device)


def caption_image(image_bytes: bytes) -> dict:
    """
    Takes raw image bytes and returns a dictionary with:
    - caption: BLIP-generated description
    - image_base64: base64-encoded PNG preview
    """
    image = Image.open(io.BytesIO(image_bytes)).convert("RGB")
    inputs = processor(image, return_tensors="pt").to(device)
    out = model.generate(**inputs,  max_new_tokens=50)
    caption = processor.decode(out[0], skip_special_tokens=True)

    buffered = io.BytesIO()
    image.save(buffered, format="PNG")
    img_str = base64.b64encode(buffered.getvalue()).decode("utf-8")

    return {
        "caption": caption,
        "image_base64": img_str
    }
