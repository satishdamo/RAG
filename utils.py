import pytesseract
import base64
import io
from PIL import Image
import fitz
import hashlib
import os
from openai import OpenAI
import boto3
from dotenv import load_dotenv

load_dotenv()

# Initialize S3 client
s3 = boto3.client(
    "s3",
    aws_access_key_id=os.getenv("AWS_ACCESS_KEY_ID"),
    aws_secret_access_key=os.getenv("AWS_SECRET_ACCESS_KEY"),
    region_name=os.getenv("AWS_DEFAULT_REGION")
)


def upload_base64_image(base64_string, key, public=True):
    # Decode base64 string
    image_bytes = base64.b64decode(base64_string)
    # Upload to S3
    s3.put_object(
        Bucket=os.getenv("AWS_S3_BUCKET_NAME"),
        Key=key,
        Body=image_bytes
    )
    # Construct the public URL (no presigned URL)
    bucket = os.getenv("AWS_S3_BUCKET_NAME")
    region = os.getenv("AWS_DEFAULT_REGION", "us-east-1")
    url = f"https://{bucket}.s3.{region}.amazonaws.com/{key}"
    return url


# Initialize OpenAI client (expects OPENAI_API_KEY in environment)
client = OpenAI()

OCR_CACHE_DIR = os.environ.get("OCR_CACHE_DIR")
os.makedirs(OCR_CACHE_DIR, exist_ok=True)


def get_pdf_hash(pdf_path: str) -> str:
    with open(pdf_path, "rb") as f:
        pdf_bytes = f.read()
    return hashlib.md5(pdf_bytes).hexdigest()


def extract_text_with_fitz_ocr(pdf_path: str) -> str:
    """
    Extract text from PDF using PyMuPDF + Tesseract OCR.
    Results cached by PDF hash to avoid repeated OCR.
    """
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
    """
    Simple heuristic: check fallback phrases and token length.
    """
    fallback_phrases = ["i'm not sure", "i do not know",
                        "cannot answer", "no relevant context"]
    lowered = answer.lower()

    if any(p in lowered for p in fallback_phrases):
        return False, 0.0

    token_count = len(answer.split())
    score = min(1.0, token_count / 100)  # Cap at 1.0
    return True, round(score, 2)


def caption_image(image_bytes: bytes) -> dict:
    """
    Takes raw image bytes and returns a dictionary with:
    - caption: OpenAI-generated description
    - image_base64: base64-encoded PNG preview
    """
    img_str = base64.b64encode(image_bytes).decode("utf-8")

    response = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": "Describe this image in one sentence."},
                    {
                        "type": "image_url",
                        "image_url": {"url": f"data:image/png;base64,{img_str}"}
                    }
                ],
            }
        ],
    )

    caption = response.choices[0].message.content

    return {
        "caption": caption,
        "image_base64": img_str
    }
