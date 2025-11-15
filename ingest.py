import os
from dotenv import load_dotenv
import fitz  # PyMuPDF
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.vectorstores import Chroma
from langchain.embeddings import HuggingFaceEmbeddings
from utils import caption_image, extract_text_with_fitz_ocr
from langchain.schema import Document


load_dotenv()

VECTOR_DB_ROOT = os.environ.get("VECTOR_DB_ROOT", "chroma_db")


def ingest_pdf(pdf_path: str):
    """
    Extracts text from a PDF (with OCR fallback), splits it into chunks,
    and stores embeddings in Chroma vector DB.
    """
    doc = fitz.open(pdf_path)
    all_text = ""

    for page in doc:
        text_dict = page.get_text("dict")
        for block in text_dict.get("blocks", []):
            if "lines" in block:
                for line in block["lines"]:
                    for span in line["spans"]:
                        all_text += span["text"] + " "
                all_text += "\n"

    if not all_text.strip():
        all_text = extract_text_with_fitz_ocr(pdf_path)

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=1500, chunk_overlap=200)
    docs = splitter.split_documents([Document(page_content=all_text)])

    embedding = HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-MiniLM-L6-v2")
    vectordb = Chroma.from_documents(
        documents=docs, embedding=embedding, persist_directory=VECTOR_DB_ROOT)
    vectordb.persist()


def extract_images_with_captions(pdf_path: str):
    """
    Extracts images from a PDF and generates captions using BLIP.
    Returns a list of dicts with base64-encoded images and captions.
    """
    doc = fitz.open(pdf_path)
    captions = []

    for page_index in range(len(doc)):
        page = doc.load_page(page_index)
        images = page.get_images(full=True)

        for img_index, img in enumerate(images):
            xref = img[0]
            base_image = doc.extract_image(xref)
            image_bytes = base_image["image"]

            result = caption_image(image_bytes)

            captions.append({
                "caption": result["caption"],
                "image_base64": result["image_base64"],
                "page": page_index + 1
            })

    return captions


def ingest_pdf_with_images(pdf_path: str):
    """
    Ingests a PDF by extracting text and images with captions,
    then storing text embeddings in Chroma vector DB.
    """
    # 1. Extract images/captions for each page
    # returns list of dicts: {"page": int, "caption": str, "image_base64": str}
    captions = extract_images_with_captions(pdf_path)

    # 2. Load and split text into chunks, keeping track of page numbers
    doc = fitz.open(pdf_path)
    all_text = ""

    for page in doc:
        text_dict = page.get_text("dict")
        for block in text_dict.get("blocks", []):
            if "lines" in block:
                for line in block["lines"]:
                    for span in line["spans"]:
                        all_text += span["text"] + " "
                all_text += "\n"

    if not all_text.strip():
        all_text = extract_text_with_fitz_ocr(pdf_path)

    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=1500, chunk_overlap=200)
    docs = []

    for i, page in enumerate(text_splitter.split_documents([Document(page_content=all_text)])):
        page_chunks = text_splitter.split_text(page.page_content)
        # Find image/caption for this page
        img_info = next((c for c in captions if c["page"] == i+1), None)
        for chunk in page_chunks:
            metadata = {"page": i+1}
            if img_info:
                metadata["caption"] = img_info["caption"]
                metadata["image_base64"] = img_info["image_base64"]
            docs.append(Document(page_content=chunk, metadata=metadata))

    # 3. Embed and persist chunks
    embeddings = HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-MiniLM-L6-v2")
    vectordb = Chroma.from_documents(
        docs, embedding=embeddings, persist_directory=VECTOR_DB_ROOT)
    vectordb.persist()
    return captions
