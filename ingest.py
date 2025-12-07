from rag_pipeline import PINECONE_INDEX_NAME
import os
import fitz  # PyMuPDF
from dotenv import load_dotenv
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_openai import OpenAIEmbeddings
from langchain_pinecone import PineconeVectorStore
from langchain.schema import Document
from utils import caption_image, extract_text_with_fitz_ocr, upload_base64_image
from rag_pipeline import PINECONE_INDEX_NAME, PINECONE_API_KEY, PINECONE_ENVIRONMENT

load_dotenv()

if not PINECONE_API_KEY:
    raise RuntimeError("PINECONE_API_KEY is required. Set it in your .env")

if not PINECONE_ENVIRONMENT:
    raise RuntimeError("PINECONE_ENVIRONMENT is required. Set it in your .env")


def ingest_pdf(pdf_path: str):
    """
    Extracts text from a PDF (with OCR fallback), splits it into chunks,
    and stores embeddings in Pinecone vector DB.
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

    embeddings = OpenAIEmbeddings()
    vector_store = PineconeVectorStore.from_existing_index(
        index_name=PINECONE_INDEX_NAME, embedding=embeddings
    )
    vector_store.add_documents(docs)

    print(f"[ingest] Upserted {len(docs)} chunks from {pdf_path}")
    return True


def extract_images_with_captions(pdf_path: str):
    """
    Extracts images from a PDF and generates captions.
    Returns a list of dicts with base64-encoded images and captions.
    """
    doc = fitz.open(pdf_path)
    captions = []

    for page_index in range(len(doc)):
        page = doc.load_page(page_index)
        images = page.get_images(full=True)

        print(f"[debug] Page {page_index+1} has {len(images)} images")

        for img_index, img in enumerate(images):
            xref = img[0]
            base_image = doc.extract_image(xref)
            image_bytes = base_image["image"]

            result = caption_image(image_bytes)

            captions.append({
                "page": page_index + 1,
                "image_index": img_index + 1,
                "caption": result["caption"],
                "image_base64": result["image_base64"]
            })

    return captions


def ingest_pdf_with_images(pdf_path: str):
    """
    Ingests a PDF by extracting text and images with captions,
    then storing text embeddings in Pinecone vector DB.
    """
    doc = fitz.open(pdf_path)
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000, chunk_overlap=100)
    embeddings = OpenAIEmbeddings()

    # Extract all images with captions
    captions = extract_images_with_captions(pdf_path)
    page_media = {}
    for item in captions:
        page = item["page"]
        if page not in page_media:
            page_media[page] = []
        # Upload each image with unique naming convention
        try:
            image_url = upload_base64_image(
                item["image_base64"],
                f"images/{os.path.basename(pdf_path)}_page_{page}_img_{item['image_index']}.png"
            )
        except Exception as e:
            print(
                f"[warn] Upload failed for page {page}, image {item['image_index']}: {e}")
            image_url = None
        page_media[page].append({
            "caption": item["caption"],
            "image_url": image_url,
            "image_index": item["image_index"]
        })

    docs = []
    summary = []

    for page_num in range(len(doc)):
        page = doc.load_page(page_num)
        text = page.get_text("text")

        if not text.strip():
            text = extract_text_with_fitz_ocr(pdf_path)

        chunks = text_splitter.split_text(text)
        page_chunks = []

        for idx, chunk in enumerate(chunks):
            metadata = {"page": page_num + 1, "chunk_index": idx + 1}

            # Attach all images for this page (flattened)
            if page_num + 1 in page_media:
                captions_list = [img["caption"]
                                 for img in page_media[page_num + 1]]
                urls_list = [img["image_url"]
                             for img in page_media[page_num + 1] if img["image_url"]]
                metadata["image_captions"] = captions_list
                metadata["image_urls"] = urls_list

            docs.append(Document(page_content=chunk, metadata=metadata))
            page_chunks.append(chunk)

        summary.append({
            "page": page_num + 1,
            "chunks": len(page_chunks),
            "images": len(page_media.get(page_num + 1, []))
        })

    # Upsert into Pinecone
    vector_store = PineconeVectorStore.from_existing_index(
        index_name=PINECONE_INDEX_NAME, embedding=embeddings
    )
    vector_store.add_documents(docs)

    print(f"[ingest] PDF: {pdf_path}")
    for s in summary:
        print(
            f"  Page {s['page']}: {s['chunks']} chunks, {s['images']} images attached")
    print(f"[ingest] Total chunks upserted: {len(docs)}")

    return summary
