from langchain.chains import RetrievalQA
from langchain.chains.question_answering import load_qa_chain
from langchain.llms import OpenAI
from langchain.vectorstores import Chroma
from langchain.embeddings import HuggingFaceEmbeddings
from fastapi.responses import HTMLResponse
from fastapi import Query, FastAPI, UploadFile, File, Depends, HTTPException
from fastapi.security import OAuth2PasswordRequestForm
from auth import authenticate_user, create_access_token, get_current_user
from rag_pipeline import build_qa_chain_with_memory
from ingest import ingest_pdf, extract_images_with_captions, ingest_pdf_with_images
from utils import extract_text_with_fitz_ocr
import shutil
import os
from dotenv import load_dotenv

load_dotenv()

# LangChain imports used by the admin ingest endpoint

# Simple defaults used by the admin endpoint; adjust paths/registry as needed
VECTOR_DB_ROOT = os.environ.get("VECTOR_DB_ROOT", "chroma_db")
vectorstore_registry = {}

app = FastAPI()
qa_chain = build_qa_chain_with_memory()
session_store = {}

# --- cached public vectordb / QA chain (initialized at app startup) ---
public_vectordb = None
public_qa_chain = None


def init_public_rag():
    """
    Initialize and cache the persisted Chroma vectorstore and a RetrievalQA chain.
    Called at application startup. If VECTOR_DB_ROOT does not exist, leaves the
    globals as None (service will return 503 until admin ingests).
    """
    global public_vectordb, public_qa_chain

    if public_vectordb is not None and public_qa_chain is not None:
        return  # already initialized

    if not os.path.isdir(VECTOR_DB_ROOT):
        # no vector DB persisted yet; admin must ingest
        print(
            f"[startup] Vector DB not found at {VECTOR_DB_ROOT}; public RAG unavailable")
        return

    # create embeddings and load persisted Chroma
    embeddings_model = os.environ.get(
        "VECTOR_EMBEDDING_MODEL", "sentence-transformers/all-MiniLM-L6-v2")
    embeddings = HuggingFaceEmbeddings(model_name=embeddings_model)
    vectordb = Chroma(persist_directory=VECTOR_DB_ROOT,
                      embedding_function=embeddings)

    # build retriever + QA chain (temperature 0 for deterministic answers)
    retriever = vectordb.as_retriever(search_kwargs={"k": 5})
    llm = OpenAI(temperature=0)

    # or "map_reduce", "refine", etc.
    qa_chain = load_qa_chain(llm, chain_type="stuff")

    qa = RetrievalQA(combine_documents_chain=qa_chain, retriever=retriever,
                     return_source_documents=True)

    public_vectordb = vectordb
    public_qa_chain = qa
    vectorstore_registry[VECTOR_DB_ROOT] = vectordb
    print(
        f"[startup] Loaded public vector DB from {VECTOR_DB_ROOT} and initialized QA chain")


@app.on_event("startup")
async def startup_event():
    try:
        init_public_rag()
    except Exception as e:
        print(f"[startup] Failed to initialize public RAG: {e}")


@app.post("/token")
async def login(form_data: OAuth2PasswordRequestForm = Depends()):
    user = authenticate_user(form_data.username, form_data.password)
    if not user:
        raise HTTPException(status_code=401, detail="Incorrect credentials")

    token = create_access_token(
        {"sub": user["username"], "role": user["role"]})

    # ✅ Safe initialization
    username = user["username"]
    if username not in session_store:
        session_store[username] = {
            "captions": [],
            "history": [],
            "file_path": None,
            "chunks": [],
            "ingestion_stats": {}
        }

    return {"access_token": token, "token_type": "bearer"}


@app.post("/upload-pdf/")
async def upload_pdf(file: UploadFile = File(...), user=Depends(get_current_user)):
    username = user["username"]

    # ✅ Ensure session exists
    if username not in session_store:
        session_store[username] = {
            "captions": [],
            "history": [],
            "file_path": None,
            "chunks": [],
            "ingestion_stats": {}
        }

    file_path = f"data/{file.filename}"
    with open(file_path, "wb") as f:
        shutil.copyfileobj(file.file, f)

    captions = ingest_pdf_with_images(file_path)

    session_store[username]["captions"] = captions
    session_store[username]["file_path"] = file_path

    return {"message": f"{file.filename} ingested with {len(captions)} images."}


@app.get("/ask/")
async def ask_question(q: str, user=Depends(get_current_user)):
    result = qa_chain({"question": q})

    answer = result.get("answer", "I'm not sure.")
    score = result.get("confidence_score", 0.0)
    retrieved_chunks = result.get("retrieved_chunks", [])
    confident = score > 0.5

    response = {
        "answer": answer,
        "confidence_score": score,
        "retrieved_chunks": retrieved_chunks
    }

    if confident:
        user_data = session_store.get(user["username"], {})
        captions = user_data.get("captions", [])

        # Fallback: regenerate captions if missing
        if not captions and "file_path" in user_data:
            captions = extract_images_with_captions(user_data["file_path"])
            session_store[user["username"]]["captions"] = captions

        response["image_captions"] = captions[:3]

    return response


@app.post("/debug-ingest/")
async def debug_ingest(file: UploadFile = File(...), full_text: bool = Query(False), user=Depends(get_current_user)):
    file_path = f"data/{file.filename}"
    with open(file_path, "wb") as f:
        shutil.copyfileobj(file.file, f)

    try:
        ingest_pdf(file_path)
        source = "pdf"
    except ValueError:
        extracted_text = extract_text_with_fitz_ocr(file_path)
        source = "ocr"
    else:
        extracted_text = extract_text_with_fitz_ocr(
            file_path) if source == "ocr" else None

    captions = extract_images_with_captions(file_path)
    return {
        "summary": extracted_text if full_text else (extracted_text[:2000] + ("..." if extracted_text and len(extracted_text) > 2000 else "")),
        "source": source,
        "image_captions": captions[:5]
    }


@app.post("/preview-html/")
async def preview_html(file: UploadFile = File(...), user=Depends(get_current_user)):
    file_path = f"data/{file.filename}"
    with open(file_path, "wb") as f:
        shutil.copyfileobj(file.file, f)

    try:
        ingest_pdf(file_path)
        extracted_text = extract_text_with_fitz_ocr(file_path)
    except ValueError:
        extracted_text = extract_text_with_fitz_ocr(file_path)

    captions = extract_images_with_captions(file_path)

    html = f"""
    <html>
    <head>
        <title>Preview: {file.filename}</title>
        <style>
            body {{
                font-family: Arial, sans-serif;
                padding: 2em;
                background: #f9f9f9;
                color: #333;
                transition: background 0.3s, color 0.3s;
            }}
            .dark-mode {{
                background: #1e1e1e;
                color: #eee;
            }}
            h1 {{ margin-top: 2em; }}
            .caption {{ margin-top: 1em; font-weight: bold; }}
            .image {{ margin-bottom: 2em; }}
            img {{ max-width: 600px; border: 1px solid #ccc; }}
            pre {{
                background: #fff;
                padding: 1em;
                border: 1px solid #ccc;
                overflow-x: auto;
                white-space: pre-wrap;
            }}
            .dark-mode pre {{
                background: #2e2e2e;
                border-color: #444;
            }}
            .controls {{
                position: fixed;
                top: 10px;
                right: 10px;
            }}
            button {{
                margin-left: 10px;
                padding: 0.5em 1em;
                font-size: 14px;
                cursor: pointer;
            }}
        </style>
    </head>
    <body>
        <div class="controls">
            <button onclick="toggleDarkMode()">Toggle Dark Mode</button>
            <button onclick="downloadText()">Download Text</button>
        </div>
        <h1>Extracted Text</h1>
        <pre id="text-block">{extracted_text[:5000]}</pre>
        <h1>Image Captions</h1>
    """

    for item in captions[:5]:
        html += f"""
        <div class="image">
            <div class="caption">Page {item['page']}: {item['caption']}</div>
            <img src="data:image/png;base64,{item['image_base64']}" />
        </div>
        """

    html += """
        <script>
            function toggleDarkMode() {
                document.body.classList.toggle('dark-mode');
            }
            function downloadText() {
                const text = document.getElementById('text-block').innerText;
                const blob = new Blob([text], { type: 'text/plain' });
                const link = document.createElement('a');
                link.href = URL.createObjectURL(blob);
                link.download = 'extracted_text.txt';
                link.click();
            }
        </script>
    </body>
    </html>
    """

    return HTMLResponse(content=html)

# Public, no-auth endpoint for front-end chat widget to query admin-ingested RAG


@app.get("/public-ask/")
async def public_ask(q: str):
    """
    Public, no-auth endpoint that answers using the admin-ingested persisted vector DB.
    If the vector DB isn't available returns 503 so front-end can show a friendly message.
    """
    if public_qa_chain is None or public_vectordb is None:
        raise HTTPException(
            status_code=503, detail="Public RAG is not available. Admin must ingest documents.")

    try:
        # run QA chain to produce the textual answer
        result = public_qa_chain.invoke({"query": q})
        answer = result.get("result", "")
        # fetch the retrieved documents for transparency
        docs = result.get("source_documents", [])
    except Exception as e:
        raise HTTPException(
            status_code=500, detail=f"Error running public RAG: {e}")

    snippets = []
    for d in docs[:4]:
        snippets.append({
            "text": d.page_content[:2000],
            "metadata": getattr(d, "metadata", {}) or {}
        })

    return {"answer": answer, "retrieved": snippets}
