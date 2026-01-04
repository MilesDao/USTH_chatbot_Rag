import os
import argparse
import tempfile
from pathlib import Path
from typing import List, Optional, Tuple

from tqdm import tqdm

# Google Document AI
from google.cloud import documentai_v1 as documentai
from google.api_core.client_options import ClientOptions

# Embedding + Chroma
from sentence_transformers import SentenceTransformer
import chromadb

# PDF split
from pypdf import PdfReader, PdfWriter

# Local LLM (Ollama)
import requests


# ----------------------------
# Document AI OCR
# ----------------------------
def make_docai_client(location: str) -> documentai.DocumentProcessorServiceClient:
    endpoint = f"{location}-documentai.googleapis.com"
    return documentai.DocumentProcessorServiceClient(
        client_options=ClientOptions(api_endpoint=endpoint)
    )


def process_pdf_with_docai(
    client: documentai.DocumentProcessorServiceClient,
    project_id: str,
    location: str,
    processor_id: str,
    pdf_path: Path,
) -> documentai.Document:
    """No process_options to avoid 'Unknown field ... imageless_mode'."""
    name = client.processor_path(project_id, location, processor_id)
    raw = pdf_path.read_bytes()

    req = documentai.ProcessRequest(
        name=name,
        raw_document=documentai.RawDocument(content=raw, mime_type="application/pdf"),
    )
    result = client.process_document(request=req)
    return result.document


# ----------------------------
# Text extraction helpers
# ----------------------------
def _get_text_from_anchor(doc: documentai.Document, text_anchor: documentai.Document.TextAnchor) -> str:
    if not text_anchor.text_segments:
        return ""
    pieces = []
    full = doc.text or ""
    for seg in text_anchor.text_segments:
        start = int(seg.start_index) if seg.start_index is not None else 0
        end = int(seg.end_index) if seg.end_index is not None else 0
        if 0 <= start < end <= len(full):
            pieces.append(full[start:end])
    return "".join(pieces)


def doc_to_paragraph_text(doc: documentai.Document) -> str:
    """Prefer paragraphs (keeps line/paragraph structure). Fallback to doc.text."""
    out_lines: List[str] = []
    if doc.pages:
        for page in doc.pages:
            if page.paragraphs:
                for p in page.paragraphs:
                    t = _get_text_from_anchor(doc, p.layout.text_anchor).strip()
                    if t:
                        out_lines.append(t)
                out_lines.append("")  # blank line between pages
            else:
                if page.lines:
                    for ln in page.lines:
                        t = _get_text_from_anchor(doc, ln.layout.text_anchor).strip()
                        if t:
                            out_lines.append(t)
                    out_lines.append("")
    text = "\n".join(out_lines).strip()
    return text if text else (doc.text or "").strip()


# ----------------------------
# Chunking for embedding
# ----------------------------
def chunk_text(text: str, chunk_size: int = 300, overlap: int = 60) -> List[str]:
    text = text.replace("\r\n", "\n").replace("\r", "\n")
    text = "\n".join([ln.rstrip() for ln in text.splitlines()])
    text = text.strip()
    if not text:
        return []
    step = max(1, chunk_size - overlap)
    chunks: List[str] = []
    for i in range(0, len(text), step):
        ch = text[i:i + chunk_size].strip()
        if ch:
            chunks.append(ch)
    return chunks


# ----------------------------
# PDF split helpers (safe for page limits)
# ----------------------------
def get_pdf_num_pages(pdf_path: Path) -> int:
    reader = PdfReader(str(pdf_path))
    return len(reader.pages)


def split_pdf_into_parts(
    pdf_path: Path,
    out_dir: Path,
    max_pages: int,
) -> List[Tuple[Path, int, int]]:
    """
    Split pdf into parts of <= max_pages.
    Returns list of (part_path, page_start_1based, page_end_1based).
    """
    reader = PdfReader(str(pdf_path))
    total = len(reader.pages)
    parts: List[Tuple[Path, int, int]] = []

    part_idx = 0
    for start in range(0, total, max_pages):
        end = min(start + max_pages, total)  # end exclusive
        writer = PdfWriter()
        for i in range(start, end):
            writer.add_page(reader.pages[i])

        part_idx += 1
        part_path = out_dir / f"{pdf_path.stem}.part{part_idx:03d}_p{start+1}-{end}.pdf"
        with open(part_path, "wb") as f:
            writer.write(f)

        parts.append((part_path, start + 1, end))
    return parts


# ----------------------------
# Vietnamese refine (FREE) using local Ollama
# ----------------------------
def split_for_ollama(text: str, max_chars: int = 9000) -> List[str]:
    """
    Split by blank lines, then pack into blocks <= max_chars.
    Keeps paragraph structure.
    """
    text = text.strip()
    if not text:
        return []
    paras = [p.strip() for p in text.split("\n\n") if p.strip()]
    blocks: List[str] = []
    cur = ""
    for p in paras:
        candidate = (cur + "\n\n" + p).strip() if cur else p
        if len(candidate) <= max_chars:
            cur = candidate
        else:
            if cur:
                blocks.append(cur)
            if len(p) > max_chars:
                # hard cut if a paragraph is too long
                for i in range(0, len(p), max_chars):
                    blocks.append(p[i:i + max_chars])
                cur = ""
            else:
                cur = p
    if cur:
        blocks.append(cur)
    return blocks


def ollama_refine_block(
    block: str,
    model: str,
    base_url: str,
    timeout: int = 120,
) -> str:
    """
    Call Ollama generate API (local).
    """
    prompt = (
        "Bạn là công cụ hiệu đính văn bản OCR tiếng Việt.\n"
        "YÊU CẦU BẮT BUỘC:\n"
        "1) Chỉ sửa lỗi OCR/chính tả/ký tự, thêm dấu câu, xuống dòng cho dễ đọc.\n"
        "2) KHÔNG được bịa, KHÔNG thêm dữ kiện mới, KHÔNG suy đoán.\n"
        "3) Chỗ nào không chắc do OCR mờ/sai, GIỮ NGUYÊN và thêm [UNCLEAR] ngay sau cụm đó.\n"
        "4) Giữ nguyên ý nghĩa, giữ nguyên ngôn ngữ (tiếng Việt là chính).\n"
        "5) Trả về CHỈ văn bản đã hiệu đính, không giải thích.\n"
        "\n---\n\n"
        f"{block}\n"
    )

    url = base_url.rstrip("/") + "/api/generate"
    payload = {
        "model": model,
        "prompt": prompt,
        "stream": False,
        "options": {
            "temperature": 0.1,
            "top_p": 0.9,
        },
    }
    r = requests.post(url, json=payload, timeout=timeout)
    r.raise_for_status()
    data = r.json()
    return (data.get("response") or "").strip()


def refine_text_ollama(
    text: str,
    model: str,
    base_url: str,
    max_chars: int = 9000,
) -> str:
    if not text.strip():
        return text

    blocks = split_for_ollama(text, max_chars=max_chars)
    if not blocks:
        return text

    refined_blocks: List[str] = []
    for b in blocks:
        try:
            refined_blocks.append(ollama_refine_block(b, model=model, base_url=base_url))
        except Exception:
            # if one block fails, keep original block
            refined_blocks.append(b)

    return "\n\n".join([x for x in refined_blocks if x]).strip()


def check_ollama_alive(base_url: str, timeout: int = 5) -> bool:
    try:
        r = requests.get(base_url.rstrip("/") + "/api/tags", timeout=timeout)
        return r.status_code == 200
    except Exception:
        return False


# ----------------------------
# Main
# ----------------------------
def main():
    ap = argparse.ArgumentParser()

    ap.add_argument("--project_id", required=True)
    ap.add_argument("--location", required=True, help="e.g. us, eu, asia-southeast1")
    ap.add_argument("--processor_id", required=True)

    ap.add_argument("--input_dir", default="raw", help="Folder chứa PDF scan")
    ap.add_argument("--out_text_dir", default="output/docai_text", help="Lưu text OCR")
    ap.add_argument("--chunk_size", type=int, default=300)
    ap.add_argument("--overlap", type=int, default=60)

    ap.add_argument("--embed_model", default="sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2")
    ap.add_argument("--chroma_dir", default="chroma_db")
    ap.add_argument("--collection", default="usth_docs")

    # Split controls (IMPORTANT for PDFs > 30 pages)
    ap.add_argument("--split_large_pdfs", action="store_true", help="Tự split PDF theo max_pages_per_request")
    ap.add_argument("--max_pages_per_request", type=int, default=15,
                    help="Set 15 để luôn an toàn với page limit non-imageless")

    # Refine controls
    ap.add_argument("--refine", choices=["none", "ollama"], default="ollama",
                    help="none: không sửa | ollama: refine tiếng Việt bằng LLM local (FREE)")
    ap.add_argument("--ollama_model", default="qwen2.5:7b")
    ap.add_argument("--ollama_url", default="http://localhost:11434")
    ap.add_argument("--ollama_max_chars", type=int, default=9000)

    args = ap.parse_args()

    input_dir = Path(args.input_dir)
    out_text_dir = Path(args.out_text_dir)
    out_text_dir.mkdir(parents=True, exist_ok=True)

    pdfs = sorted(input_dir.glob("*.pdf"))
    if not pdfs:
        print(f"Không thấy PDF trong: {input_dir.resolve()}")
        return

    # 1) DocAI client
    client = make_docai_client(args.location)

    # 2) Embedder
    embedder = SentenceTransformer(args.embed_model)

    # 3) Chroma
    chroma_client = chromadb.PersistentClient(path=args.chroma_dir)
    col = chroma_client.get_or_create_collection(name=args.collection)

    # 4) Ollama availability
    ollama_ok = False
    if args.refine == "ollama":
        ollama_ok = check_ollama_alive(args.ollama_url)
        if not ollama_ok:
            print("[WARN] Không kết nối được Ollama tại", args.ollama_url)
            print("[WARN] Hãy chạy: ollama serve  (và đảm bảo model đã pull).")
            print("[WARN] Tạm thời chuyển refine=none.")
            args.refine = "none"

    # Accumulate and upsert at end (simple). If your corpus is huge, we can stream-upsert.
    all_ids: List[str] = []
    all_docs: List[str] = []
    all_metas: List[dict] = []
    all_embeds: List[list] = []

    for pdf in tqdm(pdfs, desc="Document AI OCR"):
        try:
            num_pages = get_pdf_num_pages(pdf)

            # Determine parts
            parts: List[Tuple[Path, int, int]] = [(pdf, 1, num_pages)]
            tempdir_obj: Optional[tempfile.TemporaryDirectory] = None

            if args.split_large_pdfs and num_pages > args.max_pages_per_request:
                tempdir_obj = tempfile.TemporaryDirectory()
                split_dir = Path(tempdir_obj.name)
                parts = split_pdf_into_parts(pdf, split_dir, args.max_pages_per_request)

            final_text_parts: List[str] = []

            for part_idx, (part_path, pstart, pend) in enumerate(parts, start=1):
                doc = process_pdf_with_docai(
                    client,
                    args.project_id,
                    args.location,
                    args.processor_id,
                    part_path,
                )
                part_text_raw = doc_to_paragraph_text(doc).strip()

                # Save raw part
                raw_path = out_text_dir / f"{pdf.stem}.p{pstart}-{pend}.raw.txt"
                raw_path.write_text(part_text_raw, encoding="utf-8")

                # Refine (Vietnamese) via Ollama
                part_text = part_text_raw
                if args.refine == "ollama" and part_text_raw:
                    part_text = refine_text_ollama(
                        part_text_raw,
                        model=args.ollama_model,
                        base_url=args.ollama_url,
                        max_chars=args.ollama_max_chars,
                    )

                out_path = out_text_dir / f"{pdf.stem}.p{pstart}-{pend}.txt"
                out_path.write_text(part_text, encoding="utf-8")

                header = f"\n\n===== {pdf.name} | PART {part_idx}/{len(parts)} | PAGES {pstart}-{pend} =====\n\n"
                final_text_parts.append(header + part_text)

            if tempdir_obj is not None:
                tempdir_obj.cleanup()

            final_text = "\n".join(final_text_parts).strip()
            whole_out = out_text_dir / f"{pdf.stem}.ALL.txt"
            whole_out.write_text(final_text, encoding="utf-8")

            # Chunk -> embed
            chunks = chunk_text(final_text, chunk_size=args.chunk_size, overlap=args.overlap)
            if not chunks:
                continue

            embeds = embedder.encode(chunks, show_progress_bar=False, normalize_embeddings=True)

            for idx, (ch, emb) in enumerate(zip(chunks, embeds)):
                cid = f"{pdf.stem}::chunk{idx}"
                all_ids.append(cid)
                all_docs.append(ch)
                all_metas.append({
                    "source_file": pdf.name,
                    "chunk_index": idx,
                    "chunk_size": args.chunk_size,
                    "overlap": args.overlap,
                    "refine": args.refine,
                    "ollama_model": args.ollama_model if args.refine == "ollama" else "",
                    "split_large_pdfs": bool(args.split_large_pdfs),
                    "max_pages_per_request": int(args.max_pages_per_request),
                    "pdf_pages": int(num_pages),
                })
                all_embeds.append(emb.tolist())

        except Exception as e:
            print(f"[ERROR] {pdf.name}: {e}")

    if all_ids:
        col.upsert(ids=all_ids, documents=all_docs, metadatas=all_metas, embeddings=all_embeds)
        print(f"\nDone. Upserted {len(all_ids)} chunks into Chroma collection '{args.collection}'.")
        print(f"Text saved to: {out_text_dir.resolve()}")
        print(f"Chroma persisted at: {Path(args.chroma_dir).resolve()}")
        print("Files: *.pX-Y.raw.txt, *.pX-Y.txt, và *.ALL.txt (ghép toàn bộ).")
    else:
        print("Không có chunk nào để lưu (text rỗng hoặc OCR lỗi).")


if __name__ == "__main__":
    cred = os.environ.get("GOOGLE_APPLICATION_CREDENTIALS")
    if not cred:
        print("ERROR: GOOGLE_APPLICATION_CREDENTIALS chưa được set.")
        print("Ví dụ: export GOOGLE_APPLICATION_CREDENTIALS=$HOME/keys/introai-483108-xxxx.json")
    else:
        main()
