import os
import argparse
import tempfile
from pathlib import Path
from typing import List, Optional, Tuple

from tqdm import tqdm
from google.cloud import documentai_v1 as documentai
from google.api_core.client_options import ClientOptions

from pypdf import PdfReader, PdfWriter

import requests

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
    """Prefer paragraphs, fallback to lines, fallback to doc.text."""
    out_lines: List[str] = []
    if doc.pages:
        for page in doc.pages:
            if page.paragraphs:
                for p in page.paragraphs:
                    t = _get_text_from_anchor(doc, p.layout.text_anchor).strip()
                    if t:
                        out_lines.append(t)
                out_lines.append("")
            elif page.lines:
                for ln in page.lines:
                    t = _get_text_from_anchor(doc, ln.layout.text_anchor).strip()
                    if t:
                        out_lines.append(t)
                out_lines.append("")
    text = "\n".join(out_lines).strip()
    return text if text else (doc.text or "").strip()

def get_pdf_num_pages(pdf_path: Path) -> int:
    reader = PdfReader(str(pdf_path))
    return len(reader.pages)


def split_pdf_into_parts(
    pdf_path: Path,
    out_dir: Path,
    max_pages: int,
) -> List[Tuple[Path, int, int]]:
    reader = PdfReader(str(pdf_path))
    total = len(reader.pages)
    parts: List[Tuple[Path, int, int]] = []

    part_idx = 0
    for start in range(0, total, max_pages):
        end = min(start + max_pages, total)
        writer = PdfWriter()
        for i in range(start, end):
            writer.add_page(reader.pages[i])

        part_idx += 1
        part_path = out_dir / f"{pdf_path.stem}.part{part_idx:03d}_p{start+1}-{end}.pdf"
        with open(part_path, "wb") as f:
            writer.write(f)

        parts.append((part_path, start + 1, end))
    return parts

def split_for_ollama(text: str, max_chars: int = 9000) -> List[str]:
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
    prompt = (
        "Bạn là công cụ hiệu đính văn bản OCR tiếng Việt.\n"
        "YÊU CẦU BẮT BUỘC:\n"
        "1) Chỉ sửa lỗi OCR/chính tả/ký tự, thêm dấu câu, xuống dòng.\n"
        "2) KHÔNG bịa, KHÔNG thêm dữ kiện.\n"
        "3) Không chắc thì giữ nguyên và thêm [UNCLEAR].\n"
        "4) Trả về CHỈ văn bản đã hiệu đính.\n"
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
    return (r.json().get("response") or "").strip()


def refine_text_ollama(
    text: str,
    model: str,
    base_url: str,
    max_chars: int = 9000,
) -> str:
    blocks = split_for_ollama(text, max_chars=max_chars)
    if not blocks:
        return text
    refined: List[str] = []
    for b in blocks:
        try:
            refined.append(ollama_refine_block(b, model, base_url))
        except Exception:
            refined.append(b)
    return "\n\n".join(refined).strip()


def check_ollama_alive(base_url: str) -> bool:
    try:
        r = requests.get(base_url.rstrip("/") + "/api/tags", timeout=5)
        return r.status_code == 200
    except Exception:
        return False

def main():
    ap = argparse.ArgumentParser()

    ap.add_argument("--project_id", required=True)
    ap.add_argument("--location", required=True)
    ap.add_argument("--processor_id", required=True)

    ap.add_argument("--input_dir", default="raw")
    ap.add_argument("--out_text_dir", default="output/docai_text")

    ap.add_argument("--split_large_pdfs", action="store_true")
    ap.add_argument("--max_pages_per_request", type=int, default=15)

    ap.add_argument("--refine", choices=["none", "ollama"], default="ollama")
    ap.add_argument("--ollama_model", default="qwen2.5:7b")
    ap.add_argument("--ollama_url", default="http://localhost:11434")
    ap.add_argument("--ollama_max_chars", type=int, default=9000)

    args = ap.parse_args()

    input_dir = Path(args.input_dir)
    out_text_dir = Path(args.out_text_dir)
    out_text_dir.mkdir(parents=True, exist_ok=True)

    pdfs = sorted(input_dir.glob("*.pdf"))
    if not pdfs:
        print("Không tìm thấy PDF.")
        return

    client = make_docai_client(args.location)

    if args.refine == "ollama" and not check_ollama_alive(args.ollama_url):
        print("[WARN] Ollama không chạy → chuyển refine=none")
        args.refine = "none"

    for pdf in tqdm(pdfs, desc="OCR PDFs"):
        try:
            num_pages = get_pdf_num_pages(pdf)
            parts = [(pdf, 1, num_pages)]
            tmp: Optional[tempfile.TemporaryDirectory] = None

            if args.split_large_pdfs and num_pages > args.max_pages_per_request:
                tmp = tempfile.TemporaryDirectory()
                parts = split_pdf_into_parts(pdf, Path(tmp.name), args.max_pages_per_request)

            merged: List[str] = []

            for i, (part_path, pstart, pend) in enumerate(parts, 1):
                doc = process_pdf_with_docai(
                    client,
                    args.project_id,
                    args.location,
                    args.processor_id,
                    part_path,
                )

                raw_text = doc_to_paragraph_text(doc).strip()
                raw_file = out_text_dir / f"{pdf.stem}.p{pstart}-{pend}.raw.txt"
                raw_file.write_text(raw_text, encoding="utf-8")

                final_text = raw_text
                if args.refine == "ollama" and raw_text:
                    final_text = refine_text_ollama(
                        raw_text,
                        args.ollama_model,
                        args.ollama_url,
                        args.ollama_max_chars,
                    )

                out_file = out_text_dir / f"{pdf.stem}.p{pstart}-{pend}.txt"
                out_file.write_text(final_text, encoding="utf-8")

                merged.append(
                    f"\n\n===== {pdf.name} | PART {i}/{len(parts)} | PAGES {pstart}-{pend} =====\n\n"
                    + final_text
                )

            if tmp:
                tmp.cleanup()

            all_text = "\n".join(merged).strip()
            (out_text_dir / f"{pdf.stem}.ALL.txt").write_text(all_text, encoding="utf-8")

        except Exception as e:
            print(f"[ERROR] {pdf.name}: {e}")

    print("DONE. OCR + refine hoàn tất.")
    print("Output:", out_text_dir.resolve())


if __name__ == "__main__":
    cred = os.environ.get("GOOGLE_APPLICATION_CREDENTIALS")
    if not cred:
        print("ERROR: GOOGLE_APPLICATION_CREDENTIALS chưa được set.")
    else:
        main()
