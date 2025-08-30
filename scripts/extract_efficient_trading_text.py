from __future__ import annotations

import argparse
from pathlib import Path
from typing import Optional


def extract_with_pdfminer(input_path: Path) -> Optional[str]:
    try:
        from pdfminer.high_level import extract_text  # type: ignore
    except Exception:
        return None
    try:
        return extract_text(str(input_path))
    except Exception:
        return None


def extract_with_pypdf(input_path: Path) -> Optional[str]:
    try:
        from pypdf import PdfReader  # type: ignore
    except Exception:
        return None
    try:
        reader = PdfReader(str(input_path))
        pieces = []
        for page in reader.pages:
            txt = page.extract_text() or ""
            pieces.append(txt)
        return "\n".join(pieces)
    except Exception:
        return None


def extract_with_pymupdf(input_path: Path) -> Optional[str]:
    try:
        import fitz  # PyMuPDF
    except Exception:
        return None
    try:
        text_parts: list[str] = []
        with fitz.open(str(input_path)) as doc:
            for page in doc:
                # 'text' uses a fast text extraction; 'textpage' or 'blocks' variants can be used if needed
                txt = page.get_text("text")
                if not txt:
                    # Try a different extractor variant
                    txt = page.get_text("blocks")
                    if isinstance(txt, list):
                        txt = "\n".join(
                            block[4]
                            for block in txt
                            if len(block) > 4 and isinstance(block[4], str)
                        )
                text_parts.append(txt or "")
        combined = "\n".join(text_parts)
        return combined
    except Exception:
        return None


def extract_with_ocr(input_path: Path) -> Optional[str]:
    """Rasterize pages and OCR with Tesseract. Requires Pillow and pytesseract and system tesseract binary."""
    try:
        import fitz  # PyMuPDF
        from PIL import Image
        import pytesseract
    except Exception:
        return None
    try:
        text_parts: list[str] = []
        # Render at a moderate DPI for OCR quality/speed balance
        zoom = 2.0  # ~144 DPI if base is 72
        mat = fitz.Matrix(zoom, zoom)
        with fitz.open(str(input_path)) as doc:
            for page in doc:
                pix = page.get_pixmap(matrix=mat, alpha=False)
                img = Image.frombytes("RGB", [pix.width, pix.height], pix.samples)
                txt = pytesseract.image_to_string(img)
                text_parts.append(txt or "")
        combined = "\n".join(text_parts)
        return combined
    except Exception:
        return None


def main() -> None:
    default_input = Path(
        "/Users/jibranbohra/Downloads/price-impact/literature/Efficient Trading with Price Impact.pdf"
    )
    default_output = default_input.with_suffix(".txt")

    parser = argparse.ArgumentParser(
        description="Extract text from a PDF into a .txt file."
    )
    parser.add_argument(
        "--input",
        type=Path,
        default=default_input,
        help="Absolute path to the input PDF",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=default_output,
        help="Absolute path to the output .txt",
    )
    args = parser.parse_args()

    input_path: Path = args.input
    output_path: Path = args.output

    if not input_path.exists():
        raise FileNotFoundError(f"Input PDF not found: {input_path}")

    # Try pdfminer first (better text fidelity), then fall back to pypdf, then PyMuPDF, then OCR.
    text = extract_with_pdfminer(input_path)
    if text is None or not text.strip():
        text = extract_with_pypdf(input_path)
    if text is None or not text.strip():
        text = extract_with_pymupdf(input_path)
    if text is None or not text.strip():
        text = extract_with_ocr(input_path)

    if text is None or not text.strip():
        raise RuntimeError(
            "Failed to extract text. Please ensure either 'pdfminer.six' or 'pypdf' is installed."
        )

    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(text, encoding="utf-8")
    print(f"Wrote text to: {output_path}")


if __name__ == "__main__":
    main()
