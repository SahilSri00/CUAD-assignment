"""PDF text extraction and normalization for legal contracts.

Extracts text from contract PDFs using PyMuPDF (fitz), normalizes the text,
and handles chunking for contracts that exceed LLM context windows.
"""

import re
import logging
from pathlib import Path
from typing import List, Optional

import fitz  # PyMuPDF

logger = logging.getLogger(__name__)

# Approximate token limit for gpt-4o-mini (leave headroom for prompts)
MAX_CHUNK_CHARS = 40_000  # ~10K tokens
CHUNK_OVERLAP_CHARS = 2_000  # ~500 tokens overlap


def extract_text_from_pdf(pdf_path: Path) -> str:
    """Extract raw text from a PDF file using PyMuPDF.

    Args:
        pdf_path: Path to the PDF file.

    Returns:
        Extracted raw text from all pages.

    Raises:
        FileNotFoundError: If the PDF file does not exist.
        RuntimeError: If text extraction fails.
    """
    if not pdf_path.exists():
        raise FileNotFoundError(f"PDF not found: {pdf_path}")

    try:
        doc = fitz.open(str(pdf_path))
        pages = []
        for page in doc:
            text = page.get_text("text")
            if text.strip():
                pages.append(text)
        doc.close()
        return "\n\n".join(pages)
    except Exception as e:
        raise RuntimeError(f"Failed to extract text from {pdf_path}: {e}") from e


def extract_text_from_txt(txt_path: Path) -> str:
    """Read text from a pre-extracted TXT file as fallback.

    Args:
        txt_path: Path to the TXT file.

    Returns:
        Content of the text file.
    """
    encodings = ["utf-8", "latin-1", "cp1252"]
    for enc in encodings:
        try:
            return txt_path.read_text(encoding=enc)
        except (UnicodeDecodeError, UnicodeError):
            continue
    raise RuntimeError(f"Could not decode {txt_path} with any encoding.")


def normalize_text(text: str) -> str:
    """Clean and normalize extracted contract text.

    Performs the following normalizations:
    - Remove confidentiality legends and SEC boilerplate
    - Fix encoding artifacts and special characters
    - Collapse excessive whitespace and blank lines
    - Remove page numbers and headers/footers
    - Normalize quotes and dashes

    Args:
        text: Raw extracted text.

    Returns:
        Cleaned and normalized text.
    """
    # Remove common confidentiality legends
    legends = [
        r"THIS EXHIBIT HAS BEEN REDACTED.*?SECURITIES AND EXCHANGE COMMISSION\.?",
        r"CONFIDENTIAL TREATMENT REQUESTED.*?SECURITIES AND EXCHANGE COMMISSION\.?",
        r"\[\*\s*\*\s*\*\]",
        r"\*\*\*",
    ]
    for pattern in legends:
        text = re.sub(pattern, "[REDACTED]", text, flags=re.IGNORECASE | re.DOTALL)

    # Fix common encoding artifacts
    replacements = {
        "\u2019": "'",  # Right single quote
        "\u2018": "'",  # Left single quote
        "\u201c": '"',  # Left double quote
        "\u201d": '"',  # Right double quote
        "\u2013": "-",  # En dash
        "\u2014": "--", # Em dash
        "\u2022": "*",  # Bullet
        "\u25aa": "*",  # Black small square
        "\xa0": " ",    # Non-breaking space
        "\u00a7": "Section ",  # Section symbol
    }
    for old, new in replacements.items():
        text = text.replace(old, new)

    # Remove standalone page numbers (e.g., lines that are just "12" or "Page 12")
    text = re.sub(r"\n\s*(?:Page\s+)?\d{1,3}\s*(?:of\s+\d{1,3})?\s*\n", "\n", text, flags=re.IGNORECASE)

    # Collapse multiple blank lines into one
    text = re.sub(r"\n{3,}", "\n\n", text)

    # Collapse multiple spaces into one
    text = re.sub(r"[ \t]{2,}", " ", text)

    # Strip leading/trailing whitespace from each line
    lines = [line.strip() for line in text.split("\n")]
    text = "\n".join(lines)

    return text.strip()


def chunk_text(text: str, max_chars: int = MAX_CHUNK_CHARS, overlap: int = CHUNK_OVERLAP_CHARS) -> List[str]:
    """Split text into overlapping chunks for LLM processing.

    Attempts to split at paragraph boundaries to preserve clause integrity.

    Args:
        text: The full normalized text.
        max_chars: Maximum characters per chunk.
        overlap: Number of characters to overlap between chunks.

    Returns:
        List of text chunks.
    """
    if len(text) <= max_chars:
        return [text]

    chunks = []
    paragraphs = text.split("\n\n")
    current_chunk = ""

    for para in paragraphs:
        if len(current_chunk) + len(para) + 2 <= max_chars:
            current_chunk += ("\n\n" + para) if current_chunk else para
        else:
            if current_chunk:
                chunks.append(current_chunk)
            # Start new chunk with overlap from end of previous chunk
            if chunks and overlap > 0:
                prev = chunks[-1]
                overlap_text = prev[-overlap:] if len(prev) > overlap else prev
                current_chunk = overlap_text + "\n\n" + para
            else:
                current_chunk = para

            # Handle single paragraphs longer than max_chars
            if len(current_chunk) > max_chars:
                # Force split at sentence boundaries
                sentences = re.split(r'(?<=[.!?])\s+', current_chunk)
                current_chunk = ""
                for sent in sentences:
                    if len(current_chunk) + len(sent) + 1 <= max_chars:
                        current_chunk += (" " + sent) if current_chunk else sent
                    else:
                        if current_chunk:
                            chunks.append(current_chunk)
                        current_chunk = sent

    if current_chunk:
        chunks.append(current_chunk)

    logger.info("Split text (%d chars) into %d chunks.", len(text), len(chunks))
    return chunks


def parse_contract(
    contract_id: str, pdf_path: Path, txt_path: Optional[Path] = None
) -> dict:
    """Full parsing pipeline for a single contract.

    Extracts text from PDF (with TXT fallback), normalizes it,
    and prepares chunks if needed.

    Args:
        contract_id: Identifier for the contract.
        pdf_path: Path to the PDF file.
        txt_path: Optional path to the pre-extracted TXT file.

    Returns:
        Dictionary with keys:
        - 'contract_id': The contract identifier.
        - 'full_text': The full normalized text.
        - 'chunks': List of text chunks for LLM processing.
        - 'char_count': Total character count.
        - 'source': Whether text came from 'pdf' or 'txt'.
    """
    # Try PDF first, fallback to TXT
    source = "pdf"
    try:
        raw_text = extract_text_from_pdf(pdf_path)
        if not raw_text.strip():
            raise RuntimeError("PDF extraction returned empty text.")
    except Exception as e:
        logger.warning("PDF extraction failed for %s: %s. Trying TXT fallback.", contract_id, e)
        if txt_path and txt_path.exists():
            raw_text = extract_text_from_txt(txt_path)
            source = "txt"
        else:
            raise RuntimeError(f"No text source available for {contract_id}") from e

    normalized = normalize_text(raw_text)
    chunks = chunk_text(normalized)

    return {
        "contract_id": contract_id,
        "full_text": normalized,
        "chunks": chunks,
        "char_count": len(normalized),
        "source": source,
    }
