"""Dataset downloader and manager for the CUAD contract dataset.

Handles downloading the CUAD v1 dataset from Zenodo, extracting it,
and selecting a deterministic subset of 50 contracts for processing.
"""

import os
import zipfile
import logging
from pathlib import Path
from typing import List, Tuple

import requests
import pandas as pd
from tqdm import tqdm

logger = logging.getLogger(__name__)

CUAD_URL = "https://zenodo.org/records/4595826/files/CUAD_v1.zip"
SUBSET_SIZE = 50


def download_dataset(data_dir: Path, zip_path: Path) -> None:
    """Download the CUAD v1 dataset from Zenodo.

    Args:
        data_dir: Directory to store extracted data.
        zip_path: Path for the downloaded zip file.
    """
    if zip_path.exists():
        logger.info("Dataset zip already exists at %s, skipping download.", zip_path)
        return

    logger.info("Downloading CUAD v1 dataset from %s ...", CUAD_URL)
    response = requests.get(CUAD_URL, stream=True, timeout=300)
    response.raise_for_status()

    total_size = int(response.headers.get("content-length", 0))
    with open(zip_path, "wb") as f, tqdm(
        total=total_size, unit="B", unit_scale=True, desc="Downloading CUAD"
    ) as pbar:
        for chunk in response.iter_content(chunk_size=1024 * 1024):
            f.write(chunk)
            pbar.update(len(chunk))

    logger.info("Download complete: %s", zip_path)


def extract_dataset(zip_path: Path, data_dir: Path) -> Path:
    """Extract the CUAD zip archive to the data directory.

    Args:
        zip_path: Path to the downloaded zip file.
        data_dir: Directory to extract into.

    Returns:
        Path to the extracted CUAD_v1 directory.
    """
    cuad_dir = data_dir / "CUAD_v1"
    if cuad_dir.exists():
        logger.info("Dataset already extracted at %s", cuad_dir)
        return cuad_dir

    logger.info("Extracting dataset to %s ...", data_dir)
    with zipfile.ZipFile(zip_path, "r") as zf:
        zf.extractall(data_dir)

    logger.info("Extraction complete.")
    return cuad_dir


def _find_pdf_files(cuad_dir: Path) -> List[Path]:
    """Find all PDF files in the CUAD dataset directory."""
    pdf_dir = cuad_dir / "full_contract_pdf"
    if not pdf_dir.exists():
        raise FileNotFoundError(f"PDF directory not found: {pdf_dir}")

    pdf_files = sorted(pdf_dir.rglob("*.pdf"))
    logger.info("Found %d PDF files in dataset.", len(pdf_files))
    return pdf_files


def _find_txt_file(cuad_dir: Path, pdf_name: str) -> Path | None:
    """Find the corresponding TXT file for a given PDF filename."""
    txt_dir = cuad_dir / "full_contract_txt"
    txt_name = pdf_name.replace(".pdf", ".txt")
    txt_path = txt_dir / txt_name
    return txt_path if txt_path.exists() else None


def select_subset(
    cuad_dir: Path, n: int = SUBSET_SIZE
) -> List[Tuple[str, Path, Path | None]]:
    """Select a deterministic subset of contracts.

    Selects every (total // n)-th contract from the alphabetically sorted list
    to ensure diversity across contract types.

    Args:
        cuad_dir: Path to the extracted CUAD_v1 directory.
        n: Number of contracts to select.

    Returns:
        List of (contract_id, pdf_path, txt_path) tuples.
    """
    all_pdfs = _find_pdf_files(cuad_dir)
    if len(all_pdfs) < n:
        logger.warning(
            "Only %d contracts found, using all of them.", len(all_pdfs)
        )
        selected = all_pdfs
    else:
        step = len(all_pdfs) // n
        selected = [all_pdfs[i * step] for i in range(n)]

    result = []
    for pdf_path in selected:
        contract_id = pdf_path.stem
        txt_path = _find_txt_file(cuad_dir, pdf_path.name)
        result.append((contract_id, pdf_path, txt_path))

    logger.info("Selected %d contracts for processing.", len(result))
    return result


def load_ground_truth(cuad_dir: Path) -> pd.DataFrame:
    """Load the CUAD master_clauses.csv ground truth file.

    Args:
        cuad_dir: Path to the extracted CUAD_v1 directory.

    Returns:
        DataFrame with ground truth annotations.
    """
    csv_path = cuad_dir / "master_clauses.csv"
    if not csv_path.exists():
        raise FileNotFoundError(f"Ground truth CSV not found: {csv_path}")

    df = pd.read_csv(csv_path)
    logger.info("Loaded ground truth: %d contracts, %d columns.", len(df), len(df.columns))
    return df


def prepare_data(project_root: Path) -> Tuple[List[Tuple[str, Path, Path | None]], pd.DataFrame, Path]:
    """Full data preparation pipeline: download, extract, select subset.

    Args:
        project_root: Root directory of the project.

    Returns:
        Tuple of (selected_contracts, ground_truth_df, cuad_dir).
    """
    data_dir = project_root / "data"
    data_dir.mkdir(exist_ok=True)

    zip_path = project_root / "CUAD_v1.zip"

    download_dataset(data_dir, zip_path)
    cuad_dir = extract_dataset(zip_path, data_dir)
    contracts = select_subset(cuad_dir)
    ground_truth = load_ground_truth(cuad_dir)

    return contracts, ground_truth, cuad_dir
