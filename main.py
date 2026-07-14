"""Legal Contract Analysis Pipeline — Main Entrypoint.

A pipeline that uses Large Language Models to analyze legal contracts from
the CUAD dataset, extract key clauses, and generate summaries.

Usage:
    python main.py                          # Run full pipeline
    python main.py --search "query text"    # Semantic search over clauses
    python main.py --contracts 20           # Process 20 contracts instead of 50
    python main.py --model gpt-4o-mini      # Use a different model
"""

import argparse
import json
import logging
import sys
from pathlib import Path

import pandas as pd
from tqdm import tqdm

from core.downloader import prepare_data
from core.parser import parse_contract
from core.extractor import process_contract
from core.search import build_search_index, save_index, load_index, search
from core.evaluator import evaluate_extractions, print_evaluation_report

# Project root
PROJECT_ROOT = Path(__file__).resolve().parent
OUTPUT_DIR = PROJECT_ROOT / "output"


def setup_logging(verbose: bool = False) -> None:
    """Configure logging for the pipeline."""
    level = logging.DEBUG if verbose else logging.INFO
    logging.basicConfig(
        level=level,
        format="%(asctime)s | %(levelname)-8s | %(name)s | %(message)s",
        datefmt="%H:%M:%S",
    )


def run_pipeline(args: argparse.Namespace) -> None:
    """Execute the full contract analysis pipeline.

    Steps:
        1. Download and extract CUAD dataset
        2. Parse contract PDFs
        3. Extract clauses and generate summaries using LLM
        4. Evaluate against CUAD ground truth
        5. Build semantic search index (bonus)
        6. Save results to CSV and JSON
    """
    logger = logging.getLogger("pipeline")

    # --- Step 1: Data Preparation ---
    logger.info("=" * 60)
    logger.info("STEP 1/6: Preparing dataset...")
    logger.info("=" * 60)
    contracts, ground_truth, cuad_dir = prepare_data(PROJECT_ROOT)

    # Limit to requested number of contracts
    if args.contracts < len(contracts):
        contracts = contracts[: args.contracts]
        logger.info("Limited to %d contracts as requested.", args.contracts)

    # --- Step 2: Parse Contracts ---
    logger.info("=" * 60)
    logger.info("STEP 2/6: Parsing %d contracts...", len(contracts))
    logger.info("=" * 60)
    parsed_contracts = []
    for contract_id, pdf_path, txt_path in tqdm(contracts, desc="Parsing PDFs"):
        try:
            parsed = parse_contract(contract_id, pdf_path, txt_path)
            parsed_contracts.append(parsed)
            logger.debug(
                "Parsed %s: %d chars, %d chunks, source=%s",
                contract_id,
                parsed["char_count"],
                len(parsed["chunks"]),
                parsed["source"],
            )
        except Exception as e:
            logger.error("Failed to parse %s: %s", contract_id, e)

    logger.info(
        "Successfully parsed %d/%d contracts.",
        len(parsed_contracts),
        len(contracts),
    )

    # --- Step 3: LLM Extraction & Summarization ---
    logger.info("=" * 60)
    logger.info("STEP 3/6: Running LLM extraction & summarization...")
    logger.info("=" * 60)
    results = []
    for parsed in tqdm(parsed_contracts, desc="LLM Processing"):
        try:
            result = process_contract(
                contract_id=parsed["contract_id"],
                full_text=parsed["full_text"],
                chunks=parsed["chunks"],
                model=args.model,
            )
            results.append(result)
        except Exception as e:
            logger.error("LLM processing failed for %s: %s", parsed["contract_id"], e)

    logger.info("Successfully processed %d contracts.", len(results))

    # --- Step 4: Save Results ---
    logger.info("=" * 60)
    logger.info("STEP 4/6: Saving results...")
    logger.info("=" * 60)
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    # Save as CSV
    df = pd.DataFrame(results)
    csv_path = OUTPUT_DIR / "contracts_analysis.csv"
    df.to_csv(csv_path, index=False, encoding="utf-8")
    logger.info("Saved CSV: %s", csv_path)

    # Save as JSON
    json_path = OUTPUT_DIR / "contracts_analysis.json"
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2, ensure_ascii=False)
    logger.info("Saved JSON: %s", json_path)

    # --- Step 5: Evaluation ---
    logger.info("=" * 60)
    logger.info("STEP 5/6: Evaluating against CUAD ground truth...")
    logger.info("=" * 60)
    try:
        metrics = evaluate_extractions(results, ground_truth)
        print_evaluation_report(metrics)

        # Save metrics
        with open(OUTPUT_DIR / "evaluation_metrics.json", "w", encoding="utf-8") as f:
            json.dump(metrics, f, indent=2)
    except Exception as e:
        logger.error("Evaluation failed: %s", e)

    # --- Step 6: Build Search Index ---
    logger.info("=" * 60)
    logger.info("STEP 6/6: Building semantic search index...")
    logger.info("=" * 60)
    try:
        metadata, tfidf_matrix, idf, vocab = build_search_index(results)
        save_index(metadata, tfidf_matrix, idf, vocab, OUTPUT_DIR)
        logger.info("Search index saved. Use --search to query.")
    except Exception as e:
        logger.error("Search index creation failed: %s", e)

    # --- Summary ---
    print("\n" + "=" * 60)
    print("  PIPELINE COMPLETE")
    print("=" * 60)
    print(f"  Contracts processed:  {len(results)}")
    print(f"  Output CSV:           {csv_path}")
    print(f"  Output JSON:          {json_path}")
    print(f"  Search index:         {OUTPUT_DIR}")
    print("=" * 60 + "\n")


def run_search(args: argparse.Namespace) -> None:
    """Run semantic search over previously extracted clauses."""
    logger = logging.getLogger("search")

    try:
        metadata, tfidf_matrix, idf, vocab = load_index(OUTPUT_DIR)
    except FileNotFoundError:
        logger.error(
            "Search index not found. Run the pipeline first: python main.py"
        )
        sys.exit(1)

    query = args.search
    logger.info("Searching for: '%s'", query)

    results = search(query, metadata, tfidf_matrix, idf, vocab, top_k=5)

    print(f"\n{'=' * 70}")
    print(f"  Search Results for: \"{query}\"")
    print(f"{'=' * 70}\n")

    if not results:
        print("  No results found.\n")
        return

    for i, result in enumerate(results, 1):
        print(f"  [{i}] Contract: {result['contract_id']}")
        print(f"      Type:     {result['clause_type'].replace('_', ' ').title()}")
        print(f"      Score:    {result['similarity']:.4f}")
        print(f"      Snippet:  {result['text_snippet']}")
        print()

    print(f"{'=' * 70}\n")


def main() -> None:
    """Main entry point with argument parsing."""
    parser = argparse.ArgumentParser(
        description="Legal Contract Analysis Pipeline using LLMs",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=(
            "Examples:\n"
            "  python main.py                                  # Run full pipeline\n"
            "  python main.py --contracts 10                   # Process 10 contracts\n"
            "  python main.py --search 'liability cap'         # Semantic search\n"
            "  python main.py --model gpt-4o --verbose         # Use GPT-4o with debug logs\n"
        ),
    )

    parser.add_argument(
        "--search",
        type=str,
        default=None,
        help="Run semantic search with the given query instead of the full pipeline.",
    )
    parser.add_argument(
        "--contracts",
        type=int,
        default=50,
        help="Number of contracts to process (default: 50).",
    )
    parser.add_argument(
        "--model",
        type=str,
        default="llama-3.3-70b-versatile",
        help="LLM model to use (default: llama-3.3-70b-versatile for Groq).",
    )
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Enable verbose/debug logging.",
    )

    args = parser.parse_args()
    setup_logging(verbose=args.verbose)

    if args.search:
        run_search(args)
    else:
        run_pipeline(args)


if __name__ == "__main__":
    main()
