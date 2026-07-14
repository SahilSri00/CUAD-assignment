"""Evaluation module for comparing LLM extractions against CUAD ground truth.

Computes text overlap metrics between LLM-extracted clauses and the
human-annotated ground truth from the CUAD master_clauses.csv.
"""

import logging
from typing import Dict, List, Set, Tuple

import pandas as pd

logger = logging.getLogger(__name__)

# Mapping from our clause types to CUAD CSV column names
CLAUSE_TO_CUAD_COLUMNS = {
    "termination_clause": ["Termination For Convenience"],
    "liability_clause": ["Uncapped Liability", "Cap On Liability"],
}


def _tokenize(text: str) -> Set[str]:
    """Simple whitespace tokenization with lowercasing."""
    if not text or text in ("Not found", "Extraction failed"):
        return set()
    return set(text.lower().split())


def _jaccard_similarity(set_a: Set[str], set_b: Set[str]) -> float:
    """Compute Jaccard similarity between two token sets."""
    if not set_a and not set_b:
        return 1.0  # Both empty = perfect match
    if not set_a or not set_b:
        return 0.0
    intersection = set_a & set_b
    union = set_a | set_b
    return len(intersection) / len(union)


def _get_ground_truth_text(
    row: pd.Series, cuad_columns: List[str]
) -> str:
    """Extract and combine ground truth text from CUAD columns."""
    parts = []
    for col in cuad_columns:
        if col in row.index:
            val = row[col]
            if pd.notna(val) and str(val).strip():
                parts.append(str(val).strip())
    return " ".join(parts)


def _detection_match(
    extracted: str, ground_truth: str
) -> Tuple[bool, bool, bool]:
    """Check detection accuracy.

    Returns:
        Tuple of (gt_has_clause, extracted_has_clause, is_correct).
    """
    gt_has = bool(ground_truth.strip()) and ground_truth.strip().lower() not in ("no", "n/a", "")
    ex_has = bool(extracted.strip()) and extracted.strip() not in ("Not found", "Extraction failed")
    return gt_has, ex_has, gt_has == ex_has


def evaluate_extractions(
    results: List[Dict[str, str]],
    ground_truth: pd.DataFrame,
) -> Dict:
    """Evaluate LLM clause extractions against CUAD ground truth.

    Args:
        results: List of extraction result dicts from the pipeline.
        ground_truth: CUAD master_clauses DataFrame.

    Returns:
        Dictionary with evaluation metrics.
    """
    metrics = {
        "total_contracts": len(results),
        "clause_metrics": {},
        "overall_detection_accuracy": 0.0,
    }

    total_correct = 0
    total_comparisons = 0

    for clause_type, cuad_cols in CLAUSE_TO_CUAD_COLUMNS.items():
        similarities = []
        detection_correct = 0
        detection_total = 0

        for result in results:
            contract_id = result["contract_id"]
            extracted_text = result.get(clause_type, "Not found")

            # Find matching row in ground truth
            gt_rows = ground_truth[
                ground_truth["Filename"].str.contains(
                    contract_id[:50], case=False, na=False, regex=False
                )
            ]

            if gt_rows.empty:
                logger.debug("No ground truth found for %s", contract_id)
                continue

            gt_row = gt_rows.iloc[0]
            gt_text = _get_ground_truth_text(gt_row, cuad_cols)

            # Compute Jaccard similarity
            extracted_tokens = _tokenize(extracted_text)
            gt_tokens = _tokenize(gt_text)
            sim = _jaccard_similarity(extracted_tokens, gt_tokens)
            similarities.append(sim)

            # Detection accuracy
            _, _, correct = _detection_match(extracted_text, gt_text)
            if correct:
                detection_correct += 1
            detection_total += 1
            total_correct += int(correct)
            total_comparisons += 1

        avg_similarity = sum(similarities) / len(similarities) if similarities else 0.0
        det_accuracy = detection_correct / detection_total if detection_total > 0 else 0.0

        metrics["clause_metrics"][clause_type] = {
            "avg_jaccard_similarity": round(avg_similarity, 4),
            "detection_accuracy": round(det_accuracy, 4),
            "contracts_evaluated": detection_total,
        }

    metrics["overall_detection_accuracy"] = (
        round(total_correct / total_comparisons, 4) if total_comparisons > 0 else 0.0
    )

    return metrics


def print_evaluation_report(metrics: Dict) -> None:
    """Print a formatted evaluation report to the console.

    Args:
        metrics: Evaluation metrics dictionary.
    """
    print("\n" + "=" * 70)
    print("  EVALUATION REPORT: LLM Extraction vs CUAD Ground Truth")
    print("=" * 70)
    print(f"\n  Contracts evaluated: {metrics['total_contracts']}")
    print(f"  Overall detection accuracy: {metrics['overall_detection_accuracy']:.1%}")
    print("\n" + "-" * 70)
    print(f"  {'Clause Type':<30} {'Jaccard Sim':>15} {'Detection Acc':>15}")
    print("-" * 70)

    for clause_type, m in metrics["clause_metrics"].items():
        label = clause_type.replace("_", " ").title()
        print(
            f"  {label:<30} {m['avg_jaccard_similarity']:>14.4f} {m['detection_accuracy']:>14.1%}"
        )

    print("-" * 70)
    print("\n  Note: Confidentiality clauses are not evaluated against ground truth")
    print("  as CUAD does not contain a dedicated Confidentiality category.")
    print("=" * 70 + "\n")
