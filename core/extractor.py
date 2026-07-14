"""LLM-powered clause extraction and contract summarization.

Uses OpenAI's GPT models to extract key legal clauses (Termination,
Confidentiality, Liability) and generate concise contract summaries.
Implements few-shot prompting for improved extraction quality.
"""

import json
import time
import logging
from typing import Dict, List, Optional

import openai

logger = logging.getLogger(__name__)

DEFAULT_MODEL = "gpt-4o-mini"
MAX_RETRIES = 3
RETRY_DELAY = 2  # seconds

# --- Few-Shot Examples for Clause Extraction (Bonus) ---

FEW_SHOT_EXAMPLES = [
    {
        "contract_snippet": (
            "12. TERMINATION. Either party may terminate this Agreement upon "
            "thirty (30) days' prior written notice to the other party. This "
            "Agreement shall automatically terminate if either party files for "
            "bankruptcy or becomes insolvent. Upon termination, all licenses "
            "granted herein shall cease immediately.\n\n"
            "13. CONFIDENTIAL INFORMATION. Each party agrees that all information "
            "disclosed by the other party that is designated as confidential or "
            "that reasonably should be understood to be confidential shall not be "
            "disclosed to any third party for a period of five (5) years following "
            "the termination of this Agreement.\n\n"
            "14. LIMITATION OF LIABILITY. IN NO EVENT SHALL EITHER PARTY BE LIABLE "
            "TO THE OTHER FOR ANY INDIRECT, INCIDENTAL, SPECIAL, OR CONSEQUENTIAL "
            "DAMAGES. THE TOTAL LIABILITY OF EITHER PARTY SHALL NOT EXCEED THE "
            "TOTAL FEES PAID UNDER THIS AGREEMENT DURING THE TWELVE (12) MONTHS "
            "PRECEDING THE CLAIM."
        ),
        "extraction": {
            "termination_clause": (
                "Either party may terminate this Agreement upon thirty (30) days' "
                "prior written notice to the other party. This Agreement shall "
                "automatically terminate if either party files for bankruptcy or "
                "becomes insolvent. Upon termination, all licenses granted herein "
                "shall cease immediately."
            ),
            "confidentiality_clause": (
                "Each party agrees that all information disclosed by the other party "
                "that is designated as confidential or that reasonably should be "
                "understood to be confidential shall not be disclosed to any third "
                "party for a period of five (5) years following the termination of "
                "this Agreement."
            ),
            "liability_clause": (
                "IN NO EVENT SHALL EITHER PARTY BE LIABLE TO THE OTHER FOR ANY "
                "INDIRECT, INCIDENTAL, SPECIAL, OR CONSEQUENTIAL DAMAGES. THE TOTAL "
                "LIABILITY OF EITHER PARTY SHALL NOT EXCEED THE TOTAL FEES PAID "
                "UNDER THIS AGREEMENT DURING THE TWELVE (12) MONTHS PRECEDING THE CLAIM."
            ),
        },
    },
    {
        "contract_snippet": (
            "8. Term and Termination. The initial term of this Agreement shall be "
            "two (2) years from the Effective Date. Either party may terminate "
            "this Agreement for cause if the other party materially breaches any "
            "provision and fails to cure such breach within sixty (60) days after "
            "written notice thereof.\n\n"
            "9. Limitation of Liability. Except for breaches of confidentiality "
            "obligations, neither party's aggregate liability under this Agreement "
            "shall exceed one million dollars ($1,000,000)."
        ),
        "extraction": {
            "termination_clause": (
                "The initial term of this Agreement shall be two (2) years from the "
                "Effective Date. Either party may terminate this Agreement for cause "
                "if the other party materially breaches any provision and fails to "
                "cure such breach within sixty (60) days after written notice thereof."
            ),
            "confidentiality_clause": "Not found",
            "liability_clause": (
                "Except for breaches of confidentiality obligations, neither party's "
                "aggregate liability under this Agreement shall exceed one million "
                "dollars ($1,000,000)."
            ),
        },
    },
]


def _build_extraction_prompt(contract_text: str) -> List[Dict[str, str]]:
    """Build the messages for clause extraction with few-shot examples.

    Args:
        contract_text: The normalized contract text.

    Returns:
        List of message dicts for the OpenAI API.
    """
    system_msg = (
        "You are an expert legal contract analyst. Your task is to carefully read "
        "commercial legal contracts and extract specific clauses verbatim from the text.\n\n"
        "For each contract, extract these three types of clauses:\n"
        "1. **Termination conditions**: Any clauses describing how, when, or under what "
        "circumstances the agreement can be terminated, including notice periods and "
        "termination for cause/convenience.\n"
        "2. **Confidentiality clauses**: Any clauses related to non-disclosure obligations, "
        "protection of confidential or proprietary information, trade secrets, and the "
        "duration of confidentiality obligations.\n"
        "3. **Liability clauses**: Any clauses describing limitations on liability, caps "
        "on damages, indemnification obligations, uncapped liability provisions, or "
        "exclusions of consequential damages.\n\n"
        "IMPORTANT RULES:\n"
        "- Extract the EXACT text from the contract. Do not paraphrase or summarize.\n"
        "- If multiple relevant clauses exist for one type, combine them with a newline separator.\n"
        "- If a clause type is not present in the contract, use \"Not found\".\n"
        "- Return a valid JSON object with exactly these keys: "
        "\"termination_clause\", \"confidentiality_clause\", \"liability_clause\".\n"
    )

    messages = [{"role": "system", "content": system_msg}]

    # Add few-shot examples
    for example in FEW_SHOT_EXAMPLES:
        messages.append({
            "role": "user",
            "content": f"Extract the key clauses from this contract:\n\n{example['contract_snippet']}"
        })
        messages.append({
            "role": "assistant",
            "content": json.dumps(example["extraction"], indent=2)
        })

    # Add the actual contract
    messages.append({
        "role": "user",
        "content": f"Extract the key clauses from this contract:\n\n{contract_text}"
    })

    return messages


def _build_summary_prompt(contract_text: str) -> List[Dict[str, str]]:
    """Build the messages for contract summarization.

    Args:
        contract_text: The normalized contract text.

    Returns:
        List of message dicts for the OpenAI API.
    """
    system_msg = (
        "You are an expert legal analyst. Your task is to generate a concise summary "
        "of a commercial legal contract.\n\n"
        "The summary MUST:\n"
        "- Be exactly 100 to 150 words long\n"
        "- Cover the PURPOSE of the agreement\n"
        "- Identify the KEY OBLIGATIONS of each party\n"
        "- Highlight any NOTABLE RISKS or PENALTIES\n\n"
        "Return a valid JSON object with a single key \"summary\" containing the summary text.\n"
        "Do not include any other keys or text outside the JSON."
    )

    messages = [
        {"role": "system", "content": system_msg},
        {
            "role": "user",
            "content": f"Summarize this contract in 100-150 words:\n\n{contract_text}",
        },
    ]

    return messages


def _build_merge_prompt(chunk_results: List[Dict]) -> List[Dict[str, str]]:
    """Build a prompt to merge clause extractions from multiple chunks.

    Args:
        chunk_results: List of extraction results from individual chunks.

    Returns:
        List of message dicts for the OpenAI API.
    """
    system_msg = (
        "You are an expert legal analyst. You have been given clause extractions from "
        "multiple chunks of the same legal contract. Your task is to merge and deduplicate "
        "these extractions into a single coherent result.\n\n"
        "Rules:\n"
        "- Combine related clauses from different chunks, removing exact duplicates.\n"
        "- If all chunks say \"Not found\" for a clause type, output \"Not found\".\n"
        "- Preserve the original clause text as much as possible.\n"
        "- Return a valid JSON object with keys: \"termination_clause\", "
        "\"confidentiality_clause\", \"liability_clause\".\n"
    )

    chunks_text = "\n\n---\n\n".join(
        f"Chunk {i+1}:\n{json.dumps(r, indent=2)}" for i, r in enumerate(chunk_results)
    )

    messages = [
        {"role": "system", "content": system_msg},
        {
            "role": "user",
            "content": f"Merge these clause extractions:\n\n{chunks_text}",
        },
    ]

    return messages


def _call_llm(
    messages: List[Dict[str, str]],
    model: str = DEFAULT_MODEL,
    temperature: float = 0.0,
) -> Dict:
    """Call the OpenAI API with retry logic.

    Args:
        messages: List of message dicts.
        model: Model name to use.
        temperature: Sampling temperature (0 for deterministic).

    Returns:
        Parsed JSON response as a dictionary.
    """
    client = openai.OpenAI()

    for attempt in range(MAX_RETRIES):
        try:
            response = client.chat.completions.create(
                model=model,
                messages=messages,
                temperature=temperature,
                response_format={"type": "json_object"},
            )
            content = response.choices[0].message.content
            return json.loads(content)

        except openai.RateLimitError:
            wait = RETRY_DELAY * (2 ** attempt)
            logger.warning("Rate limit hit. Retrying in %ds... (attempt %d/%d)", wait, attempt + 1, MAX_RETRIES)
            time.sleep(wait)

        except openai.APIError as e:
            logger.error("API error: %s", e)
            if attempt < MAX_RETRIES - 1:
                time.sleep(RETRY_DELAY)
            else:
                raise

        except json.JSONDecodeError as e:
            logger.error("Failed to parse LLM response as JSON: %s", e)
            if attempt < MAX_RETRIES - 1:
                time.sleep(RETRY_DELAY)
            else:
                return {
                    "termination_clause": "Extraction failed",
                    "confidentiality_clause": "Extraction failed",
                    "liability_clause": "Extraction failed",
                }

    return {
        "termination_clause": "Extraction failed",
        "confidentiality_clause": "Extraction failed",
        "liability_clause": "Extraction failed",
    }


def extract_clauses(
    contract_text: str,
    chunks: List[str],
    model: str = DEFAULT_MODEL,
) -> Dict[str, str]:
    """Extract key clauses from a contract using an LLM.

    If the contract fits in a single chunk, it's processed directly.
    Otherwise, each chunk is processed separately and results are merged.

    Args:
        contract_text: Full normalized contract text.
        chunks: List of text chunks (may be a single chunk).
        model: OpenAI model to use.

    Returns:
        Dictionary with keys: termination_clause, confidentiality_clause,
        liability_clause.
    """
    if len(chunks) == 1:
        messages = _build_extraction_prompt(chunks[0])
        return _call_llm(messages, model=model)

    # Process each chunk separately
    chunk_results = []
    for i, chunk in enumerate(chunks):
        logger.debug("Processing chunk %d/%d", i + 1, len(chunks))
        messages = _build_extraction_prompt(chunk)
        result = _call_llm(messages, model=model)
        chunk_results.append(result)

    # Merge results from all chunks
    merge_messages = _build_merge_prompt(chunk_results)
    merged = _call_llm(merge_messages, model=model)
    return merged


def summarize_contract(
    contract_text: str,
    chunks: List[str],
    model: str = DEFAULT_MODEL,
) -> str:
    """Generate a 100-150 word summary of a contract.

    For long contracts, uses the first and last chunks to capture
    the agreement overview and key terms.

    Args:
        contract_text: Full normalized contract text.
        chunks: List of text chunks.
        model: OpenAI model to use.

    Returns:
        Contract summary string (100-150 words).
    """
    # For summarization, use full text if short, otherwise first+last chunks
    if len(chunks) == 1:
        text_for_summary = chunks[0]
    else:
        # Use first chunk (usually has agreement overview) + last chunk (usually has
        # liability, termination, general provisions)
        text_for_summary = chunks[0] + "\n\n[...]\n\n" + chunks[-1]

    messages = _build_summary_prompt(text_for_summary)
    result = _call_llm(messages, model=model)

    summary = result.get("summary", "Summary generation failed.")

    # Validate word count
    word_count = len(summary.split())
    if word_count < 80 or word_count > 170:
        logger.warning(
            "Summary word count (%d) outside target range. Regenerating...",
            word_count,
        )
        # Retry with stricter instruction
        messages[-1]["content"] += (
            f"\n\nIMPORTANT: Your previous summary was {word_count} words. "
            "It MUST be between 100 and 150 words. Adjust accordingly."
        )
        result = _call_llm(messages, model=model)
        summary = result.get("summary", summary)

    return summary


def process_contract(
    contract_id: str,
    full_text: str,
    chunks: List[str],
    model: str = DEFAULT_MODEL,
) -> Dict[str, str]:
    """Full extraction + summarization pipeline for one contract.

    Args:
        contract_id: Identifier for the contract.
        full_text: Full normalized contract text.
        chunks: List of text chunks.
        model: OpenAI model to use.

    Returns:
        Dictionary with keys: contract_id, summary, termination_clause,
        confidentiality_clause, liability_clause.
    """
    logger.info("Processing contract: %s", contract_id)

    # Extract clauses
    clauses = extract_clauses(full_text, chunks, model=model)

    # Generate summary
    summary = summarize_contract(full_text, chunks, model=model)

    return {
        "contract_id": contract_id,
        "summary": summary,
        "termination_clause": clauses.get("termination_clause", "Not found"),
        "confidentiality_clause": clauses.get("confidentiality_clause", "Not found"),
        "liability_clause": clauses.get("liability_clause", "Not found"),
    }
